using System.Collections.ObjectModel;
using System.Text.Json;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;

namespace Ramvi.Desktop.ViewModels;

/// <summary>
/// Экран чата (UI-02): список диалогов, поток ответа, Stop/Regenerate, копирование,
/// очистка, выбор модели, thinking и перетаскивание текстовых файлов.
/// Не зависит от окна: проверяется тестами без Avalonia.
/// </summary>
public sealed class MainViewModel : ObservableObject, IDisposable
{
    private readonly IChatApi _api;
    private readonly Func<string, Task> _copy;
    private ChatItemViewModel? _selectedChat;
    private ModelInfo? _selectedModel;
    private string _input = "";
    private string _status;
    private bool _isError;
    private bool _isGenerating;
    private bool _ignoreModelChange;
    private Limits _limits = new(32_000, 5, 128 * 1024, 256 * 1024, 255);
    private CancellationTokenSource? _streamCancel;
    private string? _generatingChatId;

    public MainViewModel(IChatApi api, Func<string, Task>? copy = null)
    {
        _api = api;
        _copy = copy ?? (_ => Task.CompletedTask);
        _status = L["status.connecting"];
        NewChatCommand = new Command(NewChatAsync, () => !IsGenerating);
        SendCommand = new Command(SendAsync, CanSend);
        StopCommand = new Command(StopAsync, () => IsGenerating);
        RegenerateCommand = new Command(RegenerateAsync, CanRegenerate);
        ClearCommand = new Command(ClearAsync, () => SelectedChat is not null && !IsGenerating && Messages.Count > 0);
        DeleteChatCommand = new Command(
            p => DeleteChatAsync(p as ChatItemViewModel),
            p => !IsGenerating && (p is ChatItemViewModel || SelectedChat is not null));
        CopyCommand = new Command(p => CopyAsync(p as MessageViewModel));
        RemoveAttachmentCommand = new Command(p =>
        {
            if (p is AttachmentViewModel item)
            {
                Attachments.Remove(item);
                RefreshCommands();
            }

            return Task.CompletedTask;
        });
    }

    private static Localizer L => Localizer.Current;

    public void Dispose() => _streamCancel?.Cancel();

    public ObservableCollection<ChatItemViewModel> Chats { get; } = [];

    public ObservableCollection<ModelInfo> Models { get; } = [];

    public ObservableCollection<MessageViewModel> Messages { get; } = [];

    public ObservableCollection<AttachmentViewModel> Attachments { get; } = [];

    public Command NewChatCommand { get; }

    public Command SendCommand { get; }

    public Command StopCommand { get; }

    public Command RegenerateCommand { get; }

    public Command ClearCommand { get; }

    public Command DeleteChatCommand { get; }

    public Command CopyCommand { get; }

    public Command RemoveAttachmentCommand { get; }

    public Limits Limits => _limits;

    public bool HasChats => Chats.Count > 0;

    public ChatItemViewModel? SelectedChat
    {
        get => _selectedChat;
        set
        {
            if (Set(ref _selectedChat, value))
            {
                _ = LoadChatAsync(value);
            }
        }
    }

    public ModelInfo? SelectedModel
    {
        get => _selectedModel;
        set
        {
            if (Set(ref _selectedModel, value) && !_ignoreModelChange && value is not null)
            {
                _ = ChangeModelAsync(value);
            }
        }
    }

    public string Input
    {
        get => _input;
        set
        {
            if (Set(ref _input, value))
            {
                SendCommand.Refresh();
            }
        }
    }

    public string Status
    {
        get => _status;
        private set => Set(ref _status, value);
    }

    public bool IsError
    {
        get => _isError;
        private set => Set(ref _isError, value);
    }

    public bool IsGenerating
    {
        get => _isGenerating;
        private set
        {
            if (Set(ref _isGenerating, value))
            {
                Raise(nameof(IsIdle));
                RefreshCommands();
            }
        }
    }

    public bool IsIdle => !IsGenerating;

    /// <summary>Загрузить лимиты, модели и диалоги; открыть последний диалог.</summary>
    public async Task InitializeAsync()
    {
        try
        {
            var status = await _api.StatusAsync().ConfigureAwait(true);
            _limits = status.Limits;
            Raise(nameof(Limits));
            foreach (var model in await _api.ModelsAsync().ConfigureAwait(true))
            {
                Models.Add(model);
            }

            await ReloadChatsAsync().ConfigureAwait(true);
            SelectedChat = Chats.FirstOrDefault();
            if (SelectedChat is null)
            {
                ShowModel(status.Engine.ModelId ?? Models.FirstOrDefault()?.Id);
            }

            SetStatus(L["status.ready"]);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            ShowError(error);
        }
    }

    /// <summary>Перетащенные файлы: лимиты проверяются до отправки.</summary>
    public async Task AddFilesAsync(IReadOnlyList<DroppedFile> files)
    {
        var result = await AttachmentReader
            .ReadAsync([.. Attachments.Select(a => a.Attachment)], files, _limits)
            .ConfigureAwait(true);
        foreach (var attachment in result.Accepted)
        {
            Attachments.Add(new AttachmentViewModel(attachment));
        }

        if (result.ErrorKey is not null)
        {
            SetStatus(L.Format(result.ErrorKey, result.ErrorParams), error: true);
        }

        RefreshCommands();
    }

    private bool CanSend() =>
        !IsGenerating && (!string.IsNullOrWhiteSpace(Input) || Attachments.Count > 0);

    private bool CanRegenerate() =>
        !IsGenerating && SelectedChat is not null && Messages.Any(m => m.IsUser);

    private async Task NewChatAsync()
    {
        try
        {
            var chat = await _api.CreateChatAsync(SelectedModel?.Id).ConfigureAwait(true);
            var item = new ChatItemViewModel(chat);
            Chats.Insert(0, item);
            Raise(nameof(HasChats));
            SelectedChat = item;
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            ShowError(error);
        }
    }

    private async Task SendAsync()
    {
        if (SelectedChat is null)
        {
            await NewChatAsync().ConfigureAwait(true);
            if (SelectedChat is null)
            {
                return;
            }
        }

        var text = Input;
        var files = Attachments.Select(a => a.Attachment).ToList();
        var chatId = SelectedChat.Id;
        var started = await RunStreamAsync(chatId, cancel => _api.SendAsync(chatId, text, files, cancel))
            .ConfigureAwait(true);
        if (started)
        {
            Input = "";
            Attachments.Clear();
        }
    }

    private async Task RegenerateAsync()
    {
        if (SelectedChat is { } chat)
        {
            await RunStreamAsync(chat.Id, cancel => _api.RegenerateAsync(chat.Id, cancel)).ConfigureAwait(true);
        }
    }

    /// <summary>Читает поток событий; true — сервис принял запрос.</summary>
    private async Task<bool> RunStreamAsync(string chatId, Func<CancellationToken, IAsyncEnumerable<ChatEvent>> start)
    {
        IsGenerating = true;
        _generatingChatId = chatId;
        _streamCancel = new CancellationTokenSource();
        SetStatus(L["status.generating"]);
        MessageViewModel? answer = null;
        var accepted = false;
        try
        {
            await foreach (var item in start(_streamCancel.Token).ConfigureAwait(true))
            {
                accepted = true;
                var visible = SelectedChat?.Id == chatId;
                switch (item.Type)
                {
                    case "user" when item.Message is not null && visible:
                        Messages.Add(new MessageViewModel(item.Message));
                        break;
                    case "regenerate" when visible:
                        foreach (var gone in Messages.Where(m => item.Removed?.Contains(m.Id) == true).ToList())
                        {
                            Messages.Remove(gone);
                        }

                        break;
                    case "status" when item.State == "loading":
                        SetStatus(L.Format("status.loading_model", ("model_id", item.ModelId)));
                        break;
                    case "reasoning" or "text":
                        if (answer is null)
                        {
                            answer = new MessageViewModel("assistant");
                            if (visible)
                            {
                                Messages.Add(answer);
                            }

                            SetStatus(L["status.generating"]);
                        }

                        if (item.Type == "text")
                        {
                            answer.Content += item.Text;
                        }
                        else
                        {
                            answer.Reasoning += item.Text;
                        }

                        break;
                    case "done":
                        Finish(item, ref answer, visible);
                        break;
                }
            }

            await RefreshChatAsync(chatId).ConfigureAwait(true);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException or IOException)
        {
            if (answer is not null)
            {
                answer.IsStreaming = false;
                answer.Error = L["ui.connection_lost"];
            }

            ShowError(error);
        }
        catch (OperationCanceledException)
        {
        }
        finally
        {
            _streamCancel.Dispose();
            _streamCancel = null;
            _generatingChatId = null;
            IsGenerating = false;
        }

        return accepted;
    }

    private void Finish(ChatEvent item, ref MessageViewModel? answer, bool visible)
    {
        if (item.Message is { } saved)
        {
            if (answer is null)
            {
                answer = new MessageViewModel(saved);
                if (visible)
                {
                    Messages.Add(answer);
                }
            }

            answer.Id = saved.Id;
            answer.Content = saved.Content;
            answer.Reasoning = saved.Reasoning;
        }

        if (answer is not null)
        {
            answer.IsStreaming = false;
            answer.Finish = item.Finish;
        }

        if (item.Error is { } error)
        {
            var text = L.Format(error.Code, Params(error.Params));
            if (answer is not null)
            {
                answer.Error = text;
            }

            SetStatus(text, error: true);
        }
        else
        {
            SetStatus(L["status.ready"]);
        }
    }

    private async Task StopAsync()
    {
        if (_generatingChatId is { } chatId)
        {
            try
            {
                await _api.StopAsync(chatId).ConfigureAwait(true);
            }
            catch (Exception error) when (error is ApiException or HttpRequestException)
            {
                _streamCancel?.Cancel(); // сервис недоступен: хотя бы закрыть поток
                ShowError(error);
            }
        }
    }

    private async Task ClearAsync()
    {
        if (SelectedChat is { } chat)
        {
            try
            {
                await _api.ClearAsync(chat.Id).ConfigureAwait(true);
                Messages.Clear();
                RefreshCommands();
            }
            catch (Exception error) when (error is ApiException or HttpRequestException)
            {
                ShowError(error);
            }
        }
    }

    private async Task DeleteChatAsync(ChatItemViewModel? chat)
    {
        chat ??= SelectedChat;
        if (chat is null)
        {
            return;
        }

        try
        {
            await _api.DeleteChatAsync(chat.Id).ConfigureAwait(true);
            var index = Chats.IndexOf(chat);
            Chats.Remove(chat);
            Raise(nameof(HasChats));
            if (SelectedChat == chat || SelectedChat is null)
            {
                SelectedChat = Chats.Count == 0 ? null : Chats[Math.Min(index, Chats.Count - 1)];
            }
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            ShowError(error);
        }
    }

    private async Task CopyAsync(MessageViewModel? message)
    {
        if (message is not null && !string.IsNullOrEmpty(message.Content))
        {
            await _copy(message.Content).ConfigureAwait(true);
            SetStatus(L["chat.copied"]);
        }
    }

    private async Task ChangeModelAsync(ModelInfo model)
    {
        if (SelectedChat is not { } chat || chat.ModelId == model.Id)
        {
            return;
        }

        try
        {
            chat.Update(await _api.SetModelAsync(chat.Id, model.Id).ConfigureAwait(true));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            ShowModel(chat.ModelId);
            ShowError(error);
        }
    }

    private async Task LoadChatAsync(ChatItemViewModel? chat)
    {
        Messages.Clear();
        RefreshCommands();
        if (chat is null)
        {
            return;
        }

        ShowModel(chat.ModelId);
        try
        {
            var messages = await _api.MessagesAsync(chat.Id).ConfigureAwait(true);
            if (SelectedChat != chat)
            {
                return; // пока грузили, выбрали другой диалог
            }

            foreach (var message in messages)
            {
                Messages.Add(new MessageViewModel(message));
            }

            RefreshCommands();
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            ShowError(error);
        }
    }

    private async Task ReloadChatsAsync()
    {
        Chats.Clear();
        foreach (var chat in await _api.ChatsAsync().ConfigureAwait(true))
        {
            Chats.Add(new ChatItemViewModel(chat));
        }

        Raise(nameof(HasChats));
    }

    /// <summary>После ответа: новый заголовок и порядок диалогов, без перезагрузки сообщений.</summary>
    private async Task RefreshChatAsync(string chatId)
    {
        var chats = await _api.ChatsAsync().ConfigureAwait(true);
        if (chats.FirstOrDefault(c => c.Id == chatId) is { } info &&
            Chats.FirstOrDefault(c => c.Id == chatId) is { } item)
        {
            item.Update(info);
            var index = Chats.IndexOf(item);
            if (index > 0)
            {
                Chats.Move(index, 0);
            }
        }
    }

    private void ShowModel(string? modelId)
    {
        _ignoreModelChange = true;
        SelectedModel = Models.FirstOrDefault(m => m.Id == modelId);
        _ignoreModelChange = false;
    }

    private void ShowError(Exception error)
    {
        var text = error is ApiException api
            ? L.Format(api.Code, Params(api.Parameters))
            : L["ui.connection_lost"];
        SetStatus(text, error: true);
    }

    private void SetStatus(string text, bool error = false)
    {
        Status = text;
        IsError = error;
    }

    private void RefreshCommands()
    {
        foreach (var command in new[]
                 {
                     NewChatCommand, SendCommand, StopCommand, RegenerateCommand, ClearCommand, DeleteChatCommand,
                 })
        {
            command.Refresh();
        }
    }

    private static Dictionary<string, object?> Params(IReadOnlyDictionary<string, JsonElement>? values) =>
        values?.ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal)
        ?? new Dictionary<string, object?>(StringComparer.Ordinal);
}
