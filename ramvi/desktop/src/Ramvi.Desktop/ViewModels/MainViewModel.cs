using System.Collections.ObjectModel;
using System.Text.Json;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;

namespace Ramvi.Desktop.ViewModels;

/// <summary>
/// Экран чата (UI-02): список диалогов, поток ответа, Stop/Regenerate, копирование,
/// очистка, выбор модели, thinking и перетаскивание текстовых файлов.
/// Состояния UI-03 (3.5): <see cref="State"/> и плашка <see cref="Banner"/> с действиями.
/// Не зависит от окна: проверяется тестами без Avalonia.
/// </summary>
public sealed class MainViewModel : ObservableObject, IDisposable
{
    private readonly Func<Task<IChatApi>> _connect;
    private readonly Func<string, Task> _copy;
    private readonly Func<string, Task> _openFolder;
    private readonly Func<string?, Task<string?>>? _pickFolder;
    private readonly Func<string, Task>? _chooseDataDir;
    private readonly TimeSpan _loadPoll;
    private IChatApi? _connected;
    private Problem? _problem;
    private string? _loadingModel;
    private UiState _state = UiState.Connecting;
    private StateBanner? _banner;
    private bool _isModelPickerOpen;
    private ChatItemViewModel? _selectedChat;
    private ModelInfo? _selectedModel;
    private string _input = "";
    private Func<string> _status;
    private bool _isError;
    private bool _isGenerating;
    private bool _ignoreModelChange;
    private bool _voiceForAgent;
    private Limits _limits = new(32_000, 5, 128 * 1024, 256 * 1024, 255);
    private CancellationTokenSource? _streamCancel;
    private string? _generatingChatId;
    private int _loadWatch;

    public MainViewModel(
        IChatApi api,
        Func<string, Task>? copy = null,
        Func<string, Task>? openFolder = null,
        Func<Task<string?>>? pickModelFile = null,
        Func<string?, Task<string?>>? pickFolder = null,
        Func<Task>? restartApp = null,
        Func<string, Task>? chooseDataDir = null,
        Func<Task<IReadOnlyList<DroppedFile>>>? pickFiles = null,
        Action<string>? applyTheme = null,
        TimeSpan? loadPoll = null,
        CharacterWindowStore? characterWindow = null)
        : this(() => Task.FromResult(api), copy, openFolder, pickModelFile, pickFolder, restartApp, chooseDataDir,
            pickFiles, applyTheme, loadPoll, characterWindow)
    {
    }

    /// <param name="connect">Запустить сервис и вернуть клиент; повторяется по «Повторить».</param>
    /// <param name="pickModelFile">Выбор файла GGUF в мастере; null — отмена.</param>
    /// <param name="pickFiles">Кнопка «Прикрепить файл»: выбранные файлы (пусто — отмена).</param>
    /// <param name="applyTheme">Применить тему окна сразу после сохранения настроек (system/light/dark).</param>
    /// <param name="loadPoll">Как часто спрашивать сервис, закончилась ли загрузка модели (3.17).</param>
    public MainViewModel(
        Func<Task<IChatApi>> connect,
        Func<string, Task>? copy = null,
        Func<string, Task>? openFolder = null,
        Func<Task<string?>>? pickModelFile = null,
        Func<string?, Task<string?>>? pickFolder = null,
        Func<Task>? restartApp = null,
        Func<string, Task>? chooseDataDir = null,
        Func<Task<IReadOnlyList<DroppedFile>>>? pickFiles = null,
        Action<string>? applyTheme = null,
        TimeSpan? loadPoll = null,
        CharacterWindowStore? characterWindow = null)
    {
        _connect = connect;
        _loadPoll = loadPoll ?? TimeSpan.FromMilliseconds(300);
        Agent = new AgentViewModel(() => _api, pickFolder ?? (_ => Task.FromResult<string?>(null)), Models, pickFiles);
        Character = new CharacterViewModel(() => _api, Models, () => SelectedModel, pickFolder, pickFiles, () => _limits);
        Agent.PropertyChanged += (_, e) => { if (e.PropertyName == nameof(AgentViewModel.IsRunning)) Raise(nameof(CanSelectModel)); };
        Character.PropertyChanged += (_, e) => { if (e.PropertyName == nameof(CharacterViewModel.Busy)) Raise(nameof(CanSelectModel)); };
        PropertyChanged += (_, e) => { if (e.PropertyName == nameof(IsIdle)) Raise(nameof(CanSelectModel)); };
        Sections = new SectionsViewModel(() => _api, Agent.LoadAsync, Character.LoadAsync);
        CharacterWindow = new CharacterWindowViewModel(Character, Sections.OpenCharacterAsync, characterWindow);
        ModelsScreen = new ModelsViewModel(() => _api, ReloadModelListAsync, pickFile: pickModelFile);
        _pickFolder = pickFolder;
        _chooseDataDir = chooseDataDir;
        Settings = new SettingsViewModel(
            () => _api,
            pickFolder ?? (_ => Task.FromResult<string?>(null)),
            restartApp ?? (() => Task.CompletedTask),
            applyTheme,
            () => Onboarding!.OpenAsync(SelectedModel?.Id),
            OnModelUnloaded,
            ApplyLanguage,
            Models,
            () => ModelsScreen.LoadAsync(),
            ReloadModelListAsync,
            async () =>
            {
                await Voice!.LoadAsync().ConfigureAwait(true);
                await Character.RefreshInputAsync().ConfigureAwait(true);
            },
            Agent.LoadAsync,
            copy: text => _copy!(text),
            pickFile: pickModelFile,
            characterLoad: async () =>
            {
                try
                {
                    await _api.OpenSectionAsync("character").ConfigureAwait(true);
                    await Character.LoadSettingsAsync().ConfigureAwait(true);

                }
                catch (Exception error) when (error is ApiException or HttpRequestException)
                { ShowError(error); }
            },
            cloudChanged: ReloadModelListAsync);
        Settings.FileLimitsChanged += async () =>
        {
            try { _ = await RefreshLimitsAsync().ConfigureAwait(true); }
            catch (Exception error) when (error is ApiException or HttpRequestException) { ShowError(error); }
        };
        Onboarding = new OnboardingViewModel(
            () => _api,
            Models,
            pickModelFile ?? (() => Task.FromResult<string?>(null)),
            OnboardingCompletedAsync,
            () => Settings.OpenAsync("connections"));
        _copy = copy ?? (_ => Task.CompletedTask);
        _openFolder = openFolder ?? (_ => Task.CompletedTask);
        _status = () => L["status.connecting"];
        Voice = new VoiceViewModel(() => _api, InsertVoiceText, e => { if (Sections.IsAgentSelected) Agent.VoiceError(e); else ShowError(e); }, () => IsConnected, send: SendVoiceTextAsync);
        Agent.Voice = Voice;
        Voice.PropertyChanged += (_, e) => { if (e.PropertyName == nameof(VoiceViewModel.IsRecording) && Voice.IsRecording) _voiceForAgent = Sections.IsAgentSelected; };
        Sections.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(SectionsViewModel.IsChatSelected) && Voice.IsRecording) _ = Voice.CancelCommand.RunAsync();
            // Ушли из раздела «Персонаж» — он замолкает, только если показан в окне RAMVI; вне окна
            // (отдельное окно, оверлей) он продолжает разговор, пока вы в чате (3.13/9.26).
            if (e.PropertyName == nameof(SectionsViewModel.IsCharacterSelected) && !Sections.IsCharacterSelected && CharacterWindow.IsEmbedded) { if(Character.Stop.CanExecute(null)) _ = Character.StopAsync(); _ = Character.Voice.StopAsync(); _ = Character.Speech.StopAsync(); }
        };
        NewChatCommand = new Command(NewChatAsync, () => !IsGenerating && IsConnected);
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
        AttachFilesCommand = new Command(
            async () =>
            {
                if (pickFiles is not null && await pickFiles().ConfigureAwait(true) is { Count: > 0 } picked)
                {
                    await AddFilesAsync(picked).ConfigureAwait(true);
                }
            },
            () => pickFiles is not null && IsConnected);
        Messages.CollectionChanged += (_, _) => MarkLastAnswer();
        RetryCommand = new Command(RetryAsync, () => !IsGenerating);
        OpenSettingsCommand = new Command(() => Settings.OpenAsync(), () => IsConnected);
        ChooseDataDirCommand = new Command(ChooseDataDirAsync);
        ChooseModelCommand = new Command(() =>
        {
            IsModelPickerOpen = true;
            return Task.CompletedTask;
        });
        RefreshModelsCommand = new Command(RefreshModelsAsync, () => IsConnected && !IsGenerating);
        OpenFolderCommand = new Command(p => p is string path ? _openFolder(path) : Task.CompletedTask);
        DismissCommand = new Command(() =>
        {
            SetProblem(null);
            return Task.CompletedTask;
        });
        UpdateState();
    }

    private static Localizer L => Localizer.Current;

    private IChatApi _api => _connected ?? throw new InvalidOperationException("the service is not connected");

    public void Dispose() { _streamCancel?.Cancel(); Agent.Dispose(); Character.Dispose(); }

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

    /// <summary>«Прикрепить файл» (то же, что перетаскивание, с теми же лимитами).</summary>
    public Command AttachFilesCommand { get; }

    public Command RetryCommand { get; }

    public Command ChooseModelCommand { get; }

    public Command RefreshModelsCommand { get; }

    public Command OpenFolderCommand { get; }

    public Command DismissCommand { get; }

    /// <summary>Мастер первого запуска (3.6).</summary>
    public OnboardingViewModel Onboarding { get; }

    /// <summary>Разделы интерфейса (3.11).</summary>
    public SectionsViewModel Sections { get; }

    /// <summary>Экран «Модели» (4.11).</summary>
    public ModelsViewModel ModelsScreen { get; }

    /// <summary>Экран настроек (3.7).</summary>
    public SettingsViewModel Settings { get; }
    public AgentViewModel Agent { get; }
    public CharacterViewModel Character { get; }
    /// <summary>3.13/9.26: где показан персонаж (окно RAMVI, отдельное окно, поверх экрана).</summary>
    public CharacterWindowViewModel CharacterWindow { get; }

    /// <summary>5.2: кнопка микрофона у поля ввода.</summary>
    public VoiceViewModel Voice { get; }

    public Command OpenSettingsCommand { get; }

    /// <summary>«Выбрать папку данных…» при недоступной папке данных (3.9, USB только на чтение).</summary>
    public Command ChooseDataDirCommand { get; }

    public bool IsConnected => _connected is not null;

    /// <summary>Текущее состояние UI-03.</summary>
    public UiState State
    {
        get => _state;
        private set => Set(ref _state, value);
    }

    /// <summary>Плашка состояния или null, когда всё готово.</summary>
    public StateBanner? Banner
    {
        get => _banner;
        private set
        {
            if (Set(ref _banner, value))
            {
                Raise(nameof(HasBanner));
            }
        }
    }

    public bool HasBanner => _banner is not null;

    public Problem? Problem => _problem;

    /// <summary>Открыть список моделей (действие «Выбрать другую модель»).</summary>
    public bool IsModelPickerOpen
    {
        get => _isModelPickerOpen;
        set => Set(ref _isModelPickerOpen, value);
    }

    public Limits Limits => _limits;

    /// <summary>B.15: лимиты файлов из сервиса (настройки) — для чата, агента и персонажа.</summary>
    private async Task<ServiceStatus> RefreshLimitsAsync()
    {
        var status = await _api.StatusAsync().ConfigureAwait(true);
        _limits = status.Limits;
        Agent.MessageLimits = status.Limits;
        Raise(nameof(Limits));
        return status;
    }

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

    public bool CanSelectModel => IsIdle && !Agent.IsRunning && !Character.Busy;

    /// <summary>7.8: выбрана облачная модель — сообщения, задачи агента и реплики персонажу уходят в сервис.</summary>
    public bool IsCloudModel => SelectedModel?.IsCloud == true;

    public string CloudNotice => SelectedModel?.Cloud is { } cloud
        ? L.Format("cloud.sending", ("host", cloud.Host))
        : SelectedModel?.IsCloud == true ? L["cloud.sending_unknown"] : "";

    public ModelInfo? SelectedModel
    {
        get => _selectedModel;
        set
        {
            if (!Set(ref _selectedModel, value)) return;
            Raise(nameof(IsCloudModel));
            Raise(nameof(CloudNotice));
            Agent.Model = value;
            Character.Model = value;
            if (!_ignoreModelChange && value is not null) _ = ChangeModelAsync(value);
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

    /// <summary>Строка состояния; текст собирается заново на текущем языке (3.14).</summary>
    public string Status => _status();

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
                UpdateState();
            }
        }
    }

    public bool IsIdle => !IsGenerating;

    /// <summary>3.17: идёт загрузка модели — вместо текста крутится значок у списка «Модель».</summary>
    public bool IsModelLoading => _loadingModel is not null;

    /// <summary>Подсказка к значку загрузки: какая модель загружается.</summary>
    public string LoadingTip => L.Format("status.loading_model", ("model_id", _loadingModel ?? ""));

    /// <summary>Загрузить лимиты, модели и диалоги; открыть последний диалог.</summary>
    public async Task InitializeAsync()
    {
        _connected = null;
        Models.Clear();
        Chats.Clear();
        Messages.Clear();
        Raise(nameof(HasChats));
        Raise(nameof(IsConnected));
        SetProblem(null);
        SetStatus(() => L["status.connecting"]);
        try
        {
            _connected = await _connect().ConfigureAwait(true);
            Raise(nameof(IsConnected));
            var status = await RefreshLimitsAsync().ConfigureAwait(true);
            foreach (var model in await _api.ModelsAsync().ConfigureAwait(true))
            {
                Models.Add(model);
            }

            await ReloadChatsAsync().ConfigureAwait(true);
            Localizer.SetModuleTexts(await _api.ModuleTextsAsync().ConfigureAwait(true)); // 3.14
            var onboarding = await _api.OnboardingAsync().ConfigureAwait(true);
            await Sections.LoadAsync().ConfigureAwait(true);
            if (Sections.Items.Any(s => s.View == "agent")) Settings.AddTab("agent", "section.agent");
            if (Sections.Items.Any(s => s.View == "character")) Settings.AddTab("character", "section.character");
            if (Sections.Items.Any(s => s.View == "character")) _ = CharacterWindow.RestoreAsync(); // 9.26
            else CharacterWindow.Available = false;
            SelectedChat = Chats.FirstOrDefault();
            if (SelectedChat is null)
            {
                ShowModel(status.Engine.ModelId ?? onboarding.DefaultModel ?? Models.FirstOrDefault()?.Id);
            }

            SetStatus(() => L["status.ready"]);
            UpdateState();
            await Voice.LoadAsync().ConfigureAwait(true); // 5.7: метка, если звук уходит в API
            await LoadModelAsync(SelectedModel?.Id).ConfigureAwait(true); // 3.17: сразу при запуске
            if (!onboarding.Done)
            {
                await Onboarding.OpenAsync(onboarding.DefaultModel).ConfigureAwait(true);
            }
        }
        catch (ServiceStartException error)
        {
            SetProblem(new Problem(error.Code, error.Parameters));
        }
        catch (Exception error) when (error is InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            SetProblem(new Problem("status.service_failed", new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["detail"] = error.Message,
            }));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            ShowError(error);
        }
        finally
        {
            RefreshCommands();
            UpdateState();
        }
    }

    /// <summary>Перетащенные файлы: лимиты проверяются до отправки.</summary>
    public async Task AddFilesAsync(IReadOnlyList<DroppedFile> files)
    {
        if (Sections.IsAgentSelected) { await Agent.AddFilesAsync(files).ConfigureAwait(true); return; }
        if (Sections.IsCharacterSelected) { await Character.AddFilesAsync(files).ConfigureAwait(true); return; }
        var result = await AttachmentReader
            .ReadAsync([.. Attachments.Select(a => a.Attachment)], files, _limits)
            .ConfigureAwait(true);
        foreach (var attachment in result.Accepted)
        {
            Attachments.Add(new AttachmentViewModel(attachment));
        }

        if (result.ErrorKey is not null)
        {
            SetStatus(() => L.Format(result.ErrorKey, result.ErrorParams), error: true);
        }

        RefreshCommands();
    }

    private bool CanSend() =>
        IsConnected && !IsGenerating && (!string.IsNullOrWhiteSpace(Input) || Attachments.Count > 0);

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
        // Поле очищается, как только сервис принял запрос (первое событие потока), а не после
        // всего ответа: иначе вопрос висел в поле, пока модель печатает, и стирал то, что
        // пользователь успел набрать к концу ответа. Отказ сервиса оставляет текст и файлы.
        await RunStreamAsync(chatId, cancel => _api.SendAsync(chatId, text, files, cancel), onAccepted: () =>
        {
            if (Input == text)
            {
                Input = "";
            }

            foreach (var sent in Attachments.Where(a => files.Contains(a.Attachment)).ToList())
            {
                Attachments.Remove(sent);
            }
        }).ConfigureAwait(true);
    }

    private async Task RegenerateAsync()
    {
        if (SelectedChat is { } chat)
        {
            await RunStreamAsync(chat.Id, cancel => _api.RegenerateAsync(chat.Id, cancel)).ConfigureAwait(true);
        }
    }

    /// <summary>Читает поток событий; true — сервис принял запрос.</summary>
    private async Task<bool> RunStreamAsync(
        string chatId, Func<CancellationToken, IAsyncEnumerable<ChatEvent>> start, Action? onAccepted = null)
    {
        SetProblem(null);
        IsGenerating = true;
        _generatingChatId = chatId;
        _streamCancel = new CancellationTokenSource();
        SetStatus(() => L["status.generating"]);
        MessageViewModel? answer = null;
        var accepted = false;
        try
        {
            await foreach (var item in start(_streamCancel.Token).ConfigureAwait(true))
            {
                if (!accepted)
                {
                    accepted = true;
                    onAccepted?.Invoke();
                }

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
                        SetLoading(item.ModelId ?? ""); // значок загрузки вместо текста (3.17)
                        break;
                    case "reasoning" or "text":
                        SetLoading(null);
                        if (answer is null)
                        {
                            answer = new MessageViewModel("assistant");
                            if (visible)
                            {
                                Messages.Add(answer);
                            }

                            SetStatus(() => L["status.generating"]);
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
                        SetLoading(null);
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
            SetLoading(null);
            IsGenerating = false;
            MarkLastAnswer();
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
            var problem = new Problem(error.Code, Params(error.Params));
            if (answer is not null)
            {
                answer.Error = L.Format(problem.Code, problem.Params);
            }

            SetProblem(problem);
        }
        else
        {
            SetStatus(() => L["status.ready"]);
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
            SetStatus(() => L["chat.copied"]);
        }
    }

    /// <summary>Модель выбрана в списке: запомнить её у диалога и сразу начать загрузку (3.17).</summary>
    private async Task ChangeModelAsync(ModelInfo model)
    {
        if (SelectedChat is { } chat && chat.ModelId != model.Id)
        {
            try
            {
                chat.Update(await _api.SetModelAsync(chat.Id, model.Id).ConfigureAwait(true));
            }
            catch (Exception error) when (error is ApiException or HttpRequestException)
            {
                ShowModel(chat.ModelId);
                ShowError(error);
                return;
            }
        }

        if (_problem?.State is UiState.LowMemory or UiState.Incompatible or UiState.NoAccess or UiState.NoModel)
        {
            SetProblem(null); // другая модель — прежняя причина к ней не относится
        }

        await LoadModelAsync(model.Id).ConfigureAwait(true);
    }

    /// <summary>
    /// 3.17: начать загрузку модели в сервисе и, пока она идёт, крутить значок. Сервис держит
    /// модель загруженной до выбора другой, «Выгрузить модель» или выхода из программы.
    /// </summary>
    private async Task LoadModelAsync(string? modelId)
    {
        if (modelId is null || !IsConnected)
        {
            return;
        }

        var watch = ++_loadWatch;
        try
        {
            var engine = (await _api.LoadModelAsync(modelId).ConfigureAwait(true)).Engine;
            if (ShowLoad(watch, engine, modelId))
            {
                _ = WatchLoadAsync(watch, modelId); // окно не ждёт конца загрузки
            }
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            LoadFailed(watch, error);
        }
    }

    /// <summary>Спрашивать сервис, пока загрузка не закончится; новый выбор прекращает опрос.</summary>
    private async Task WatchLoadAsync(int watch, string modelId)
    {
        try
        {
            do
            {
                await Task.Delay(_loadPoll).ConfigureAwait(true);
                if (watch != _loadWatch)
                {
                    return;
                }
            }
            while (ShowLoad(watch, (await _api.StatusAsync().ConfigureAwait(true)).Engine, modelId));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            LoadFailed(watch, error);
        }
    }

    /// <summary>Показать ход загрузки; true — ещё грузится.</summary>
    private bool ShowLoad(int watch, EngineStatus engine, string modelId)
    {
        if (watch != _loadWatch)
        {
            return false;
        }

        if (engine.Loading is null && engine.State != "starting")
        {
            if (!IsGenerating)
            {
                SetLoading(null);
                if (_problem is null && !IsError)
                {
                    SetStatus(() => L["status.ready"]);
                }
            }

            if (engine.Problem is { } problem)
            {
                SetProblem(new Problem(problem.Code, Params(problem.Params)));
            }

            return false;
        }

        if (!IsGenerating && _problem is null)
        {
            SetStatus(() => ""); // не «Готово», пока крутится значок загрузки
        }

        SetLoading(engine.Loading ?? engine.ModelId ?? modelId);
        return true;
    }

    private void LoadFailed(int watch, Exception error)
    {
        if (watch == _loadWatch)
        {
            SetLoading(null);
            ShowError(error);
        }
    }

    /// <summary>3.14: язык из настроек — сразу, без перезапуска.</summary>
    public void ApplyLanguage(string language)
    {
        Localizer.Current = Localizer.Load(language);
        Relocalize();
        Hotkeys.Current.Refresh(); // подсказки с клавишами — на новом языке
    }

    /// <summary>Все тексты окна — на текущем языке: разметка обновляется сама, здесь — тексты моделей.</summary>
    public override void Relocalize()
    {
        base.Relocalize();
        UpdateState(); // плашка собирается заново
        foreach (var item in new IEnumerable<ObservableObject>[] { Chats, Messages, Attachments }.SelectMany(x => x))
        {
            item.Relocalize();
        }

        Sections.Relocalize();
        Agent.Relocalize();
        Character.Relocalize();
        ModelsScreen.Relocalize();
        Settings.Relocalize();
        Onboarding.Relocalize();
    }

    /// <summary>Из настроек нажали «Выгрузить модель»: значок загрузки больше не нужен.</summary>
    private void OnModelUnloaded()
    {
        _loadWatch++;
        SetLoading(null);
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

    /// <summary>«Ответить заново» показывается только у последнего ответа модели.</summary>
    private void MarkLastAnswer()
    {
        var last = Messages.Count > 0 && Messages[^1].IsAssistant ? Messages[^1] : null;
        foreach (var message in Messages)
        {
            message.IsLastAnswer = message == last;
        }
    }

    private void ShowModel(string? modelId)
    {
        _ignoreModelChange = true;
        SelectedModel = Models.FirstOrDefault(m => m.Id == modelId);
        _ignoreModelChange = false;
    }

    private void ShowError(Exception error) =>
        SetProblem(error is ApiException api
            ? new Problem(api.Code, Params(api.Parameters))
            : new Problem("ui.connection_lost", new Dictionary<string, object?>(StringComparer.Ordinal)));

    private void SetProblem(Problem? problem)
    {
        _problem = problem;
        Raise(nameof(Problem));
        if (problem is not null)
        {
            SetStatus(() => L.Format(problem.Code, problem.Params), error: true);
        }
        else if (IsError)
        {
            var key = IsConnected ? "status.ready" : "status.connecting";
            SetStatus(() => L[key]);
        }

        UpdateState();
    }

    private void SetLoading(string? modelId)
    {
        if (_loadingModel != modelId)
        {
            _loadingModel = modelId;
            Raise(nameof(IsModelLoading));
            Raise(nameof(LoadingTip));
            UpdateState();
        }
    }

    /// <summary>«Повторить»: нет связи — заново запустить сервис; иначе ответить заново.</summary>
    private async Task RetryAsync()
    {
        var problem = _problem;
        SetProblem(null);
        if (!IsConnected || problem?.Code is "ui.connection_lost" or "storage.no_access" or "status.service_failed")
        {
            await InitializeAsync().ConfigureAwait(true);
        }
        else if (CanRegenerate())
        {
            await RegenerateAsync().ConfigureAwait(true);
        }
        else
        {
            await RefreshModelsAsync().ConfigureAwait(true);
        }
    }

    private async Task ChooseDataDirAsync()
    {
        if (_pickFolder is null || _chooseDataDir is null)
        {
            return;
        }

        var path = await _pickFolder(_problem?.Path).ConfigureAwait(true);
        if (string.IsNullOrEmpty(path))
        {
            return;
        }

        await _chooseDataDir(path).ConfigureAwait(true);
        await InitializeAsync().ConfigureAwait(true); // сервис запускается заново уже с новой папкой
    }

    /// <summary>Мастер закрыт: выбранная модель — для нового диалога.</summary>
    private Task OnboardingCompletedAsync(OnboardingState state)
    {
        UpdateState();
        if (SelectedChat is null && state.DefaultModel is { } model)
        {
            ShowModel(model);
            return LoadModelAsync(model); // модель из мастера загружается сразу (3.17)
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// «Обновить список»: сервис просматривает папку моделей из настроек и подключает новые
    /// модели на месте, затем список перечитывается (модель могли добавить, пока окно открыто).
    /// </summary>
    private async Task RefreshModelsAsync()
    {
        try
        {
            var scan = await _api.ScanModelsAsync().ConfigureAwait(true);
            var models = scan.Models;
            var selected = SelectedChat?.ModelId ?? SelectedModel?.Id;
            Models.Clear();
            foreach (var model in models)
            {
                Models.Add(model);
            }

            ShowModel(selected ?? Models.FirstOrDefault()?.Id);
            if (_problem?.State == UiState.NoModel && Models.Count > 0)
            {
                SetProblem(null);
            }

            if (_problem is null)
            {
                SetStatus(() => L.Format(
                    "models.scanned",
                    ("added", scan.Added.Count), ("skipped", scan.Skipped.Count), ("folder", scan.Folder)));
            }

            UpdateState();
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            ShowError(error);
        }
    }

    /// <summary>Открыть настройки на вкладке «Модели» (раньше — раздел бокового меню).</summary>
    public Task OpenModelsAsync() => Settings.OpenAsync("models");

    /// <summary>
    /// 4.11: список моделей чата после скачивания, удаления или смены модели по умолчанию
    /// на экране «Модели» — без просмотра папки, выбранная модель сохраняется, если осталась.
    /// </summary>
    private async Task ReloadModelListAsync()
    {
        try
        {
            var models = await _api.ModelsAsync().ConfigureAwait(true);
            var selected = SelectedChat?.ModelId ?? SelectedModel?.Id;
            Models.Clear();
            foreach (var model in models)
            {
                Models.Add(model);
            }

            var onboarding = await _api.OnboardingAsync().ConfigureAwait(true);
            var keep = Models.Any(m => m.Id == selected) ? selected : onboarding.DefaultModel;
            ShowModel(keep ?? Models.FirstOrDefault()?.Id);
            if (_problem?.State == UiState.NoModel && Models.Count > 0)
            {
                SetProblem(null);
            }

            UpdateState();
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            ShowError(error);
        }
    }

    /// <summary>Состояние из причины, связи, генерации и списка моделей; затем плашка.</summary>
    private void UpdateState()
    {
        var state = _problem?.State
            ?? (!IsConnected ? UiState.Connecting
                : _loadingModel is not null ? UiState.Loading
                : IsGenerating ? UiState.Generating
                : Models.Count == 0 ? UiState.NoModel
                : UiState.Ready);
        Banner = MakeBanner(state); // плашка раньше состояния: кто ждёт State, видит готовую плашку
        State = state;
    }

    private StateBanner? MakeBanner(UiState state)
    {
        var actions = new List<StateAction>();
        void Add(string id, string label, Command command, object? parameter = null) =>
            actions.Add(new StateAction(id, L[label], command, parameter));

        string text;
        switch (state)
        {
            case UiState.Ready or UiState.Generating or UiState.Connecting or UiState.Loading:
                // Строка состояния и кнопка «Стоп» уже есть; загрузку показывает значок (3.17).
                return null;
            case UiState.NoModel:
                text = _problem is { } missing ? L.Format(missing.Code, missing.Params) : L["state.no_model.text"];
                Add("refresh", "action.refresh", RefreshModelsCommand);
                if (Models.Count > 0)
                {
                    Add("choose_model", "action.choose_model", ChooseModelCommand);
                }

                break;
            default:
                var problem = _problem!;
                text = L.Format(problem.Code, problem.Params);
                if (state == UiState.LowMemory)
                {
                    text += "\n" + L["state.low_memory.hint"];
                }

                if (problem.Code == "storage.no_access" && _pickFolder is not null && _chooseDataDir is not null)
                {
                    Add("choose_data_dir", "action.choose_data_dir", ChooseDataDirCommand);
                }

                if (state == UiState.NoAccess && FolderOf(problem.Path) is { } folder)
                {
                    Add("open_folder", "action.open_folder", OpenFolderCommand, folder);
                }

                if (state is UiState.LowMemory or UiState.Incompatible or UiState.NoAccess && (IsConnected && Models.Count > 1))
                {
                    Add("choose_model", "action.choose_model", ChooseModelCommand);
                }

                if (state != UiState.Incompatible)
                {
                    Add("retry", "action.retry", RetryCommand);
                }

                if (state == UiState.Error && IsConnected)
                {
                    Add("dismiss", "action.dismiss", DismissCommand);
                }

                break;
        }

        return new StateBanner(state, L[$"state.{StateKey(state)}.title"], text, actions);
    }

    private static string? FolderOf(string? path)
    {
        if (string.IsNullOrEmpty(path))
        {
            return null;
        }

        if (Directory.Exists(path))
        {
            return path;
        }

        var parent = System.IO.Path.GetDirectoryName(path);
        return parent is not null && Directory.Exists(parent) ? parent : null;
    }

    private static string StateKey(UiState state) => state switch
    {
        UiState.NoModel => "no_model",
        UiState.LowMemory => "low_memory",
        UiState.ApiBusy => "api_busy",
        UiState.NoAccess => "no_access",
        _ => state.ToString().ToLowerInvariant(),
    };

    private void SetStatus(Func<string> text, bool error = false)
    {
        _status = text;
        Raise(nameof(Status));
        IsError = error;
    }

    private void RefreshCommands()
    {
        foreach (var command in new[]
                 {
                     NewChatCommand, SendCommand, StopCommand, RegenerateCommand, ClearCommand, DeleteChatCommand,
                     RetryCommand, RefreshModelsCommand, OpenSettingsCommand, AttachFilesCommand,
                 })
        {
            command.Refresh();
        }

        Voice.Refresh();
    }

    /// <summary>5.2: распознанный текст дописывается в поле ввода; человек правит и отправляет сам.</summary>
    private void InsertVoiceText(string text)
    {
        if (_voiceForAgent) { Agent.InsertVoiceText(text); return; }
        Input = string.IsNullOrWhiteSpace(Input) ? text : Input.TrimEnd() + " " + text;
        if (_problem?.Code.StartsWith("voice.", StringComparison.Ordinal) == true)
        {
            SetProblem(null); // прошлая ошибка голоса больше не актуальна
        }
    }

    /// <summary>«Отправлять автоматически»: распознанное уходит в чат, если отправка сейчас возможна.</summary>
    private Task SendVoiceTextAsync() => _voiceForAgent
        ? Sections.IsAgentSelected && Agent.StartCommand.CanExecute(null) ? Agent.StartCommand.RunAsync() : Task.CompletedTask
        : Sections.IsChatSelected && SendCommand.CanExecute(null) ? SendCommand.RunAsync() : Task.CompletedTask;

    private static Dictionary<string, object?> Params(IReadOnlyDictionary<string, JsonElement>? values) =>
        values?.ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal)
        ?? new Dictionary<string, object?>(StringComparer.Ordinal);
}
