using System.Collections.ObjectModel;
using System.Globalization;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Право нового ключа: chat, models, agent, remote (из сети).</summary>
public sealed class KeyRight(string code, bool enabled) : ObservableObject
{
    private bool _enabled = enabled;

    public string Code { get; } = code;

    public string Label => Localizer.Current[$"connections.permission.{Code}"];

    public bool Enabled
    {
        get => _enabled;
        set => Set(ref _enabled, value);
    }
}

/// <summary>Строка списка клиентов: без ключа — подсказка, права, счётчики, отзыв в два нажатия.</summary>
public sealed class ClientRow : ObservableObject
{
    private readonly Func<ClientRow, Task> _revoke;
    private bool _confirming;

    public ClientRow(ClientInfo info, Func<ClientRow, Task> revoke)
    {
        Info = info;
        _revoke = revoke;
        RevokeCommand = new Command(RevokeAsync, () => IsActive);
    }

    private static Localizer L => Localizer.Current;

    public ClientInfo Info { get; }

    public string Name => Info.Name;

    public string KeyHint => Info.KeyHint;

    public bool IsActive => Info.State == "active";

    public string State => L[$"connections.state.{Info.State}"];

    public string Permissions => string.Join(", ", Info.Permissions.Select(p => L[$"connections.permission.{p}"]));

    public string Usage => L.Format(
        "connections.usage",
        ("requests", Info.Requests),
        ("prompt", Info.PromptTokens),
        ("completion", Info.CompletionTokens),
        ("seconds", Math.Round(Info.BusySeconds, 1).ToString(CultureInfo.CurrentCulture)),
        ("errors", Info.Errors));

    public string LastSeen => Info.LastSeen is { } seen
        ? L.Format("connections.last_seen", ("time", DateTimeOffset.FromUnixTimeMilliseconds((long)(seen * 1000))
            .ToLocalTime().ToString("g", CultureInfo.CurrentCulture)))
        : L["connections.never_seen"];

    public bool HasRunning => Info.Running > 0;

    public string Running => L.Format("connections.running", ("count", Info.Running));

    /// <summary>Первое нажатие спрашивает подтверждение: отзыв не отменить.</summary>
    public bool Confirming
    {
        get => _confirming;
        private set
        {
            if (Set(ref _confirming, value))
            {
                Raise(nameof(RevokeLabel));
            }
        }
    }

    public string RevokeLabel => _confirming ? L["connections.revoke_confirm"] : L["connections.revoke"];

    public Command RevokeCommand { get; }

    private async Task RevokeAsync()
    {
        if (!Confirming)
        {
            Confirming = true;
            return;
        }

        Confirming = false;
        await _revoke(this).ConfigureAwait(true);
    }
}

/// <summary>
/// Вкладка «Подключения» (7.1–7.4): адрес внешнего API и готовые настройки для IDE,
/// клиенты с правами и счётчиками, новый ключ (показывается один раз), отзыв.
/// Порт, доступ из сети и список адресов редактирует <see cref="SettingsViewModel"/>
/// и сохраняет общей кнопкой «Сохранить».
/// </summary>
public sealed class ConnectionsViewModel : ObservableObject
{
    private readonly Func<IChatApi> _api;
    private readonly Func<string, Task> _copy;
    private string _newName = "";
    private string? _newKey;
    private string? _url;
    private bool _running;
    private string? _queue;
    private string? _error;
    private string? _message;

    public ConnectionsViewModel(Func<IChatApi> api, Func<string, Task>? copy = null)
    {
        _api = api;
        _copy = copy ?? (_ => Task.CompletedTask);
        NewPermissions =
        [
            new("chat", true), new("models", true), new("agent", false), new("remote", false),
        ];
        CreateCommand = new Command(CreateAsync, () => NewName.Trim().Length > 0);
        CopyKeyCommand = new Command(() => _newKey is { } key ? _copy(key) : Task.CompletedTask, () => HasNewKey);
        DismissKeyCommand = new Command(() =>
        {
            NewKey = null;
            return Task.CompletedTask;
        });
        CopyUrlCommand = new Command(() => _url is { } url ? _copy(OpenAiUrl) : Task.CompletedTask, () => _url is not null);
        RefreshCommand = new Command(LoadAsync);
    }

    private static Localizer L => Localizer.Current;

    public ObservableCollection<ClientRow> Clients { get; } = [];

    public ObservableCollection<KeyRight> NewPermissions { get; }

    public Command CreateCommand { get; }

    public Command CopyKeyCommand { get; }

    public Command DismissKeyCommand { get; }

    public Command CopyUrlCommand { get; }

    public Command RefreshCommand { get; }

    public string NewName
    {
        get => _newName;
        set
        {
            if (Set(ref _newName, value))
            {
                CreateCommand.Refresh();
            }
        }
    }

    /// <summary>Только что созданный ключ — целиком только сейчас; после «Готово» больше не виден.</summary>
    public string? NewKey
    {
        get => _newKey;
        private set
        {
            if (Set(ref _newKey, value))
            {
                Raise(nameof(HasNewKey));
                CopyKeyCommand.Refresh();
            }
        }
    }

    public bool HasNewKey => _newKey is not null;

    public bool HasClients => Clients.Count > 0;

    public bool IsRunning => _running;

    public string Status => _running && _url is not null
        ? L.Format("connections.api_running", ("url", _url))
        : L["connections.api_stopped"];

    /// <summary>Готовые адреса для IDE: OpenAI-совместимые (…/v1), Ollama и Anthropic — корень.</summary>
    public string OpenAiUrl => (_url ?? "") + "/v1";

    public string BaseUrl => _url ?? "";

    public bool HasUrl => _url is not null;

    public string? Queue => _queue;

    public string? Error
    {
        get => _error;
        private set
        {
            if (Set(ref _error, value))
            {
                Raise(nameof(HasError));
            }
        }
    }

    public bool HasError => _error is not null;

    public string? Message
    {
        get => _message;
        private set
        {
            if (Set(ref _message, value))
            {
                Raise(nameof(HasMessage));
            }
        }
    }

    public bool HasMessage => _message is not null;

    public async Task LoadAsync()
    {
        Error = null;
        try
        {
            Show(await _api().ConnectionsAsync().ConfigureAwait(true));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException or NotSupportedException)
        {
            Error = Text(error);
        }
    }

    private void Show(ConnectionsInfo info)
    {
        _url = info.Api.Running ? info.Api.Url : null;
        _running = info.Api.Running;
        _queue = info.Queue is { } q
            ? L.Format("connections.queue", ("concurrent", q.MaxConcurrent), ("per_client", q.PerClientQueue),
                ("clients", q.MaxClients))
            : null;
        Clients.Clear();
        foreach (var client in info.Clients.OrderBy(c => c.State != "active").ThenBy(c => c.Created))
        {
            Clients.Add(new ClientRow(client, RevokeAsync));
        }

        foreach (var name in new[]
                 {
                     nameof(HasClients), nameof(IsRunning), nameof(Status), nameof(OpenAiUrl), nameof(BaseUrl),
                     nameof(HasUrl), nameof(Queue),
                 })
        {
            Raise(name);
        }

        CopyUrlCommand.Refresh();
    }

    private async Task CreateAsync()
    {
        Error = null;
        Message = null;
        var permissions = NewPermissions.Where(p => p.Enabled).Select(p => p.Code).ToList();
        if (permissions.Count == 0)
        {
            Error = L["connections.no_permissions"];
            return;
        }

        try
        {
            var created = await _api().CreateKeyAsync(NewName.Trim(), permissions).ConfigureAwait(true);
            NewKey = created.Key;
            NewName = "";
            await LoadAsync().ConfigureAwait(true);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException or NotSupportedException)
        {
            Error = Text(error);
        }
    }

    private async Task RevokeAsync(ClientRow row)
    {
        Error = null;
        try
        {
            var result = await _api().RevokeKeyAsync(row.Info.Id, cancelRunning: true).ConfigureAwait(true);
            Message = L.Format("connections.revoked", ("name", row.Name), ("stopped", result.Stopped));
            await LoadAsync().ConfigureAwait(true);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    public override void Relocalize()
    {
        base.Relocalize();
        foreach (var permission in NewPermissions)
        {
            permission.Relocalize();
        }

        foreach (var row in Clients)
        {
            row.Relocalize();
        }
    }

    private static string Text(Exception error) => error switch
    {
        ApiException api => L.Format(
            api.Code, api.Parameters.ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal)),
        NotSupportedException => L["api.not_found"],
        _ => L["ui.connection_lost"],
    };
}
