using System.Globalization;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Вариант языка в списке: код и подпись.</summary>
public sealed record LanguageOption(string Code, string Label);

/// <summary>
/// Экран настроек (3.7): папка данных, папка моделей, язык, порт внешнего API, доступ из сети.
/// Проверяет и сохраняет сервис; отправляются только изменённые поля. Смена папки данных
/// или языка применяется перезапуском приложения — экран предлагает перезапустить сразу.
/// </summary>
public sealed class SettingsViewModel : ObservableObject
{
    private readonly Func<IChatApi> _api;
    private readonly Func<string?, Task<string?>> _pickFolder;
    private readonly Func<Task> _restart;
    private AppSettings? _saved;
    private bool _isOpen;
    private string _dataDir = "";
    private string _modelsDir = "";
    private LanguageOption? _language;
    private string _apiPort = "";
    private bool _networkAccess;
    private string _currentDataDir = "";
    private string? _error;
    private string? _message;
    private bool _restartNeeded;

    public SettingsViewModel(Func<IChatApi> api, Func<string?, Task<string?>> pickFolder, Func<Task> restart)
    {
        _api = api;
        _pickFolder = pickFolder;
        _restart = restart;
        SaveCommand = new Command(SaveAsync, () => _saved is not null);
        CloseCommand = new Command(() =>
        {
            IsOpen = false;
            return Task.CompletedTask;
        });
        BrowseDataDirCommand = new Command(async () => DataDir = await _pickFolder(DataDir).ConfigureAwait(true) ?? DataDir);
        BrowseModelsDirCommand = new Command(async () =>
            ModelsDir = await _pickFolder(ModelsDir).ConfigureAwait(true) ?? ModelsDir);
        RestartCommand = new Command(() => _restart(), () => RestartNeeded);
    }

    private static Localizer L => Localizer.Current;

    public Command SaveCommand { get; }

    public Command CloseCommand { get; }

    public Command BrowseDataDirCommand { get; }

    public Command BrowseModelsDirCommand { get; }

    public Command RestartCommand { get; }

    public List<LanguageOption> Languages { get; } = [];

    public bool IsOpen
    {
        get => _isOpen;
        private set => Set(ref _isOpen, value);
    }

    public string DataDir
    {
        get => _dataDir;
        set => Set(ref _dataDir, value);
    }

    public string ModelsDir
    {
        get => _modelsDir;
        set => Set(ref _modelsDir, value);
    }

    public LanguageOption? Language
    {
        get => _language;
        set => Set(ref _language, value);
    }

    public string ApiPort
    {
        get => _apiPort;
        set => Set(ref _apiPort, value);
    }

    public bool NetworkAccess
    {
        get => _networkAccess;
        set => Set(ref _networkAccess, value);
    }

    /// <summary>Какая папка данных используется сейчас (может отличаться от сохранённой до перезапуска).</summary>
    public string DataDirNote => L.Format("settings.data_dir_note", ("path", _currentDataDir));

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

    public bool RestartNeeded
    {
        get => _restartNeeded;
        private set
        {
            if (Set(ref _restartNeeded, value))
            {
                RestartCommand.Refresh();
            }
        }
    }

    public async Task OpenAsync()
    {
        Error = null;
        Message = null;
        IsOpen = true;
        try
        {
            Show(await _api().SettingsAsync().ConfigureAwait(true));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    private void Show(SettingsInfo info)
    {
        _saved = info.Settings;
        _currentDataDir = info.CurrentDataDir;
        Languages.Clear();
        Languages.AddRange(info.Languages.Select(code => new LanguageOption(code, L[$"settings.lang.{code}"])));
        Raise(nameof(Languages));
        DataDir = info.Settings.DataDir;
        ModelsDir = info.Settings.ModelsDir;
        Language = Languages.FirstOrDefault(l => l.Code == info.Settings.Language);
        ApiPort = info.Settings.ApiPort.ToString(CultureInfo.InvariantCulture);
        NetworkAccess = info.Settings.NetworkAccess;
        Raise(nameof(DataDirNote));
        SaveCommand.Refresh();
    }

    private async Task SaveAsync()
    {
        if (_saved is not { } saved)
        {
            return;
        }

        Error = null;
        Message = null;
        if (!int.TryParse(ApiPort.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out var port))
        {
            Error = L["settings.port_not_number"];
            return;
        }

        var update = new SettingsUpdate(
            DataDir: DataDir != saved.DataDir ? DataDir : null,
            ModelsDir: ModelsDir != saved.ModelsDir ? ModelsDir : null,
            Language: Language is { } language && language.Code != saved.Language ? language.Code : null,
            ApiPort: port != saved.ApiPort ? port : null,
            NetworkAccess: NetworkAccess != saved.NetworkAccess ? NetworkAccess : null);
        if (update == new SettingsUpdate())
        {
            Message = L["settings.saved"];
            return;
        }

        try
        {
            var info = await _api().SaveSettingsAsync(update).ConfigureAwait(true);
            Show(info);
            if (info.RestartRequired is { Count: > 0 })
            {
                RestartNeeded = true;
            }

            Message = RestartNeeded ? L["settings.restart_needed"] : L["settings.saved"];
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    private static string Text(Exception error) => error is ApiException api
        ? L.Format(api.Code, api.Parameters.ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal))
        : L["ui.connection_lost"];
}
