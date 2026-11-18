using System.Globalization;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Вариант языка в списке: код и подпись.</summary>
public sealed record LanguageOption(string Code, string Label);

/// <summary>Вариант темы: system (как в ОС), light, dark (3.16).</summary>
public sealed record ThemeOption(string Code, string Label);

/// <summary>
/// Экран настроек (3.7): папка данных, папка моделей, язык, порт внешнего API, доступ из сети.
/// Проверяет и сохраняет сервис; отправляются только изменённые поля. Смена папки данных
/// применяется перезапуском приложения — экран предлагает перезапустить сразу; язык и
/// тема — сразу, без перезапуска (3.14, 3.16).
/// </summary>
public sealed class SettingsViewModel : ObservableObject
{
    private readonly Func<IChatApi> _api;
    private readonly Func<string?, Task<string?>> _pickFolder;
    private readonly Func<Task> _restart;
    private readonly Action<string>? _applyTheme;
    private readonly Func<Task>? _openOnboarding;
    private readonly Action? _onUnloaded;
    private readonly Action<string>? _applyLanguage;
    private string? _loadedModel;
    private ThemeOption? _theme;
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
    private readonly IReadOnlyCollection<ModelInfo> _models;
    private readonly Func<Task>? _openModels;
    private readonly Func<Task>? _defaultChanged;
    private ModelInfo? _defaultModel;
    private string? _savedDefault;
    private readonly Func<Task>? _soundChanged;

    public SettingsViewModel(
        Func<IChatApi> api,
        Func<string?, Task<string?>> pickFolder,
        Func<Task> restart,
        Action<string>? applyTheme = null,
        Func<Task>? openOnboarding = null,
        Action? onUnloaded = null,
        Action<string>? applyLanguage = null,
        IReadOnlyCollection<ModelInfo>? models = null,
        Func<Task>? openModels = null,
        Func<Task>? defaultChanged = null,
        Func<Task>? soundChanged = null)
    {
        _soundChanged = soundChanged;
        _api = api;
        Sound = new SoundSettingsViewModel(api);
        _models = models ?? [];
        _openModels = openModels;
        _defaultChanged = defaultChanged;
        ManageModelsCommand = new Command(
            async () =>
            {
                IsOpen = false;
                await _openModels!().ConfigureAwait(true);
            },
            () => _openModels is not null);
        _applyLanguage = applyLanguage;
        _onUnloaded = onUnloaded;
        UnloadModelCommand = new Command(UnloadModelAsync, () => _loadedModel is not null);
        _applyTheme = applyTheme;
        _openOnboarding = openOnboarding;
        OpenOnboardingCommand = new Command(
            async () =>
            {
                IsOpen = false;
                await _openOnboarding!().ConfigureAwait(true);
            },
            () => _openOnboarding is not null);
        Themes = [.. ThemeOptions()];
        _pickFolder = pickFolder;
        _restart = restart;
        SaveCommand = new Command(SaveAsync, () => _saved is not null);
        CloseCommand = new Command(() =>
        {
            IsOpen = false;
            Sound.Stop();
            return Task.CompletedTask;
        });
        BrowseDataDirCommand = new Command(async () => DataDir = await _pickFolder(DataDir).ConfigureAwait(true) ?? DataDir);
        BrowseModelsDirCommand = new Command(async () =>
            ModelsDir = await _pickFolder(ModelsDir).ConfigureAwait(true) ?? ModelsDir);
        RestartCommand = new Command(() => _restart(), () => RestartNeeded);
    }

    private static Localizer L => Localizer.Current;

    public Command SaveCommand { get; }

    /// <summary>5.2: раздел «Звук» — микрофон, язык и модель распознавания речи.</summary>
    public SoundSettingsViewModel Sound { get; }

    public Command CloseCommand { get; }

    public Command BrowseDataDirCommand { get; }

    public Command BrowseModelsDirCommand { get; }

    public Command RestartCommand { get; }

    /// <summary>Снова открыть мастер первого запуска: проверка компьютера и подключение модели.</summary>
    public Command OpenOnboardingCommand { get; }

    /// <summary>4.11: открыть экран «Модели» (скачать, удалить, по умолчанию).</summary>
    public Command ManageModelsCommand { get; }

    /// <summary>4.11: установленные модели для выбора модели по умолчанию.</summary>
    public IReadOnlyCollection<ModelInfo> Models => _models;

    /// <summary>4.11: модель новых чатов; сохраняется кнопкой «Сохранить».</summary>
    public ModelInfo? DefaultModel
    {
        get => _defaultModel;
        set => Set(ref _defaultModel, value);
    }

    /// <summary>3.17: «Выгрузить модель» — освободить память; активна, только когда модель загружена.</summary>
    public Command UnloadModelCommand { get; }

    /// <summary>Какая модель сейчас в памяти (или загружается); null — не загружена.</summary>
    public string? LoadedModel
    {
        get => _loadedModel;
        private set
        {
            if (Set(ref _loadedModel, value))
            {
                Raise(nameof(LoadedModelText));
                UnloadModelCommand.Refresh();
            }
        }
    }

    public string LoadedModelText => _loadedModel is { } model
        ? L.Format("settings.model_loaded", ("model_id", model))
        : L["settings.model_not_loaded"];

    public List<LanguageOption> Languages { get; private set; } = [];

    public static IReadOnlyList<string> ThemeCodes { get; } = ["system", "light", "dark"];

    public List<ThemeOption> Themes { get; private set; }

    /// <summary>Тема применяется сразу после сохранения, без перезапуска.</summary>
    public ThemeOption? Theme
    {
        get => _theme;
        set => Set(ref _theme, value);
    }

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
            var info = await _api().SettingsAsync().ConfigureAwait(true);
            Show(info);
            await Sound.LoadAsync(info.Settings).ConfigureAwait(true);
            ShowEngine((await _api().StatusAsync().ConfigureAwait(true)).Engine);
            _savedDefault = (await _api().OnboardingAsync().ConfigureAwait(true)).DefaultModel;
            Raise(nameof(Models));
            DefaultModel = _models.FirstOrDefault(m => m.Id == _savedDefault);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    private void ShowEngine(EngineStatus engine) =>
        LoadedModel = engine.Loading ?? (engine.State is "stopped" or "failed" ? null : engine.ModelId);

    private async Task UnloadModelAsync()
    {
        Error = null;
        Message = null;
        try
        {
            var result = await _api().UnloadModelAsync().ConfigureAwait(true);
            ShowEngine(result.Engine);
            _onUnloaded?.Invoke();
            Message = L["settings.model_unloaded"];
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    private static IEnumerable<ThemeOption> ThemeOptions() =>
        ThemeCodes.Select(code => new ThemeOption(code, L[$"settings.theme.{code}"]));

    /// <summary>Подписи языков и тем — на новом языке; выбор сохраняется.</summary>
    public override void Relocalize()
    {
        var theme = Theme?.Code;
        var language = Language?.Code;
        Themes = [.. ThemeOptions()];
        Languages = [.. Languages.Select(l => new LanguageOption(l.Code, L[$"settings.lang.{l.Code}"]))];
        base.Relocalize();
        Sound.Relocalize();
        Theme = Themes.FirstOrDefault(t => t.Code == theme);
        Language = Languages.FirstOrDefault(l => l.Code == language);
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
        Theme = Themes.FirstOrDefault(t => t.Code == info.Settings.Theme) ?? Themes[0];
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
            NetworkAccess: NetworkAccess != saved.NetworkAccess ? NetworkAccess : null,
            Theme: Theme is { } theme && theme.Code != saved.Theme ? theme.Code : null);
        var sound = Sound.Changes(saved);
        var api = Sound.ApiChanges(saved);
        update = update with
        {
            VoiceDevice = sound.Device, VoiceLanguage = sound.Language, VoiceModel = sound.Model,
            VoiceBackend = api.Backend, VoiceApiUrl = api.Url, VoiceApiModel = api.Model,
        };
        var newDefault = DefaultModel?.Id is { } id && id != _savedDefault ? id : null;
        if (update == new SettingsUpdate() && newDefault is null)
        {
            Message = L["settings.saved"];
            return;
        }

        try
        {
            if (newDefault is not null)
            {
                await _api().SetDefaultModelAsync(newDefault).ConfigureAwait(true);
                _savedDefault = newDefault;
                if (_defaultChanged is not null)
                {
                    await _defaultChanged().ConfigureAwait(true);
                }
            }

            if (update == new SettingsUpdate())
            {
                Message = L["settings.saved"];
                return;
            }

            var info = await _api().SaveSettingsAsync(update).ConfigureAwait(true);
            Show(info);
            Sound.Show(info.Settings);
            if (_soundChanged is not null)
            {
                await _soundChanged().ConfigureAwait(true); // 5.7: метка «звук отправляется в …»
            }
            if (update.Theme is not null)
            {
                _applyTheme?.Invoke(info.Settings.Theme);
            }

            if (update.Language is not null)
            {
                _applyLanguage?.Invoke(info.Settings.Language); // сразу, без перезапуска
            }

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
