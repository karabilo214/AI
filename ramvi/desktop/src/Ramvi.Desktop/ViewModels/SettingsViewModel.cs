using System.Collections.ObjectModel;
using System.Globalization;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Вариант языка в списке: код и подпись.</summary>
public sealed record LanguageOption(string Code, string Label);

/// <summary>Вариант темы: system (как в ОС), light, dark (3.16).</summary>
public sealed record ThemeOption(string Code, string Label);

/// <summary>Вкладка настроек: встроенная (general, sound, models) или модуля (module:&lt;id&gt;).</summary>
public sealed class SettingsTab(string id, string titleKey) : ObservableObject
{
    public string Id { get; } = id;

    public string TitleKey { get; } = titleKey;

    public string Title => Localizer.Current[TitleKey];
}

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
    private string _networkAllow = "";
    private string? _modelsDirWarning;
    private readonly Func<Task<string?>> _pickFile;
    private UpdateStatus? _update;
    private string _currentDataDir = "";
    private string? _error;
    private string? _message;
    private bool _restartNeeded;
    private readonly IReadOnlyCollection<ModelInfo> _models;
    private readonly Func<Task>? _openModels;
    private readonly Func<Task>? _defaultChanged;
    private ModelInfo? _defaultModel;
    private string? _savedDefault;
    private SettingsTab? _selectedTab;
    private readonly Func<Task>? _soundChanged;
    private readonly Func<Task>? _agentLoad;

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
        Func<Task>? soundChanged = null,
        Func<Task>? agentLoad = null,
        Func<string, Task>? copy = null,
        Func<Task<string?>>? pickFile = null)
    {
        _pickFile = pickFile ?? (() => Task.FromResult<string?>(null));
        InstallUpdateCommand = new Command(InstallUpdateAsync);
        RestartToUpdateCommand = new Command(() => _restart!(), () => _update?.Staged is not null);
        Connections = new ConnectionsViewModel(api, copy);
        _soundChanged = soundChanged;
        _agentLoad = agentLoad;
        _api = api;
        Sound = new SoundSettingsViewModel(api);
        _models = models ?? [];
        _openModels = openModels;
        _defaultChanged = defaultChanged;
        Tabs =
        [
            new("general", "settings.tab.general"), new("sound", "settings.tab.sound"),
            new("models", "settings.tab.models"), new("connections", "settings.tab.connections"),
        ];
        _selectedTab = Tabs[0];
        // «Управлять моделями…» — та же панель, вкладка «Модели» (раньше — раздел бокового меню).
        ManageModelsCommand = new Command(() =>
        {
            SelectedTab = Tabs.First(t => t.Id == "models");
            return Task.CompletedTask;
        });
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

    /// <summary>Вкладки: «Общие», «Звук», «Модели»; модули добавляют свои (<see cref="AddTab"/>).</summary>
    public ObservableCollection<SettingsTab> Tabs { get; }

    public SettingsTab? SelectedTab
    {
        get => _selectedTab;
        set
        {
            if (value is null || !Set(ref _selectedTab, value))
            {
                return;
            }

            foreach (var name in new[] { nameof(IsGeneralTab), nameof(IsSoundTab), nameof(IsModelsTab), nameof(IsConnectionsTab), nameof(IsModuleTab), nameof(IsAgentTab), nameof(HasFooter) })
            {
                Raise(name);
            }

            if (value.Id == "module:agent" && IsOpen && _agentLoad is not null) _ = _agentLoad();
            if (value.Id == "connections" && IsOpen) _ = Connections.LoadAsync();
            if (value.Id == "models" && IsOpen && _openModels is not null)
            {
                _ = _openModels(); // экран моделей перечитывает каталог и загрузки
            }
        }
    }

    public bool IsGeneralTab => _selectedTab?.Id == "general";

    public bool IsSoundTab => _selectedTab?.Id == "sound";

    public bool IsModelsTab => _selectedTab?.Id == "models";

    /// <summary>7.1–7.4: внешний API — порт, сеть, разрешённые адреса, ключи программ.</summary>
    public bool IsConnectionsTab => _selectedTab?.Id == "connections";

    public ConnectionsViewModel Connections { get; }

    public bool IsModuleTab => _selectedTab?.Id.StartsWith("module:", StringComparison.Ordinal) == true && !IsAgentTab;
    public bool IsAgentTab => _selectedTab?.Id == "module:agent";

    /// <summary>«Сохранить» — у вкладок с полями; у «Модели» свои кнопки на карточках.</summary>
    public bool HasFooter => IsGeneralTab || IsSoundTab || IsConnectionsTab;

    /// <summary>Вкладка модуля (этапы 6 и 9): добавляется один раз по id.</summary>
    public void AddTab(string moduleId, string titleKey)
    {
        var id = "module:" + moduleId;
        if (Tabs.All(t => t.Id != id))
        {
            Tabs.Add(new SettingsTab(id, titleKey));
        }
    }

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

    /// <summary>8.3: «Установить из файла…» — подготовить пакет обновления.</summary>
    public Command InstallUpdateCommand { get; }

    /// <summary>8.3: перезапуск применяет подготовленное обновление (неудачное — откатится).</summary>
    public Command RestartToUpdateCommand { get; }

    /// <summary>Обновления доступны (комплект с папкой приложения).</summary>
    public bool HasUpdates => _update is not null;

    public string UpdateText => _update switch
    {
        null => "",
        { Staged: { } staged } => L.Format("settings.update_staged", ("version", staged), ("current", _update.Current)),
        { RolledBack: { } back } => L.Format("settings.update_rolled_back", ("version", back), ("current", _update.Current)),
        { Applied: { } applied, Health: "good" } => L.Format("settings.update_applied", ("version", applied)),
        _ => L.Format("settings.update_current", ("version", _update.Current)),
    };

    private void ShowUpdate(UpdateStatus? status)
    {
        _update = status;
        Raise(nameof(HasUpdates));
        Raise(nameof(UpdateText));
        RestartToUpdateCommand.Refresh();
    }

    private async Task InstallUpdateAsync()
    {
        Error = null;
        Message = null;
        if (await _pickFile().ConfigureAwait(true) is not { } path)
        {
            return;
        }

        try
        {
            ShowUpdate(await _api().StageUpdateAsync(path).ConfigureAwait(true));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    /// <summary>8.2: папка моделей на сетевом или съёмном диске — загрузка дольше.</summary>
    public string? ModelsDirWarning
    {
        get => _modelsDirWarning;
        private set
        {
            if (Set(ref _modelsDirWarning, value))
            {
                Raise(nameof(HasModelsDirWarning));
            }
        }
    }

    public bool HasModelsDirWarning => _modelsDirWarning is not null;

    /// <summary>7.4: с каких адресов можно при включённой сети (CIDR через запятую).</summary>
    public string NetworkAllow
    {
        get => _networkAllow;
        set => Set(ref _networkAllow, value);
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

    /// <summary>Открыть настройки на вкладке <paramref name="tab"/> (general, sound, models).</summary>
    public async Task OpenAsync(string tab = "general")
    {
        Error = null;
        Message = null;
        IsOpen = true;
        var target = Tabs.FirstOrDefault(t => t.Id == tab) ?? Tabs[0];
        if (target == _selectedTab && target.Id == "models" && _openModels is not null)
        {
            _ = _openModels();
        }

        SelectedTab = target;
        if (target.Id == "connections")
        {
            _ = Connections.LoadAsync();
        }

        try
        {
            var info = await _api().SettingsAsync().ConfigureAwait(true);
            Show(info);
            await Sound.LoadAsync(info.Settings).ConfigureAwait(true);
            ShowEngine((await _api().StatusAsync().ConfigureAwait(true)).Engine);
            _savedDefault = (await _api().OnboardingAsync().ConfigureAwait(true)).DefaultModel;
            try
            {
                ShowUpdate(await _api().UpdatesAsync().ConfigureAwait(true));
            }
            catch (Exception error) when (error is ApiException or NotSupportedException)
            {
                ShowUpdate(null); // без папки приложения обновлений нет
            }
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
        Connections.Relocalize();
        foreach (var t in Tabs)
        {
            t.Relocalize();
        }
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
        NetworkAllow = info.Settings.NetworkAllow;
        ModelsDirWarning = info.ModelsDirWarning is { } warning ? L[warning] : null;
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
            Theme: Theme is { } theme && theme.Code != saved.Theme ? theme.Code : null,
            NetworkAllow: NetworkAllow.Trim() != saved.NetworkAllow ? NetworkAllow.Trim() : null);
        var sound = Sound.Changes(saved);
        var api = Sound.ApiChanges(saved);
        update = update with
        {
            VoiceDevice = sound.Device, VoiceLanguage = sound.Language, VoiceModel = sound.Model,
            VoiceBackend = api.Backend, VoiceApiUrl = api.Url, VoiceApiModel = api.Model,
            VoiceAutoSend = Sound.AutoSendChange(saved),
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
            if (info.ApiError is { } apiError)
            {
                Message = null;
                Error = L.Format(apiError, ("port", info.Settings.ApiPort)); // 7.4: порт занят
            }

            if (update.ApiPort is not null || update.NetworkAccess is not null || update.NetworkAllow is not null)
            {
                await Connections.LoadAsync().ConfigureAwait(true); // адрес и состояние API
            }
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
