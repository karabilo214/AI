using System.Globalization;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Вариант в списке раздела «Звук»: код и подпись.</summary>
public sealed record VoiceOption(string Code, string Label);

/// <summary>
/// 5.2: раздел «Звук» в настройках — микрофон, язык распознавания, модель распознавания
/// и её скачивание. Выбор сохраняется вместе с остальными настройками кнопкой «Сохранить».
/// </summary>
public sealed class SoundSettingsViewModel : ObservableObject
{
    public static IReadOnlyList<string> LanguageCodes { get; } = ["auto", "ru", "en", "uk"];

    private readonly Func<IChatApi> _api;
    private readonly TimeSpan _poll;
    private VoiceOption? _device;
    private VoiceOption? _language;
    private VoiceOption? _model;
    private VoiceStatus? _status;
    private string? _error;
    private int _watch;
    private IReadOnlyList<VoiceDevice> _devices = [];

    public SoundSettingsViewModel(Func<IChatApi> api, TimeSpan? poll = null)
    {
        _api = api;
        _poll = poll ?? TimeSpan.FromMilliseconds(500);
        Languages = [.. LanguageOptions()];
        DownloadCommand = new Command(DownloadAsync, () => SelectedModel is { Installed: false } && !IsDownloading);
        CancelDownloadCommand = new Command(CancelDownloadAsync, () => IsDownloading);
        DownloadAccelCommand = new Command(
            DownloadAccelAsync, () => _status?.Accel is { Offered: true, Installed: false } && !IsDownloading);
    }

    private static Localizer L => Localizer.Current;

    public Command DownloadCommand { get; }

    public Command CancelDownloadCommand { get; }

    /// <summary>5.4: скачать ускоритель распознавания на видеокарте.</summary>
    public Command DownloadAccelCommand { get; }

    /// <summary>5.4: строка ускорителя видна только на Windows с NVIDIA.</summary>
    public bool AccelOffered => _status?.Accel is { Offered: true };

    /// <summary>5.4: ускоритель предлагается, но не скачан — видна кнопка скачивания.</summary>
    public bool AccelMissing => _status?.Accel is { Offered: true, Installed: false };

    /// <summary>5.4: «Готово…», «Не скачано (685 МБ)…», «Скачивается: 35 %» или ошибка.</summary>
    public string AccelStatus
    {
        get
        {
            if (_status?.Accel is not { } accel)
            {
                return "";
            }

            if (Progress(_status.Download, accel.Id) is { } progress)
            {
                return progress;
            }

            return accel.Installed
                ? L["voice.accel_installed"]
                : L.Format("voice.accel_not_installed", ("size", Megabytes(accel.Size)));
        }
    }

    /// <summary>5.4: где прошло последнее распознавание и за сколько.</summary>
    public string LastText => _status?.Last is { } last
        ? L.Format(last.Backend == "gpu" ? "voice.last_gpu" : "voice.last_cpu",
            ("seconds", last.ProcS.ToString("0.0", L.Language == "en" ? CultureInfo.InvariantCulture : CultureInfo.GetCultureInfo(L.Language))))
        : "";

    public bool HasLast => _status?.Last is not null;

    /// <summary>Микрофоны; первый вариант (код «») — системный по умолчанию.</summary>
    public List<VoiceOption> Devices { get; private set; } = [];

    public List<VoiceOption> Languages { get; private set; }

    public List<VoiceOption> Models { get; private set; } = [];

    public VoiceOption? Device
    {
        get => _device;
        set => Set(ref _device, value);
    }

    public VoiceOption? Language
    {
        get => _language;
        set => Set(ref _language, value);
    }

    public VoiceOption? Model
    {
        get => _model;
        set
        {
            if (Set(ref _model, value))
            {
                RaiseModel();
            }
        }
    }

    private VoiceModelOption? SelectedModel =>
        _status?.Models.FirstOrDefault(m => m.Id == _model?.Code);

    public bool IsDownloading => _status?.Download.State == "running";

    /// <summary>«Скачана», «Не скачана (574 МБ)», «Скачивается: 35 %» или ошибка скачивания.</summary>
    public string ModelStatus
    {
        get
        {
            if (_status is not { } status || SelectedModel is not { } model)
            {
                return "";
            }

            if (Progress(status.Download, model.Id) is { } progress)
            {
                return progress;
            }

            return model.Installed
                ? L["voice.model_installed"]
                : L.Format("voice.model_not_installed", ("size", Megabytes(model.Size)));
        }
    }

    /// <summary>Ход или ошибка скачивания для `what` (модель или ускоритель); null — не оно.</summary>
    private static string? Progress(VoiceDownload download, string what)
    {
        if (download.Model != what)
        {
            return null;
        }

        if (download.State == "running")
        {
            var percent = download.Total is > 0 ? (int)(100 * (download.Done ?? 0) / download.Total.Value) : 0;
            return L.Format("voice.model_downloading", ("percent", percent));
        }

        return download.State == "failed" && download.Error is { } error ? L.Format(error.Code, Params(error.Params)) : null;
    }

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

    /// <summary>Загрузить микрофоны и состояние модели; выбрать сохранённые значения.</summary>
    public async Task LoadAsync(AppSettings saved)
    {
        Error = null;
        Show(saved, []);
        try
        {
            SetStatus(await _api().VoiceStatusAsync().ConfigureAwait(true));
            Show(saved);
            Show(saved, await _api().VoiceDevicesAsync().ConfigureAwait(true));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error); // нет микрофона или звуковой подсистемы — настройки всё равно работают
        }

        if (IsDownloading)
        {
            _ = WatchDownloadAsync(++_watch);
        }
    }

    /// <summary>Показать сохранённый выбор; микрофон, которого нет в системе, остаётся в списке.</summary>
    public void Show(AppSettings saved, IReadOnlyList<VoiceDevice>? devices = null)
    {
        _devices = devices ?? _devices;
        Devices = [new VoiceOption("", L["voice.device_default"]), .. _devices.Select(d => new VoiceOption(d.Name, d.Name))];
        if (saved.VoiceDevice.Length > 0 && Devices.All(d => d.Code != saved.VoiceDevice))
        {
            Devices.Add(new VoiceOption(saved.VoiceDevice, L.Format("voice.device_absent", ("device", saved.VoiceDevice))));
        }

        Raise(nameof(Devices));
        Device = Devices.First(d => d.Code == saved.VoiceDevice || (saved.VoiceDevice.Length == 0 && d.Code.Length == 0));
        Language = Languages.FirstOrDefault(l => l.Code == saved.VoiceLanguage) ?? Languages[0];
        Model = Models.FirstOrDefault(m => m.Code == saved.VoiceModel) ?? Models.FirstOrDefault();
    }

    /// <summary>Изменённые поля для сохранения (null — без изменений).</summary>
    public (string? Device, string? Language, string? Model) Changes(AppSettings saved) => (
        Device is { } device && device.Code != saved.VoiceDevice ? device.Code : null,
        Language is { } language && language.Code != saved.VoiceLanguage ? language.Code : null,
        Model is { } model && model.Code != saved.VoiceModel ? model.Code : null);

    public void Stop() => _watch++;

    public override void Relocalize()
    {
        var device = Device?.Code;
        var language = Language?.Code;
        var model = Model?.Code;
        Languages = [.. LanguageOptions()];
        Models = [.. ModelOptions()];
        if (Devices.Count > 0)
        {
            Devices[0] = new VoiceOption("", L["voice.device_default"]);
        }

        base.Relocalize();
        Device = Devices.FirstOrDefault(d => d.Code == device);
        Language = Languages.FirstOrDefault(l => l.Code == language);
        Model = Models.FirstOrDefault(m => m.Code == model);
    }

    private void SetStatus(VoiceStatus status)
    {
        var model = _model?.Code;
        var first = _status is null;
        _status = status;
        if (first || Models.Count != status.Models.Count)
        {
            Models = [.. ModelOptions()];
            Raise(nameof(Models));
            Model = Models.FirstOrDefault(m => m.Code == model);
        }

        Raise(nameof(IsDownloading));
        RaiseModel();
    }

    private void RaiseModel()
    {
        Raise(nameof(ModelStatus));
        Raise(nameof(AccelOffered));
        Raise(nameof(AccelMissing));
        Raise(nameof(AccelStatus));
        Raise(nameof(LastText));
        Raise(nameof(HasLast));
        DownloadCommand.Refresh();
        CancelDownloadCommand.Refresh();
        DownloadAccelCommand.Refresh();
    }

    private static IEnumerable<VoiceOption> LanguageOptions() =>
        LanguageCodes.Select(code => new VoiceOption(code, L[$"voice.lang.{code}"]));

    private IEnumerable<VoiceOption> ModelOptions() =>
        (_status?.Models ?? []).Select(m => new VoiceOption(
            m.Id, L.Format($"voice.model.{m.Id.Replace('-', '_')}", ("size", Megabytes(m.Size)))));

    private async Task DownloadAsync()
    {
        Error = null;
        try
        {
            SetStatus(await _api().DownloadVoiceModelAsync(_model?.Code).ConfigureAwait(true));
            _ = WatchDownloadAsync(++_watch);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    private async Task DownloadAccelAsync()
    {
        Error = null;
        try
        {
            SetStatus(await _api().DownloadVoiceAccelAsync().ConfigureAwait(true));
            _ = WatchDownloadAsync(++_watch);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    private async Task CancelDownloadAsync()
    {
        try
        {
            SetStatus(await _api().CancelVoiceDownloadAsync().ConfigureAwait(true));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    private async Task WatchDownloadAsync(int watch)
    {
        try
        {
            while (watch == _watch)
            {
                await Task.Delay(_poll).ConfigureAwait(true);
                if (watch != _watch)
                {
                    return;
                }

                SetStatus(await _api().VoiceStatusAsync().ConfigureAwait(true));
                if (!IsDownloading)
                {
                    return;
                }
            }
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
    }

    private static string Megabytes(long bytes) =>
        (bytes / 1_000_000).ToString(CultureInfo.InvariantCulture);

    private static Dictionary<string, object?> Params(Dictionary<string, System.Text.Json.JsonElement>? values) =>
        values?.ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal)
        ?? new Dictionary<string, object?>(StringComparer.Ordinal);

    private static string Text(Exception error) => error is ApiException api
        ? L.Format(api.Code, api.Parameters.ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal))
        : L["ui.connection_lost"];
}
