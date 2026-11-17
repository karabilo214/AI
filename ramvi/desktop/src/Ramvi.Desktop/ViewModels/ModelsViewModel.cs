using System.Collections.ObjectModel;
using System.Text.Json;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>
/// Карточка модели на экране «Модели» (4.11): установленная модель или запись каталога.
/// Одна и та же модель из каталога и установленная — одна карточка.
/// </summary>
public sealed class ModelCardViewModel : ObservableObject
{
    private static readonly string[] Active = ["queued", "running", "verifying"];
    private readonly ModelsViewModel _owner;
    private DownloadInfo? _download;
    private bool _confirmDelete;

    public ModelCardViewModel(
        ModelsViewModel owner, string key, string name, string? quant, long bytes, CompatInfo? compat,
        string? modelId, string? entryId, bool ownsFile)
    {
        _owner = owner;
        (Key, Name, Quant, Bytes, Compat, ModelId, EntryId, OwnsFile) =
            (key, name, quant, bytes, compat, modelId, entryId, ownsFile);
        DownloadCommand = new Command(() => _owner.StartDownloadAsync(this), () => CanDownload);
        PauseCommand = new Command(() => _owner.DownloadActionAsync(this, "pause"), () => IsDownloading);
        ResumeCommand = new Command(() => _owner.DownloadActionAsync(this, "resume"), () => IsPaused);
        CancelCommand = new Command(() => _owner.DownloadActionAsync(this, "cancel"), () => IsDownloading || IsPaused);
        MakeDefaultCommand = new Command(() => _owner.MakeDefaultAsync(this), () => IsInstalled && !IsDefault);
        DeleteCommand = new Command(DeleteAsync, () => IsInstalled);
    }

    private static Localizer L => Localizer.Current;

    public string Key { get; }

    public string Name { get; }

    public string? Quant { get; }

    public long Bytes { get; }

    public CompatInfo? Compat { get; }

    /// <summary>Установленная модель (null — только в каталоге).</summary>
    public string? ModelId { get; }

    /// <summary>Запись каталога, из которой модель можно скачать.</summary>
    public string? EntryId { get; }

    /// <summary>Файл скачал RAMVI — при удалении удаляется и файл.</summary>
    public bool OwnsFile { get; }

    public bool IsInstalled => ModelId is not null;

    public bool IsDefault => ModelId is not null && ModelId == _owner.DefaultModel;

    public DownloadInfo? Download
    {
        get => _download;
        set
        {
            if (Set(ref _download, value))
            {
                Raise(string.Empty);
                RefreshCommands();
            }
        }
    }

    public bool IsDownloading => _download is not null && Active.Contains(_download.State);

    public bool IsPaused => _download?.State == "paused";

    public bool CanDownload => !IsInstalled && EntryId is not null && !IsDownloading && !IsPaused;

    public bool HasProgress => IsDownloading || IsPaused;

    public double Progress => _download is { TotalBytes: > 0 } d ? 100.0 * d.BytesDone / d.TotalBytes.Value : 0;

    public string ProgressText => _download is null
        ? ""
        : L.Format("models.progress", ("percent", Math.Round(Progress)), ("done", Gb(_download.BytesDone)),
            ("total", Gb(_download.TotalBytes ?? 0)));

    /// <summary>Причина ошибки загрузки (перевод ключа сервиса).</summary>
    public string? DownloadError => _download?.Error is { } error ? ModelsViewModel.Text(error) : null;

    public bool HasDownloadError => DownloadError is not null && !IsDownloading;

    /// <summary>Плашка состояния, как на иллюстрации ТЗ.</summary>
    public string StateText =>
        IsDefault ? L["models.state.current"]
        : IsInstalled ? L["models.state.installed"]
        : IsDownloading ? L["models.state.downloading"]
        : IsPaused ? L["models.state.paused"]
        : L["models.state.online"];

    public bool IsCurrentChip => IsDefault;

    /// <summary>Статус совместимости с этим компьютером (4.4).</summary>
    public string CompatText => Compat is null ? "" : L[Compat.Status];

    public bool IsRecommended => Compat?.Status == "compat.recommended";

    public bool IsSlow => Compat?.Status == "compat.slow";

    public bool IsUnfit => Compat is { Status: not ("compat.recommended" or "compat.slow" or "compat.unknown") };

    public bool SuitsTask => Compat?.Task?.Suits ?? true;

    /// <summary>«Гибрид GPU+RAM · оценка ~8–12 токенов/с».</summary>
    public string Detail
    {
        get
        {
            var parts = new List<string>();
            if (Compat?.Plan is { } plan)
            {
                parts.Add(plan.Gpu >= plan.Layers ? L["models.plan.gpu"]
                    : plan.Gpu == 0 ? L["models.plan.cpu"]
                    : L.Format("models.plan.hybrid", ("gpu", plan.Gpu), ("layers", plan.Layers)));
            }

            if (Compat?.Speed is { } speed)
            {
                parts.Add(L.Message(speed));
            }

            parts.Add(Gb(Bytes) + " " + L["models.gb"]);
            return string.Join(" · ", parts);
        }
    }

    /// <summary>4.7 / экзамен 2 ТЗ: фактические ресурсы из контрольного прогона на этом ПК.</summary>
    public string? Measured => Compat?.Control is { } c
        ? L.Format("models.measured",
            ("vram", Gb(c.PeakVram ?? 0)), ("ram", Gb(c.PeakRam ?? 0)),
            ("ttft", (c.TtftS ?? 0).ToString("0.00", System.Globalization.CultureInfo.CurrentCulture)))
        : null;

    public bool HasMeasured => Measured is not null;

    /// <summary>Подсказка: все причины статуса и почему не подходит задаче.</summary>
    public string Reasons
    {
        get
        {
            var reasons = (Compat?.Reasons ?? []).Concat(Compat?.Task?.Reasons ?? []);
            return string.Join("\n", reasons.Select(L.Message));
        }
    }

    public bool IsConfirmingDelete
    {
        get => _confirmDelete;
        private set
        {
            if (Set(ref _confirmDelete, value))
            {
                Raise(nameof(DeleteLabel));
            }
        }
    }

    public string DeleteLabel => IsConfirmingDelete
        ? L[OwnsFile ? "models.delete_confirm_file" : "models.delete_confirm"]
        : L["models.delete"];

    public string DeleteHint => L[OwnsFile ? "models.delete_hint_file" : "models.delete_hint"];

    public Command DownloadCommand { get; }

    public Command PauseCommand { get; }

    public Command ResumeCommand { get; }

    public Command CancelCommand { get; }

    public Command MakeDefaultCommand { get; }

    public Command DeleteCommand { get; }

    /// <summary>Удаление — вторым нажатием: первое только спрашивает «точно?».</summary>
    private Task DeleteAsync()
    {
        if (!IsConfirmingDelete)
        {
            IsConfirmingDelete = true;
            return Task.CompletedTask;
        }

        IsConfirmingDelete = false;
        return _owner.DeleteAsync(this);
    }

    public void CancelConfirm() => IsConfirmingDelete = false;

    public void RefreshCommands()
    {
        foreach (var command in new[] { DownloadCommand, PauseCommand, ResumeCommand, CancelCommand, MakeDefaultCommand, DeleteCommand })
        {
            command.Refresh();
        }
    }

    private static string Gb(long bytes) => (bytes / 1e9).ToString("0.0", System.Globalization.CultureInfo.CurrentCulture);
}

/// <summary>
/// Экран «Модели» (UI-CATALOG-01, пункт 4.11) по иллюстрации ТЗ «Каталог моделей»:
/// поиск, фильтры «Все / Установленные / Подходят», карточки с оценкой скорости и статусом
/// совместимости, «Скачать» с прогрессом и паузой, модель по умолчанию, удаление.
/// Данные — из сервиса (каталог 4.2, загрузки 4.3, совместимость 4.4, выбор 4.6).
/// </summary>
public sealed class ModelsViewModel : ObservableObject
{
    public const string All = "all";
    public const string Installed = "installed";
    public const string Suitable = "suitable";

    private readonly Func<IChatApi> _api;
    private readonly Func<Task> _modelsChanged;
    private readonly TimeSpan _poll;
    private readonly List<ModelCardViewModel> _all = [];
    private string _search = "";
    private string _filter = All;
    private string? _error;
    private string? _recommendation;
    private bool _stale;
    private bool _busy;
    private bool _polling;

    /// <param name="modelsChanged">Список установленных моделей изменился (скачали, удалили, по умолчанию).</param>
    /// <param name="poll">Как часто обновлять прогресс загрузок.</param>
    public ModelsViewModel(Func<IChatApi> api, Func<Task> modelsChanged, TimeSpan? poll = null)
    {
        _api = api;
        _modelsChanged = modelsChanged;
        _poll = poll ?? TimeSpan.FromSeconds(1);
        RefreshCommand = new Command(RefreshCatalogAsync, () => !IsBusy);
        FilterCommand = new Command(p =>
        {
            Filter = p as string ?? All;
            return Task.CompletedTask;
        });
    }

    private static Localizer L => Localizer.Current;

    public ObservableCollection<ModelCardViewModel> Cards { get; } = [];

    public string? DefaultModel { get; private set; }

    public string Search
    {
        get => _search;
        set
        {
            if (Set(ref _search, value ?? ""))
            {
                ApplyFilter();
            }
        }
    }

    public string Filter
    {
        get => _filter;
        set
        {
            if (Set(ref _filter, value))
            {
                Raise(nameof(IsAll));
                Raise(nameof(IsInstalledFilter));
                Raise(nameof(IsSuitableFilter));
                ApplyFilter();
            }
        }
    }

    public bool IsAll => Filter == All;

    public bool IsInstalledFilter => Filter == Installed;

    public bool IsSuitableFilter => Filter == Suitable;

    public bool IsEmpty => Cards.Count == 0;

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

    public bool HasError => Error is not null;

    /// <summary>Нижняя строка: какая модель выбрана для профиля задачи и почему (4.6).</summary>
    public string? Recommendation
    {
        get => _recommendation;
        private set => Set(ref _recommendation, value);
    }

    /// <summary>Каталог не обновился (нет сети) — показана сохранённая копия.</summary>
    public bool IsStale
    {
        get => _stale;
        private set => Set(ref _stale, value);
    }

    public bool IsBusy
    {
        get => _busy;
        private set
        {
            if (Set(ref _busy, value))
            {
                RefreshCommand.Refresh();
            }
        }
    }

    public Command RefreshCommand { get; }

    public Command FilterCommand { get; }

    /// <summary>Перечитать всё из сервиса (без сети): каталог, установленные, загрузки, выбор.</summary>
    public async Task LoadAsync()
    {
        try
        {
            var api = _api();
            var catalog = await api.CatalogAsync().ConfigureAwait(true);
            var installed = await api.ModelsAsync().ConfigureAwait(true);
            var downloads = await api.DownloadsAsync().ConfigureAwait(true);
            DefaultModel = (await api.OnboardingAsync().ConfigureAwait(true)).DefaultModel;
            Build(catalog, installed, downloads);
            IsStale = catalog.Stale;
            Recommendation = await RecommendationTextAsync(api).ConfigureAwait(true);
            Error = null;
            PollWhileDownloading();
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Describe(error);
        }
    }

    private static async Task<string?> RecommendationTextAsync(IChatApi api)
    {
        var rec = await api.RecommendationsAsync().ConfigureAwait(true);
        var reason = L.Message(rec.Reason);
        return rec.Best is null ? reason : L.Format("models.recommend", ("name", rec.Best.Name), ("reason", reason));
    }

    /// <summary>«Обновить каталог»: спросить провайдеров (сеть), затем перечитать экран.</summary>
    private async Task RefreshCatalogAsync()
    {
        IsBusy = true;
        try
        {
            await _api().RefreshCatalogAsync().ConfigureAwait(true);
            await LoadAsync().ConfigureAwait(true);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Describe(error);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void Build(CatalogView catalog, IReadOnlyList<ModelInfo> installed, IReadOnlyList<DownloadInfo> downloads)
    {
        _all.Clear();
        var latest = new Dictionary<string, DownloadInfo>(StringComparer.Ordinal);
        foreach (var item in downloads)
        {
            latest[item.EntryId] = item; // по порядку: последняя загрузка записи
        }

        var taken = new HashSet<string>(StringComparer.Ordinal);
        foreach (var model in installed)
        {
            var key = model.Compat?.Model?.Key is { Length: > 0 } k ? k : model.Id;
            var catalogId = Capability(model, "catalog_id");
            var owns = catalogId is not null && Capability(model, "imported") == "False";
            _all.Add(new ModelCardViewModel(
                this, key, model.Id, model.Quant, model.Bytes ?? 0, model.Compat, model.Id, catalogId, owns));
            taken.Add(key);
            if (catalogId is not null)
            {
                taken.Add(catalogId);
            }
        }

        foreach (var entry in catalog.Entries)
        {
            var key = entry.Compat?.Model?.Key is { Length: > 0 } k ? k : entry.Id;
            if (taken.Contains(entry.Id) || taken.Contains(key))
            {
                continue; // уже установлена — одна карточка
            }

            taken.Add(key);
            var card = new ModelCardViewModel(
                this, key, entry.Name, entry.Quantization, entry.Bytes, entry.Compat, null, entry.Id, false);
            if (latest.TryGetValue(entry.Id, out var download) && download.State != "done")
            {
                card.Download = download;
            }

            _all.Add(card);
        }

        ApplyFilter();
    }

    private static string? Capability(ModelInfo model, string name)
    {
        if (model.Capabilities is null || !model.Capabilities.TryGetValue(name, out var value))
        {
            return null;
        }

        return value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            JsonValueKind.True => "True",
            JsonValueKind.False => "False",
            _ => value.GetRawText(),
        };
    }

    private void ApplyFilter()
    {
        var text = Search.Trim();
        var shown = _all.Where(card =>
            (Filter != Installed || card.IsInstalled)
            && (Filter != Suitable || ((card.IsRecommended || card.IsSlow) && card.SuitsTask))
            && (text.Length == 0
                || card.Name.Contains(text, StringComparison.OrdinalIgnoreCase)
                || (card.Quant?.Contains(text, StringComparison.OrdinalIgnoreCase) ?? false)));
        // Сначала установленные, потом подходящие этому компьютеру, потом остальные.
        var ordered = shown
            .OrderBy(c => c.IsInstalled ? 0 : c.IsRecommended ? 1 : c.IsSlow ? 2 : 3)
            .ThenBy(c => c.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
        Cards.Clear();
        foreach (var card in ordered)
        {
            Cards.Add(card);
        }

        Raise(nameof(IsEmpty));
    }

    internal async Task StartDownloadAsync(ModelCardViewModel card)
    {
        if (card.EntryId is null)
        {
            return;
        }

        try
        {
            card.Download = await _api().StartDownloadAsync(card.EntryId).ConfigureAwait(true);
            PollWhileDownloading();
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Describe(error);
        }
    }

    internal async Task DownloadActionAsync(ModelCardViewModel card, string action)
    {
        if (card.Download is null)
        {
            return;
        }

        try
        {
            card.Download = await _api().DownloadActionAsync(card.Download.Id, action).ConfigureAwait(true);
            if (card.Download.State == "cancelled")
            {
                card.Download = null;
            }

            PollWhileDownloading();
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Describe(error);
        }
    }

    internal async Task MakeDefaultAsync(ModelCardViewModel card)
    {
        if (card.ModelId is null)
        {
            return;
        }

        try
        {
            await _api().SetDefaultModelAsync(card.ModelId).ConfigureAwait(true);
            DefaultModel = card.ModelId;
            foreach (var item in _all)
            {
                item.Relocalize();
                item.RefreshCommands();
            }

            await _modelsChanged().ConfigureAwait(true);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Describe(error);
        }
    }

    internal async Task DeleteAsync(ModelCardViewModel card)
    {
        if (card.ModelId is null)
        {
            return;
        }

        try
        {
            await _api().DeleteModelAsync(card.ModelId, card.OwnsFile).ConfigureAwait(true);
            await LoadAsync().ConfigureAwait(true);
            await _modelsChanged().ConfigureAwait(true);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Describe(error);
        }
    }

    /// <summary>Пока есть идущие загрузки — обновлять прогресс; скачанная модель появляется в списке.</summary>
    private async void PollWhileDownloading()
    {
        if (_polling)
        {
            return;
        }

        _polling = true;
        try
        {
            while (_all.Any(c => c.IsDownloading))
            {
                await Task.Delay(_poll).ConfigureAwait(true);
                IReadOnlyList<DownloadInfo> downloads;
                try
                {
                    downloads = await _api().DownloadsAsync().ConfigureAwait(true);
                }
                catch (Exception error) when (error is ApiException or HttpRequestException)
                {
                    Error = Describe(error);
                    return;
                }

                var finished = false;
                foreach (var card in _all.Where(c => c.Download is not null))
                {
                    var now = downloads.LastOrDefault(d => d.Id == card.Download!.Id);
                    if (now is null)
                    {
                        continue;
                    }

                    finished |= now.State == "done";
                    card.Download = now.State is "done" or "cancelled" ? null : now;
                }

                if (finished)
                {
                    await LoadAsync().ConfigureAwait(true);
                    await _modelsChanged().ConfigureAwait(true);
                }
            }
        }
        finally
        {
            _polling = false;
        }
    }

    internal static string Text(ApiError error)
    {
        var values = (error.Params ?? []).ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal);
        return L.Format(error.Code, values);
    }

    private static string Describe(Exception error) => error switch
    {
        ApiException api => Text(new ApiError(api.Code, new Dictionary<string, JsonElement>(api.Parameters))),
        _ => L.Format("models.error", ("detail", error.Message)),
    };

    public override void Relocalize()
    {
        base.Relocalize();
        foreach (var card in _all)
        {
            card.Relocalize();
        }
    }
}
