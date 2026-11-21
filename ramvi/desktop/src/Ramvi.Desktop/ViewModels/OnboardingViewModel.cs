using System.Collections.ObjectModel;
using System.Text.Json;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Строка проверки компьютера: значок статуса и текст (по ключу — на текущем языке).</summary>
public sealed class CheckItem(string id, string status, string code, IReadOnlyDictionary<string, object?> parameters)
    : ObservableObject
{
    public string Id { get; } = id;

    public string Status { get; } = status;

    public string Text => Localizer.Current.Format(code, parameters);

    public string Mark => Status switch
    {
        "ok" => "✓",
        "warn" => "!",
        "fail" => "✗",
        _ => "?",
    };

    public bool IsOk => Status == "ok";

    public bool IsWarn => Status == "warn";

    public bool IsFail => Status == "fail";
}

/// <summary>
/// Мастер первого запуска (UI-01, пункт 3.6): шаг 1 — проверка компьютера,
/// шаг 2 — выбор установленной модели или подключение файла GGUF без копирования.
/// Пропустить можно на любом шаге; модель выбирается и позже.
/// </summary>
public sealed class OnboardingViewModel : ObservableObject
{
    private readonly Func<IChatApi> _api;
    private readonly Func<Task<string?>> _pickFile;
    private readonly Func<OnboardingState, Task> _completed;
    private readonly Func<Task> _openCloud;
    private bool _cloudOnly;
    private bool _isOpen;
    private int _step = 1;
    private bool _isChecking;
    private bool _isImporting;
    private string? _error;
    private ModelInfo? _selected;

    public OnboardingViewModel(
        Func<IChatApi> api,
        ObservableCollection<ModelInfo> models,
        Func<Task<string?>> pickFile,
        Func<OnboardingState, Task> completed,
        Func<Task>? openCloud = null)
    {
        _api = api;
        _openCloud = openCloud ?? (() => Task.CompletedTask);
        Models = models;
        _pickFile = pickFile;
        _completed = completed;
        NextCommand = new Command(() =>
        {
            Step = 2;
            return Task.CompletedTask;
        }, () => Step == 1 && !CloudOnly);
        BackCommand = new Command(() =>
        {
            Step = 1;
            return Task.CompletedTask;
        }, () => Step == 2);
        ImportCommand = new Command(ImportAsync, () => !IsImporting);
        StartCommand = new Command(() => FinishAsync(Selected?.Id), () => Selected is not null && !IsImporting);
        SkipCommand = new Command(() => FinishAsync(null), () => !IsImporting);
        CloudCommand = new Command(async () =>
        {
            if (await FinishAsync(null).ConfigureAwait(true))
            {
                await _openCloud().ConfigureAwait(true); // «Подключения» → «Облачные модели»
            }
        });
    }

    /// <summary>B.12: облачная модель вместо локальной — мастер закрывается, открываются «Подключения».</summary>
    public Command CloudCommand { get; }

    /// <summary>Локальные модели здесь не работают (Mac): шаг выбора модели не нужен.</summary>
    public bool CloudOnly
    {
        get => _cloudOnly;
        private set
        {
            if (Set(ref _cloudOnly, value))
            {
                Refresh();
            }
        }
    }

    private static Localizer L => Localizer.Current;

    public ObservableCollection<CheckItem> Checks { get; } = [];

    public ObservableCollection<ModelInfo> Models { get; }

    public Command NextCommand { get; }

    public Command BackCommand { get; }

    public Command ImportCommand { get; }

    public Command StartCommand { get; }

    public Command SkipCommand { get; }

    public bool IsOpen
    {
        get => _isOpen;
        private set => Set(ref _isOpen, value);
    }

    public int Step
    {
        get => _step;
        private set
        {
            if (Set(ref _step, value))
            {
                Raise(nameof(IsCheckStep));
                Raise(nameof(IsModelStep));
                Refresh();
            }
        }
    }

    public bool IsCheckStep => Step == 1;

    public bool IsModelStep => Step == 2;

    public bool IsChecking
    {
        get => _isChecking;
        private set => Set(ref _isChecking, value);
    }

    public bool IsImporting
    {
        get => _isImporting;
        private set
        {
            if (Set(ref _isImporting, value))
            {
                Refresh();
            }
        }
    }

    public bool HasModels => Models.Count > 0;

    public ModelInfo? Selected
    {
        get => _selected;
        set
        {
            if (Set(ref _selected, value))
            {
                StartCommand.Refresh();
            }
        }
    }

    /// <summary>Ошибка импорта или проверки (текст на языке интерфейса).</summary>
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

    /// <summary>Показать мастер и запустить проверку компьютера.</summary>
    public async Task OpenAsync(string? defaultModel)
    {
        Step = 1;
        Error = null;
        Selected = Models.FirstOrDefault(m => m.Id == defaultModel) ?? Models.FirstOrDefault();
        Raise(nameof(HasModels));
        IsOpen = true;
        await CheckAsync().ConfigureAwait(true);
    }

    private async Task CheckAsync()
    {
        IsChecking = true;
        Checks.Clear();
        try
        {
            var report = await _api().SystemAsync().ConfigureAwait(true);
            CloudOnly = report.CloudOnly;
            foreach (var check in report.Checks)
            {
                Checks.Add(new CheckItem(check.Id, check.Status, check.Code, Params(check.Params)));
            }
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
        finally
        {
            IsChecking = false;
        }
    }

    /// <summary>Выбор файла — согласие подключить его на месте (без копирования).</summary>
    private async Task ImportAsync()
    {
        var path = await _pickFile().ConfigureAwait(true);
        if (string.IsNullOrEmpty(path))
        {
            return;
        }

        IsImporting = true;
        Error = null;
        try
        {
            var model = await _api().ImportModelAsync(path).ConfigureAwait(true);
            var existing = Models.FirstOrDefault(m => m.Id == model.Id);
            if (existing is null)
            {
                Models.Add(model);
                Raise(nameof(HasModels));
            }

            Selected = existing ?? model;
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
        }
        finally
        {
            IsImporting = false;
        }
    }

    private async Task<bool> FinishAsync(string? modelId)
    {
        try
        {
            var state = await _api().CompleteOnboardingAsync(modelId).ConfigureAwait(true);
            IsOpen = false;
            await _completed(state).ConfigureAwait(true);
            return true;
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            Error = Text(error);
            return false;
        }
    }

    private void Refresh()
    {
        foreach (var command in new[] { NextCommand, BackCommand, ImportCommand, StartCommand, SkipCommand })
        {
            command.Refresh();
        }
    }

    private static string Text(Exception error) => error is ApiException api
        ? L.Format(api.Code, Params(api.Parameters))
        : L["ui.connection_lost"];

    public override void Relocalize()
    {
        base.Relocalize();
        foreach (var check in Checks)
        {
            check.Relocalize();
        }
    }

    private static Dictionary<string, object?> Params(IReadOnlyDictionary<string, JsonElement>? values) =>
        values?.ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal)
        ?? new Dictionary<string, object?>(StringComparer.Ordinal);
}
