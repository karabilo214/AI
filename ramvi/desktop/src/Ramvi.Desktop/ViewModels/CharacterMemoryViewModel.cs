using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>9.17: просмотр, исправление, экспорт и удаление памяти и истории персонажа.</summary>
public sealed class CharacterMemoryViewModel : ObservableObject
{
    // Читаемый файл: с отступами и кириллицей как есть.
    private static readonly JsonWriterOptions ExportFormat = new() { Indented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
    private readonly Func<string, JsonObject?, Task<JsonElement>> _call;
    private readonly Func<Task> _historyChanged;
    private string _status = "";
    private string _error = "";
    private bool _available;
    private bool _clearMemoryArmed;
    private bool _clearHistoryArmed;
    private bool _resetArmed;
    private bool _adaptationEnabled = true;
    private bool _adaptationAvailable;
    private bool _loadingAdaptation;

    public CharacterMemoryViewModel(Func<string, JsonObject?, Task<JsonElement>> call, Func<Task> historyChanged)
    {
        _call = call;
        _historyChanged = historyChanged;
        RefreshCommand = new Command(() => RefreshAsync());
        ClearMemoryCommand = new Command(() => ArmOrRunAsync(true));
        ClearHistoryCommand = new Command(() => ArmOrRunAsync(false));
        ClearFeedbackCommand = new Command(() => FeedbackAsync("feedback_clear"));
        CancelClearCommand = new Command(() => { ClearMemoryArmed = false; ClearHistoryArmed = false; ResetArmed = false; return Task.CompletedTask; });
        SaveAdaptationCommand = new Command(() => AdaptationAsync("adaptation_save", null));
        UndoAdaptationCommand = new Command(() => AdaptationAsync("adaptation_restore", null), () => AdaptationHistory.Count > 0);
        ResetAdaptationCommand = new Command(ResetOrArmAsync, () => Corrections.Count > 0);
        ForgetBehaviorCommand = new Command(ForgetBehaviorAsync);
    }

    // 9.34: что персонаж понял о владельце (таблица поведения) и его настроение.
    private bool _forgetBehaviorArmed;
    private string _mood = "";
    public ObservableCollection<BehaviorItem> Behavior { get; } = [];
    public bool NoBehavior => Behavior.Count == 0;
    public string Mood { get => _mood; private set => Set(ref _mood, value); }
    public Command ForgetBehaviorCommand { get; }
    public bool ForgetBehaviorArmed { get => _forgetBehaviorArmed; private set { if (Set(ref _forgetBehaviorArmed, value)) Raise(nameof(ForgetBehaviorLabel)); } }
    public string ForgetBehaviorLabel => Localizer.Current[_forgetBehaviorArmed ? "character.behavior_forget_confirm" : "character.behavior_forget"];
    private JsonElement? _moodState;

    private async Task BehaviorAsync()
    {
        try { ReadBehavior(await _call("behavior", null).ConfigureAwait(true)); }
        catch (Exception error) when (CharacterViewModel.Handled(error)) { Error = Localizer.Current["character.memory_error"]; }
    }

    private async Task ForgetBehaviorAsync()
    {
        if (!ForgetBehaviorArmed) { ForgetBehaviorArmed = true; return; }
        ForgetBehaviorArmed = false;
        try
        {
            await _call("brain_forget", null).ConfigureAwait(true);
            await BehaviorAsync().ConfigureAwait(true);
        }
        catch (Exception error) when (CharacterViewModel.Handled(error)) { Error = Localizer.Current["character.memory_error"]; }
    }

    internal void ReadBehavior(JsonElement result)
    {
        Behavior.Clear();
        foreach (var row in result.GetProperty("rows").EnumerateArray()) Behavior.Add(new BehaviorItem(row));
        Raise(nameof(NoBehavior));
        _moodState = result.GetProperty("mood").Clone();
        Mood = MoodText(_moodState.Value);
    }

    private static string MoodText(JsonElement mood)
    {
        var parts = new List<string>();
        static string N(double v) => v.ToString("0.00", System.Globalization.CultureInfo.CurrentCulture);
        var irritation = mood.GetProperty("irritation").GetDouble();
        var cheer = mood.GetProperty("cheer").GetDouble();
        if (irritation >= 0.05) parts.Add(Localizer.Current["character.mood_irritated"].Replace("{value}", N(irritation), StringComparison.Ordinal));
        if (cheer >= 0.05) parts.Add(Localizer.Current["character.mood_cheer"].Replace("{value}", N(cheer), StringComparison.Ordinal));
        return parts.Count == 0 ? Localizer.Current["character.mood_none"] : string.Join(" · ", parts);
    }

    // 9.20: подстройка характера — переключатель, текущие поправки, журнал, точки сохранения.
    public ObservableCollection<CorrectionItem> Corrections { get; } = [];
    public ObservableCollection<AdaptationChangeItem> AdaptationHistory { get; } = [];
    public ObservableCollection<AdaptationPointItem> SavedPoints { get; } = [];
    public Command SaveAdaptationCommand { get; }
    public Command UndoAdaptationCommand { get; }
    public Command ResetAdaptationCommand { get; }
    public bool NoCorrections => Corrections.Count == 0;
    public bool NoAdaptationHistory => AdaptationHistory.Count == 0;
    public bool HasSavedPoints => SavedPoints.Count > 0;
    public bool ResetArmed { get => _resetArmed; private set => Set(ref _resetArmed, value); }
    public bool AdaptationAvailable { get => _adaptationAvailable; private set => Set(ref _adaptationAvailable, value); }

    /// <summary>Выключение не сбрасывает ни поправки, ни память, ни профиль — они просто не применяются.</summary>
    public bool AdaptationEnabled
    {
        get => _adaptationEnabled;
        set
        {
            if (!Set(ref _adaptationEnabled, value) || _loadingAdaptation) return;
            _ = AdaptationAsync("adaptation_enable", new JsonObject { ["enabled"] = value });
        }
    }

    public ObservableCollection<MemoryRecordItem> Records { get; } = [];
    public ObservableCollection<HistoryTurnItem> History { get; } = [];
    public ObservableCollection<PreferenceItem> Preferences { get; } = [];
    public Command ClearFeedbackCommand { get; }
    public bool NoPreferences => Preferences.Count == 0;
    public Command RefreshCommand { get; }
    public Command ClearMemoryCommand { get; }
    public Command ClearHistoryCommand { get; }
    public Command CancelClearCommand { get; }
    public string Status { get => _status; private set => Set(ref _status, value); }
    public string Error { get => _error; private set => Set(ref _error, value); }
    public bool Available { get => _available; private set => Set(ref _available, value); }
    public bool ClearMemoryArmed { get => _clearMemoryArmed; private set => Set(ref _clearMemoryArmed, value); }
    public bool ClearHistoryArmed { get => _clearHistoryArmed; private set => Set(ref _clearHistoryArmed, value); }
    public bool NoRecords => Records.Count == 0;
    public bool NoHistory => History.Count == 0;

    public override void Relocalize()
    {
        base.Relocalize();
        foreach (var record in Records) record.Relocalize();
        foreach (var preference in Preferences) preference.Relocalize();
        foreach (var item in Corrections) item.Relocalize();
        foreach (var item in AdaptationHistory) item.Relocalize();
        foreach (var item in SavedPoints) item.Relocalize();
        foreach (var item in Behavior) item.Relocalize();
        if (_moodState is { } mood) Mood = MoodText(mood);
        Raise(nameof(ForgetBehaviorLabel));
    }

    /// <summary>Перечитать память и историю; status — уже полученный ответ, если есть.</summary>
    public async Task RefreshAsync(JsonElement? status = null)
    {
        await RunAsync("memory", null).ConfigureAwait(true);
        try
        {
            ReadHistory(status ?? await _call("status", null).ConfigureAwait(true));
            if (Available)
            {
                await FeedbackAsync("feedback_view").ConfigureAwait(true);
                await AdaptationAsync("adaptation", null).ConfigureAwait(true);
                await BehaviorAsync().ConfigureAwait(true);
            }
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        { Error = Localizer.Current["character.memory_error"]; }
    }

    /// <summary>Документ экспорта (JSON) для сохранения в файл, выбранный пользователем.</summary>
    public async Task<string?> ExportAsync()
    {
        try
        {
            var result = await _call("memory_export", null).ConfigureAwait(true);
            Error = "";
            Status = Localizer.Current["character.memory_exported"];
            using var buffer = new MemoryStream();
            using (var writer = new Utf8JsonWriter(buffer, ExportFormat)) result.GetProperty("memory").WriteTo(writer);
            return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        { Error = Localizer.Current["character.memory_error"]; return null; }
    }

    private async Task ArmOrRunAsync(bool memory)
    {
        // Полная очистка — в два нажатия: сначала вопрос, потом подтверждение.
        if (memory && !ClearMemoryArmed) { ClearMemoryArmed = true; ClearHistoryArmed = false; return; }
        if (!memory && !ClearHistoryArmed) { ClearHistoryArmed = true; ClearMemoryArmed = false; return; }
        ClearMemoryArmed = false; ClearHistoryArmed = false;
        if (memory) await RunAsync("memory_clear", null).ConfigureAwait(true);
        else await HistoryAsync("history_clear", null).ConfigureAwait(true);
    }

    private async Task RunAsync(string operation, JsonObject? arguments)
    {
        try
        {
            var result = await _call(operation, arguments).ConfigureAwait(true);
            var memory = result.GetProperty("memory");
            Available = true; Error = "";
            Records.Clear();
            foreach (var record in memory.GetProperty("records").EnumerateArray())
                Records.Add(new MemoryRecordItem(record, this));
            Raise(nameof(NoRecords));
            if (operation != "memory") Status = Localizer.Current["character.memory_done"];
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        {
            // Список не открылся — у модуля нет своей базы памяти; иначе — ошибка действия.
            if (operation == "memory") { Available = false; Error = Localizer.Current["character.memory_unavailable"]; }
            else Error = Localizer.Current["character.memory_error"];
        }
    }

    private async Task HistoryAsync(string operation, JsonObject? arguments)
    {
        try
        {
            var result = await _call(operation, arguments).ConfigureAwait(true);
            if (!result.TryGetProperty("history", out var history)) { Error = Localizer.Current["character.memory_error"]; return; }
            var forgotten = history.GetProperty("memory_forgotten").GetInt32();
            Status = forgotten > 0 ? Localizer.Current["character.history_removed_memory"].Replace("{count}", forgotten.ToString(System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal) : Localizer.Current["character.memory_done"];
            Error = "";
            await RefreshAsync().ConfigureAwait(true);
            await _historyChanged().ConfigureAwait(true);
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        { Error = Localizer.Current["character.memory_error"]; }
    }

    private async Task FeedbackAsync(string operation)
    {
        try { ReadPreferences((await _call(operation, null).ConfigureAwait(true)).GetProperty("feedback")); }
        catch (Exception error) when (CharacterViewModel.Handled(error)) { Error = Localizer.Current["character.memory_error"]; }
    }

    private async Task ResetOrArmAsync()
    {
        // Сброс — в два нажатия; прежняя настройка остаётся в журнале и возвращается откатом.
        if (!ResetArmed) { ResetArmed = true; return; }
        ResetArmed = false;
        await AdaptationAsync("adaptation_reset", null).ConfigureAwait(true);
    }

    private async Task AdaptationAsync(string operation, JsonObject? arguments)
    {
        try
        {
            var result = await _call(operation, arguments).ConfigureAwait(true);
            if (!result.TryGetProperty("adaptation", out var view))
            {
                if (operation == "adaptation") AdaptationAvailable = false;
                else Error = Localizer.Current["character.memory_error"];
                return;
            }
            ReadAdaptation(view);
            if (operation != "adaptation") { Status = Localizer.Current["character.memory_done"]; Error = ""; }
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        {
            if (operation == "adaptation") AdaptationAvailable = false;
            else Error = Localizer.Current["character.memory_error"];
        }
    }

    /// <summary>Состояние подстройки от модуля: включена ли, поправки, журнал, точки сохранения.</summary>
    public void ReadAdaptation(JsonElement view)
    {
        AdaptationAvailable = true;
        _loadingAdaptation = true;
        try { AdaptationEnabled = view.GetProperty("enabled").GetBoolean(); }
        finally { _loadingAdaptation = false; }
        Corrections.Clear();
        foreach (var item in view.GetProperty("corrections").EnumerateObject())
            Corrections.Add(new CorrectionItem(item.Name, item.Value.GetDouble()));
        AdaptationHistory.Clear();
        foreach (var item in view.GetProperty("history").EnumerateArray())
            AdaptationHistory.Add(new AdaptationChangeItem(item, this));
        SavedPoints.Clear();
        foreach (var item in view.GetProperty("snapshots").EnumerateArray())
            if (item.GetProperty("kind").GetString() == "saved") SavedPoints.Add(new AdaptationPointItem(item, this));
        Raise(nameof(NoCorrections));
        Raise(nameof(NoAdaptationHistory));
        Raise(nameof(HasSavedPoints));
        UndoAdaptationCommand.Refresh();
        ResetAdaptationCommand.Refresh();
    }

    internal Task RestoreAsync(string snapshot) => AdaptationAsync("adaptation_restore", new JsonObject { ["snapshot_id"] = snapshot });

    /// <summary>Сводка оценок: куда указывают достоверные сигналы и установлено ли предпочтение.</summary>
    public void ReadPreferences(JsonElement feedback)
    {
        Preferences.Clear();
        foreach (var item in feedback.GetProperty("preferences").EnumerateArray())
            Preferences.Add(new PreferenceItem(item));
        Raise(nameof(NoPreferences));
    }

    private void ReadHistory(JsonElement status)
    {
        History.Clear();
        if (status.TryGetProperty("history", out var turns))
            foreach (var turn in turns.EnumerateArray())
                History.Add(new HistoryTurnItem(turn, this));
        Raise(nameof(NoHistory));
    }

    internal Task ConfirmAsync(string id) => RunAsync("memory_confirm", new JsonObject { ["id"] = id });
    internal Task ForgetAsync(string id) => RunAsync("memory_forget", new JsonObject { ["id"] = id });
    internal Task CorrectAsync(string id, string value) => RunAsync("memory_correct", new JsonObject { ["id"] = id, ["value"] = value });
    internal Task ForgetTurnAsync(string id) => HistoryAsync("history_forget", new JsonObject { ["ids"] = new JsonArray(id) });
}

/// <summary>Одна запись памяти: значение можно исправить, запись — подтвердить или забыть.</summary>
public sealed class MemoryRecordItem : ObservableObject
{
    private string _value;

    public MemoryRecordItem(JsonElement record, CharacterMemoryViewModel owner)
    {
        Id = record.GetProperty("id").GetString()!;
        Subject = record.GetProperty("subject").GetString() ?? "";
        Original = record.GetProperty("value").GetString() ?? "";
        _value = Original;
        var status = record.GetProperty("status").GetString() ?? "";
        var origin = record.GetProperty("origin").GetString() ?? "";
        var kind = record.GetProperty("kind").GetString() ?? "";
        IsActive = status == "active";
        _codes = (kind, status, origin);
        ConfirmCommand = new Command(() => owner.ConfirmAsync(Id), () => !IsActive);
        SaveCommand = new Command(() => owner.CorrectAsync(Id, Value.Trim()), () => Value.Trim().Length > 0 && Value != Original);
        ForgetCommand = new Command(() => owner.ForgetAsync(Id));
    }

    private static readonly Dictionary<string, string> Kinds = new()
    {
        ["working"] = "character.memory_kind_working", ["profile"] = "character.memory_kind_profile",
        ["episodic"] = "character.memory_kind_episodic", ["semantic"] = "character.memory_kind_semantic",
        ["behavioral"] = "character.memory_kind_behavioral",
    };
    private static readonly Dictionary<string, string> Statuses = new()
    {
        ["active"] = "character.memory_status_active", ["candidate"] = "character.memory_status_candidate",
        ["disputed"] = "character.memory_status_disputed", ["superseded"] = "character.memory_status_superseded",
    };
    private static readonly Dictionary<string, string> Origins = new()
    {
        ["user_said"] = "character.memory_origin_user_said", ["observed"] = "character.memory_origin_observed",
        ["interpretation"] = "character.memory_origin_interpretation", ["assumption"] = "character.memory_origin_assumption",
    };
    private static string Label(Dictionary<string, string> keys, string value) =>
        keys.TryGetValue(value, out var key) ? Localizer.Current[key] : value;

    public string Id { get; }
    public string Subject { get; }
    public string Original { get; }
    private readonly (string Kind, string Status, string Origin) _codes;
    public string Details => string.Join(" · ", Label(Kinds, _codes.Kind), Label(Statuses, _codes.Status), Label(Origins, _codes.Origin));
    public bool IsActive { get; }
    public string Value { get => _value; set { if (Set(ref _value, value)) SaveCommand.Refresh(); } }
    public Command ConfirmCommand { get; }
    public Command SaveCommand { get; }
    public Command ForgetCommand { get; }
}

/// <summary>Одна реплика истории диалога с удалением.</summary>
public sealed class HistoryTurnItem
{
    public HistoryTurnItem(JsonElement turn, CharacterMemoryViewModel owner)
    {
        Id = turn.GetProperty("id").GetString()!;
        User = turn.GetProperty("user").GetString() ?? "";
        var assistant = turn.GetProperty("assistant").GetString() ?? "";
        Assistant = assistant.Length > 160 ? assistant[..160] + "…" : assistant;
        ForgetCommand = new Command(() => owner.ForgetTurnAsync(Id));
    }

    public string Id { get; }
    public string User { get; }
    public string Assistant { get; }
    public Command ForgetCommand { get; }
}

/// <summary>Одна сторона манеры: направление достоверных оценок, их число и установлено ли.</summary>
public sealed class PreferenceItem : ObservableObject
{
    internal static readonly Dictionary<string, string> Dimensions = new()
    {
        ["overall"] = "character.dim_overall", ["warmth"] = "character.dim_warmth", ["humor"] = "character.dim_humor",
        ["sarcasm"] = "character.dim_sarcasm", ["strictness"] = "character.dim_strictness", ["curiosity"] = "character.dim_curiosity",
        ["verbosity"] = "character.dim_verbosity", ["initiative"] = "character.dim_initiative", ["empathy"] = "character.dim_empathy",
    };
    private readonly string _dimension;
    private readonly int _direction;
    private readonly bool _established;
    private readonly int _reliable;
    private readonly int _unreliable;

    public PreferenceItem(JsonElement item)
    {
        _dimension = item.GetProperty("dimension").GetString() ?? "";
        _direction = item.GetProperty("direction").GetInt32();
        _established = item.GetProperty("established").GetBoolean();
        _reliable = item.GetProperty("reliable").GetInt32();
        _unreliable = item.GetProperty("unreliable").GetInt32();
    }

    public string Text
    {
        get
        {
            var name = Dimensions.TryGetValue(_dimension, out var key) ? Localizer.Current[key] : _dimension;
            var way = Localizer.Current[_direction > 0 ? "character.pref_more" : _direction < 0 ? "character.pref_less" : "character.pref_unclear"];
            var state = Localizer.Current[_established ? "character.pref_established" : "character.pref_tentative"];
            var counts = Localizer.Current["character.pref_counts"]
                .Replace("{reliable}", _reliable.ToString(System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal)
                .Replace("{unreliable}", _unreliable.ToString(System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal);
            return name + ": " + way + " · " + state + " · " + counts;
        }
    }
}

/// <summary>9.34: строка таблицы поведения — событие, моя реакция, как он отреагировал.</summary>
public sealed class BehaviorItem(JsonElement row) : ObservableObject
{
    private readonly string _event = row.GetProperty("event").GetString() ?? "";
    private readonly string _strategy = row.GetProperty("strategy").GetString() ?? "";
    private readonly int _direction = row.GetProperty("direction").GetInt32();
    private readonly bool _sure = row.GetProperty("status").GetString() == "sure";
    private readonly int _reactions = row.GetProperty("reactions").GetInt32();

    public string Text
    {
        get
        {
            var l = Localizer.Current;
            return l["character.behavior_row"]
                .Replace("{event}", l["character.behavior_event." + _event], StringComparison.Ordinal)
                .Replace("{strategy}", l["character.behavior_strategy." + _strategy], StringComparison.Ordinal)
                .Replace("{reaction}", l[_direction > 0 ? "character.behavior_liked" : "character.behavior_disliked"], StringComparison.Ordinal)
                .Replace("{status}", l[_sure ? "character.behavior_sure" : "character.behavior_guess"], StringComparison.Ordinal)
                .Replace("{count}", _reactions.ToString(System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal);
        }
    }
}

/// <summary>Число поправки по-человечески: «−0,03» по правилам языка интерфейса.</summary>
internal static class Trait
{
    public static string Name(string trait) =>
        PreferenceItem.Dimensions.TryGetValue(trait, out var key) ? Localizer.Current[key] : trait;

    public static string Number(double value) =>
        (value > 0 ? "+" : value < 0 ? "−" : "") + Math.Abs(value).ToString("0.00", System.Globalization.CultureInfo.GetCultureInfo(Localizer.Current.Language));
}

/// <summary>Текущая поправка одной черты поверх профиля.</summary>
public sealed class CorrectionItem(string trait, double value) : ObservableObject
{
    public string Text => Trait.Name(trait) + ": " + Trait.Number(value);
}

/// <summary>Одна запись журнала подстройки с возвратом к состоянию до неё.</summary>
public sealed class AdaptationChangeItem : ObservableObject
{
    private static readonly Dictionary<string, string> Actions = new()
    {
        ["step"] = "character.adapt_action_step", ["reset"] = "character.adapt_action_reset", ["restore"] = "character.adapt_action_restore",
    };
    private readonly string _trait;
    private readonly string _action;
    private readonly double _before;
    private readonly double _after;

    public AdaptationChangeItem(JsonElement item, CharacterMemoryViewModel owner)
    {
        _trait = item.GetProperty("trait").GetString() ?? "";
        _action = item.GetProperty("action").GetString() ?? "";
        _before = item.GetProperty("before").GetDouble();
        _after = item.GetProperty("after").GetDouble();
        When = item.GetProperty("created_at").GetString() ?? "";
        var snapshot = item.TryGetProperty("snapshot_id", out var id) && id.ValueKind == JsonValueKind.String ? id.GetString() : null;
        RestoreCommand = new Command(() => snapshot is null ? Task.CompletedTask : owner.RestoreAsync(snapshot), () => snapshot is not null);
    }

    public string When { get; }
    public Command RestoreCommand { get; }
    public string Text =>
        Trait.Name(_trait) + ": " + Trait.Number(_before) + " → " + Trait.Number(_after) + " · " +
        (Actions.TryGetValue(_action, out var key) ? Localizer.Current[key] : _action) + " · " + When;
}

/// <summary>Точка, сохранённая владельцем, — к ней можно вернуться.</summary>
public sealed class AdaptationPointItem : ObservableObject
{
    private readonly Dictionary<string, double> _traits = [];

    public AdaptationPointItem(JsonElement item, CharacterMemoryViewModel owner)
    {
        var id = item.GetProperty("id").GetString()!;
        When = item.GetProperty("created_at").GetString() ?? "";
        foreach (var trait in item.GetProperty("traits").EnumerateObject()) _traits[trait.Name] = trait.Value.GetDouble();
        RestoreCommand = new Command(() => owner.RestoreAsync(id));
    }

    public string When { get; }
    public Command RestoreCommand { get; }
    public string Text => When + " · " + (_traits.Count == 0
        ? Localizer.Current["character.adapt_base"]
        : string.Join(", ", _traits.Select(t => Trait.Name(t.Key) + " " + Trait.Number(t.Value))));
}
