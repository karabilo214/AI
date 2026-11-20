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

    public CharacterMemoryViewModel(Func<string, JsonObject?, Task<JsonElement>> call, Func<Task> historyChanged)
    {
        _call = call;
        _historyChanged = historyChanged;
        RefreshCommand = new Command(() => RefreshAsync());
        ClearMemoryCommand = new Command(() => ArmOrRunAsync(true));
        ClearHistoryCommand = new Command(() => ArmOrRunAsync(false));
        CancelClearCommand = new Command(() => { ClearMemoryArmed = false; ClearHistoryArmed = false; return Task.CompletedTask; });
    }

    public ObservableCollection<MemoryRecordItem> Records { get; } = [];
    public ObservableCollection<HistoryTurnItem> History { get; } = [];
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
    }

    /// <summary>Перечитать память и историю; status — уже полученный ответ, если есть.</summary>
    public async Task RefreshAsync(JsonElement? status = null)
    {
        await RunAsync("memory", null).ConfigureAwait(true);
        try
        {
            ReadHistory(status ?? await _call("status", null).ConfigureAwait(true));
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
