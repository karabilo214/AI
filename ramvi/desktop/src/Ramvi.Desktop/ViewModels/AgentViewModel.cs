using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;

namespace Ramvi.Desktop.ViewModels;

public sealed class AgentOption(string group, string key, bool enabled) : ObservableObject
{
    private bool _enabled = enabled;
    public string Group { get; } = group;
    public string Key { get; } = key;
    public string Label => Localizer.Current["agent.option." + Key];
    public bool Enabled { get => _enabled; set => Set(ref _enabled, value); }
}

public sealed class AgentLimit(string group, string key, JsonNode? initial) : ObservableObject
{
    private string _value = group == "retention" && key == "max_bytes" && initial is not null
        ? (initial.GetValue<double>() / 1048576).ToString(CultureInfo.InvariantCulture) : initial?.ToString() ?? "1";
    private bool _unlimited = initial is null;
    public string Group { get; } = group;
    public string Key { get; } = key;
    public string Label => Localizer.Current["agent.limit." + Key];
    public bool SupportsUnlimited => Key is not ("max_completion_tokens" or "max_calls_per_step" or "max_nudges" or "max_series" or "context");
    public bool Unlimited { get => _unlimited; set { if (Set(ref _unlimited, value)) Raise(nameof(IsLimited)); } }
    public bool IsLimited => !Unlimited;
    public string Value { get => _value; set => Set(ref _value, value); }
}

public sealed record AgentRunItem(string Id, string Label);
public sealed record AgentChangeItem(string Id, string Label);

/// <summary>Agent screen and trusted settings. Polling survives navigation; Stop is a server action.</summary>
public sealed class AgentViewModel : ObservableObject, IDisposable
{
    private readonly Func<IChatApi> _api;
    private readonly Func<string?, Task<string?>> _pickFolder;
    private readonly Func<Task<IReadOnlyList<DroppedFile>>>? _pickFiles;
    private VoiceViewModel? _voice;
    private readonly CancellationTokenSource _lifetime = new();
    private bool _polling;
    private bool _disposed;
    private readonly CancellationToken _token;
    private JsonObject? _policy;
    private JsonElement? _pending;
    private string? _taskId;
    private string _workspace = "";
    private string _input = "";
    private string _status = "";
    private string _log = "";
    private string _result = "";
    private string _error = "";
    private bool _running;
    private ModelInfo? _model;
    private AgentRunItem? _selectedRun;
    private AgentChangeItem? _selectedChange;
    private bool _clearConfirmed;
    private int _lastCount = -1;
    private string? _loadedRun;
    private bool _finishedLoaded;
    public AgentViewModel(Func<IChatApi> api, Func<string?, Task<string?>> pickFolder, IReadOnlyCollection<ModelInfo> models, Func<Task<IReadOnlyList<DroppedFile>>>? pickFiles = null)
    {
        _token = _lifetime.Token;
        _api = api; _pickFolder = pickFolder; Models = models; _pickFiles = pickFiles;
        BrowseCommand = new Command(async () => Workspace = await _pickFolder(Workspace).ConfigureAwait(true) ?? Workspace);
        StartCommand = new Command(StartAsync, () => !IsRunning && Model is not null && Workspace.Length > 0 && (Input.Trim().Length > 0 || Attachments.Count > 0) && Input.Length <= MessageLimits.MaxTextChars);
        AttachFilesCommand = new Command(async () => { if (_pickFiles is not null) await AddFilesAsync(await _pickFiles().ConfigureAwait(true)).ConfigureAwait(true); }, () => _pickFiles is not null);
        RemoveAttachmentCommand = new Command(p => { if (p is AttachmentViewModel item) Attachments.Remove(item); return Task.CompletedTask; });
        Attachments.CollectionChanged += (_, _) => StartCommand.Refresh();
        StopCommand = new Command(() => ActAsync("stop", new JsonObject()), () => IsRunning);
        SaveSettingsCommand = new Command(SaveAsync);
        RefreshHistoryCommand = new Command(LoadHistoryAsync);
        ApproveCommand = new Command(() => DecideAsync("approve"));
        DenyCommand = new Command(() => DecideAsync("deny"));
        StopPendingCommand = new Command(() => DecideAsync("stop"));
        RollbackCommand = new Command(RollbackAsync, () => SelectedChange is not null && !IsRunning);
        ClearCommand = new Command(ClearAsync, () => ClearConfirmed);
        WhyCommand = new Command(WhyAsync);
        PickTargetCommand = new Command(PickTargetAsync);
        SaveGrantsCommand = new Command(() => ActAsync("target/grants", new JsonObject { ["ui"] = GrantUi, ["screen"] = GrantScreen, ["control"] = GrantControl }));
    }
    private static Localizer L => Localizer.Current;
    public IReadOnlyCollection<ModelInfo> Models { get; }
    public Ramvi.Desktop.Api.Limits MessageLimits { get; set; } = new(32_000, 5, 128 * 1024, 256 * 1024, 255);
    public ObservableCollection<AttachmentViewModel> Attachments { get; } = [];
    public VoiceViewModel? Voice { get => _voice; set => Set(ref _voice, value); }
    public Command AttachFilesCommand { get; }
    public Command RemoveAttachmentCommand { get; }
    public bool IsIdle => !IsRunning;
    public ObservableCollection<AgentOption> Rights { get; } = [];
    public ObservableCollection<AgentOption> Restrictions { get; } = [];
    public ObservableCollection<AgentOption> Confirmations { get; } = [];
    public ObservableCollection<AgentOption> Tools { get; } = [];
    public ObservableCollection<AgentLimit> Limits { get; } = [];
    public ObservableCollection<AgentRunItem> Runs { get; } = [];
    public ObservableCollection<AgentChangeItem> Changes { get; } = [];
    public Command BrowseCommand { get; }
    public Command StartCommand { get; }
    public Command StopCommand { get; }
    public Command SaveSettingsCommand { get; }
    public Command RefreshHistoryCommand { get; }
    public Command ApproveCommand { get; }
    public Command DenyCommand { get; }
    public Command StopPendingCommand { get; }
    public Command RollbackCommand { get; }
    public Command ClearCommand { get; }
    public Command WhyCommand { get; }
    public Command PickTargetCommand { get; }
    public Command SaveGrantsCommand { get; }
    public string Workspace { get => _workspace; set { Set(ref _workspace, value); StartCommand.Refresh(); } }
    public string Input { get => _input; set { Set(ref _input, value); StartCommand.Refresh(); } }
    public ModelInfo? Model { get => _model; set { Set(ref _model, value); StartCommand.Refresh(); } }
    public string Status { get => _status; private set => Set(ref _status, value); }
    public string Error { get => _error; private set { Set(ref _error, value); Raise(nameof(HasError)); } }
    public bool HasError => Error.Length > 0;
    public string Log { get => _log; private set => Set(ref _log, value); }
    public string Result { get => _result; private set => Set(ref _result, value); }
    public bool IsRunning { get => _running; private set { Set(ref _running, value); Raise(nameof(IsIdle)); StartCommand.Refresh(); StopCommand.Refresh(); RollbackCommand.Refresh(); } }
    public bool HasPending => _pending is not null;
    public bool IsControlPending => _pending is { } p && Text(p, "kind") == "control";
    public string PendingTitle => _pending is { } p ? (Text(p, "source_module") == "character" ? L["agent.from_character"] + " · " : "") + L["agent.kind." + Text(p, "kind")] + " · " + Text(p, "title") + (Text(p, "reason").Length > 0 ? " · " + L["agent.checkpoint." + Text(p, "reason")] : "") : "";
    public string PendingDetail => _pending is { } p && Text(p, "kind") == "limit" ? L["agent.metrics"] + ": " + Text(p, "steps") + " · " + Text(p, "elapsed") + " s · " + Text(p, "tokens") + "\n" + Text(p, "detail") : _pending is { } q ? Text(q, "diff") + Text(q, "command") + Text(q, "cwd") + Text(q, "application") + "\n" + Text(q, "plan") + "\n" + Text(q, "value") : "";
    public string ApproveLabel => _pending is { } p && Text(p, "kind") == "limit" ? L["agent.continue"] : L["agent.approve"];
    public string Reason { get; set; } = "";
    public string Series { get; set; } = "0";
    public string SecretPatterns { get; set; } = "";
    public bool UseUi { get; set; }
    public bool GrantUi { get; set; }
    public bool GrantScreen { get; set; }
    public bool GrantControl { get; set; }
    public string Target { get; private set; } = "";
    public string HistoryPath { get; set; } = "";
    public string ClearBefore { get; set; } = "";
    public bool ForceRollback { get; set; }
    public bool ClearConfirmed { get => _clearConfirmed; set { Set(ref _clearConfirmed, value); ClearCommand.Refresh(); } }
    public AgentRunItem? SelectedRun
    {
        get => _selectedRun;
        set { if (Set(ref _selectedRun, value) && value is not null) _ = ShowRunAsync(value.Id); }
    }
    public AgentChangeItem? SelectedChange { get => _selectedChange; set { Set(ref _selectedChange, value); RollbackCommand.Refresh(); } }
    public async Task LoadAsync()
    {
        try
        {
            Error = "";
            var settings = await _api().AgentAsync("settings", cancel: _token).ConfigureAwait(true);
            _policy = JsonNode.Parse(settings.GetProperty("settings").GetRawText())!.AsObject();
            Rights.Clear(); Restrictions.Clear(); Confirmations.Clear(); Tools.Clear(); Limits.Clear();
            foreach (var group in _policy)
            {
                foreach (var field in group.Value!.AsObject())
                {
                    if (field.Value is JsonValue val && val.TryGetValue<bool>(out var flag))
                    {
                        var collection = group.Key == "permissions" ? Rights : group.Key == "restrictions" ? Restrictions : Confirmations;
                        collection.Add(new AgentOption(group.Key, field.Key, flag));
                    }
                    else if (field.Key == "secret_patterns")
                        SecretPatterns = string.Join('\n', field.Value!.AsArray().Select(n => n!.ToString()));
                    else if (field.Key != "disabled") Limits.Add(new AgentLimit(group.Key, field.Key, field.Value));
                }
            }
            var disabled = _policy["access"]!["disabled"]!.AsArray().Select(n => n!.ToString()).ToHashSet(StringComparer.Ordinal);
            foreach (var tool in settings.GetProperty("tools").EnumerateArray()) Tools.Add(new AgentOption("tools", tool.GetString()!, !disabled.Contains(tool.GetString()!)));
            Raise(nameof(SecretPatterns));
            Model ??= Models.FirstOrDefault();
            await LoadHistoryAsync().ConfigureAwait(true);
            if (!_polling) { _polling = true; _ = PollAsync(); }
        }
        catch (Exception e) when (IsHandled(e)) { ShowError(e); }
    }
    public async Task AddFilesAsync(IReadOnlyList<DroppedFile> files)
    {
        var result = await AttachmentReader.ReadAsync([.. Attachments.Select(a => a.Attachment)], files, MessageLimits).ConfigureAwait(true);
        foreach (var file in result.Accepted) Attachments.Add(new AttachmentViewModel(file));
        if (result.ErrorKey is { } key) Error = L.Format(key, result.ErrorParams);
    }
    public void InsertVoiceText(string text) { Input = string.IsNullOrWhiteSpace(Input) ? text : Input.TrimEnd() + " " + text; }
    public void VoiceError(Exception error) => ShowError(error);
    private async Task StartAsync()
    {
        try
        {
            Error = ""; Result = ""; Log = ""; SelectedRun = null; Changes.Clear(); _loadedRun = null; _lastCount = -1; _finishedLoaded = false;
            if (UseUi) { Status = L["agent.target_countdown"]; await Task.Delay(5000, _token).ConfigureAwait(true); }
            var response = await CallAsync("start", new JsonObject { ["goal"] = Input.Trim().Length > 0 ? Input : L["agent.attachments_goal"], ["attachments"] = new JsonArray([.. Attachments.Select(a => (JsonNode?)new JsonObject { ["name"] = a.Attachment.Name, ["text"] = a.Attachment.Text })]), ["workspace"] = Workspace, ["model_id"] = Model!.Id, ["use_ui"] = UseUi }).ConfigureAwait(true);
            _taskId = response.GetProperty("task_id").GetString();
            Input = ""; Attachments.Clear();
            IsRunning = true; Status = L["agent.running"];
        }
        catch (Exception e) when (IsHandled(e)) { ShowError(e); }
    }
    private async Task PollAsync()
    {
        try
        {
            while (!_lifetime.IsCancellationRequested)
            {
                try
                {
                    var state = await _api().AgentAsync("state", cancel: _token).ConfigureAwait(true);
                    _pending = state.TryGetProperty("pending", out var p) && p.ValueKind == JsonValueKind.Object ? p.Clone() : null;
                    foreach (var name in new[] { nameof(HasPending), nameof(PendingTitle), nameof(PendingDetail), nameof(ApproveLabel), nameof(IsControlPending) }) Raise(name);
                    if (state.TryGetProperty("task", out var task) && task.ValueKind == JsonValueKind.Object)
                    {
                        _taskId = Text(task, "task_id");
                        IsRunning = Text(task, "state") == "running";
                        Status = L["agent.state." + (HasPending ? "waiting_approval" : Text(task, "state"))];
                        if (task.TryGetProperty("result", out var result) && result.ValueKind == JsonValueKind.Object)
                        {
                            Result = Text(result, "result");
                            var error = Text(result, "error");
                            if (error.StartsWith("agent.ready.", StringComparison.Ordinal)) Error = L[error];
                        }
                        if (_taskId is not null && _selectedRun is null) await ShowRunAsync(_taskId).ConfigureAwait(true);
                        if (!IsRunning && !_finishedLoaded) { _finishedLoaded = true; await LoadHistoryAsync().ConfigureAwait(true); }
                    }
                    else if (IsRunning)
                    {
                        IsRunning = false; Status = L["agent.state.cancelled"];
                        await LoadHistoryAsync().ConfigureAwait(true);
                    }
                }
                catch (Exception e) when (IsHandled(e)) { ShowError(e); }
                await Task.Delay(500, _token).ConfigureAwait(true);
            }
        }
        catch (OperationCanceledException) { }
    }
    private async Task DecideAsync(string action)
    {
        if (_pending is not { } p) return;
        if (!int.TryParse(Series, NumberStyles.None, CultureInfo.InvariantCulture, out var series)) { Error = L["agent.bad_limit"]; return; }
        await ActAsync("decision", new JsonObject { ["id"] = Text(p, "id"), ["action"] = action, ["series"] = series, ["reason"] = Reason }).ConfigureAwait(true);
    }
    private async Task SaveAsync()
    {
        if (_policy is null) return;
        try
        {
            var policy = (JsonObject)_policy.DeepClone();
            foreach (var option in Rights.Concat(Restrictions).Concat(Confirmations)) policy[option.Group]![option.Key] = option.Enabled;
            policy["restrictions"]!["secret_patterns"] = new JsonArray([.. SecretPatterns.Split('\n', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).Select(s => (JsonNode?)JsonValue.Create(s))]);
            policy["access"]!["disabled"] = new JsonArray([.. Tools.Where(t => !t.Enabled).Select(t => (JsonNode?)JsonValue.Create(t.Key))]);
            foreach (var limit in Limits)
            {
                if (limit.Unlimited && limit.SupportsUnlimited) { policy[limit.Group]![limit.Key] = null; continue; }
                if (!double.TryParse(limit.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var number) || !double.IsFinite(number) || number <= 0) throw new FormatException();
                if (limit.Group == "retention" && limit.Key == "max_bytes") number *= 1048576;
                policy[limit.Group]![limit.Key] = Math.Truncate(number) == number ? JsonValue.Create((long)number) : JsonValue.Create(number);
            }
            await CallAsync("settings", policy).ConfigureAwait(true);
            Status = L["settings.saved"];
        }
        catch (Exception e) when (IsHandled(e)) { ShowError(e); }
    }
    private async Task LoadHistoryAsync()
    {
        try
        {
            var data = await _api().AgentAsync("history", cancel: _token).ConfigureAwait(true);
            var selected = SelectedRun?.Id;
            Runs.Clear();
            foreach (var run in data.GetProperty("runs").EnumerateArray()) Runs.Add(new AgentRunItem(Text(run, "id"), Text(run, "objective") + " · " + L["agent.state." + Text(run, "status")]));
            SelectedRun = Runs.FirstOrDefault(r => r.Id == selected) ?? Runs.FirstOrDefault();
        }
        catch (Exception e) when (IsHandled(e)) { ShowError(e); }
    }
    private async Task ShowRunAsync(string id)
    {
        try
        {
            var data = await _api().AgentAsync("runs/" + Uri.EscapeDataString(id), cancel: _token).ConfigureAwait(true);
            var events = data.GetProperty("events");
            if (_loadedRun == id && _lastCount == events.GetArrayLength()) return;
            _loadedRun = id; _lastCount = events.GetArrayLength();
            var summary = data.GetProperty("summary");
            var sb = new StringBuilder();
            sb.Append(L["agent.metrics"]).Append(": ").Append(Text(summary, "steps")).Append(" · ").Append(Text(summary, "duration")).Append(" s · ").Append(Text(summary, "tokens")).Append(" · ").Append(Text(summary, "context_peak")).AppendLine();
            foreach (var e in events.EnumerateArray().TakeLast(200))
            {
                var kind = Text(e, "kind"); var payload = e.GetProperty("payload");
                if (kind is "agent.plan" or "agent.message") sb.AppendLine(Text(payload, "text"));
                else if (kind == "agent.tool_call") sb.Append(L["agent.option." + Text(payload, "tool")]).Append(' ').AppendLine(Text(payload, "args"));
                else if (kind == "agent.tool_result") sb.AppendLine(Text(payload, "content"));
                else if (kind == "agent.readiness")
                {
                    foreach (var list in new[] { "errors", "warnings" })
                        if (payload.TryGetProperty(list, out var items))
                            foreach (var item in items.EnumerateArray()) sb.AppendLine(L[Text(item, "code")]);
                }
                else if (kind == "agent.done") sb.AppendLine(L["agent.state." + Text(payload, "status")]);
            }
            Log = sb.ToString();
            Changes.Clear();
            foreach (var change in data.GetProperty("changes").EnumerateArray()) Changes.Add(new AgentChangeItem(Text(change, "id"), string.Join(", ", change.GetProperty("files").EnumerateArray().Select(f => f.GetString()))));
            SelectedChange = Changes.FirstOrDefault();
        }
        catch (Exception e) when (IsHandled(e)) { ShowError(e); }
    }
    private async Task RollbackAsync()
    {
        if (SelectedChange is not { } change || _selectedRun is not { } run) return;
        await ActAsync("rollback/" + Uri.EscapeDataString(change.Id), new JsonObject { ["run_id"] = run.Id, ["force"] = ForceRollback }).ConfigureAwait(true);
    }
    private async Task ClearAsync()
    {
        JsonNode? before = null;
        if (ClearBefore.Trim().Length > 0)
        {
            if (!DateTimeOffset.TryParseExact(ClearBefore, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var date)) { Error = L["agent.bad_limit"]; return; }
            before = JsonValue.Create(date.ToUnixTimeSeconds());
        }
        await ActAsync("clear", new JsonObject { ["confirm"] = ClearConfirmed, ["before"] = before }).ConfigureAwait(true);
        ClearConfirmed = false;
        await LoadHistoryAsync().ConfigureAwait(true);
    }
    private async Task WhyAsync()
    {
        try
        {
            var data = await CallAsync("why", new JsonObject { ["path"] = HistoryPath, ["run_id"] = SelectedRun?.Id }).ConfigureAwait(true);
            var sb = new StringBuilder();
            foreach (var reason in data.GetProperty("reasons").EnumerateArray())
                sb.AppendLine(Text(reason, "objective")).AppendLine(Text(reason, "plan")).AppendLine(Text(reason, "tool")).AppendLine(Text(reason, "approval")).AppendLine(Text(reason, "diff")).AppendLine(Text(reason, "rollback"));
            Log = sb.Length == 0 ? L["agent.no_changes"] : sb.ToString();
        }
        catch (Exception e) when (IsHandled(e)) { ShowError(e); }
    }
    private async Task PickTargetAsync()
    {
        try
        {
            Status = L["agent.target_countdown"];
            await Task.Delay(5000, _token).ConfigureAwait(true);
            var response = await CallAsync("target", new JsonObject()).ConfigureAwait(true);
            Target = Text(response.GetProperty("target"), "title");
            GrantUi = false; GrantScreen = false; GrantControl = false;
            Raise(nameof(GrantUi)); Raise(nameof(GrantScreen)); Raise(nameof(GrantControl));
            Raise(nameof(Target));
            Status = L["agent.target_selected"];
        }
        catch (Exception e) when (IsHandled(e)) { ShowError(e); }
    }
    public override void Relocalize()
    {
        base.Relocalize();
        foreach (var item in Rights.Concat(Restrictions).Concat(Confirmations).Concat(Tools)) item.Relocalize();
        foreach (var item in Limits) item.Relocalize();
    }
    private Task<JsonElement> CallAsync(string path, JsonObject body)
    {
        using var document = JsonDocument.Parse(body.ToJsonString());
        return _api().AgentAsync(path, document.RootElement.Clone(), _token);
    }
    private async Task ActAsync(string path, JsonObject body)
    {
        try { Error = ""; await CallAsync(path, body).ConfigureAwait(true); }
        catch (Exception e) when (IsHandled(e)) { ShowError(e); }
    }
    private static bool IsHandled(Exception e) => e is ApiException or HttpRequestException or NotSupportedException or FormatException or InvalidOperationException or OperationCanceledException or JsonException;
    private void ShowError(Exception e) => Error = e is ApiException a ? L.Format(a.Code, a.Parameters.ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal)) : e is FormatException ? L["agent.bad_limit"] : L["agent.unavailable"];
    private static string Text(JsonElement element, string name) => element.TryGetProperty(name, out var value) && value.ValueKind != JsonValueKind.Null ? value.ValueKind == JsonValueKind.String ? value.GetString() ?? "" : value.ToString() : "";
    public void Dispose() { if (_disposed) return; _disposed = true; _lifetime.Cancel(); _lifetime.Dispose(); }
}
