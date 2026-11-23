using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>9.21: явные поручения персонажа агенту. Папку, права, подтверждения и журнал задаёт
/// раздел «Агент»; персонаж получает только привязку своей области и предлагает задачу.</summary>
public sealed class CharacterAgentViewModel : ObservableObject
{
    private readonly Func<string, JsonObject?, Task<JsonElement>> _call;
    private readonly Func<string, JsonObject, Task<JsonElement>> _agent;
    private readonly Func<string?, Task<string?>> _pickFolder;
    private readonly Func<ModelInfo?> _model;
    private readonly Func<Task> _finished;
    private readonly TimeSpan _poll;
    private string _folder = "";
    private string _status = "";
    private string _error = "";
    private bool _enabled;
    private bool _running;
    private bool _waiting;
    private bool _loading;
    private bool _window; // 9.28: агент действует в выбранном окне
    private string _goal = "";
    private string _state = "";

    public CharacterAgentViewModel(
        Func<string, JsonObject?, Task<JsonElement>> call,
        Func<string, JsonObject, Task<JsonElement>> agent,
        Func<string?, Task<string?>> pickFolder,
        Func<ModelInfo?> model,
        Func<Task> finished,
        TimeSpan? poll = null)
    {
        _call = call; _agent = agent; _pickFolder = pickFolder; _model = model; _finished = finished;
        _poll = poll ?? TimeSpan.FromMilliseconds(500);
        BrowseCommand = new Command(async () => Folder = await _pickFolder(Folder).ConfigureAwait(true) ?? Folder);
        CancelCommand = new Command(CancelAsync, () => Running);
    }

    public Command BrowseCommand { get; }
    public Command CancelCommand { get; }
    public string Folder { get => _folder; set { if (Set(ref _folder, value) && Enabled && !_loading) _ = SwitchAsync(true); } }
    public string Status { get => _status; private set => Set(ref _status, value); }
    public string Error { get => _error; private set => Set(ref _error, value); }
    public bool Running { get => _running; private set { if (Set(ref _running, value)) CancelCommand.Refresh(); } }
    public bool Waiting { get => _waiting; private set => Set(ref _waiting, value); }
    public bool HasTask => _goal.Length > 0;

    /// <summary>Включение — явное решение владельца: сервис делает привязку с правилами раздела «Агент».</summary>
    public bool Enabled
    {
        get => _enabled;
        set { if (Set(ref _enabled, value) && !_loading) _ = SwitchAsync(value); }
    }

    public bool CanDelegate(string goal) => Enabled && !Running && goal.Trim().Length > 0;

    private async Task SwitchAsync(bool on)
    {
        try
        {
            if (on && (Folder.Length == 0 || _model() is null))
            {
                Error = Localizer.Current["character.agent_need_folder"];
                SetQuietly(false);
                return;
            }
            var body = new JsonObject { ["module"] = "character", ["enabled"] = on };
            if (on) { body["workspace"] = Folder; body["model_id"] = _model()!.Id; body["use_ui"] = _window; }
            var binding = (await _agent("delegate", body).ConfigureAwait(true)).GetProperty("binding_id");
            var id = binding.ValueKind == JsonValueKind.String ? binding.GetString() : null;
            Read((await _call("agent_bind", new JsonObject { ["binding_id"] = id }).ConfigureAwait(true)).GetProperty("delegation"));
            Error = "";
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        {
            Error = Localizer.Current["character.agent_unavailable"];
            SetQuietly(false);
        }
    }

    /// <summary>9.28: поручения выполняются в выбранном окне (привязка пересоздаётся).</summary>
    public async Task SetWindowAsync(bool inWindow)
    {
        if (_window == inWindow) return;
        _window = inWindow;
        if (Enabled) await SwitchAsync(true).ConfigureAwait(true);
    }

    private void SetQuietly(bool value)
    {
        _loading = true;
        try { Enabled = value; }
        finally { _loading = false; }
    }

    /// <summary>Отдать задачу агенту; true — задача принята (черновик можно очистить).</summary>
    public async Task<bool> DelegateAsync(string goal, CancellationToken cancel = default)
    {
        if (!CanDelegate(goal)) return false;
        try
        {
            var view = (await _call("delegate", new JsonObject { ["goal"] = goal.Trim() }).ConfigureAwait(true)).GetProperty("delegation");
            Read(view);
            if (!Running) return false;
            Error = "";
            _ = FollowAsync(cancel);
            return true;
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        { Error = Localizer.Current["character.agent_unavailable"]; return false; }
    }

    private async Task FollowAsync(CancellationToken cancel)
    {
        try
        {
            while (Running && !cancel.IsCancellationRequested)
            {
                await Task.Delay(_poll, cancel).ConfigureAwait(true);
                await RefreshAsync().ConfigureAwait(true);
            }
            if (!cancel.IsCancellationRequested) await _finished().ConfigureAwait(true);
        }
        catch (OperationCanceledException) { }
    }

    public async Task RefreshAsync()
    {
        try { Read((await _call("delegation", null).ConfigureAwait(true)).GetProperty("delegation")); }
        catch (Exception error) when (CharacterViewModel.Handled(error)) { Error = Localizer.Current["character.agent_unavailable"]; Running = false; }
    }

    private async Task CancelAsync()
    {
        try { Read((await _call("delegate_cancel", null).ConfigureAwait(true)).GetProperty("delegation")); }
        catch (Exception error) when (CharacterViewModel.Handled(error)) { Error = Localizer.Current["character.agent_unavailable"]; }
    }

    private static readonly Dictionary<string, string> Errors = new()
    {
        ["agent_off"] = "character.agent_off", ["agent_busy"] = "character.agent_busy",
        ["agent_unavailable"] = "character.agent_unavailable",
    };
    private static readonly Dictionary<string, string> States = new()
    {
        ["running"] = "character.agent_running", ["done"] = "character.agent_done",
        ["cancelled"] = "character.agent_cancelled", ["failed"] = "character.agent_failed",
    };

    /// <summary>Состояние поручения от модуля.</summary>
    public void Read(JsonElement view)
    {
        SetQuietly(view.GetProperty("enabled").GetBoolean());
        if (view.TryGetProperty("error", out var error) && error.ValueKind == JsonValueKind.String)
            Error = Errors.TryGetValue(error.GetString()!, out var key) ? Localizer.Current[key] : error.GetString()!;
        var task = view.GetProperty("task");
        if (task.ValueKind != JsonValueKind.Object) { Running = false; Waiting = false; _goal = ""; _state = ""; Raise(nameof(HasTask)); Status = ""; return; }
        _goal = task.GetProperty("goal").GetString() ?? "";
        _state = task.GetProperty("state").GetString() ?? "";
        Running = _state == "running";
        Waiting = Running && task.GetProperty("phase").GetString() == "waiting_approval";
        Raise(nameof(HasTask));
        Status = Text();
    }

    private string Text()
    {
        var state = Waiting ? Localizer.Current["character.agent_waiting"]
            : States.TryGetValue(_state, out var key) ? Localizer.Current[key] : _state;
        return Localizer.Current["character.agent_task"].Replace("{goal}", _goal, StringComparison.Ordinal) + " · " + state;
    }

    public override void Relocalize()
    {
        if (HasTask) Status = Text();
        base.Relocalize();
    }
}
