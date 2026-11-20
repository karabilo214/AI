using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Explicit Character session, separate from Chat's push-to-record control.</summary>
public sealed class CharacterVoiceViewModel : ObservableObject, IDisposable
{
    private readonly Func<string, JsonObject?, Task<JsonElement>> _call;
    private readonly Func<bool> _canStart;
    private readonly Action<JsonElement> _follow;
    private readonly CancellationTokenSource _lifetime = new();
    private readonly CancellationToken _token;
    private string _state = "inactive", _partial = "", _error = "";
    private bool _available, _ending, _disposed;
    private string? _remoteHost;
    private string? _lastReply;
    private int _watch;
    public CharacterVoiceViewModel(Func<string, JsonObject?, Task<JsonElement>> call, Func<bool> canStart, Action<JsonElement> follow)
    {
        _call = call; _canStart = canStart; _follow = follow; _token = _lifetime.Token;
        ToggleCommand = new Command(() => IsActive ? StopAsync() : StartAsync(), () => _state != "stopping" && (IsActive || (_available && _canStart())));
        EndCommand = new Command(StopAsync, () => IsActive);
    }
    public Command ToggleCommand { get; }
    public Command EndCommand { get; }
    public bool IsActive => _state is not ("inactive" or "error");
    public string ButtonText => Localizer.Current[IsActive ? "character.voice_stop" : "character.voice_start"];
    public string Status => Localizer.Current["character.voice_state." + _state];
    public string Partial { get => _partial; private set { if (Set(ref _partial, value)) Raise(nameof(HasPartial)); } }
    public bool HasPartial => !string.IsNullOrEmpty(Partial);
    public bool HasRemote => _remoteHost is not null;
    public string Error { get => _error; private set { if (Set(ref _error, value)) Raise(nameof(HasError)); } }
    private string _timing = "";
    public string Timing { get => _timing; private set => Set(ref _timing, value); }
    public bool HasError => !string.IsNullOrEmpty(Error);
    public string RemoteText => _remoteHost is null ? "" : Localizer.Current.Format("voice.api_sending", ("host", _remoteHost));
    public async Task LoadAsync()
    {
        try
        {
            var caps = await _call("capabilities", null).ConfigureAwait(true);
            _available = caps.TryGetProperty("available", out var available) && available.GetBoolean();
            _remoteHost = caps.TryGetProperty("remote_host", out var host) && host.ValueKind == JsonValueKind.String ? host.GetString() : null;
            Raise(nameof(RemoteText)); Raise(nameof(HasRemote)); Refresh();
        }
        catch (Exception e) when (Handled(e)) { _available = false; Refresh(); }
    }
    public void Refresh() { ToggleCommand.Refresh(); EndCommand.Refresh(); }
    private void SetState(string value)
    {
        _state = value; Raise(nameof(IsActive)); Raise(nameof(ButtonText)); Raise(nameof(Status)); Refresh();
    }
    private async Task StartAsync()
    {
        var watch=++_watch;
        _ending = false; _lastReply = null; Error = ""; Partial = ""; SetState("starting");
        try
        {
            await _call("start", null).ConfigureAwait(true);
            if(watch==_watch && !_ending && !_disposed) _ = WatchAsync(watch);
        }
        catch (Exception e) when (Handled(e)) { if(watch==_watch && !_disposed) {SetState("error"); Error = Localizer.Current["character.voice_error"];} }
    }
    private async Task WatchAsync(int watch)
    {
        try
        {
            while (!_token.IsCancellationRequested && watch == _watch)
            {
                var result = await _call("status", null).ConfigureAwait(true);
                if (watch != _watch || _token.IsCancellationRequested) return;
                if (result.TryGetProperty("recognition_ms", out var timing) && timing.ValueKind == JsonValueKind.Number)
                    Timing = Localizer.Current.Format("character.stt_ms", ("ms", timing.GetDouble().ToString("0", System.Globalization.CultureInfo.InvariantCulture)));
                var state = result.GetProperty("state").GetString() ?? "error";
                SetState(_ending && state is not ("inactive" or "error") ? "stopping" : state);
                Partial = _ending ? "" : result.GetProperty("partial").GetString() ?? "";
                if (!_ending && result.TryGetProperty("reply", out var reply) && reply.ValueKind == JsonValueKind.Object)
                {
                    var id = reply.GetProperty("response_id").GetString();
                    if (id != _lastReply) { _lastReply = id; _follow(reply); }
                }
                if (state is "inactive" or "error")
                {
                    if (state == "error") Error = Localizer.Current["character.voice_error"];
                    return;
                }
                await Task.Delay(150, _token).ConfigureAwait(true);
            }
        }
        catch (Exception e) when (Handled(e))
        {
            if (!_token.IsCancellationRequested) { SetState("error"); Error = Localizer.Current["character.voice_error"]; }
        }
    }
    public async Task StopAsync()
    {
        if (!IsActive) return;
        var watch=++_watch;
        _ending=true;Partial="";SetState("stopping");
        try {
            await _call("stop",null).ConfigureAwait(true);
            if(watch==_watch && !_disposed) _=WatchAsync(watch);
        }
        catch(Exception e) when(Handled(e)) { if(watch==_watch && !_disposed) {SetState("error");Error=Localizer.Current["character.voice_error"];} }
    }
    private static bool Handled(Exception e) => e is ApiException or HttpRequestException or InvalidOperationException or KeyNotFoundException or JsonException or OperationCanceledException or NotSupportedException;
    public void Dispose() { if (_disposed) return; _disposed = true; _watch++; _lifetime.Cancel(); _lifetime.Dispose(); }
}
