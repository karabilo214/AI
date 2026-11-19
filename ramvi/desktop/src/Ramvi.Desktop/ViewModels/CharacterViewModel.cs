using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Text-only Character client. Receives complete validated segments from its module.</summary>
public sealed class CharacterViewModel : ObservableObject, IDisposable
{
    private readonly Func<IChatApi> _api;
    private readonly CancellationTokenSource _lifetime = new();
    private readonly CancellationToken _token;
    private string _draft = "", _transcript = "", _error = "";
    private bool _busy, _loading, _disposed, _ignoreEvents;
    private ModelInfo? _model;
    private CharacterProfile? _profile;
    private string? _responseId;
    public CharacterViewModel(Func<IChatApi> api, ObservableCollection<ModelInfo> models)
    {
        _token = _lifetime.Token;
        _api = api; Models = models;
        Voice = new CharacterVoiceViewModel((operation, args) => ModuleCallAsync("voice", operation, args), () => Model is not null, FollowVoiceReply);
        Speech = new CharacterSpeechViewModel((operation, args) => ModuleCallAsync("speech", operation, args));
        Send = new Command(SendAsync, () => !Busy && !string.IsNullOrWhiteSpace(Draft) && Model is not null);
        Stop = new Command(StopAsync, () => Busy);
    }
    public ObservableCollection<ModelInfo> Models { get; }
    public ObservableCollection<CharacterProfile> Profiles { get; } = [];
    public CharacterVoiceViewModel Voice { get; }
    public CharacterSpeechViewModel Speech { get; }
    public Command Send { get; }
    public Command Stop { get; }
    public string Draft { get => _draft; set { if (Set(ref _draft, value)) Send.Refresh(); } }
    public string Transcript { get => _transcript; private set => Set(ref _transcript, value); }
    public string Error { get => _error; private set => Set(ref _error, value); }
    public bool Busy { get => _busy; private set { Set(ref _busy, value); Send.Refresh(); Stop.Refresh(); } }
    public ModelInfo? Model
    {
        get => _model;
        set { if (Set(ref _model, value)) { Send.Refresh(); Voice.Refresh(); if (!_loading) { _ignoreEvents = true; _ = ConfigureAsync(); } } }
    }
    public CharacterProfile? Profile
    {
        get => _profile;
        set { if (Set(ref _profile, value) && !_loading && value is not null) _ = SelectProfileAsync(value.Id); }
    }
    private Task<JsonElement> CallAsync(string operation, JsonObject? arguments = null) => ModuleCallAsync("dialogue", operation, arguments);
    private async Task<JsonElement> ModuleCallAsync(string command, string operation, JsonObject? arguments = null)
    {
        using var document = JsonDocument.Parse(new JsonObject { ["operation"] = operation, ["arguments"] = arguments ?? new JsonObject() }.ToJsonString());
        var body = document.RootElement.Clone();
        if (command == "voice" && operation == "start") await ConfigureAsync().ConfigureAwait(true);
        var result = await _api().ModuleCommandAsync("character", command, body, _token).ConfigureAwait(true);
        if (result.TryGetProperty("error", out _)) throw new InvalidOperationException("Character request failed");
        return result;
    }
    public async Task LoadAsync()
    {
        try
        {
            var result = await CallAsync("status").ConfigureAwait(true);
            _loading = true;
            try
            {
                var character = result.GetProperty("character");
                var selected = character.GetProperty("model").GetString();
                Model = Models.FirstOrDefault(m => m.Id == selected) ?? Models.FirstOrDefault();
                if (!Busy) Draft = character.GetProperty("draft").GetString() ?? "";
                Profiles.Clear();
                foreach (var item in result.GetProperty("profiles").EnumerateArray())
                    Profiles.Add(new(item.GetProperty("id").GetString()!, item.GetProperty("name").GetString()!));
                Profile = Profiles.FirstOrDefault(p => p.Id == character.GetProperty("profile_id").GetString());
                if (!Busy) ReadHistory(result);
            }
            finally { _loading = false; }
            if (!Busy && result.TryGetProperty("job", out var job) && job.ValueKind == JsonValueKind.Object && job.GetProperty("state").GetString() == "running")
            {
                Busy = true; _responseId = job.GetProperty("id").GetString();
                _ = PollAsync(_responseId!);
            }
            Error = "";
            await Voice.LoadAsync().ConfigureAwait(true);
            await Speech.LoadAsync().ConfigureAwait(true);
        }
        catch (Exception e) when (Handled(e)) { Error = Localizer.Current["character.request_error"]; }
    }
    private void ReadHistory(JsonElement result)
    {
        Transcript = string.Join("\n\n", result.GetProperty("history").EnumerateArray().Select(turn =>
            Localizer.Current["character.owner"] + ": " + turn.GetProperty("user").GetString() + "\nRAMVI: " + turn.GetProperty("assistant").GetString()));
    }
    private async Task ConfigureAsync()
    {
        try { await CallAsync("configure", new JsonObject { ["draft"] = Draft, ["model"] = Model?.Id }).ConfigureAwait(true); }
        catch (Exception e) when (Handled(e)) { Error = Localizer.Current["character.request_error"]; }
    }
    private async Task SelectProfileAsync(string id)
    {
        _ignoreEvents = true;
        try { await CallAsync("profile", new JsonObject { ["id"] = id }).ConfigureAwait(true); }
        catch (Exception e) when (Handled(e)) { Error = Localizer.Current["character.request_error"]; }
    }
    private async Task SendAsync()
    {
        Busy = true; _ignoreEvents = false; Error = "";
        var input = Draft;
        string? sentId = null;
        try
        {
            var result = await CallAsync("start", new JsonObject { ["text"] = input, ["model"] = Model!.Id }).ConfigureAwait(true);
            _responseId = result.GetProperty("response_id").GetString(); sentId = _responseId; Draft = "";
            Transcript += (Transcript.Length == 0 ? "" : "\n\n") + Localizer.Current["character.owner"] + ": " + input + "\nRAMVI: ";
            await PollAsync(_responseId!).ConfigureAwait(true);
        }
        catch (Exception e) when (Handled(e)) { Error = Localizer.Current["character.request_error"]; }
        finally { if (sentId is null || _responseId == sentId) Busy = false; }
    }
    private async Task PollAsync(string id)
    {
        var cursor = 0;
        try
        {
            while (!_lifetime.IsCancellationRequested && _responseId == id)
            {
                var result = await CallAsync("poll", new JsonObject { ["response_id"] = id, ["cursor"] = cursor }).ConfigureAwait(true);
                if (_responseId != id) break;
                var job = result.GetProperty("job");
                if (job.ValueKind != JsonValueKind.Object) break;
                foreach (var record in job.GetProperty("events").EnumerateArray())
                    if (!_ignoreEvents && record.GetProperty("response_id").GetString() == id && record.GetProperty("type").GetString() == "segment") Transcript += record.GetProperty("text").GetString() + "\n";
                cursor = job.GetProperty("cursor").GetInt32();
                var state = job.GetProperty("state").GetString();
                var settled = !job.TryGetProperty("settled", out var stopped) || stopped.GetBoolean();
                if (state != "running" && settled)
                {
                    if (state == "error") Error = Localizer.Current["character.reply_error"];
                    break;
                }
                await Task.Delay(100, _token).ConfigureAwait(true);
            }
        }
        catch (Exception e) when (Handled(e)) { if (!_lifetime.IsCancellationRequested) Error = Localizer.Current["character.request_error"]; }
        finally { if (_responseId == id) Busy = false; }
    }
    private void FollowVoiceReply(JsonElement reply)
    {
        var id = reply.GetProperty("response_id").GetString()!;
        _responseId = id; _ignoreEvents = false; Busy = true;
        Transcript += (Transcript.Length == 0 ? "" : "\n\n") + Localizer.Current["character.owner"] + ": " + reply.GetProperty("text").GetString() + "\nRAMVI: ";
        _ = PollAsync(id);
    }
    private async Task StopAsync()
    {
        _ignoreEvents = true;
        try { await CallAsync("cancel").ConfigureAwait(true); }
        catch (Exception e) when (Handled(e)) { Error = Localizer.Current["character.request_error"]; }
    }
    private static bool Handled(Exception e) => e is ApiException or HttpRequestException or InvalidOperationException or KeyNotFoundException or JsonException or OperationCanceledException or NotSupportedException;
    public override void Relocalize() { base.Relocalize(); Voice.Relocalize(); Speech.Relocalize(); }
    public void Dispose() { if (_disposed) return; _disposed = true; Voice.Dispose(); Speech.Dispose(); _lifetime.Cancel(); _lifetime.Dispose(); }
}
public sealed record CharacterProfile(string Id, string Name);
