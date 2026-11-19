using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Character;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Character dialogue client: validated segments and generation-fenced presentation.</summary>
public sealed class CharacterViewModel : ObservableObject, IDisposable
{
    private readonly Func<IChatApi> _api;
    private readonly CancellationTokenSource _lifetime = new();
    private readonly CancellationToken _token;
    private string _draft = "", _transcript = "", _error = "";
    private bool _busy, _loading, _disposed, _ignoreEvents;
    private ModelInfo? _model;
    private CharacterProfile? _profile;
    private string? _responseId, _serverSession;
    private int _loadVersion;
    private readonly SemaphoreSlim _commands = new(1, 1);
    private long _requestRevision, _presentationEpoch, _serverGeneration = -1;
    private readonly HashSet<string> _segments = [];
    private CharacterDirective? _presentation;
    public CharacterDirective? Presentation { get => _presentation; private set => Set(ref _presentation,value); }
    private long InvalidateResponse()
    {
        _requestRevision++;
        _responseId=null; _ignoreEvents=true; _segments.Clear(); Busy=false;
        Presentation=new(++_presentationEpoch,null,null,"neutral","idle");
        Stop.Refresh(); return _requestRevision;
    }
    private void ObserveSession(string? session)
    {
        if(session is null || session==_serverSession) return;
        if(_serverSession is not null && _responseId is not null) InvalidateResponse();
        _serverSession=session; _serverGeneration=-1;
    }
    private void ReadSession(JsonElement packet)
    {
        if(packet.TryGetProperty("session_id",out var session)) ObserveSession(session.GetString());
    }
    private void ObserveGeneration(long generation,string? response,string? session=null)
    {
        ObserveSession(session);
        if(generation<=_serverGeneration) return;
        _serverGeneration=generation;
        if(_responseId is not null && _responseId!=response) { InvalidateResponse(); Busy=false; }
    }
    public CharacterViewModel(Func<IChatApi> api, ObservableCollection<ModelInfo> models)
    {
        _token = _lifetime.Token;
        _api = api; Models = models;
        Voice = new CharacterVoiceViewModel((operation, args) => ModuleCallAsync("voice", operation, args), () => Model is not null, FollowVoiceReply);
        Speech = new CharacterSpeechViewModel((operation, args) => ModuleCallAsync("speech", operation, args), ObserveGeneration);
        Send = new Command(SendAsync, () => !string.IsNullOrWhiteSpace(Draft) && Model is not null);
        Stop = new Command(StopAsync, () => Busy || _responseId is not null);
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
        set { if (Set(ref _model, value)) { Send.Refresh(); Voice.Refresh(); if (!_loading) { InvalidateResponse(); _ = ConfigureAsync(); } } }
    }
    public CharacterProfile? Profile
    {
        get => _profile;
        set { if (Set(ref _profile, value) && !_loading && value is not null) _ = SelectProfileAsync(value.Id); }
    }
    private Task<JsonElement> CallAsync(string operation, JsonObject? arguments = null) => ModuleCallAsync("dialogue", operation, arguments);
    private async Task<JsonElement> ModuleCallAsync(string command, string operation, JsonObject? arguments = null)
    {
        var serial=(command=="speech" && operation is "stop" or "configure" or "preview") || (command=="voice" && operation is "start" or "stop");
        if(serial) await _commands.WaitAsync(_token).ConfigureAwait(true);
        try {
            using var document=JsonDocument.Parse(new JsonObject { ["operation"]=operation,["arguments"]=arguments ?? new JsonObject() }.ToJsonString());
            if(command=="voice" && operation=="start") await CallAsync("configure",new JsonObject { ["draft"]=Draft,["model"]=Model?.Id }).ConfigureAwait(true);
            var result=await _api().ModuleCommandAsync("character",command,document.RootElement.Clone(),_token).ConfigureAwait(true);
            if(result.TryGetProperty("error",out var error) && error.ValueKind is not (JsonValueKind.Null or JsonValueKind.Undefined) && !result.TryGetProperty("state",out _)) throw new InvalidOperationException("Character request failed");
            return result;
        } finally { if(serial) _commands.Release(); }
    }

    public async Task LoadAsync()
    {
        var load=++_loadVersion;
        try
        {
            var result = await CallAsync("status").ConfigureAwait(true);
            if(load!=_loadVersion || _token.IsCancellationRequested) return;
            ReadSession(result);
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
        await _commands.WaitAsync(_token).ConfigureAwait(true);
        try { await CallAsync("configure",new JsonObject { ["draft"]=Draft,["model"]=Model?.Id }).ConfigureAwait(true); }
        catch(Exception e) when(Handled(e)) { Error=Localizer.Current["character.request_error"]; }
        finally { _commands.Release(); }
    }
    private async Task SelectProfileAsync(string id)
    {
        InvalidateResponse();
        await _commands.WaitAsync(_token).ConfigureAwait(true);
        try { await CallAsync("profile",new JsonObject { ["id"]=id }).ConfigureAwait(true); }
        catch(Exception e) when(Handled(e)) { Error=Localizer.Current["character.request_error"]; }
        finally { _commands.Release(); }
    }
    private async Task SendAsync()
    {
        var revision=InvalidateResponse();
        Busy=true; Error="";
        var input=Draft; var model=Model!.Id;
        await _commands.WaitAsync(_token).ConfigureAwait(true);
        try
        {
            if(revision!=_requestRevision) return;
            // Fence audio before starting a new LLM response. Wait asynchronously for
            // the old model worker; UI/polling/Stop stay live during this wait.
            await CallAsync("cancel").ConfigureAwait(true);
            var deadline=DateTime.UtcNow.AddSeconds(5);
            while(true) {
                var status=await CallAsync("status").ConfigureAwait(true);
                if(revision!=_requestRevision) return;
                var previous=status.GetProperty("job");
                if(previous.ValueKind!=JsonValueKind.Object || !previous.TryGetProperty("settled",out var settled) || settled.GetBoolean()) break;
                if(DateTime.UtcNow>=deadline) throw new InvalidOperationException("Previous response is stopping");
                await Task.Delay(40,_token).ConfigureAwait(true);
            }
            var result=await CallAsync("start",new JsonObject { ["text"]=input,["model"]=model }).ConfigureAwait(true);
            if(revision!=_requestRevision) return;
            ReadSession(result);
            var id=result.GetProperty("response_id").GetString()!;
            if(result.TryGetProperty("generation",out var generation)) {
                if(generation.GetInt64()<_serverGeneration) { Busy=false; return; }
                ObserveGeneration(generation.GetInt64(),id);
            }
            _responseId=id; _ignoreEvents=false; Draft="";
            Presentation=new(++_presentationEpoch,id,null,"neutral","idle");
            Transcript+=(Transcript.Length==0?"":"\n\n")+Localizer.Current["character.owner"]+": "+input+"\nRAMVI: ";
            _=PollAsync(id);
        }
        catch(Exception e) when(Handled(e)) { if(revision==_requestRevision) { Error=Localizer.Current["character.request_error"]; Busy=false; } }
        finally { _commands.Release(); }
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
                ReadSession(job);
                if(_responseId!=id) break;
                if(job.TryGetProperty("generation",out var generation)) {
                    if(generation.GetInt64()<_serverGeneration) break;
                    ObserveGeneration(generation.GetInt64(),id);
                }
                foreach(var record in job.GetProperty("events").EnumerateArray()) {
                    if(_ignoreEvents || record.GetProperty("response_id").GetString()!=id || record.GetProperty("type").GetString()!="segment") continue;
                    var segment=record.TryGetProperty("id",out var segmentId) ? segmentId.GetString()! : "cursor-"+cursor;
                    if(!_segments.Add(segment)) continue;
                    Transcript+=record.GetProperty("text").GetString()+"\n";
                    if(record.TryGetProperty("animation",out var animation))
                        Presentation=new(_presentationEpoch,id,segment,record.GetProperty("emotion").GetString()!,animation.GetString()!);
                }
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
        catch (Exception e) when (Handled(e)) { if (!_lifetime.IsCancellationRequested && _responseId==id) Error = Localizer.Current["character.request_error"]; }
        finally { if (_responseId == id) Busy = false; }
    }
    private void FollowVoiceReply(JsonElement reply)
    {
        InvalidateResponse();
        ReadSession(reply);
        var id = reply.GetProperty("response_id").GetString()!;
        Presentation=new(++_presentationEpoch,id,null,"neutral","idle");
        _responseId = id; _ignoreEvents = false; Busy = true;
        Transcript += (Transcript.Length == 0 ? "" : "\n\n") + Localizer.Current["character.owner"] + ": " + reply.GetProperty("text").GetString() + "\nRAMVI: ";
        _ = PollAsync(id);
    }
    public async Task StopAsync()
    {
        var revision=InvalidateResponse();
        await _commands.WaitAsync(_token).ConfigureAwait(true);
        try { await CallAsync("cancel").ConfigureAwait(true); }
        catch(Exception e) when(Handled(e)) { if(revision==_requestRevision) Error=Localizer.Current["character.request_error"]; }
        finally { _commands.Release(); if(revision==_requestRevision) Busy=false; }
    }
    private static bool Handled(Exception e) => e is ApiException or HttpRequestException or InvalidOperationException or KeyNotFoundException or JsonException or OperationCanceledException or NotSupportedException;
    public override void Relocalize() { base.Relocalize(); Voice.Relocalize(); Speech.Relocalize(); }
    public void Dispose() { if (_disposed) return; _disposed = true; Voice.Dispose(); Speech.Dispose(); _lifetime.Cancel(); _lifetime.Dispose(); }
}
public sealed record CharacterProfile(string Id, string Name);
