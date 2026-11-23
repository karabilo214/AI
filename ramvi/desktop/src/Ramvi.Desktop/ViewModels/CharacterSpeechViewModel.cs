using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

public sealed record SpeechVoice(string Id, string Name, string[] Languages);
public sealed record SpeechProvider(string Id, string Name, bool External, string EmotionControl, JsonElement Pricing, SpeechVoice[] Voices);

/// <summary>Explicit recovery for future answers. Never retries a partly spoken segment.</summary>
public sealed class CharacterSpeechViewModel : ObservableObject, IDisposable
{
    private readonly Func<string, JsonObject?, Task<JsonElement>> _call;
    private readonly Action<long,string?,string?>? _generationChanged;
    private readonly Action<JsonElement,double>? _playbackChanged;
    private readonly CancellationTokenSource _lifetime = new();
    private readonly CancellationToken _token;
    private SpeechProvider? _provider;
    private SpeechVoice? _voice;
    private bool _enabled, _remoteText, _loading, _selectingProvider, _disposed, _available, _textOnly;
    private string _key = "", _error = "", _state = "off", _fallback = "", _inventory = "";
    private int _watch, _catalogVersion, _selectionVersion;
    private Task _selectionStop = Task.CompletedTask;
    public CharacterSpeechViewModel(Func<string, JsonObject?, Task<JsonElement>> call, Action<long,string?,string?>? generationChanged = null, Action<JsonElement,double>? playbackChanged = null)
    {
        _call=call; _generationChanged=generationChanged; _playbackChanged=playbackChanged; _token=_lifetime.Token;
        SaveCommand=new Command(()=>ApplyAsync(false),()=>_available && Provider is not null);
        KeyCommand=new Command(SaveKeyAsync,()=>IsExternal);
        PreviewCommand=new Command(()=>ApplyAsync(true),()=>_available && Provider is not null);
        ResumeCommand=new Command(()=>ApplyAsync(false),()=>_available && Provider is not null && Enabled);
        ToggleMuteCommand=new Command(ToggleMuteAsync,()=>_available && Provider is not null);
        RefreshCommand=new Command(()=>ReadCatalogAsync(true,true));
        StopCommand=new Command(StopAsync);
    }
    public ObservableCollection<SpeechProvider> Providers { get; }=[];
    public ObservableCollection<SpeechVoice> Voices { get; }=[];
    public Command SaveCommand { get; }
    public Command KeyCommand { get; }
    public Command PreviewCommand { get; }
    public Command StopCommand { get; }
    public Command ResumeCommand { get; }
    /// <summary>9.26: кнопка-динамик у персонажа — выключить или снова включить озвучку.</summary>
    public Command ToggleMuteCommand { get; }
    public string MuteText=>Localizer.Current[Enabled ? "character.speech_mute" : "character.speech_unmute"];
    public Command RefreshCommand { get; }
    private VoiceOption? _deliveryMode;
    private VoiceOption? _buffer;
    private bool _reactions = true;
    private string _timing = "";
    private string _syncState = "";
    public string Timing { get => _timing; private set => Set(ref _timing, value); }
    public string SyncState { get => _syncState; private set => Set(ref _syncState, value); }
    public List<VoiceOption> DeliveryModes { get; private set; } = [new("sentences", Localizer.Current["character.delivery.sentences"]), new("complete", Localizer.Current["character.delivery.complete"])];
    public List<VoiceOption> Buffers { get; } = [new("1", "1"), new("2", "2"), new("3", "3")];
    public VoiceOption? DeliveryMode { get => _deliveryMode ?? DeliveryModes[0]; set => Set(ref _deliveryMode, value); }
    public VoiceOption? Buffer { get => _buffer ?? Buffers[1]; set => Set(ref _buffer, value); }
    public bool Reactions { get => _reactions; set => Set(ref _reactions, value); }
    public bool Enabled { get=>_enabled;set { if(Set(ref _enabled,value)) { Raise(nameof(MuteText)); RefreshCommands(); if(!value && !_loading) QueueStop(); } } }
    public bool RemoteText { get=>_remoteText;set=>Set(ref _remoteText,value); }
    public bool TextOnly { get=>_textOnly;private set=>Set(ref _textOnly,value); }
    public string Key { get=>_key;set=>Set(ref _key,value); }
    public string Error { get=>_error=="" ? "" : Localizer.Current[_error]; }
    public string InventoryStatus=>_inventory=="" ? "" : Localizer.Current[_inventory];
    public string Status=>Localizer.Current["character.speech_state."+_state];
    /// <summary>Голос сейчас готовится или звучит: есть что останавливать.</summary>
    public bool Speaking=>_state is "queued" or "synthesizing" or "playing";
    public string Fallback { get=>_fallback;private set=>Set(ref _fallback,value); }
    public bool IsExternal=>Provider?.External==true;
    public SpeechVoice? Voice { get=>_voice;set { if(Set(ref _voice,value) && !_loading && !_selectingProvider) QueueStop(); } }
    public SpeechProvider? Provider
    {
        get=>_provider;
        set {
            var previousId=_provider?.Id;
            if(!Set(ref _provider,value)) return;
            _selectingProvider=true;
            try {
                Voices.Clear();foreach(var voice in value?.Voices ?? []) Voices.Add(voice);
                Voice=Voices.FirstOrDefault(v=>v.Languages.Contains(Localizer.Current.Language)) ?? Voices.FirstOrDefault();
            } finally {_selectingProvider=false;}
            Raise(nameof(IsExternal));Raise(nameof(Price));Raise(nameof(Source));if(previousId!=value?.Id) Key="";RefreshCommands();
            if(!_loading) { if(previousId!=value?.Id) RemoteText=false; QueueStop(); }
        }
    }
    private void RefreshCommands() { SaveCommand.Refresh();PreviewCommand.Refresh();ResumeCommand.Refresh();KeyCommand.Refresh();ToggleMuteCommand?.Refresh(); }
    private async Task ToggleMuteAsync()
    {
        // Выключение: текущий звук останавливается сразу, ответы дальше только текстом. Включение —
        // прежние голос и поставщик, без захода в настройки.
        if (Enabled) { await TextModeAsync().ConfigureAwait(true); return; }
        Enabled = true;
        await ApplyAsync(false).ConfigureAwait(true);
    }
    private void SetError(string key) { _error=key;Raise(nameof(Error)); }
    private void QueueStop()
    {
        _selectionVersion++;
        async Task Then(Task previous) { await previous.ConfigureAwait(true);if(!_disposed) await StopAsync().ConfigureAwait(true); }
        _selectionStop=Then(_selectionStop);
    }
    public string Source=>Provider?.Pricing.TryGetProperty("source",out var value)==true ? value.GetString() ?? "" : "";
    public string Price {
        get {
            if(Provider is not {} provider) return "";
            if(!provider.External) return Localizer.Current["character.speech_free"];
            var price=provider.Pricing;var date=price.GetProperty("checked_on").GetString()!;
            if(price.TryGetProperty("text_input_per_million_tokens",out var input))
                return Localizer.Current.Format("character.speech_token_price",("input",input.ToString()),("output",price.GetProperty("audio_output_per_million_tokens").ToString()),("date",date));
            return Localizer.Current.Format("character.speech_plan_price",("price",price.GetProperty("published_from_per_1000_characters").ToString()),("date",date));
        }
    }
    public async Task LoadAsync()
    {
        await ReadCatalogAsync(false,false).ConfigureAwait(true);
        if(!_available || _disposed) return;
        try {
            var policy = await _call("policy", null).ConfigureAwait(true);
            if (policy.TryGetProperty("settings", out var saved))
            {
                DeliveryMode = DeliveryModes.FirstOrDefault(x => x.Code == saved.GetProperty("mode").GetString());
                Buffer = Buffers.FirstOrDefault(x => x.Code == saved.GetProperty("buffer").ToString());
                Reactions = saved.GetProperty("reactions").GetBoolean();
            }
            await _call("resume",null).ConfigureAwait(true);Watch();
        }
        catch(Exception e) when(Handled(e)) { SetError("character.speech_error");TextOnly=true; }
    }
    private async Task ReadCatalogAsync(bool refresh,bool preserve)
    {
        var version=++_catalogVersion;
        try {
            for(var attempt=0;attempt<30;attempt++) {
                var result=await _call("catalog",new JsonObject { ["refresh"]=refresh && attempt==0 }).ConfigureAwait(true);
                if(version!=_catalogVersion || _disposed) return;
                _available=result.TryGetProperty("available",out var available) && available.GetBoolean();
                if(!_available) { SetError("character.speech_error");RefreshCommands();return; }
                var settings=result.GetProperty("settings");
                var selectedProvider=preserve ? Provider?.Id : settings.GetProperty("provider").GetString();
                var selectedVoice=preserve ? Voice?.Id : settings.GetProperty("voice").GetString();
                _loading=true;
                try {
                    Providers.Clear();
                    foreach(var item in result.GetProperty("providers").EnumerateArray()) {
                        var voices=item.GetProperty("voices").EnumerateArray().Select(v=>new SpeechVoice(v.GetProperty("id").GetString()!,v.GetProperty("name").GetString()!,v.GetProperty("languages").EnumerateArray().Select(l=>l.GetString()!).ToArray())).ToArray();
                        Providers.Add(new(item.GetProperty("id").GetString()!,item.GetProperty("name").GetString()!,item.GetProperty("external").GetBoolean(),item.GetProperty("emotion_control").GetString()!,item.GetProperty("pricing").Clone(),voices));
                    }
                    Provider=Providers.FirstOrDefault(p=>p.Id==selectedProvider) ?? Providers.FirstOrDefault();
                    if(!string.IsNullOrEmpty(selectedVoice)) {
                        Voice=Voices.FirstOrDefault(v=>v.Id==selectedVoice);
                        if(Voice is null) { var missing=new SpeechVoice(selectedVoice,Localizer.Current.Format("character.speech_missing_voice",("voice",selectedVoice)),[]);Voices.Add(missing);Voice=missing; }
                    }
                    if(!preserve) { Enabled=settings.GetProperty("enabled").GetBoolean();RemoteText=settings.GetProperty("remote_text_enabled").GetBoolean(); }
                } finally {_loading=false;}
                RefreshCommands();preserve=true;
                var inventory=result.TryGetProperty("local_voices",out var local) ? local.GetProperty("state").GetString() : "ready";
                _inventory=inventory=="loading" ? "character.speech_voices_loading" : inventory=="error" ? "character.speech_voices_failed" : "";
                Raise(nameof(InventoryStatus));
                if(inventory!="loading") return;
                await Task.Delay(200,_token).ConfigureAwait(true);
            }
            _inventory="character.speech_voices_failed";Raise(nameof(InventoryStatus));
        }
        catch(Exception e) when(Handled(e)) { SetError("character.speech_error"); }
    }
    private JsonObject Settings()=>new() { ["enabled"]=Enabled,["provider"]=Provider?.Id ?? "system",["voice"]=Voice?.Id ?? "",["remote_text_enabled"]=RemoteText };
    private async Task ApplyAsync(bool preview)
    {
        var settings=Settings();var selection=_selectionVersion;
        try {
            await _selectionStop.ConfigureAwait(true);
            if(selection!=_selectionVersion || _disposed) return;
            await _call("policy", new JsonObject { ["settings"] = new JsonObject { ["mode"] = DeliveryMode?.Code ?? "sentences", ["buffer"] = int.Parse(Buffer?.Code ?? "2", System.Globalization.CultureInfo.InvariantCulture), ["reactions"] = Reactions } }).ConfigureAwait(true);
            if(selection!=_selectionVersion || _disposed) return;
            await _call("configure",new JsonObject { ["settings"]=settings }).ConfigureAwait(true);
            if(selection!=_selectionVersion || _disposed) return;
            if(preview) await _call("preview",null).ConfigureAwait(true);
            SetError("");TextOnly=false;Fallback="";Watch();
        }
        catch(Exception e) when(Handled(e)) { if(selection==_selectionVersion && !_disposed) {SetError("character.speech_error");TextOnly=true;} }
    }
    private async Task SaveKeyAsync()
    {
        try { await _call("key",new JsonObject { ["provider"]=Provider!.Id,["value"]=Key }).ConfigureAwait(true);SetError(""); }
        catch(Exception e) when(Handled(e)) { SetError("character.speech_key_error"); }
        finally {Key="";}
    }
    private static string ErrorKey(string? code)=>code switch {
        "speech.device_unavailable" or "speech.device_lost" or "speech.device_timeout"=>"character.speech_device_failed",
        "speech.provider_timeout" or "speech.audio_timeout" or "speech.deadline"=>"character.speech_timeout",
        "speech.key_missing" or "speech.api_http_401" or "speech.api_http_403"=>"character.speech_auth_failed",
        "speech.remote_text_disabled"=>"character.speech_permission_failed",
        "speech.voice_missing" or "speech.voice_not_supported"=>"character.speech_voice_failed",
        _=>"character.speech_error"
    };
    private void Watch()=>_=WatchAsync(++_watch);
    private async Task WatchAsync(int watch)
    {
        try {
            while(!_token.IsCancellationRequested && watch==_watch) {
                var requestedAt = System.Diagnostics.Stopwatch.GetTimestamp();
                var result=await _call("status",null).ConfigureAwait(true);
                if(watch!=_watch || _token.IsCancellationRequested) return;
                if(result.TryGetProperty("generation",out var generation))
                    _generationChanged?.Invoke(generation.GetInt64(),result.TryGetProperty("response_id",out var response) && response.ValueKind==JsonValueKind.String ? response.GetString() : null,result.TryGetProperty("session_id",out var session) ? session.GetString() : null);
                _playbackChanged?.Invoke(result, System.Diagnostics.Stopwatch.GetElapsedTime(requestedAt).TotalMilliseconds);
                _state=result.GetProperty("state").GetString()!;Raise(nameof(Status));Raise(nameof(Speaking));
                TextOnly=result.TryGetProperty("mode",out var mode) && mode.GetString()=="text";
                if(result.TryGetProperty("error",out var error) && error.ValueKind==JsonValueKind.String) SetError(ErrorKey(error.GetString()));
                if(result.TryGetProperty("last",out var last) && last.ValueKind==JsonValueKind.Object)
                {
                    Fallback=last.GetProperty("fallback").GetBoolean() ? Localizer.Current["character.speech_fallback"] : "";
                    var values = new List<string>();
                    if (last.TryGetProperty("first_pcm_ms", out var pcm)) values.Add(Localizer.Current.Format("character.tts_ms", ("ms", pcm.GetDouble().ToString("0", System.Globalization.CultureInfo.InvariantCulture))));
                    if (last.TryGetProperty("first_dac_ms", out var dac)) values.Add(Localizer.Current.Format("character.dac_ms", ("ms", dac.GetDouble().ToString("0", System.Globalization.CultureInfo.InvariantCulture))));
                    Timing = string.Join(" · ", values);
                }
                else Timing = "";
                SyncState = result.TryGetProperty("sync", out var sync) && sync.ValueKind == JsonValueKind.Object
                    ? sync.GetProperty("emotion").GetString() + " · " + sync.GetProperty("segment_id").GetString() : "";
                if (sync.ValueKind == JsonValueKind.Object && sync.TryGetProperty("reaction_label", out var reason) && reason.ValueKind == JsonValueKind.String && !string.IsNullOrEmpty(reason.GetString()))
                    SyncState += " · " + Localizer.Current["character.reaction_label." + reason.GetString()];
                await Task.Delay(40,_token).ConfigureAwait(true);
            }
        }
        catch(Exception e) when(Handled(e)) { if(!_token.IsCancellationRequested) { _playbackChanged?.Invoke(default, 0); SetError("character.speech_error");TextOnly=true;} }
    }
    public async Task TextModeAsync()
    {
        Enabled = false;
        await StopAsync().ConfigureAwait(true);
        if (_available && Provider is not null) await ApplyAsync(false).ConfigureAwait(true);
    }
    public async Task StopAsync()
    {
        _watch++;
        _playbackChanged?.Invoke(default, 0);
        if(!_available) return;
        try { await _call("stop",null).ConfigureAwait(true);_state="off";TextOnly=Enabled;Raise(nameof(Status));Raise(nameof(Speaking)); }
        catch(Exception e) when(Handled(e)) { if(!_token.IsCancellationRequested) SetError("character.speech_error"); }
    }
    private static bool Handled(Exception e)=>e is ApiException or HttpRequestException or InvalidOperationException or KeyNotFoundException or JsonException or OperationCanceledException or NotSupportedException;
    public override void Relocalize()
    {
        var mode = DeliveryMode?.Code;
        DeliveryModes = [new("sentences", Localizer.Current["character.delivery.sentences"]), new("complete", Localizer.Current["character.delivery.complete"])];
        Raise(nameof(DeliveryModes));
        DeliveryMode = DeliveryModes.FirstOrDefault(x => x.Code == mode);
        base.Relocalize();
    }
    public void Dispose() { if(_disposed) return;_disposed=true;_watch++;_catalogVersion++;_lifetime.Cancel();_lifetime.Dispose();Key=""; }
}
