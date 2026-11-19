using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

public sealed record SpeechVoice(string Id, string Name, string[] Languages);
public sealed record SpeechProvider(string Id, string Name, bool External, string EmotionControl, JsonElement Pricing, SpeechVoice[] Voices);

/// <summary>Separate output credentials and opt-in. Never reuses STT credentials or microphone audio.</summary>
public sealed class CharacterSpeechViewModel : ObservableObject, IDisposable
{
    private readonly Func<string, JsonObject?, Task<JsonElement>> _call;
    private readonly CancellationTokenSource _lifetime = new();
    private readonly CancellationToken _token;
    private SpeechProvider? _provider;
    private SpeechVoice? _voice;
    private bool _enabled, _remoteText, _loading, _disposed, _available;
    private string _key = "", _error = "", _state = "off", _fallback = "";
    private int _watch;
    public CharacterSpeechViewModel(Func<string, JsonObject?, Task<JsonElement>> call)
    {
        _call = call; _token = _lifetime.Token;
        SaveCommand = new Command(SaveAsync, () => _available && Provider is not null);
        KeyCommand = new Command(SaveKeyAsync, () => IsExternal);
        PreviewCommand = new Command(PreviewAsync, () => _available && Provider is not null);
        StopCommand = new Command(StopAsync);
    }
    public ObservableCollection<SpeechProvider> Providers { get; } = [];
    public ObservableCollection<SpeechVoice> Voices { get; } = [];
    public Command SaveCommand { get; }
    public Command KeyCommand { get; }
    public Command PreviewCommand { get; }
    public Command StopCommand { get; }
    public bool Enabled { get => _enabled; set => Set(ref _enabled, value); }
    public bool RemoteText { get => _remoteText; set => Set(ref _remoteText, value); }
    public string Key { get => _key; set => Set(ref _key, value); }
    public string Error { get => _error; private set => Set(ref _error, value); }
    public string Status => Localizer.Current["character.speech_state." + _state];
    public string Fallback { get => _fallback; private set => Set(ref _fallback, value); }
    public bool IsExternal => Provider?.External == true;
    public SpeechVoice? Voice { get => _voice; set => Set(ref _voice, value); }
    public SpeechProvider? Provider
    {
        get => _provider;
        set
        {
            if (!Set(ref _provider, value)) return;
            Voices.Clear();
            foreach (var voice in value?.Voices ?? []) Voices.Add(voice);
            Voice = Voices.FirstOrDefault(v => v.Languages.Contains(Localizer.Current.Language)) ?? Voices.FirstOrDefault();
            Raise(nameof(IsExternal)); Raise(nameof(Price)); Raise(nameof(Source));
            Key = ""; KeyCommand.Refresh(); SaveCommand.Refresh(); PreviewCommand.Refresh();
            if (!_loading) _ = StopAsync();
        }
    }
    public string Source => Provider?.Pricing.TryGetProperty("source", out var value) == true ? value.GetString() ?? "" : "";
    public string Price
    {
        get
        {
            if (Provider is not { } provider) return "";
            var price = provider.Pricing;
            if (!provider.External) return Localizer.Current["character.speech_free"];
            var date = price.GetProperty("checked_on").GetString()!;
            if (price.TryGetProperty("text_input_per_million_tokens", out var input))
                return Localizer.Current.Format("character.speech_token_price", ("input", input.ToString()), ("output", price.GetProperty("audio_output_per_million_tokens").ToString()), ("date", date));
            return Localizer.Current.Format("character.speech_plan_price", ("price", price.GetProperty("published_from_per_1000_characters").ToString()), ("date", date));
        }
    }
    public async Task LoadAsync()
    {
        try
        {
            var result = await _call("catalog", null).ConfigureAwait(true);
            _available = result.TryGetProperty("available", out var available) && available.GetBoolean();
            if (!_available) return;
            _loading = true;
            try
            {
                Providers.Clear();
                foreach (var item in result.GetProperty("providers").EnumerateArray())
                {
                    var voices = item.GetProperty("voices").EnumerateArray().Select(v => new SpeechVoice(v.GetProperty("id").GetString()!, v.GetProperty("name").GetString()!, v.GetProperty("languages").EnumerateArray().Select(l => l.GetString()!).ToArray())).ToArray();
                    Providers.Add(new(item.GetProperty("id").GetString()!, item.GetProperty("name").GetString()!, item.GetProperty("external").GetBoolean(), item.GetProperty("emotion_control").GetString()!, item.GetProperty("pricing").Clone(), voices));
                }
                var settings = result.GetProperty("settings");
                Provider = Providers.FirstOrDefault(p => p.Id == settings.GetProperty("provider").GetString());
                Voice = Voices.FirstOrDefault(v => v.Id == settings.GetProperty("voice").GetString()) ?? Voice;
                Enabled = settings.GetProperty("enabled").GetBoolean(); RemoteText = settings.GetProperty("remote_text_enabled").GetBoolean();
            }
            finally { _loading = false; }
            SaveCommand.Refresh(); PreviewCommand.Refresh();
            await _call("resume", null).ConfigureAwait(true);
            _ = WatchAsync(++_watch);
        }
        catch (Exception e) when (Handled(e)) { _available = false; }
    }
    private JsonObject Settings() => new() { ["enabled"] = Enabled, ["provider"] = Provider?.Id ?? "system", ["voice"] = Voice?.Id ?? "", ["remote_text_enabled"] = RemoteText };
    private async Task SaveAsync()
    {
        try { await _call("configure", new JsonObject { ["settings"] = Settings() }).ConfigureAwait(true); Error = ""; _ = WatchAsync(++_watch); }
        catch (Exception e) when (Handled(e)) { Error = Localizer.Current["character.speech_error"]; }
    }
    private async Task SaveKeyAsync()
    {
        try { await _call("key", new JsonObject { ["provider"] = Provider!.Id, ["value"] = Key }).ConfigureAwait(true); Error = ""; }
        catch (Exception e) when (Handled(e)) { Error = Localizer.Current["character.speech_key_error"]; }
        finally { Key = ""; }
    }
    private async Task PreviewAsync()
    {
        try { await _call("configure", new JsonObject { ["settings"] = Settings() }).ConfigureAwait(true); await _call("preview", null).ConfigureAwait(true); Error = ""; _ = WatchAsync(++_watch); }
        catch (Exception e) when (Handled(e)) { Error = Localizer.Current["character.speech_error"]; }
    }
    private async Task WatchAsync(int watch)
    {
        try
        {
            while (!_token.IsCancellationRequested && watch == _watch)
            {
                var result = await _call("status", null).ConfigureAwait(true);
                if (watch != _watch || _token.IsCancellationRequested) return;
                _state = result.GetProperty("state").GetString()!; Raise(nameof(Status));
                if (result.TryGetProperty("error", out var error) && error.ValueKind == JsonValueKind.String) Error = Localizer.Current["character.speech_error"];
                if (result.TryGetProperty("last", out var last) && last.ValueKind == JsonValueKind.Object)
                    Fallback = last.GetProperty("fallback").GetBoolean() ? Localizer.Current["character.speech_fallback"] : "";
                await Task.Delay(200, _token).ConfigureAwait(true);
            }
        }
        catch (Exception e) when (Handled(e)) { if (!_token.IsCancellationRequested) Error = Localizer.Current["character.speech_error"]; }
    }
    public async Task StopAsync()
    {
        _watch++;
        if (!_available) return;
        try { await _call("stop", null).ConfigureAwait(true); _state = "off"; Raise(nameof(Status)); }
        catch (Exception e) when (Handled(e)) { if (!_token.IsCancellationRequested) Error = Localizer.Current["character.speech_error"]; }
    }
    private static bool Handled(Exception e) => e is ApiException or HttpRequestException or InvalidOperationException or KeyNotFoundException or JsonException or OperationCanceledException or NotSupportedException;
    public void Dispose() { if (_disposed) return; _disposed = true; _watch++; _lifetime.Cancel(); _lifetime.Dispose(); Key = ""; }
}
