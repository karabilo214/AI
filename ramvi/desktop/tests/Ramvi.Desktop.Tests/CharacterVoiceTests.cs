using System.Text.Json;
using System.Text.Json.Nodes;
using Avalonia.Headless.XUnit;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class CharacterVoiceTests
{
    private static JsonElement Json(string value) => JsonDocument.Parse(value).RootElement.Clone();
    [AvaloniaFact]
    public async Task Voice_is_explicit_partial_does_not_send_and_final_follows_once()
    {
        var starts = 0; var followed = new List<string>();
        var response = """{"state":"hearing","partial":"provisional","reply":null}""";
        Task<JsonElement> Call(string op, JsonObject? args)
        {
            if (op == "capabilities") return Task.FromResult(Json("""{"available":true,"remote_host":null}"""));
            if (op == "start") { starts++; return Task.FromResult(Json("{}")); }
            if (op == "stop") { response = """{"state":"inactive","partial":"","reply":null}"""; return Task.FromResult(Json("{}")); }
            return Task.FromResult(Json(response));
        }
        using var voice = new CharacterVoiceViewModel(Call, () => true, e => followed.Add(e.GetProperty("response_id").GetString()!));
        await voice.LoadAsync(); Assert.Equal(0, starts); Assert.False(voice.IsActive);
        await voice.ToggleCommand.RunAsync();
        Assert.Equal(1, starts); Assert.Equal("provisional", voice.Partial); Assert.Empty(followed);
        response = """{"state":"listening","partial":"","reply":{"response_id":"reply1","text":"final"}}""";
        await MainViewModelTests.Until(() => followed.Count == 1, "voice reply");
        await Task.Delay(180); Assert.Single(followed);
        await voice.EndCommand.RunAsync();
        await MainViewModelTests.Until(() => !voice.IsActive, "voice stopped");
    }
    [AvaloniaFact]
    public async Task A_late_voice_result_after_stop_does_not_start_a_reply()
    {
        var pending = new TaskCompletionSource<JsonElement>(); var followed = false;
        var stopped = false;
        Task<JsonElement> Call(string op, JsonObject? args) => op switch
        {
            "capabilities" => Task.FromResult(Json("""{"available":true}""")),
            "start" => Task.FromResult(Json("{}")),
            "stop" => Stop(),
            _ => stopped ? Task.FromResult(Json("""{"state":"inactive","partial":"","reply":null}""")) : pending.Task
        };
        Task<JsonElement> Stop() { stopped = true; return Task.FromResult(Json("{}")); }
        using var voice = new CharacterVoiceViewModel(Call, () => true, _ => followed = true);
        await voice.LoadAsync(); await voice.ToggleCommand.RunAsync();
        await voice.EndCommand.RunAsync();
        pending.SetResult(Json("""{"state":"listening","partial":"late","reply":{"response_id":"late","text":"late"}}"""));
        await MainViewModelTests.Until(() => !voice.IsActive, "late voice dropped");
        Assert.False(followed); Assert.Equal("", voice.Partial);
        voice.Dispose();
    }
}
