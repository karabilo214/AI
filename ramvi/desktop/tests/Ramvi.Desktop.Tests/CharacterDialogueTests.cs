using System.Collections.ObjectModel;
using System.Text.Json;
using Avalonia.Headless.XUnit;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class CharacterDialogueTests
{
    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();
    private const string Status = """
        {"character":{"model":"model-b","draft":"","profile_id":"friendly"},"profiles":[{"id":"friendly","name":"Friendly"}],"history":[{"user":"Earlier","assistant":"Reply"}],"job":null}
        """;
    [AvaloniaFact]
    public async Task Loads_own_history_profile_and_model_then_sends_validated_text()
    {
        var calls = new List<string>();
        var api = new FakeChatApi();
        api.ModuleHandler = (section, command, body) =>
        {
            Assert.Equal("character", section);
            if (command is "voice" or "speech") return Task.FromResult(Json("{\"available\":false}"));
            Assert.Equal("dialogue", command);
            var operation = body.GetProperty("operation").GetString()!; calls.Add(operation);
            return Task.FromResult(operation switch
            {
                "status" => Json(Status),
                "start" => Json("""{"response_id":"reply1"}"""),
                "poll" => Json("""{"job":{"state":"completed","cursor":1,"events":[{"type":"segment","response_id":"reply1","text":"Fresh reply"}]}}"""),
                _ => Json("{}")
            });
        };
        using var vm = new CharacterViewModel(() => api, new ObservableCollection<ModelInfo>(api.Models));
        await vm.LoadAsync();
        Assert.Equal("model-b", vm.Model!.Id); Assert.Equal("friendly", vm.Profile!.Id);
        Assert.Contains("Earlier", vm.Transcript);
        vm.Draft = "Hello"; await vm.Send.RunAsync();
        Assert.Contains("Fresh reply", vm.Transcript); Assert.False(vm.Busy); Assert.Equal("", vm.Draft);
        Assert.Equal(new[] { "status", "memory", "cancel", "status", "start", "poll" }, calls);
    }
    [AvaloniaFact]
    public async Task Stop_discards_a_poll_response_that_arrives_late()
    {
        var pending = new TaskCompletionSource<JsonElement>();
        var api = new FakeChatApi();
        api.ModuleHandler = (_, _, body) => body.GetProperty("operation").GetString() switch
        {
            "status" => Task.FromResult(Json(Status)),
            "start" => Task.FromResult(Json("""{"response_id":"reply1"}""")),
            "poll" => pending.Task,
            _ => Task.FromResult(Json("{}"))
        };
        using var vm = new CharacterViewModel(() => api, new ObservableCollection<ModelInfo>(api.Models));
        await vm.LoadAsync(); vm.Draft = "Hello";
        var send = vm.Send.RunAsync();
        await vm.Stop.RunAsync();
        pending.SetResult(Json("""{"job":{"state":"cancelled","cursor":1,"events":[{"type":"segment","response_id":"reply1","text":"Late text"}]}}"""));
        await send;
        Assert.DoesNotContain("Late text", vm.Transcript); Assert.False(vm.Busy);
        vm.Dispose(); // Idempotent, including disposal by MainWindow.
    }
}
