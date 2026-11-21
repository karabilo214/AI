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
    public async Task Profanity_checkbox_reads_the_module_and_sends_the_owner_boundary()
    {
        var sent = new List<string>();
        var api = new FakeChatApi();
        api.ModuleHandler = (section, command, body) =>
        {
            if (command is "voice" or "speech") return Task.FromResult(Json("{\"available\":false}"));
            var operation = body.GetProperty("operation").GetString()!;
            if (operation == "boundary") sent.Add(body.GetProperty("arguments").GetRawText());
            return Task.FromResult(operation == "status"
                ? Json(Status.Replace("\"profile_id\":\"friendly\"", "\"profile_id\":\"friendly\",\"profanity\":true"))
                : Json("{}"));
        };
        using var vm = new CharacterViewModel(() => api, new ObservableCollection<ModelInfo>(api.Models));
        await vm.LoadAsync();
        Assert.True(vm.Profanity);   // загрузка из модуля не отправляет команду
        Assert.Empty(sent);
        vm.Profanity = false;
        for (var i = 0; i < 50 && sent.Count == 0; i++) await Task.Delay(10);
        Assert.Equal(["{\"name\":\"profanity\",\"allowed\":false}"], sent);
    }
    [AvaloniaFact]
    public async Task Rating_buttons_send_dimension_and_direction_for_the_last_reply()
    {
        var sent = new List<string>();
        var api = new FakeChatApi();
        api.ModuleHandler = (section, command, body) =>
        {
            if (command is "voice" or "speech") return Task.FromResult(Json("{\"available\":false}"));
            var operation = body.GetProperty("operation").GetString()!;
            if (operation == "feedback") sent.Add(body.GetProperty("arguments").GetRawText());
            return Task.FromResult(operation switch
            {
                "status" => Json(Status),
                "feedback" => Json("""{"feedback":{"preferences":[{"dimension":"humor","direction":-1,"agreement":1,"reliable":1,"unreliable":0,"responses":1,"established":false}],"signals":[]}}"""),
                _ => Json("{}")
            });
        };
        using var vm = new CharacterViewModel(() => api, new ObservableCollection<ModelInfo>(api.Models));
        await vm.LoadAsync();
        Assert.True(vm.CanRate && vm.RateCommand.CanExecute("humor:-1"));
        await vm.RateCommand.RunAsync("humor:-1");
        Assert.Equal("""{"dimension":"humor","direction":-1}""", Assert.Single(sent));
        Assert.Equal(Ramvi.Desktop.Localization.Localizer.Current["character.feedback_saved"], vm.FeedbackStatus);
        var shown = Assert.Single(vm.Memory.Preferences).Text;
        Assert.Contains(Ramvi.Desktop.Localization.Localizer.Current["character.pref_tentative"], shown, StringComparison.Ordinal);
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
