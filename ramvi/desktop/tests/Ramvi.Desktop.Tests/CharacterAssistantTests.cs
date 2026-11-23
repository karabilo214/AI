using System.Text.Json;
using System.Text.Json.Nodes;
using Avalonia.Headless.XUnit;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>9.28: помощник в приложениях — окно и доступ выбирает только пользователь.</summary>
public sealed class CharacterAssistantTests
{
    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();

    [AvaloniaFact]
    public async Task Owner_picks_a_window_allows_reading_and_turns_it_off()
    {
        Localizer.Current = Localizer.Load("ru");
        var agentCalls = new List<(string Path, string Body)>();
        var moduleCalls = new List<(string Op, string Body)>();
        Task<JsonElement> Agent(string path, JsonObject body)
        {
            agentCalls.Add((path, body.ToJsonString()));
            return Task.FromResult(path == "target"
                ? Json("""{"target":{"application":"notepad.exe","window_id":"0x1","title":"заметки — Блокнот"}}""")
                : path == "delegate" ? Json("""{"binding_id":null}""") : Json("{}"));
        }
        Task<JsonElement> Module(string op, JsonObject? body)
        {
            moduleCalls.Add((op, body?.ToJsonString() ?? ""));
            return Task.FromResult(Json("""{"delegation":{"enabled":false,"task":null},"assistant":null}"""));
        }
        var delegation = new CharacterAgentViewModel(Module, Agent, _ => Task.FromResult<string?>(null), () => null, () => Task.CompletedTask);
        var assistant = new CharacterAssistantViewModel(Agent, Module, delegation, TimeSpan.Zero);
        Assert.False(assistant.ApplyCommand.CanExecute(null)); // без выбранного окна разрешать нечего

        await assistant.PickWindowCommand.RunAsync();
        Assert.Equal("заметки — Блокнот", assistant.Title);
        Assert.True(assistant.ReadUi); Assert.False(assistant.SendToCloud); Assert.False(assistant.AgentActs);

        await assistant.ApplyCommand.RunAsync();
        var grants = agentCalls.Single(c => c.Path == "target/module");
        Assert.Contains("\"ui\":true", grants.Body); Assert.Contains("\"egress\":false", grants.Body);
        Assert.DoesNotContain(agentCalls, c => c.Path == "target/grants"); // гранты агента не трогает
        Assert.Contains(moduleCalls, c => c.Op == "assistant" && c.Body.Contains("notepad.exe", StringComparison.Ordinal));
        Assert.True(assistant.Active);

        await assistant.OffCommand.RunAsync();
        Assert.Contains("\"ui\":false", agentCalls.Last(c => c.Path == "target/module").Body);
        Assert.Contains(moduleCalls, c => c.Op == "assistant" && c.Body.Contains("null", StringComparison.Ordinal));
        Assert.False(assistant.Active);
    }

    [AvaloniaFact]
    public async Task Without_system_windows_the_section_says_it_is_unavailable()
    {
        Localizer.Current = Localizer.Load("ru");
        Task<JsonElement> Agent(string path, JsonObject body) => throw new ApiException("api.bad_request", null);
        Task<JsonElement> Module(string op, JsonObject? body) => Task.FromResult(Json("{}"));
        var delegation = new CharacterAgentViewModel(Module, Agent, _ => Task.FromResult<string?>(null), () => null, () => Task.CompletedTask);
        var assistant = new CharacterAssistantViewModel(Agent, Module, delegation, TimeSpan.Zero);
        await assistant.PickWindowCommand.RunAsync();
        Assert.Equal(Localizer.Current["character.assistant_unavailable"], assistant.Error);
        Assert.False(assistant.HasTarget);
    }
}
