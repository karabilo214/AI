using System.Text.Json;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class AgentTests
{
    private static JsonElement Json(string value)
    {
        using var document = JsonDocument.Parse(value);
        return document.RootElement.Clone();
    }

    [AvaloniaFact]
    public async Task Agent_has_real_screen_dynamic_settings_and_reviewable_decision()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        api.Sections.Add(new SectionInfo("agent", "section.agent", "agent", 20, "agent", "agent", "available", null));
        var started = false;
        var approved = false;
        JsonElement? saved = null;
        api.AgentHandler = (path, body) => path switch
        {
            "settings" when body is null => Json("""{"settings":{"permissions":{"read":true,"write":false},"restrictions":{"secret_patterns":[".env"]},"access":{"disabled":[],"confirm_writes":true},"limits":{"max_steps":null},"retention":{}},"tools":["read_file","edit_file"]}"""),
            "settings" => Save(body),
            "history" => Json("""{"runs":[]}"""),
            "start" => Start(),
            "decision" => Decide(),
            "state" when !started => Json("""{"task":null,"pending":null}"""),
            "state" when !approved => Json("""{"task":{"task_id":"run","state":"running","result":null},"pending":{"id":"review-1","kind":"write","title":"calc.py","diff":"- return a-b\n+ return a+b"}}"""),
            "state" => Json("""{"task":{"task_id":"run","state":"done","result":{"result":"Fixed"}},"pending":null}"""),
            "runs/run" => Json("""{"summary":{"steps":3,"duration":1,"tokens":{},"context_peak":100},"events":[],"changes":[]}"""),
            _ => throw new InvalidOperationException(path),
        };
        JsonElement Save(JsonElement? payload) { saved = payload; return Json("{}"); }
        JsonElement Start() { started = true; return Json("""{"task_id":"run"}"""); }
        JsonElement Decide() { approved = true; return Json("{}"); }
        using var model = new MainViewModel(api, pickFolder: _ => Task.FromResult<string?>("/project"));
        var window = new MainWindow { DataContext = model };
        window.Show();
        try
        {
            await model.InitializeAsync();
            Assert.Contains(model.Settings.Tabs, t => t.Id == "module:agent");
            model.Sections.Selected = model.Sections.Items.Single(i => i.View == "agent");
            await MainViewModelTests.Until(() => model.Agent.Rights.Count == 2, "Agent loaded");
            Dispatcher.UIThread.RunJobs();
            Assert.True(window.FindControl<Border>("AgentPanel")!.IsVisible);
            Assert.False(window.FindControl<DockPanel>("ChatPanel")!.IsVisible);
            Assert.True(model.Agent.Limits.Single().Unlimited);
            model.Agent.Rights.Single(o => o.Key == "write").Enabled = true;
            await model.Agent.SaveSettingsCommand.RunAsync();
            Assert.True(saved!.Value.GetProperty("permissions").GetProperty("write").GetBoolean());
            Assert.Equal(JsonValueKind.Null, saved.Value.GetProperty("limits").GetProperty("max_steps").ValueKind);
            await model.Agent.BrowseCommand.RunAsync();
            model.Agent.Input = "Fix add";
            Assert.True(model.Agent.StartCommand.CanExecute(null));
            await model.Agent.StartCommand.RunAsync();
            await MainViewModelTests.Until(() => model.Agent.HasPending, "diff appeared");
            Assert.False(approved);
            Assert.Contains("return a+b", model.Agent.PendingDetail, StringComparison.Ordinal);
            Assert.True(window.GetVisualDescendants().OfType<Border>().Single(b => b.Name == "AgentApproval").IsVisible);
            await model.Agent.ApproveCommand.RunAsync();
            await MainViewModelTests.Until(() => !model.Agent.IsRunning && !model.Agent.HasPending, "task completed");
            Assert.True(approved);
            Assert.Equal("Fixed", model.Agent.Result);
            Assert.NotNull(saved);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Agent_request_to_turn_on_a_tool_is_shown_and_answered()
    {
        // 6.14: агент просит включить выключенный инструмент — пользователь видит, какой и зачем.
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        api.Sections.Add(new SectionInfo("agent", "section.agent", "agent", 20, "agent", "agent", "available", null));
        var started = false;
        JsonElement? decision = null;
        api.AgentHandler = (path, body) => path switch
        {
            "settings" when body is null => Json("""{"settings":{"permissions":{"read":true,"write":true,"terminal":true},"restrictions":{},"access":{"disabled":["run_command"]},"limits":{},"retention":{}},"tools":["read_file","run_command"]}"""),
            "history" => Json("""{"runs":[]}"""),
            "start" => Start(),
            "decision" => Decide(body),
            "state" when !started => Json("""{"task":null,"pending":null}"""),
            "state" when decision is null => Json("""{"task":{"task_id":"run","state":"waiting_approval","result":null},"pending":{"id":"tool-1","kind":"tool","title":"run_command","tool":"run_command","description":"Run tests, linters, builds or diagnostics.","why":"run the tests"}}"""),
            "state" => Json("""{"task":{"task_id":"run","state":"done","result":{"result":"Tests pass"}},"pending":null}"""),
            "runs/run" => Json("""{"summary":{"steps":3,"duration":1,"tokens":{},"context_peak":100},"events":[],"changes":[]}"""),
            _ => throw new InvalidOperationException(path),
        };
        JsonElement Start() { started = true; return Json("""{"task_id":"run"}"""); }
        JsonElement Decide(JsonElement? payload) { decision = payload; return Json("{}"); }
        using var model = new MainViewModel(api, pickFolder: _ => Task.FromResult<string?>("/project"));
        var window = new MainWindow { DataContext = model };
        window.Show();
        try
        {
            await model.InitializeAsync();
            model.Sections.Selected = model.Sections.Items.Single(i => i.View == "agent");
            await MainViewModelTests.Until(() => model.Agent.Rights.Count > 0, "Agent loaded");
            await model.Agent.BrowseCommand.RunAsync();
            model.Agent.Input = "Run tests";
            await model.Agent.StartCommand.RunAsync();
            await MainViewModelTests.Until(() => model.Agent.HasPending, "tool request appeared");
            Assert.Contains(Localizer.Current["agent.kind.tool"], model.Agent.PendingTitle, StringComparison.Ordinal);
            Assert.Contains("run_command", model.Agent.PendingTitle, StringComparison.Ordinal);
            Assert.Contains("run the tests", model.Agent.PendingDetail, StringComparison.Ordinal);
            Assert.Contains(Localizer.Current["agent.tool_request_note"], model.Agent.PendingDetail, StringComparison.Ordinal);
            Assert.Equal(Localizer.Current["agent.enable_tool"], model.Agent.ApproveLabel);
            await model.Agent.ApproveCommand.RunAsync();
            await MainViewModelTests.Until(() => !model.Agent.IsRunning && !model.Agent.HasPending, "task completed");
            Assert.Equal("approve", decision!.Value.GetProperty("action").GetString());
            Assert.Equal("tool-1", decision.Value.GetProperty("id").GetString());
        }
        finally { window.Close(); }
    }

    [Fact]
    public async Task Agent_tab_is_absent_when_module_is_not_installed()
    {
        var api = new FakeChatApi();
        using var model = new MainViewModel(api);
        await model.InitializeAsync();
        Assert.DoesNotContain(model.Settings.Tabs, t => t.Id == "module:agent");
        Assert.False(model.Sections.IsAgentSelected);
    }
    [AvaloniaFact]
    public async Task Agent_composer_shares_voice_and_file_rules_but_keeps_chat_draft()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        api.AppSettings = api.AppSettings with { VoiceAutoSend = false };
        api.Sections.Add(new SectionInfo("agent", "section.agent", "agent", 20, "agent", "agent", "available", null));
        var bytes = System.Text.Encoding.UTF8.GetBytes("specification");
        using var model = new MainViewModel(api, pickFiles: () => Task.FromResult<IReadOnlyList<Ramvi.Desktop.Services.DroppedFile>>(
            [new("spec.txt", bytes.Length, () => Task.FromResult<Stream>(new MemoryStream(bytes)))]));
        var window = new MainWindow { DataContext = model };
        window.Show();
        try
        {
            await model.InitializeAsync();
            model.Input = "chat draft";
            model.Sections.Selected = model.Sections.Items.Single(s => s.View == "agent");
            await model.Agent.AttachFilesCommand.RunAsync();
            Assert.Single(model.Agent.Attachments);
            Assert.Empty(model.Attachments);
            Assert.Same(model.Voice, model.Agent.Voice);
            await model.Voice.ToggleCommand.RunAsync();
            Dispatcher.UIThread.RunJobs();
            var overlay = window.GetVisualDescendants().OfType<Border>().Single(b => b.Name == "AgentRecordingOverlay");
            Assert.True(VoiceTests.Shown(overlay));
            await model.Voice.ToggleCommand.RunAsync();
            Assert.Equal("привет мир", model.Agent.Input);
            Assert.Equal("chat draft", model.Input);
            Assert.DoesNotContain(api.Calls, c => c.StartsWith("send", StringComparison.Ordinal));
            var mic = window.GetVisualDescendants().OfType<Button>().Single(b => b.Name == "AgentMicButton");
            Assert.True(VoiceTests.Shown(mic));
        }
        finally { window.Close(); }
    }

}
