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
