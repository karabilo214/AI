using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>9.21: поручения персонажа агенту на экране.</summary>
public sealed class CharacterAgentTests
{
    private sealed class Fake
    {
        private readonly List<(string Target, string Operation, string Arguments)> _calls = [];
        public List<(string Target, string Operation, string Arguments)> Calls { get { lock (_calls) return [.. _calls]; } }
        private void Add(string target, string operation, string arguments) { lock (_calls) _calls.Add((target, operation, arguments)); }
        public string? Binding { get; private set; }
        public System.Collections.Concurrent.ConcurrentQueue<string> States { get; } = new();
        public bool AgentMissing { get; set; }
        public int Finished { get; set; }

        public Task<JsonElement> Module(string operation, JsonObject? arguments)
        {
            Add("module", operation, arguments?.ToJsonString() ?? "{}");
            if (operation == "agent_bind") Binding = arguments!["binding_id"]?.GetValue<string>();
            var task = "null";
            string? error = null;
            if (operation == "delegate")
            {
                if (Binding is null) error = "agent_off";
                else task = Task("running", "running");
            }
            else if (operation == "delegation" && States.TryDequeue(out var state))
            {
                var next = state.Split('/');
                task = Task(next[0], next[1]);
            }
            else if (operation == "delegate_cancel") task = Task("cancelled", "cancelled");
            var json = "{\"delegation\":{\"enabled\":" + (Binding is null ? "false" : "true") + ",\"task\":" + task + ",\"error\":" + (error is null ? "null" : "\"" + error + "\"") + "}}";
            return System.Threading.Tasks.Task.FromResult(JsonDocument.Parse(json).RootElement.Clone());
        }

        private static string Task(string state, string phase) =>
            "{\"task_id\":\"t1\",\"goal\":\"Почини тесты\",\"state\":\"" + state + "\",\"phase\":\"" + phase + "\",\"steps\":1,\"result\":\"\",\"stop_reason\":\"\",\"error\":null}";

        public Task<JsonElement> Agent(string path, JsonObject body)
        {
            Add("agent", path, body.ToJsonString());
            if (AgentMissing) throw new HttpRequestException("agent section is not installed");
            var enabled = body["enabled"]!.GetValue<bool>();
            return System.Threading.Tasks.Task.FromResult(JsonDocument.Parse(enabled ? """{"binding_id":"b-1"}""" : """{"binding_id":null}""").RootElement.Clone());
        }
    }

    private static readonly ModelInfo Qwen = new("qwen", "qwen3", "Q4_K_M", 1);

    private static (CharacterAgentViewModel Model, Fake Fake) Create(ModelInfo? model = null, int pollMs = 1)
    {
        Localizer.Current = Localizer.Load("ru");
        var fake = new Fake();
        var vm = new CharacterAgentViewModel(fake.Module, fake.Agent, _ => System.Threading.Tasks.Task.FromResult<string?>("/work/project"),
            () => model, () => { fake.Finished++; return System.Threading.Tasks.Task.CompletedTask; }, TimeSpan.FromMilliseconds(pollMs));
        return (vm, fake);
    }

    private static async Task Until(Func<bool> condition)
    {
        var deadline = DateTime.UtcNow.AddSeconds(3);
        while (!condition() && DateTime.UtcNow < deadline) await System.Threading.Tasks.Task.Delay(5);
        Assert.True(condition());
    }

    [Fact]
    public async Task Enabling_needs_folder_and_model_and_binds_through_the_trusted_route()
    {
        var (model, fake) = Create(Qwen);
        model.Enabled = true;
        await Until(() => model.Error.Length > 0);
        Assert.False(model.Enabled);
        Assert.Empty(fake.Calls);  // без папки ничего не отправлено

        await model.BrowseCommand.RunAsync();
        Assert.Equal("/work/project", model.Folder);
        model.Enabled = true;
        await Until(() => fake.Binding is not null);
        Assert.Equal(("agent", "delegate", """{"module":"character","enabled":true,"workspace":"/work/project","model_id":"qwen","use_ui":false}"""), fake.Calls[0]);
        Assert.Equal(("module", "agent_bind", """{"binding_id":"b-1"}"""), fake.Calls[1]);
        Assert.True(model.Enabled);

        model.Enabled = false;
        await Until(() => fake.Binding is null);
        Assert.Contains(fake.Calls, c => c.Target == "agent" && c.Arguments == """{"module":"character","enabled":false}""");
    }

    [Fact]
    public async Task Explicit_task_is_followed_to_the_end_and_can_be_cancelled()
    {
        var (model, fake) = Create(Qwen);
        Assert.False(await model.DelegateAsync("Почини тесты", TestContext.Current.CancellationToken));  // поручения выключены
        await model.BrowseCommand.RunAsync();
        model.Enabled = true;
        await Until(() => model.Enabled && fake.Binding is not null);
        Assert.False(model.CanDelegate("   "));

        fake.States.Enqueue("running/waiting_approval");
        fake.States.Enqueue("done/done");
        Assert.True(await model.DelegateAsync("  Почини тесты  ", TestContext.Current.CancellationToken));
        Assert.Contains(fake.Calls, c => c.Target == "module" && c.Operation == "delegate" && JsonNode.Parse(c.Arguments)!["goal"]!.GetValue<string>() == "Почини тесты");
        Assert.True(model.HasTask);
        await Until(() => !model.Running);
        Assert.Equal("Задача агенту: Почини тесты · готово", model.Status);
        await Until(() => fake.Finished == 1);  // лента диалога перечитана — там итог задачи

        // Отдельная модель с редким опросом: ход задачи читаем сами, без гонки с фоновым опросом.
        (model, fake) = Create(Qwen, 60_000);
        await model.BrowseCommand.RunAsync();
        model.Enabled = true;
        await Until(() => model.Enabled);
        Assert.True(await model.DelegateAsync("Почини тесты", TestContext.Current.CancellationToken));
        Assert.True(model.Running);
        Assert.False(model.CanDelegate("ещё одна"));  // одна задача за раз
        fake.States.Enqueue("running/waiting_approval");
        await model.RefreshAsync();
        Assert.True(model.Waiting);
        Assert.Equal("Задача агенту: Почини тесты · ждёт вашего подтверждения", model.Status);
        await model.CancelCommand.RunAsync();
        Assert.Contains(fake.Calls, c => c.Operation == "delegate_cancel");
        Assert.False(model.Running);
        Localizer.Current = Localizer.Load("en");
        try
        {
            model.Relocalize();
            Assert.Equal("Agent task: Почини тесты · cancelled", model.Status);
        }
        finally { Localizer.Current = Localizer.Load("ru"); }
    }

    [Fact]
    public async Task Missing_agent_shows_a_message_and_keeps_delegation_off()
    {
        var (model, fake) = Create(Qwen);
        fake.AgentMissing = true;
        await model.BrowseCommand.RunAsync();
        model.Enabled = true;
        await Until(() => model.Error == Localizer.Current["character.agent_unavailable"]);
        Assert.False(model.Enabled);
        Assert.DoesNotContain(fake.Calls, c => c.Operation == "agent_bind");
    }
}
