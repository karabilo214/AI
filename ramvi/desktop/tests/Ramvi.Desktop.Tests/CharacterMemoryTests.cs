using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>9.17: экран памяти и истории персонажа.</summary>
public sealed class CharacterMemoryTests
{
    private const string Record = """{"id":"m1","kind":"semantic","subject":"собака","value":"Собаку зовут Бублик","status":"candidate","origin":"interpretation"}""";

    private sealed class Fake
    {
        public List<(string Operation, string Arguments)> Calls { get; } = [];
        public string Records { get; set; } = "[" + Record + "]";
        public bool NoMemory { get; set; }
        public int HistoryChanged { get; set; }
        public bool Enabled { get; set; } = true;
        public bool Reset { get; set; }
        public bool Forgotten { get; set; }

        private string Adaptation(string operation, JsonObject? arguments)
        {
            if (operation == "adaptation_enable") Enabled = arguments!["enabled"]!.GetValue<bool>();
            if (operation == "adaptation_reset") Reset = true;
            if (operation == "adaptation_restore") Reset = false;
            var corrections = Reset ? "{}" : """{"humor":-0.03}""";
            return "{\"adaptation\":{\"enabled\":" + (Enabled ? "true" : "false") + ",\"profile\":\"sarcastic@1.0.0\",\"corrections\":" + corrections
                + ",\"applied\":{},\"history\":[{\"id\":\"a1\",\"trait\":\"humor\",\"before\":0.0,\"after\":-0.03,\"action\":\"step\",\"snapshot_id\":\"s1\",\"created_at\":\"2026-11-20T10:00:00Z\",\"reason\":{}}]"
                + ",\"snapshots\":[{\"id\":\"s2\",\"kind\":\"saved\",\"traits\":{\"humor\":-0.03},\"created_at\":\"2026-11-20T11:00:00Z\"},{\"id\":\"s1\",\"kind\":\"before_step\",\"traits\":{},\"created_at\":\"2026-11-20T10:00:00Z\"}]}}";
        }

        private string ForgetBrain()
        {
            Forgotten = true;
            return """{"brain":"memory","forgotten":3}""";
        }

        public Task<JsonElement> Call(string operation, JsonObject? arguments)
        {
            Calls.Add((operation, arguments?.ToJsonString() ?? "{}"));
            if (NoMemory && operation.StartsWith("memory", StringComparison.Ordinal)) throw new InvalidOperationException("Character request failed");
            var json = operation switch
            {
                "status" => """{"history":[{"id":"t1","user":"Мою собаку зовут Бублик","assistant":"Запомню."}]}""",
                "memory_export" => """{"memory":{"schema":"ramvi.character.memory_export/1","records":[]}}""",
                "history_forget" or "history_clear" => """{"history":{"removed":["t1"],"memory_forgotten":1}}""",
                "feedback_view" => """{"feedback":{"preferences":[{"dimension":"humor","direction":-1,"agreement":1.0,"reliable":3,"unreliable":2,"responses":2,"established":true}],"signals":[]}}""",
                "feedback_clear" => """{"feedback":{"preferences":[],"signals":[]}}""",
                "behavior" => Forgotten
                    ? """{"rows":[],"mood":{"irritation":0.0,"cheer":0.0,"at":0,"cause":""}}"""
                    : """{"rows":[{"event":"mistake","strategy":"roast","direction":-1,"status":"guess","reactions":3,"agreement":1.0}],"mood":{"irritation":0.62,"cheer":0.0,"at":1,"cause":"rudeness"}}""",
                "brain_forget" => ForgetBrain(),
                "adaptation" or "adaptation_enable" or "adaptation_save" or "adaptation_reset" or "adaptation_restore" => Adaptation(operation, arguments),
                _ => "{\"memory\":{\"records\":" + Records + "}}",
            };
            return Task.FromResult(JsonDocument.Parse(json).RootElement.Clone());
        }
    }

    private static (CharacterMemoryViewModel Model, Fake Fake) Create()
    {
        Localizer.Current = Localizer.Load("ru");
        var fake = new Fake();
        var model = new CharacterMemoryViewModel(fake.Call, () => { fake.HistoryChanged++; return Task.CompletedTask; });
        return (model, fake);
    }

    [Fact]
    public async Task Refresh_shows_records_with_readable_origin_and_history()
    {
        var (model, _) = Create();
        await model.RefreshAsync();
        var record = Assert.Single(model.Records);
        Assert.True(model.Available && !model.NoRecords);
        Assert.Equal("факт · не подтверждено · догадка персонажа", record.Details);
        Assert.Equal("Мою собаку зовут Бублик", Assert.Single(model.History).User);
    }

    [Fact]
    public async Task Correction_is_sent_only_after_a_real_change()
    {
        var (model, fake) = Create();
        await model.RefreshAsync();
        var record = model.Records[0];
        Assert.False(record.SaveCommand.CanExecute(null));
        record.Value = "  Собаку зовут Тузик ";
        Assert.True(record.SaveCommand.CanExecute(null));
        await record.SaveCommand.RunAsync();
        Assert.Contains(fake.Calls, c => c.Operation == "memory_correct" && JsonNode.Parse(c.Arguments)!["value"]!.GetValue<string>() == "Собаку зовут Тузик");
        record = model.Records[0];
        await record.ConfirmCommand.RunAsync();
        await record.ForgetCommand.RunAsync();
        Assert.Contains(fake.Calls, c => c.Operation == "memory_confirm");
        Assert.Contains(fake.Calls, c => c.Operation == "memory_forget");
    }

    [Fact]
    public async Task Clearing_everything_needs_a_second_press_and_can_be_cancelled()
    {
        var (model, fake) = Create();
        await model.ClearMemoryCommand.RunAsync();
        Assert.True(model.ClearMemoryArmed);
        Assert.DoesNotContain(fake.Calls, c => c.Operation == "memory_clear");
        await model.CancelClearCommand.RunAsync();
        Assert.False(model.ClearMemoryArmed);
        await model.ClearMemoryCommand.RunAsync();
        await model.ClearMemoryCommand.RunAsync();
        Assert.Single(fake.Calls, c => c.Operation == "memory_clear");

        await model.ClearHistoryCommand.RunAsync();
        Assert.True(model.ClearHistoryArmed && !model.ClearMemoryArmed);
        await model.ClearHistoryCommand.RunAsync();
        Assert.Single(fake.Calls, c => c.Operation == "history_clear");
        Assert.Equal(1, fake.HistoryChanged);
    }

    [Fact]
    public async Task Deleting_a_turn_reports_forgotten_memory_and_refreshes_transcript()
    {
        var (model, fake) = Create();
        await model.RefreshAsync();
        await model.History[0].ForgetCommand.RunAsync();
        Assert.Contains(fake.Calls, c => c.Operation == "history_forget" && JsonNode.Parse(c.Arguments)!["ids"]![0]!.GetValue<string>() == "t1");
        Assert.Contains("1", model.Status, StringComparison.Ordinal);
        Assert.Equal(1, fake.HistoryChanged);
    }

    [Fact]
    public async Task Export_is_readable_json_and_missing_memory_is_explained()
    {
        var (model, fake) = Create();
        var json = await model.ExportAsync();
        Assert.NotNull(json);
        Assert.Contains("ramvi.character.memory_export/1", json, StringComparison.Ordinal);
        Assert.Contains("\n", json, StringComparison.Ordinal);

        fake.NoMemory = true;
        await model.RefreshAsync();
        Assert.False(model.Available);
        Assert.Equal(Localizer.Current["character.memory_unavailable"], model.Error);
    }

    [Fact]
    public async Task Preferences_show_direction_settledness_and_signal_counts()
    {
        var (model, fake) = Create();
        await model.RefreshAsync();
        var preference = Assert.Single(model.Preferences);
        Assert.Equal("юмор: меньше / не нравится · установлено · достоверных 3, неоднозначных 2", preference.Text);
        Localizer.Current = Localizer.Load("en");
        try
        {
            model.Relocalize();
            Assert.StartsWith("humor: less / disliked · settled", model.Preferences[0].Text, StringComparison.Ordinal);
        }
        finally { Localizer.Current = Localizer.Load("ru"); }
        await model.ClearFeedbackCommand.RunAsync();
        Assert.True(model.NoPreferences);
        Assert.Contains(fake.Calls, c => c.Operation == "feedback_clear");
    }

    [Fact]
    public async Task Adaptation_switch_history_reset_and_restore_send_the_right_commands()
    {
        var (model, fake) = Create();
        await model.RefreshAsync();
        Assert.True(model.AdaptationAvailable && model.AdaptationEnabled);
        Assert.Equal("юмор: −0,03", Assert.Single(model.Corrections).Text);
        var change = Assert.Single(model.AdaptationHistory);
        Assert.Equal("юмор: 0,00 → −0,03 · подстройка · 2026-11-20T10:00:00Z", change.Text);
        var point = Assert.Single(model.SavedPoints);  // автоматические снимки в список точек не попадают
        Assert.True(model.HasSavedPoints);

        model.AdaptationEnabled = false;
        await Task.Yield();
        Assert.Contains(fake.Calls, c => c.Operation == "adaptation_enable" && c.Arguments == """{"enabled":false}""");
        Assert.False(model.AdaptationEnabled);
        var sent = fake.Calls.Count;
        fake.Enabled = true;  // включили в другом окне
        await model.RefreshAsync();  // чтение состояния показывает его, но не шлёт переключение обратно
        Assert.True(model.AdaptationEnabled);
        Assert.DoesNotContain(fake.Calls.Skip(sent), c => c.Operation == "adaptation_enable");

        await model.ResetAdaptationCommand.RunAsync();  // первое нажатие только спрашивает
        Assert.True(model.ResetArmed);
        Assert.DoesNotContain(fake.Calls, c => c.Operation == "adaptation_reset");
        await model.ResetAdaptationCommand.RunAsync();
        Assert.False(model.ResetArmed);
        Assert.True(model.NoCorrections);
        Assert.False(model.ResetAdaptationCommand.CanExecute(null));  // сбрасывать уже нечего

        await change.RestoreCommand.RunAsync();
        Assert.Contains(fake.Calls, c => c.Operation == "adaptation_restore" && c.Arguments == """{"snapshot_id":"s1"}""");
        await point.RestoreCommand.RunAsync();
        Assert.Contains(fake.Calls, c => c.Operation == "adaptation_restore" && c.Arguments == """{"snapshot_id":"s2"}""");
        await model.UndoAdaptationCommand.RunAsync();
        Assert.Contains(fake.Calls, c => c.Operation == "adaptation_restore" && c.Arguments == "{}");
        await model.SaveAdaptationCommand.RunAsync();
        Assert.Contains(fake.Calls, c => c.Operation == "adaptation_save");
        Assert.Equal("", model.Error);

        Localizer.Current = Localizer.Load("en");
        try
        {
            model.Relocalize();
            Assert.Equal("humor: −0.03", model.Corrections[0].Text);
        }
        finally { Localizer.Current = Localizer.Load("ru"); }
    }

    [Fact]
    public async Task Learned_behaviour_and_mood_are_shown_and_forgotten_on_the_second_press()
    {
        var (model, fake) = Create();
        await model.RefreshAsync();
        var row = Assert.Single(model.Behavior).Text;
        Assert.Equal("Когда он ошибается, а персонаж подкалывает: вас раздражает · догадка · реакций: 3", row);
        Assert.Contains("злость 0", model.Mood, StringComparison.Ordinal);
        await model.ForgetBehaviorCommand.RunAsync();
        Assert.DoesNotContain(fake.Calls, c => c.Operation == "brain_forget"); // первое нажатие — только вопрос
        Assert.True(model.ForgetBehaviorArmed);
        await model.ForgetBehaviorCommand.RunAsync();
        Assert.Contains(fake.Calls, c => c.Operation == "brain_forget");
        Assert.True(model.NoBehavior);
        Assert.Equal("спокоен", model.Mood);
    }
}
