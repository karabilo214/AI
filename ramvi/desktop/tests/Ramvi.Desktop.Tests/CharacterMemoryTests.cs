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

        public Task<JsonElement> Call(string operation, JsonObject? arguments)
        {
            Calls.Add((operation, arguments?.ToJsonString() ?? "{}"));
            if (NoMemory && operation.StartsWith("memory", StringComparison.Ordinal)) throw new InvalidOperationException("Character request failed");
            var json = operation switch
            {
                "status" => """{"history":[{"id":"t1","user":"Мою собаку зовут Бублик","assistant":"Запомню."}]}""",
                "memory_export" => """{"memory":{"schema":"ramvi.character.memory_export/1","records":[]}}""",
                "history_forget" or "history_clear" => """{"history":{"removed":["t1"],"memory_forgotten":1}}""",
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
}
