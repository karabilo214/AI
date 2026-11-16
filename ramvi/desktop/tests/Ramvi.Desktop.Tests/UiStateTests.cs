using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;
using static Ramvi.Desktop.Tests.MainViewModelTests;

namespace Ramvi.Desktop.Tests;

/// <summary>UI-03: девять состояний различаются, у каждого есть действие восстановления.</summary>
public sealed class UiStateTests : IDisposable
{
    private static readonly Localizer Ru = Localizer.Load("ru");
    private readonly string _dir = Directory.CreateTempSubdirectory("ramvi-ui-").FullName;

    public UiStateTests() => Localizer.Current = Ru;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static CancellationToken Cancel => TestContext.Current.CancellationToken;

    private static string[] Ids(MainViewModel model) => [.. model.Banner?.Actions.Select(a => a.Id) ?? []];

    private static async Task<(FakeChatApi Api, MainViewModel Model, string Chat)> Open()
    {
        var api = new FakeChatApi();
        var chat = api.AddChat("t", "model-a", Msg("user", "вопрос"), Msg("assistant", "ответ"));
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        return (api, model, chat);
    }

    /// <summary>Отправить сообщение; сервис отвечает событием done с ошибкой.</summary>
    private static async Task FailWith(FakeChatApi api, MainViewModel model, string code, params (string, object)[] parameters)
    {
        var error = Error(code, parameters);
        await api.Stream.Writer.WriteAsync(new ChatEvent("user", Message: Msg("user", "x")), Cancel);
        await api.Stream.Writer.WriteAsync(new ChatEvent("status", State: "loading", ModelId: "model-a"), Cancel);
        await api.Stream.Writer.WriteAsync(
            new ChatEvent("done", Finish: "error", Error: new ApiError(code, error.Parameters.ToDictionary())), Cancel);
        api.Stream.Writer.Complete();
        model.Input = "x";
        await model.SendCommand.RunAsync();
    }

    [Fact]
    public void Before_connection_state_is_connecting_without_banner()
    {
        var model = new MainViewModel(new FakeChatApi());

        Assert.Equal(UiState.Connecting, model.State);
        Assert.Null(model.Banner);
        Assert.False(model.SendCommand.CanExecute(null));
    }

    [Fact]
    public async Task Ready_has_no_banner()
    {
        var (_, model, _) = await Open();

        Assert.Equal(UiState.Ready, model.State);
        Assert.Null(model.Banner);
        Assert.False(model.IsError);
    }

    [Theory]
    [InlineData("engine.no_memory", UiState.LowMemory, new[] { "choose_model", "retry" })]
    [InlineData("engine.incompatible", UiState.Incompatible, new[] { "choose_model" })]
    [InlineData("engine.model_missing", UiState.NoAccess, new[] { "choose_model", "retry" })]
    [InlineData("api.busy", UiState.ApiBusy, new[] { "retry" })]
    [InlineData("chat.busy_timeout", UiState.ApiBusy, new[] { "retry" })]
    [InlineData("engine.start_failed", UiState.Error, new[] { "retry", "dismiss" })]
    [InlineData("chat.generation_failed", UiState.Error, new[] { "retry", "dismiss" })]
    [InlineData("chat.model_not_found", UiState.NoModel, new[] { "refresh", "choose_model" })]
    public async Task Service_error_becomes_state_with_actions(string code, UiState state, string[] actions)
    {
        var (api, model, _) = await Open();

        await FailWith(api, model, code, ("model_id", "model-a"), ("detail", "подробности"), ("path", "/нет/такой/папки/m.gguf"));

        Assert.Equal(state, model.State);
        Assert.Equal(actions, Ids(model));
        var banner = model.Banner!;
        Assert.True(banner.IsProblem);
        Assert.DoesNotContain("{", banner.Text, StringComparison.Ordinal);
        Assert.DoesNotContain(".", banner.Title, StringComparison.Ordinal); // не ключ, а текст
        Assert.True(model.IsError);
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task Low_memory_explains_what_to_do()
    {
        var (api, model, _) = await Open();

        await FailWith(api, model, "engine.no_memory", ("model_id", "model-a"), ("detail", ""));

        Assert.Equal(Ru["state.low_memory.title"], model.Banner!.Title);
        Assert.Contains(Ru["state.low_memory.hint"], model.Banner.Text, StringComparison.Ordinal);
        Assert.Contains("model-a", model.Banner.Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task No_access_offers_existing_folder_of_model_file()
    {
        var (api, model, _) = await Open();
        var file = Path.Combine(_dir, "m.gguf");
        var opened = new List<string>();
        model = new MainViewModel(api, openFolder: path =>
        {
            opened.Add(path);
            return Task.CompletedTask;
        });
        await model.InitializeAsync();

        await FailWith(api, model, "engine.no_access", ("model_id", "model-a"), ("path", file), ("detail", ""));

        Assert.Equal(UiState.NoAccess, model.State);
        Assert.Equal(["open_folder", "choose_model", "retry"], Ids(model));
        var open = model.Banner!.Actions[0];
        open.Command.Execute(open.Parameter);
        await Until(() => opened.Count == 1, "папка открыта");
        Assert.Equal([_dir], opened);
    }

    [Fact]
    public async Task Loading_then_generating_then_ready()
    {
        var (api, model, _) = await Open();
        var stream = api.Stream.Writer;
        model.Input = "x";
        var sending = model.SendCommand.RunAsync();

        await stream.WriteAsync(new ChatEvent("status", State: "loading", ModelId: "model-b"), Cancel);
        await Until(() => model.State == UiState.Loading, "загрузка весов");
        Assert.Equal(["stop"], Ids(model));
        Assert.False(model.Banner!.IsProblem);
        Assert.Contains("model-b", model.Banner.Text, StringComparison.Ordinal);

        await stream.WriteAsync(new ChatEvent("text", Text: "a"), Cancel);
        await Until(() => model.State == UiState.Generating, "генерация");
        Assert.Null(model.Banner);
        Assert.True(model.StopCommand.CanExecute(null));

        await stream.WriteAsync(new ChatEvent("done", Message: Msg("assistant", "a"), Finish: "stop"), Cancel);
        stream.Complete();
        await sending;
        Assert.Equal(UiState.Ready, model.State);
    }

    [Fact]
    public async Task Stop_from_loading_banner_cancels()
    {
        var (api, model, chat) = await Open();
        var stream = api.Stream.Writer;
        model.Input = "x";
        var sending = model.SendCommand.RunAsync();
        await stream.WriteAsync(new ChatEvent("status", State: "loading", ModelId: "model-a"), Cancel);
        await Until(() => model.State == UiState.Loading, "загрузка весов");

        await ((Command)model.Banner!.Actions[0].Command).RunAsync();

        Assert.Contains($"stop {chat}", api.Calls);
        stream.Complete();
        await sending;
    }

    [Fact]
    public async Task No_models_then_refresh_finds_new_model()
    {
        var api = new FakeChatApi();
        api.Models.Clear();
        var model = new MainViewModel(api);
        await model.InitializeAsync();

        Assert.Equal(UiState.NoModel, model.State);
        Assert.Equal(["refresh"], Ids(model));
        Assert.Equal(Ru["state.no_model.text"], model.Banner!.Text);

        api.Models.Add(new ModelInfo("model-new", "qwen3", "q4", 1));
        await model.RefreshModelsCommand.RunAsync();

        Assert.Equal(UiState.Ready, model.State);
        Assert.Equal("model-new", model.SelectedModel?.Id);
        Assert.Null(model.Banner);
    }

    [Fact]
    public async Task Retry_after_failure_regenerates_and_clears_problem()
    {
        var (api, model, chat) = await Open();
        await FailWith(api, model, "engine.no_memory", ("model_id", "model-a"), ("detail", ""));
        await api.Stream.Writer.WriteAsync(new ChatEvent("done", Message: Msg("assistant", "ок"), Finish: "stop"), Cancel);
        api.Stream.Writer.Complete();

        await model.RetryCommand.RunAsync();

        Assert.Equal($"regenerate {chat}", api.Calls[^1]);
        Assert.Equal(UiState.Ready, model.State);
        Assert.Null(model.Problem);
        Assert.False(model.IsError);
    }

    [Fact]
    public async Task Choose_model_opens_picker_and_new_model_clears_problem()
    {
        var (api, model, chat) = await Open();
        await FailWith(api, model, "engine.incompatible", ("model_id", "model-a"), ("detail", "qwen9"));

        await model.ChooseModelCommand.RunAsync();
        Assert.True(model.IsModelPickerOpen);
        model.SelectedModel = model.Models[1];
        await Until(() => model.Problem is null, "причина снята");

        Assert.Equal($"model {chat} model-b", api.Calls[^1]);
        Assert.Equal(UiState.Ready, model.State);
    }

    [Fact]
    public async Task Dismiss_hides_error()
    {
        var (api, model, _) = await Open();
        await FailWith(api, model, "engine.start_failed", ("detail", "x"));

        await model.DismissCommand.RunAsync();

        Assert.Equal(UiState.Ready, model.State);
        Assert.Equal(Ru["status.ready"], model.Status);
    }

    [Fact]
    public async Task Data_folder_without_access_then_retry_starts_service_again()
    {
        var api = new FakeChatApi();
        var attempts = 0;
        var model = new MainViewModel(() =>
        {
            if (++attempts == 1)
            {
                throw new ServiceStartException("storage.no_access", new Dictionary<string, object?>
                {
                    ["path"] = _dir,
                    ["detail"] = "Permission denied",
                });
            }

            return Task.FromResult<IChatApi>(api);
        });

        await model.InitializeAsync();
        Assert.Equal(UiState.NoAccess, model.State);
        Assert.Equal(["open_folder", "retry"], Ids(model));
        Assert.Contains(_dir, model.Banner!.Text, StringComparison.Ordinal);
        Assert.False(model.IsConnected);

        await model.RetryCommand.RunAsync();
        Assert.Equal(2, attempts);
        Assert.Equal(UiState.Ready, model.State);
        Assert.True(model.IsConnected);
    }

    [Fact]
    public async Task Service_start_failure_shows_detail_and_retries()
    {
        var attempts = 0;
        var model = new MainViewModel(() =>
        {
            attempts++;
            throw new InvalidOperationException("ModuleNotFoundError: ramvi_service");
        });

        await model.InitializeAsync();

        Assert.Equal(UiState.Error, model.State);
        Assert.Contains("ModuleNotFoundError", model.Banner!.Text, StringComparison.Ordinal);
        Assert.Equal(["retry"], Ids(model)); // «Скрыть» без связи бессмысленно
        await model.RetryCommand.RunAsync();
        Assert.Equal(2, attempts);
    }

    [Fact]
    public async Task Lost_connection_retry_reconnects()
    {
        var api = new FakeChatApi();
        api.AddChat("t", "model-a", Msg("user", "вопрос"));
        var connects = 0;
        var model = new MainViewModel(() =>
        {
            connects++;
            return Task.FromResult<IChatApi>(api);
        });
        await model.InitializeAsync();
        api.Stream.Writer.Complete(new HttpRequestException("обрыв"));
        model.Input = "x";
        await model.SendCommand.RunAsync();
        Assert.Equal(UiState.Error, model.State);
        Assert.Equal(Ru["ui.connection_lost"], model.Banner!.Text);

        await model.RetryCommand.RunAsync();

        Assert.Equal(2, connects);
        Assert.Equal(UiState.Ready, model.State);
    }

    [Fact]
    public void Every_state_maps_from_codes_and_has_texts_in_both_languages()
    {
        string[] codes =
        [
            "engine.no_memory", "engine.incompatible", "engine.no_access", "engine.model_missing", "storage.no_access",
            "api.busy", "chat.busy", "chat.busy_timeout", "chat.no_model", "chat.model_not_found",
            "engine.start_failed", "ui.connection_lost", "status.service_failed",
        ];
        Assert.Equal(
            [UiState.NoModel, UiState.Error, UiState.LowMemory, UiState.Incompatible, UiState.ApiBusy, UiState.NoAccess],
            codes.Select(Problem.StateFor).Distinct().Order());
        foreach (var language in Localizer.Languages)
        {
            var catalog = Localizer.ReadCatalog(language);
            string[] states = ["no_model", "loading", "error", "low_memory", "incompatible", "api_busy", "no_access"];
            foreach (var key in codes.Concat(states.Select(s => $"state.{s}.title")))
            {
                Assert.True(catalog.ContainsKey(key), $"{language}: нет {key}");
            }
        }
    }
}
