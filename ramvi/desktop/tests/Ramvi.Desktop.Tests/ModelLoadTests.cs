using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;
using static Ramvi.Desktop.Tests.MainViewModelTests;

namespace Ramvi.Desktop.Tests;

/// <summary>
/// 3.17 (UI-MODEL-LOAD-01): модель грузится при выборе, пока грузится — крутится значок,
/// «Выгрузить модель» в настройках.
/// </summary>
public sealed class ModelLoadTests
{
    private static readonly Localizer Ru = Localizer.Load("ru");

    public ModelLoadTests() => Localizer.Current = Ru;

    private static async Task<MainViewModel> Open(FakeChatApi api)
    {
        var model = new MainViewModel(api, loadPoll: TimeSpan.FromMilliseconds(10));
        await model.InitializeAsync();
        return model;
    }

    [Fact]
    public async Task Shown_model_loads_at_startup()
    {
        var api = new FakeChatApi();
        api.AddChat("t", "model-b");

        var model = await Open(api);

        Assert.Equal(["load model-b"], api.EngineCalls);
        Assert.Equal("model-b", api.Engine.ModelId);
        Assert.False(model.IsModelLoading);
        Assert.Equal(UiState.Ready, model.State);
        Assert.Empty(api.Calls); // ни одного сообщения
    }

    [Fact]
    public async Task Choosing_model_in_list_loads_it_without_a_message()
    {
        var api = new FakeChatApi();
        var chat = api.AddChat("t", "model-a");
        var model = await Open(api);

        model.SelectedModel = model.Models.Single(m => m.Id == "model-b");
        await Until(() => api.EngineCalls.Count == 2, "загрузка выбранной модели");

        Assert.Equal(["load model-a", "load model-b"], api.EngineCalls);
        Assert.Equal([$"model {chat} model-b"], api.Calls);
        await Until(() => api.Engine.ModelId == "model-b" && !model.IsModelLoading, "модель загружена");
    }

    [Fact]
    public async Task Choice_without_chat_also_loads()
    {
        var api = new FakeChatApi();
        var model = await Open(api);

        model.SelectedModel = model.Models.Single(m => m.Id == "model-b");
        await Until(() => api.EngineCalls.Contains("load model-b"), "загрузка без диалога");
        Assert.Empty(api.Calls);
    }

    [Fact]
    public async Task Spinner_instead_of_text_while_loading()
    {
        var api = new FakeChatApi { LoadPending = true };
        api.AddChat("t", "model-a");
        var model = await Open(api);
        Assert.True(model.IsModelLoading); // загрузка при запуске ещё идёт

        Assert.Equal(UiState.Loading, model.State);
        Assert.Null(model.Banner); // плашки с текстом «Загрузка модели…» больше нет
        Assert.Equal("", model.Status);
        Assert.Equal(Ru.Format("status.loading_model", ("model_id", "model-a")), model.LoadingTip);

        api.Engine = new EngineStatus("model-a", null, "ready");
        await Until(() => !model.IsModelLoading, "конец загрузки");
        Assert.Equal(UiState.Ready, model.State);
        Assert.Equal(Ru["status.ready"], model.Status);
    }

    [Fact]
    public async Task Load_failure_shows_its_problem()
    {
        var api = new FakeChatApi { LoadPending = true };
        api.AddChat("t", "model-a");
        var model = await Open(api);

        api.Engine = new EngineStatus(null, null, "stopped", new ApiError("engine.no_memory", null));
        await Until(() => model.State == UiState.LowMemory, "нехватка памяти");
        Assert.False(model.IsModelLoading);
        Assert.NotNull(model.Banner);
    }

    [Fact]
    public async Task Switching_chats_does_not_reload_model()
    {
        var api = new FakeChatApi();
        api.AddChat("старый", "model-b");
        api.AddChat("новый", "model-a");
        var model = await Open(api);

        model.SelectedChat = model.Chats[1];
        await Until(() => model.SelectedModel?.Id == "model-b", "показана модель диалога");
        model.SelectedChat = model.Chats[0];
        await Until(() => model.SelectedModel?.Id == "model-a", "снова первый диалог");

        Assert.Equal(["load model-a"], api.EngineCalls);
        Assert.Equal("model-a", api.Engine.ModelId);
    }

    [Fact]
    public async Task Model_chosen_in_wizard_loads_at_once()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(false, null) };
        var model = await Open(api);
        var wizard = model.Onboarding;

        await wizard.NextCommand.RunAsync();
        wizard.Selected = wizard.Models.Single(m => m.Id == "model-b");
        await wizard.StartCommand.RunAsync();

        Assert.Equal("load model-b", api.EngineCalls[^1]);
        Assert.Equal("model-b", api.Engine.ModelId);
    }

    [Fact]
    public async Task Settings_unload_frees_model_and_button_is_off_when_nothing_loaded()
    {
        var api = new FakeChatApi();
        api.AddChat("t", "model-a");
        var model = await Open(api);
        var settings = model.Settings;

        await model.OpenSettingsCommand.RunAsync();
        Assert.Equal(Ru.Format("settings.model_loaded", ("model_id", "model-a")), settings.LoadedModelText);
        Assert.True(settings.UnloadModelCommand.CanExecute(null));

        await settings.UnloadModelCommand.RunAsync();

        Assert.Equal("unload", api.EngineCalls[^1]);
        Assert.Null(settings.LoadedModel);
        Assert.Equal(Ru["settings.model_not_loaded"], settings.LoadedModelText);
        Assert.False(settings.UnloadModelCommand.CanExecute(null));
        Assert.Equal(Ru["settings.model_unloaded"], settings.Message);
        Assert.Equal("model-a", model.SelectedModel?.Id); // выбор в списке остаётся
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task Unload_during_loading_stops_spinner()
    {
        var api = new FakeChatApi { LoadPending = true };
        api.AddChat("t", "model-a");
        var model = await Open(api);
        Assert.True(model.IsModelLoading);

        await model.OpenSettingsCommand.RunAsync();
        Assert.Equal(Ru.Format("settings.model_loaded", ("model_id", "model-a")), model.Settings.LoadedModelText);
        await model.Settings.UnloadModelCommand.RunAsync();

        Assert.False(model.IsModelLoading);
        Assert.Equal(UiState.Ready, model.State);
    }
}
