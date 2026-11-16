using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;

namespace Ramvi.Desktop.Tests;

/// <summary>UI-01: мастер первого запуска — проверка, импорт, выбор, пропуск.</summary>
public sealed class OnboardingTests
{
    private const string FilePath = "/models/Qwen3-8B-Q4_K_M.gguf";
    private static readonly Localizer Ru = Localizer.Load("ru");
    private static readonly ModelInfo Imported = new("qwen3-8b-q4_k_m", "qwen3", "Q4_K_M", 5_000_000_000);

    public OnboardingTests() => Localizer.Current = Ru;

    private static async Task<MainViewModel> Open(FakeChatApi api, string? picked = FilePath)
    {
        var model = new MainViewModel(api, pickModelFile: () => Task.FromResult(picked));
        await model.InitializeAsync();
        return model;
    }

    [Fact]
    public async Task First_run_opens_wizard_with_localized_checks()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(false, null) };

        var model = await Open(api);
        var wizard = model.Onboarding;

        Assert.True(wizard.IsOpen);
        Assert.True(wizard.IsCheckStep);
        Assert.False(wizard.IsChecking);
        Assert.Equal(["Система: Windows 11 (AMD64)", Ru["system.gpu_none"]], wizard.Checks.Select(c => c.Text));
        Assert.Equal(["✓", "✗"], wizard.Checks.Select(c => c.Mark));
        Assert.True(wizard.Checks[1].IsFail);
        Assert.Contains("system", api.Calls);
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task Finished_wizard_is_not_shown_again()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(true, "model-b") };

        var model = await Open(api);

        Assert.False(model.Onboarding.IsOpen);
        Assert.DoesNotContain("system", api.Calls);
        Assert.Equal("model-b", model.SelectedModel?.Id); // сохранённый выбор для нового диалога
    }

    [Fact]
    public async Task Choose_installed_model_and_start()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(false, "model-b") };
        var model = await Open(api);
        var wizard = model.Onboarding;

        await wizard.NextCommand.RunAsync();
        Assert.True(wizard.IsModelStep);
        Assert.Equal("model-b", wizard.Selected?.Id);
        wizard.Selected = wizard.Models[0];
        await wizard.StartCommand.RunAsync();

        Assert.False(wizard.IsOpen);
        Assert.Equal("complete model-a", api.Calls[^1]);
        Assert.Equal("model-a", model.SelectedModel?.Id);
        Assert.Equal(UiState.Ready, model.State);
    }

    [Fact]
    public async Task No_models_import_file_then_start()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(false, null) };
        api.Models.Clear();
        api.Imports[FilePath] = Imported;
        var model = await Open(api);
        var wizard = model.Onboarding;
        await wizard.NextCommand.RunAsync();
        Assert.False(wizard.HasModels);
        Assert.False(wizard.StartCommand.CanExecute(null));
        Assert.Equal(UiState.NoModel, model.State);

        await wizard.ImportCommand.RunAsync();

        Assert.True(wizard.HasModels);
        Assert.Equal(Imported, wizard.Selected);
        Assert.Contains(Imported, model.Models); // тот же список, что у выбора модели в чате
        await wizard.StartCommand.RunAsync();
        Assert.Equal([$"system", $"import {FilePath}", "complete qwen3-8b-q4_k_m"], api.Calls);
        Assert.Equal(Imported.Id, model.SelectedModel?.Id);
        Assert.Equal(UiState.Ready, model.State);
    }

    [Fact]
    public async Task Same_file_twice_is_not_listed_twice()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(false, null) };
        api.Imports[FilePath] = Imported;
        var model = await Open(api);

        await model.Onboarding.ImportCommand.RunAsync();
        await model.Onboarding.ImportCommand.RunAsync();

        Assert.Single(model.Models, m => m.Id == Imported.Id);
    }

    [Fact]
    public async Task Cancelled_file_dialog_does_nothing()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(false, null) };
        var model = await Open(api, picked: null);

        await model.Onboarding.ImportCommand.RunAsync();

        Assert.DoesNotContain(api.Calls, c => c.StartsWith("import", StringComparison.Ordinal));
        Assert.False(model.Onboarding.HasError);
    }

    [Fact]
    public async Task Import_error_is_shown_and_list_unchanged()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(false, null) };
        api.Imports[FilePath] = Error("import.unsupported_architecture",
            ("path", FilePath), ("architecture", "llama"), ("supported", "qwen3"));
        var model = await Open(api);

        await model.Onboarding.ImportCommand.RunAsync();

        Assert.Equal(
            Ru.Format("import.unsupported_architecture", ("architecture", "llama"), ("supported", "qwen3")),
            model.Onboarding.Error);
        Assert.Equal(2, model.Models.Count);
        Assert.True(model.Onboarding.IsOpen);
    }

    [Fact]
    public async Task Skip_closes_wizard_without_model()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(false, null) };
        api.Models.Clear();
        var model = await Open(api);

        await model.Onboarding.SkipCommand.RunAsync();

        Assert.False(model.Onboarding.IsOpen);
        Assert.Equal("complete -", api.Calls[^1]);
        Assert.Equal(UiState.NoModel, model.State); // дальше — плашка «Нет модели»
    }

    [Fact]
    public async Task Back_returns_to_checks()
    {
        var api = new FakeChatApi { Onboarding = new OnboardingState(false, null) };
        var model = await Open(api);
        var wizard = model.Onboarding;

        await wizard.NextCommand.RunAsync();
        Assert.False(wizard.NextCommand.CanExecute(null));
        await wizard.BackCommand.RunAsync();

        Assert.True(wizard.IsCheckStep);
        Assert.False(wizard.BackCommand.CanExecute(null));
    }
}
