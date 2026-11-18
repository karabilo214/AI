using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;

[assembly: AvaloniaTestApplication(typeof(Ramvi.Desktop.Tests.TestApp))]

namespace Ramvi.Desktop.Tests;

public static class TestApp
{
    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UseSkia()
            .WithInterFont()
            .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false }); // снимки окна (3.16)
}

/// <summary>Окно без экрана: разметка грузится, привязки работают, Enter отправляет.</summary>
public sealed class MainWindowTests
{
    [AvaloniaFact]
    public async Task Window_shows_chats_messages_and_sends_on_enter()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        var chat = api.AddChat("Диалог", "model-b", Msg("user", "вопрос"), Msg("assistant", "ответ", reasoning: "мысль"));
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        Dispatcher.UIThread.RunJobs();

        var chats = window.FindControl<ListBox>("ChatList")!;
        Assert.Equal(model.SelectedChat, chats.SelectedItem);
        Assert.Equal("model-b", (window.FindControl<ComboBox>("ModelBox")!.SelectedItem as Api.ModelInfo)?.Id);
        var texts = window.GetVisualDescendants().OfType<TextBlock>().Select(t => t.Text).ToList();
        Assert.Contains("Диалог", texts);
        Assert.Contains("ответ", texts);
        Assert.DoesNotContain("мысль", window.GetVisualDescendants().OfType<TextBlock>()
            .Where(t => t.IsEffectivelyVisible).Select(t => t.Text)); // thinking свёрнут
        Assert.Empty(Localizer.Current.Missing);

        var input = window.FindControl<TextBox>("InputBox")!;
        input.Focus();
        input.Text = "ещё";
        api.Stream.Writer.Complete();
        window.KeyPressQwerty(PhysicalKey.Enter, RawInputModifiers.None);
        for (var i = 0; i < 100 && api.Calls.Count == 0; i++)
        {
            Dispatcher.UIThread.RunJobs();
            await Task.Delay(5, TestContext.Current.CancellationToken);
        }

        Assert.Equal([$"send {chat} ещё files=0"], api.Calls);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Problem_banner_shows_title_text_and_action_buttons()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        api.AddChat("Диалог", "model-a", Msg("user", "вопрос"));
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        Assert.False(window.FindControl<Border>("StateBanner")!.IsVisible);

        var error = Error("engine.no_memory", ("model_id", "model-a"), ("detail", ""));
        await api.Stream.Writer.WriteAsync(
            new Api.ChatEvent("done", Finish: "error", Error: new Api.ApiError(error.Code, error.Parameters.ToDictionary())),
            TestContext.Current.CancellationToken);
        api.Stream.Writer.Complete();
        model.Input = "x";
        await model.SendCommand.RunAsync();
        Dispatcher.UIThread.RunJobs();

        var banner = window.FindControl<Border>("StateBanner")!;
        Assert.True(banner.IsVisible);
        Assert.Contains("problem", banner.Classes);
        Assert.Equal("Недостаточно памяти", window.FindControl<TextBlock>("StateTitle")!.Text);
        var buttons = banner.GetVisualDescendants().OfType<Button>().Select(b => b.Content as string).ToList();
        Assert.Equal(["Выбрать другую модель", "Повторить"], buttons);
        Assert.Empty(Localizer.Current.Missing);

        await model.ChooseModelCommand.RunAsync();
        Dispatcher.UIThread.RunJobs();
        Assert.True(window.FindControl<ComboBox>("ModelBox")!.IsDropDownOpen);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Onboarding_overlay_shows_checks_then_hides_after_start()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi { Onboarding = new Api.OnboardingState(false, null) };
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        Dispatcher.UIThread.RunJobs();

        var overlay = window.FindControl<Border>("Onboarding")!;
        Assert.True(overlay.IsVisible);
        Assert.True(window.FindControl<StackPanel>("CheckStep")!.IsVisible);
        Assert.False(window.FindControl<StackPanel>("ModelStep")!.IsVisible);
        var texts = overlay.GetVisualDescendants().OfType<TextBlock>().Select(t => t.Text).ToList();
        Assert.Contains("Добро пожаловать в RAMVI", texts);
        Assert.Contains("Система: Windows 11 (AMD64)", texts);

        window.FindControl<Button>("NextButton")!.Command!.Execute(null);
        Dispatcher.UIThread.RunJobs();
        Assert.True(window.FindControl<StackPanel>("ModelStep")!.IsVisible);
        Assert.Equal(2, window.FindControl<ListBox>("WizardModels")!.ItemCount);

        await model.Onboarding.StartCommand.RunAsync();
        Dispatcher.UIThread.RunJobs();
        Assert.False(overlay.IsVisible);
        Assert.Empty(Localizer.Current.Missing);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Settings_panel_shows_fields_and_network_warning()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        Assert.False(window.FindControl<Border>("SettingsPanel")!.IsVisible);

        window.FindControl<Button>("SettingsButton")!.Command!.Execute(null);
        await Task.Delay(50, TestContext.Current.CancellationToken);
        Dispatcher.UIThread.RunJobs();

        var panel = window.FindControl<Border>("SettingsPanel")!;
        Assert.True(panel.IsVisible);
        Assert.Equal("data", window.FindControl<TextBox>("DataDirBox")!.Text);
        Assert.Equal("11435", window.FindControl<TextBox>("ApiPortBox")!.Text);
        Assert.Equal("Русский", (window.FindControl<ComboBox>("LanguageBox")!.SelectedItem as LanguageOption)?.Label);
        var visible = panel.GetVisualDescendants().OfType<TextBlock>().Where(t => t.IsEffectivelyVisible).Select(t => t.Text).ToList();
        Assert.Contains(Localizer.Current["settings.api_note"], visible);
        Assert.DoesNotContain(Localizer.Current["settings.network_warning"], visible);

        window.FindControl<CheckBox>("NetworkBox")!.IsChecked = true;
        Dispatcher.UIThread.RunJobs();
        visible = panel.GetVisualDescendants().OfType<TextBlock>().Where(t => t.IsEffectivelyVisible).Select(t => t.Text).ToList();
        Assert.Contains(Localizer.Current["settings.network_warning"], visible);
        Assert.False(window.FindControl<Button>("RestartButton")!.IsVisible);
        Assert.Empty(Localizer.Current.Missing);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Section_rail_switches_between_chat_and_module()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        api.Sections.Add(new Api.SectionInfo("agent", "section.agent", "agent", 20, "agent", "agent", "available", null));
        api.AddChat("Диалог", "model-a", Msg("user", "вопрос"), Msg("assistant", "ответ"));
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        Dispatcher.UIThread.RunJobs();

        Assert.True(window.FindControl<Border>("SectionRail")!.IsVisible);
        Assert.True(window.FindControl<DockPanel>("ChatPanel")!.IsVisible);
        Assert.False(window.FindControl<Border>("ModuleSection")!.IsVisible);

        window.FindControl<ListBox>("SectionList")!.SelectedIndex = 1;
        await Task.Delay(50, TestContext.Current.CancellationToken);
        Dispatcher.UIThread.RunJobs();
        Assert.False(window.FindControl<DockPanel>("ChatPanel")!.IsVisible);
        Assert.False(window.FindControl<Border>("ModuleSection")!.IsVisible);
        Assert.True(window.FindControl<Border>("AgentPanel")!.IsVisible);

        window.FindControl<ListBox>("SectionList")!.SelectedIndex = 0;
        Dispatcher.UIThread.RunJobs();
        Assert.True(window.FindControl<DockPanel>("ChatPanel")!.IsVisible);
        var texts = window.GetVisualDescendants().OfType<TextBlock>().Select(t => t.Text).ToList();
        Assert.Contains("ответ", texts);
        Assert.Empty(Localizer.Current.Missing);
        window.Close();
    }
}
