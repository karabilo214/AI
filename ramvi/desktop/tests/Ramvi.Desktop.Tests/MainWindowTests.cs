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
        AppBuilder.Configure<App>().UseHeadless(new AvaloniaHeadlessPlatformOptions());
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
}
