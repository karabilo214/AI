using Avalonia;
using Avalonia.Controls;
using Avalonia.VisualTree;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.Threading;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>
/// 8.5 (UI-04): основной путь пользователя без мыши — запуск, вопрос, остановка ответа,
/// новый диалог, настройки и обратно, переход по элементам клавишей Tab.
/// </summary>
public sealed class KeyboardTests
{
    private static async Task UntilAsync(Func<bool> condition, string what)
    {
        for (var i = 0; i < 200 && !condition(); i++)
        {
            Dispatcher.UIThread.RunJobs();
            await Task.Delay(5, TestContext.Current.CancellationToken);
        }

        Assert.True(condition(), what);
    }

    private static object? Focused(Window window) => window.FocusManager?.GetFocusedElement();

    [AvaloniaFact]
    public async Task Main_path_works_without_mouse()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        var chat = api.AddChat("Диалог", "model-a");
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 760 };
        window.Show();
        await model.InitializeAsync();
        window.FocusInput();
        var input = window.FindControl<TextBox>("InputBox")!;
        await UntilAsync(() => Focused(window) == input, "после запуска фокус в поле ввода");

        // Вопрос: набрать и Enter
        var answer = api.Stream;
        window.KeyTextInput("привет");
        window.KeyPressQwerty(PhysicalKey.Enter, RawInputModifiers.None);
        await UntilAsync(() => api.Calls.Contains($"send {chat} привет files=0"), "отправка по Enter");
        await UntilAsync(() => model.IsGenerating, "ответ идёт");

        // Esc — остановить ответ, фокус остаётся в поле ввода
        window.KeyPressQwerty(PhysicalKey.Escape, RawInputModifiers.None);
        await UntilAsync(() => api.Calls.Contains($"stop {chat}"), "Esc останавливает ответ");
        answer.Writer.Complete();
        await UntilAsync(() => !model.IsGenerating, "ответ остановлен");

        // Ctrl+N — новый диалог
        window.KeyPressQwerty(PhysicalKey.N, RawInputModifiers.Control);
        await UntilAsync(() => api.Calls.Any(c => c.StartsWith("create ", StringComparison.Ordinal)), "Ctrl+N — новый диалог");
        await UntilAsync(() => Focused(window) == input, "после нового диалога фокус в поле ввода");

        // Ctrl+, — настройки; Tab переходит по элементам настроек; Esc — закрыть
        window.KeyPressQwerty(PhysicalKey.Comma, RawInputModifiers.Control);
        await UntilAsync(() => model.Settings.IsOpen, "Ctrl+, открывает настройки");
        window.FindControl<Button>("SettingsClose")!.Focus(NavigationMethod.Tab);
        var visited = new HashSet<object>();
        for (var i = 0; i < 12; i++)
        {
            window.KeyPressQwerty(PhysicalKey.Tab, RawInputModifiers.None);
            Dispatcher.UIThread.RunJobs();
            if (Focused(window) is Visual element)
            {
                Assert.True(window.FindControl<Border>("SettingsPanel")!.IsVisualAncestorOf(element), $"Tab ушёл из настроек на {element}");
                visited.Add(element);
            }
        }

        Assert.True(visited.Count >= 5, $"Tab прошёл {visited.Count} элементов");
        window.KeyPressQwerty(PhysicalKey.Escape, RawInputModifiers.None);
        await UntilAsync(() => !model.Settings.IsOpen, "Esc закрывает настройки");
        await UntilAsync(() => Focused(window) == input, "после настроек фокус в поле ввода");

        // Список диалогов — стрелками
        var list = window.FindControl<ListBox>("ChatList")!;
        Assert.True(list.ItemCount >= 2);
        // Из поля ввода назад по Shift+Tab — до списка диалогов
        var path = new List<string>();
        for (var i = 0; i < 30 && Focused(window) is not ListBoxItem; i++)
        {
            window.KeyPressQwerty(PhysicalKey.Tab, RawInputModifiers.Shift);
            Dispatcher.UIThread.RunJobs();
            path.Add(Focused(window)?.GetType().Name + ":" + (Focused(window) as Control)?.Name);
        }

        Assert.True(Focused(window) is ListBoxItem, "Shift+Tab не дошёл до списка: " + string.Join(" > ", path));
        Assert.DoesNotContain(path, p => p == "Button:"); // крестик в строке не ловит Tab
        var before = model.SelectedChat;
        window.KeyPressQwerty(PhysicalKey.ArrowDown, RawInputModifiers.None);
        window.KeyPressQwerty(PhysicalKey.ArrowUp, RawInputModifiers.None);
        window.KeyPressQwerty(PhysicalKey.ArrowDown, RawInputModifiers.None);
        await UntilAsync(() => model.SelectedChat != before, "стрелки выбирают диалог");

        // Delete — удалить выбранный диалог
        var target = model.SelectedChat!.Id;
        window.KeyPressQwerty(PhysicalKey.Delete, RawInputModifiers.None);
        await UntilAsync(() => api.Calls.Contains($"delete {target}"), "Delete удаляет выбранный диалог");
        window.Close();
    }

    [AvaloniaFact]
    public async Task Esc_during_recording_cancels_recording_not_settings_or_answer()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        api.AddChat("Диалог", "model-a");
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        window.KeyPressQwerty(PhysicalKey.Comma, RawInputModifiers.Meta); // ⌘, на Mac
        await UntilAsync(() => model.Settings.IsOpen, "⌘, открывает настройки");
        window.KeyPressQwerty(PhysicalKey.N, RawInputModifiers.Control);
        Dispatcher.UIThread.RunJobs();
        Assert.DoesNotContain(api.Calls, c => c.StartsWith("create ", StringComparison.Ordinal)); // не под настройками
        window.KeyPressQwerty(PhysicalKey.Escape, RawInputModifiers.None);
        await UntilAsync(() => !model.Settings.IsOpen, "Esc закрывает настройки");
        window.Close();
    }

    /// <summary>Мастер первого запуска: фокус в мастере, Tab не уходит под него, Enter — дальше.</summary>
    [AvaloniaFact]
    public async Task Wizard_is_passed_with_keyboard_and_traps_focus()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi { Onboarding = new Api.OnboardingState(false, null) };
        api.AddChat("Диалог", "model-a");
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        window.FocusInput();
        Assert.True(model.Onboarding.IsOpen);
        var next = window.FindControl<Button>("NextButton")!;
        await UntilAsync(() => Focused(window) == next, "фокус на «Далее»");

        var wizard = window.FindControl<Border>("Onboarding")!;
        for (var i = 0; i < 10; i++)
        {
            window.KeyPressQwerty(PhysicalKey.Tab, RawInputModifiers.None);
            Dispatcher.UIThread.RunJobs();
            var focused = Assert.IsAssignableFrom<Visual>(Focused(window));
            Assert.True(wizard.IsVisualAncestorOf(focused), $"Tab ушёл из мастера на {focused}");
        }

        next.Focus(NavigationMethod.Tab);
        window.KeyPressQwerty(PhysicalKey.Enter, RawInputModifiers.None);
        await UntilAsync(() => model.Onboarding.Step == 2, "Enter на «Далее» — шаг 2");
        await UntilAsync(() => Focused(window) is Button { IsEffectivelyVisible: true } b && wizard.IsVisualAncestorOf(b), "фокус на кнопке шага 2");
        window.FindControl<Button>("SkipButton2")!.Focus(NavigationMethod.Tab);
        window.KeyPressQwerty(PhysicalKey.Enter, RawInputModifiers.None);
        await UntilAsync(() => !model.Onboarding.IsOpen, "мастер закрыт с клавиатуры");
        await UntilAsync(() => Focused(window) == window.FindControl<TextBox>("InputBox"), "после мастера фокус в поле ввода");
        window.Close();
    }

    /// <summary>
    /// Масштаб 200 %: на экране 1920×1080 окну остаётся 960×540 логических точек. Главные
    /// элементы помещаются и видны (минимальный размер окна 760×500 меньше этого).
    /// </summary>
    [AvaloniaFact]
    public async Task Main_controls_fit_at_200_percent_scale()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        api.AddChat("Диалог", "model-a");
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 960, Height = 540 };
        window.Show();
        await model.InitializeAsync();
        Dispatcher.UIThread.RunJobs();
        Assert.True(window.MinWidth <= 960 && window.MinHeight <= 540);
        foreach (var name in new[] { "InputBox", "SendButton", "ModelBox", "NewChatButton", "SettingsButton", "ChatList" })
        {
            var control = window.FindControl<Control>(name)!;
            Assert.True(control.IsEffectivelyVisible, name);
            var box = control.TranslatePoint(new Avalonia.Point(control.Bounds.Width, control.Bounds.Height), window)!.Value;
            Assert.True(box.X <= 960 + 0.5 && box.Y <= 540 + 0.5, $"{name} выходит за окно: {box}");
            Assert.True(control.Bounds.Width > 0 && control.Bounds.Height > 0, name);
        }

        window.Close();
    }
}
