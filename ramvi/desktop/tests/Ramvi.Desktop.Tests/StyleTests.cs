using System.Text.Json;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Styling;
using Avalonia.Themes.Fluent;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;

namespace Ramvi.Desktop.Tests;

/// <summary>
/// 3.16 UI-STYLE-01: ресурсы темы = палитра, контраст текста ≥ 4.5:1 в обеих темах,
/// снимки окна и замечания владельца после экзамена 3.10.
/// </summary>
public sealed class StyleTests
{
    private static readonly ThemeVariant[] Variants = [ThemeVariant.Light, ThemeVariant.Dark];

    /// <summary>Пары «текст на фоне», которые использует стиль (Styles.axaml).</summary>
    private static readonly (string Text, string Background)[] TextPairs =
    [
        ("text", "background"), ("text-muted", "background"), ("primary", "background"),
        ("on-primary", "primary"), ("on-primary", "primary-hover"), ("on-primary", "primary-pressed"),
        ("text", "info-bg"), ("text", "error-bg"), ("error", "error-bg"), ("error", "background"),
        ("success", "background"), ("warning", "background"), ("text", "surface"),
    ];

    // Неактивные элементы (disabled-text) — исключение WCAG 1.4.3, как и в contrast-report палитры.

    /// <summary>Контуры: по ним узнаётся поле ввода в неоморфизме (PALETTE.md, правило 3), ≥ 3:1.</summary>
    private static readonly (string Line, string Background)[] OutlinePairs =
    [
        ("control-border", "background"), ("focus", "background"),
    ];

    private static string RepoFile(params string[] parts)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var path = Path.Combine([dir.FullName, .. parts]);
            if (File.Exists(path))
            {
                return path;
            }
        }

        throw new FileNotFoundException(string.Join('/', parts));
    }

    private static Dictionary<string, Dictionary<string, string>> PaletteThemes()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(RepoFile("assets", "branding", "palette.json")));
        return document.RootElement.GetProperty("themes").EnumerateObject().ToDictionary(
            theme => theme.Name,
            theme => theme.Value.EnumerateObject().ToDictionary(t => t.Name, t => t.Value.GetString()!),
            StringComparer.Ordinal);
    }

    private static Color Token(string name, ThemeVariant variant)
    {
        Assert.True(Application.Current!.TryGetResource($"ramvi-{name}", variant, out var value), name);
        return Assert.IsType<SolidColorBrush>(value).Color;
    }

    private static double Contrast(Color a, Color b)
    {
        static double Channel(byte c)
        {
            var s = c / 255.0;
            return s <= 0.04045 ? s / 12.92 : Math.Pow((s + 0.055) / 1.055, 2.4);
        }

        static double Luminance(Color c) => (0.2126 * Channel(c.R)) + (0.7152 * Channel(c.G)) + (0.0722 * Channel(c.B));
        var (x, y) = (Luminance(a), Luminance(b));
        return (Math.Max(x, y) + 0.05) / (Math.Min(x, y) + 0.05);
    }

    [AvaloniaFact]
    public void Theme_resources_equal_palette_json_in_both_themes()
    {
        var themes = PaletteThemes();
        foreach (var (name, variant) in new[] { ("light", ThemeVariant.Light), ("dark", ThemeVariant.Dark) })
        {
            foreach (var (token, hex) in themes[name])
            {
                Assert.Equal(Color.Parse(hex), Token(token, variant));
            }

            Assert.True(Application.Current!.TryGetResource("ramvi-shadow-raised", variant, out var shadow));
            Assert.Equal(Color.Parse(themes[name]["shadow-dark"]), Assert.IsType<BoxShadows>(shadow)[0].Color);
        }

        // Стандартные элементы Fluent — в тех же цветах (App.axaml)
        var fluent = Application.Current!.Styles.OfType<FluentTheme>().Single();
        foreach (var (name, variant) in new[] { ("light", ThemeVariant.Light), ("dark", ThemeVariant.Dark) })
        {
            var palette = fluent.Palettes[variant];
            Assert.Equal(Color.Parse(themes[name]["primary"]), palette.Accent);
            Assert.Equal(Color.Parse(themes[name]["background"]), palette.RegionColor);
            Assert.Equal(Color.Parse(themes[name]["text"]), palette.BaseHigh);
            Assert.Equal(Color.Parse(themes[name]["text-muted"]), palette.BaseMedium);
            Assert.Equal(Color.Parse(themes[name]["error"]), palette.ErrorText);
        }
    }

    [AvaloniaFact]
    public void Text_contrast_is_at_least_4_5_in_both_themes()
    {
        foreach (var variant in Variants)
        {
            foreach (var (text, background) in TextPairs)
            {
                var ratio = Contrast(Token(text, variant), Token(background, variant));
                Assert.True(ratio >= 4.5, $"{variant}: {text} на {background} = {ratio:F2}");
            }

            foreach (var (line, background) in OutlinePairs)
            {
                var ratio = Contrast(Token(line, variant), Token(background, variant));
                Assert.True(ratio >= 3, $"{variant}: {line} на {background} = {ratio:F2}");
            }
        }
    }

    [AvaloniaFact]
    public void App_name_is_ramvi_for_mac_menu() => Assert.Equal("RAMVI", Application.Current!.Name);

    [AvaloniaFact]
    public async Task Title_bar_is_part_of_window_with_system_buttons()
    {
        var (window, _, _) = await OpenAsync(new FakeChatApi());

        // Окно расширено в область заголовка; наша полоса — роль TitleBar (перетаскивание,
        // двойной щелчок и прилипание делает система), кнопки окна остаются системными.
        Assert.True(window.ExtendClientAreaToDecorationsHint);
        var bar = window.FindControl<Border>("TitleBar")!;
        Assert.Equal(WindowDecorationsElementRole.TitleBar, Avalonia.Controls.Chrome.WindowDecorationProperties.GetElementRole(bar));
        Assert.Equal(window.WindowDecorationMargin.Top, bar.Height);
        Assert.Equal(Grid.GetRow(bar) + 1, Grid.GetRow(window.FindControl<Border>("Onboarding")!.Parent as Control ?? bar));
        window.Close();
    }

    [AvaloniaFact]
    public void Theme_setting_maps_to_variant()
    {
        Assert.Equal(ThemeVariant.Light, App.ThemeVariantOf("light"));
        Assert.Equal(ThemeVariant.Dark, App.ThemeVariantOf("dark"));
        Assert.Equal(ThemeVariant.Default, App.ThemeVariantOf("system"));
        Assert.Equal(ThemeVariant.Default, App.ThemeVariantOf("что-то"));
    }

    private static async Task<(MainWindow Window, MainViewModel Model, FakeChatApi Api)> OpenAsync(
        FakeChatApi api, Func<Task<IReadOnlyList<DroppedFile>>>? pickFiles = null, Action<string>? applyTheme = null)
    {
        Localizer.Current = Localizer.Load("ru");
        var model = new MainViewModel(api, pickFiles: pickFiles, applyTheme: applyTheme);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 760 };
        window.Show();
        await model.InitializeAsync();
        Pump();
        return (window, model, api);
    }

    private static void Pump()
    {
        for (var i = 0; i < 3; i++)
        {
            Dispatcher.UIThread.RunJobs();
            AvaloniaHeadlessPlatform.ForceRenderTimerTick();
        }
    }

    private static async Task UntilAsync(Func<bool> condition, string what)
    {
        var deadline = DateTime.UtcNow.AddSeconds(5);
        while (!condition())
        {
            Assert.True(DateTime.UtcNow < deadline, $"не дождались: {what}");
            Pump();
            await Task.Delay(5, TestContext.Current.CancellationToken);
        }
    }

    private static bool AtBottom(ScrollViewer scroll) =>
        scroll.Offset.Y + scroll.Viewport.Height >= scroll.Extent.Height - 1;

    private static string Lines(int count) => string.Concat(Enumerable.Range(1, count).Select(i => $"строка {i}\n"));

    [AvaloniaFact]
    public async Task Window_follows_streaming_answer_until_user_scrolls_up()
    {
        var api = new FakeChatApi();
        var chat = api.AddChat("Диалог", "model-a", Msg("user", "вопрос"), Msg("assistant", Lines(10)));
        var (window, model, _) = await OpenAsync(api);
        var scroll = window.FindControl<ScrollViewer>("MessagesScroll")!;
        model.Input = "длинный ответ";
        var stream = api.Stream.Writer; // до отправки: поддельный API подменяет канал при чтении
        var sending = model.SendCommand.RunAsync();
        await stream.WriteAsync(new ChatEvent("user", Message: Msg("user", "длинный ответ") with { ChatId = chat }),
            TestContext.Current.CancellationToken);
        await stream.WriteAsync(new ChatEvent("text", Text: "начало\n"), TestContext.Current.CancellationToken);
        await UntilAsync(() => model.Messages.Count == 4, "ответ появился");

        // (1) текст растёт внутри одного сообщения — окно держится у низа
        for (var i = 0; i < 6; i++)
        {
            await stream.WriteAsync(new ChatEvent("text", Text: Lines(15)), TestContext.Current.CancellationToken);
            await UntilAsync(() => model.Messages[^1].Content.Length > (i + 1) * 100, "часть ответа");
            Pump();
            Assert.True(scroll.Extent.Height > scroll.Viewport.Height, "есть что прокручивать");
            Assert.True(AtBottom(scroll), $"шаг {i}: {scroll.Offset.Y}+{scroll.Viewport.Height} < {scroll.Extent.Height}");
        }

        // Пользователь прокрутил вверх читать — новый текст его не дёргает
        scroll.Offset = new Vector(0, 0);
        Pump();
        await stream.WriteAsync(new ChatEvent("text", Text: Lines(15)), TestContext.Current.CancellationToken);
        await UntilAsync(() => model.Messages[^1].Content.Contains("строка 15\nстрока 1\n", StringComparison.Ordinal), "ещё часть");
        Pump();
        Assert.Equal(0, scroll.Offset.Y);

        stream.Complete();
        await sending;
        window.Close();
    }

    [AvaloniaFact]
    public async Task Input_is_cleared_when_service_accepts_not_after_whole_answer()
    {
        var api = new FakeChatApi();
        var chat = api.AddChat("Диалог", "model-a");
        var (window, model, _) = await OpenAsync(api);
        var input = window.FindControl<TextBox>("InputBox")!;
        input.Text = "вопрос";
        var stream = api.Stream.Writer;
        var sending = model.SendCommand.RunAsync();
        await stream.WriteAsync(new ChatEvent("user", Message: Msg("user", "вопрос") with { ChatId = chat }),
            TestContext.Current.CancellationToken);

        // (2) ответ ещё печатается, а поле уже пусто
        await UntilAsync(() => model.Messages.Count == 1, "вопрос принят");
        Assert.True(model.IsGenerating);
        Assert.Equal("", input.Text ?? "");

        // Набранное во время ответа не стирается в конце
        input.Text = "следующий вопрос";
        stream.Complete();
        await sending;
        Pump();
        Assert.Equal("следующий вопрос", input.Text);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Typing_indicator_is_visible_only_while_answer_is_printed()
    {
        var api = new FakeChatApi();
        api.AddChat("Диалог", "model-a");
        var (window, model, _) = await OpenAsync(api);
        var typing = window.FindControl<StackPanel>("TypingIndicator")!;
        Assert.False(typing.IsVisible);

        model.Input = "привет";
        var stream = api.Stream.Writer;
        var sending = model.SendCommand.RunAsync();
        await stream.WriteAsync(new ChatEvent("text", Text: "10003"), TestContext.Current.CancellationToken);
        await UntilAsync(() => model.Messages.Count == 1, "кусок ответа");

        // (3) промежуточный текст помечен: «RAMVI печатает…»
        Assert.True(typing.IsEffectivelyVisible);
        Assert.Contains("RAMVI печатает…", typing.GetVisualDescendants().OfType<TextBlock>().Select(t => t.Text));
        stream.Complete();
        await sending;
        Pump();
        Assert.False(typing.IsVisible);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Attach_button_adds_picked_files_with_limits()
    {
        var api = new FakeChatApi();
        var picked = new List<DroppedFile> { AttachmentReaderTests.File("заметка.txt", "текст") };
        var (window, model, _) = await OpenAsync(api, pickFiles: () => Task.FromResult<IReadOnlyList<DroppedFile>>(picked));

        // (4) значок-скрепка «Прикрепить файл» с подсказкой в поле ввода
        var button = window.FindControl<Button>("AttachButton")!;
        Assert.True(button.IsEffectivelyVisible);
        Assert.IsType<PathIcon>(button.Content);
        Assert.StartsWith("Прикрепить файл", ToolTip.GetTip(button) as string, StringComparison.Ordinal);
        await model.AttachFilesCommand.RunAsync();
        Pump();
        Assert.Equal("заметка.txt", Assert.Single(model.Attachments).Attachment.Name);
        Assert.Contains(window.FindControl<Border>("Composer")!.GetVisualDescendants().OfType<TextBlock>(),
            t => t.Text?.StartsWith("заметка.txt", StringComparison.Ordinal) == true);

        // Отмена выбора ничего не добавляет; лимит числа файлов — как у перетаскивания (2)
        picked.Clear();
        await model.AttachFilesCommand.RunAsync();
        Assert.Single(model.Attachments);
        picked.AddRange([AttachmentReaderTests.File("b.txt", "b"), AttachmentReaderTests.File("c.txt", "c")]);
        await model.AttachFilesCommand.RunAsync();
        Assert.Equal(2, model.Attachments.Count);
        Assert.True(model.IsError);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Without_picker_attach_button_is_disabled()
    {
        var (window, model, _) = await OpenAsync(new FakeChatApi());
        Assert.False(model.AttachFilesCommand.CanExecute(null));
        window.Close();
    }

    [AvaloniaFact]
    public async Task Icon_buttons_have_tooltips_instead_of_text()
    {
        var api = new FakeChatApi();
        api.AddChat("Диалог", "model-a", Msg("user", "вопрос"), Msg("assistant", "ответ"));
        var (window, _, _) = await OpenAsync(api);

        // (5) Удалить, Очистить, Копировать, Настройки — значки с подсказкой
        foreach (var (name, tip) in new[]
                 {
                     ("DeleteButton", "Удалить диалог"),
                     ("ClearButton", "Очистить диалог: удалить все сообщения, сам диалог останется"),
                     ("SettingsButton", "Настройки"),
                 })
        {
            var button = window.FindControl<Button>(name)!;
            Assert.Equal(tip, ToolTip.GetTip(button));
            Assert.IsType<PathIcon>(button.Content);
        }

        var copy = window.FindControl<ItemsControl>("MessageList")!.GetVisualDescendants().OfType<Button>()
            .Single(b => Equals(ToolTip.GetTip(b), "Копировать") && b.IsEffectivelyVisible);
        Assert.IsType<PathIcon>(copy.Content);
        Assert.Empty(Localizer.Current.Missing);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Chat_list_has_delete_cross_for_each_chat()
    {
        var api = new FakeChatApi();
        var first = api.AddChat("Первый", "model-a");
        var second = api.AddChat("Второй", "model-a");
        var (window, model, _) = await OpenAsync(api);

        // (6) крестик у диалога в списке слева, с подсказкой; удаляет именно этот диалог
        var crosses = window.FindControl<ListBox>("ChatList")!.GetVisualDescendants().OfType<Button>()
            .Where(b => b.Classes.Contains("row-action")).ToList();
        Assert.Equal(2, crosses.Count);
        Assert.All(crosses, c => Assert.Equal("Удалить диалог", ToolTip.GetTip(c)));
        var target = crosses.Single(c => (c.CommandParameter as ChatItemViewModel)?.Id == first);
        Assert.NotEqual(model.SelectedChat?.Id, first); // удаляется не выбранный диалог
        target.Command!.Execute(target.CommandParameter);
        await UntilAsync(() => model.Chats.Count == 1, "диалог удалён");
        Assert.Equal([$"delete {first}"], api.Calls);
        Assert.Equal(second, model.SelectedChat?.Id);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Answer_again_is_explained_and_only_on_last_answer()
    {
        var api = new FakeChatApi();
        api.AddChat("Диалог", "model-a",
            Msg("user", "раз"), Msg("assistant", "ответ 1"), Msg("user", "два"), Msg("assistant", "ответ 2"));
        var (window, model, _) = await OpenAsync(api);

        // (7) «Ещё раз» → «Ответить заново» у последнего ответа, с пояснением
        Assert.Equal("Ответить заново", Localizer.Current["chat.regenerate"]);
        var hint = Localizer.Current["chat.regenerate_hint"];
        var visible = window.FindControl<ItemsControl>("MessageList")!.GetVisualDescendants().OfType<Button>()
            .Where(b => Equals(ToolTip.GetTip(b), hint) && b.IsEffectivelyVisible).ToList();
        var button = Assert.Single(visible);
        Assert.Same(model.Messages[^1], button.DataContext);
        Assert.False(model.Messages[1].ShowRegenerate);
        Assert.Null(window.FindControl<Button>("RegenerateButton"));

        var stream = api.Stream.Writer;
        button.Command!.Execute(null);
        await UntilAsync(() => api.Calls.Count == 1, "запрос");
        Assert.StartsWith("regenerate ", api.Calls[0], StringComparison.Ordinal);
        stream.Complete();
        await UntilAsync(() => !model.IsGenerating, "конец");
        window.Close();
    }

    [AvaloniaFact]
    public async Task Theme_is_saved_and_applied_without_restart()
    {
        var applied = new List<string>();
        var api = new FakeChatApi();
        var (window, model, _) = await OpenAsync(api, applyTheme: applied.Add);
        await model.Settings.OpenAsync();
        Pump();
        Assert.Equal("system", model.Settings.Theme?.Code);
        Assert.Equal(["Как в системе", "Светлая", "Тёмная"], model.Settings.Themes.Select(t => t.Label));

        model.Settings.Theme = model.Settings.Themes.Single(t => t.Code == "dark");
        await model.Settings.SaveCommand.RunAsync();

        Assert.Equal("dark", api.SettingsUpdates.Single().Theme);
        Assert.Equal(["dark"], applied);
        Assert.False(model.Settings.RestartNeeded);
        Assert.Empty(Localizer.Current.Missing);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Wizard_can_be_opened_again_from_settings()
    {
        var api = new FakeChatApi();
        var (window, model, _) = await OpenAsync(api);
        Assert.False(model.Onboarding.IsOpen); // мастер уже пройден

        await model.Settings.OpenAsync();
        Pump();
        var button = window.FindControl<Button>("OpenOnboardingButton")!;
        Assert.True(button.IsEffectivelyVisible);
        Assert.Equal("Запустить мастер", button.Content);
        await model.Settings.OpenOnboardingCommand.RunAsync();
        Pump();

        Assert.False(model.Settings.IsOpen);
        Assert.True(window.FindControl<Border>("Onboarding")!.IsVisible);
        Assert.Contains("system", api.Calls); // проверка компьютера запущена заново
        model.Onboarding.NextCommand.Execute(null);
        Pump();
        Assert.True(window.FindControl<Button>("ImportButton")!.IsEffectivelyVisible); // и можно подключить модель
        Assert.Empty(Localizer.Current.Missing);
        window.Close();
    }

    [AvaloniaFact]
    public async Task Refresh_list_scans_models_folder()
    {
        var api = new FakeChatApi();
        api.Models.Clear(); // моделей нет: плашка «Нет модели» с «Обновить список»
        api.FolderModels.Add(new ModelInfo("qwen3-14b-q4_k_m", "qwen3", "Q4_K_M", 1));
        var (window, model, _) = await OpenAsync(api);
        Assert.Equal(UiState.NoModel, model.State);
        var refresh = window.FindControl<Border>("StateBanner")!.GetVisualDescendants().OfType<Button>()
            .Single(b => Equals(b.Content, "Обновить список"));

        refresh.Command!.Execute(refresh.CommandParameter);
        await UntilAsync(() => model.Models.Count == 1, "модель из папки");

        Assert.Equal(1, api.Scans);
        Assert.Equal("qwen3-14b-q4_k_m", model.SelectedModel?.Id);
        Assert.Equal(UiState.Ready, model.State);
        Assert.Contains("новых 1", model.Status, StringComparison.Ordinal);
        Assert.Empty(Localizer.Current.Missing);
        window.Close();
    }

    /// <summary>3.14: язык меняется без перезапуска — разметка, строка состояния, плашка, настройки.</summary>
    [AvaloniaFact]
    public async Task Language_switch_updates_open_window_without_restart()
    {
        var api = new FakeChatApi();
        api.Models.Clear(); // плашка «Нет модели» — тексты моделей представления тоже меняются
        var (window, model, _) = await OpenAsync(api);
        var send = window.FindControl<Button>("SendButton")!;
        string SendText() => send.GetVisualDescendants().OfType<TextBlock>().Single().Text!;
        Assert.Equal("Отправить", SendText());
        Assert.Equal("Нет модели", window.FindControl<TextBlock>("StateTitle")!.Text);
        GC.Collect(); // слабые подписки не теряют живое окно
        GC.WaitForPendingFinalizers();
        try
        {
            model.ApplyLanguage("en");
            Pump();

            Assert.Equal("Send", SendText());
            Assert.Equal("Settings", ToolTip.GetTip(window.FindControl<Button>("SettingsButton")!));
            Assert.Equal("No model", window.FindControl<TextBlock>("StateTitle")!.Text);
            Assert.Equal("Ready", model.Status);
            await model.Settings.OpenAsync();
            Pump();
            Assert.Equal("Same as system", model.Settings.Theme?.Label);
            Assert.Empty(Localizer.Current.Missing);
            Shoot(window, "settings-en");
        }
        finally
        {
            model.ApplyLanguage("ru");
            Pump();
            Application.Current!.RequestedThemeVariant = ThemeVariant.Default;
            window.Close();
        }

        Assert.Equal("Отправить", SendText());
    }

    /// <summary>3.17: пока модель грузится — вращающийся значок с подсказкой, а не текст.</summary>
    [AvaloniaFact]
    public async Task Loading_spinner_rotates_and_settings_has_unload_button()
    {
        var api = new FakeChatApi { LoadPending = true };
        api.AddChat("Диалог", "model-a");
        var (window, model, _) = await OpenAsync(api);
        var spinner = window.FindControl<Border>("LoadingSpinner")!;
        Assert.True(spinner.IsEffectivelyVisible);
        Assert.Equal("Загрузка модели model-a…", ToolTip.GetTip(spinner));
        Assert.False(window.FindControl<Border>("StateBanner")!.IsVisible);
        var icon = spinner.GetVisualDescendants().OfType<PathIcon>().Single();
        var rotate = Assert.IsType<RotateTransform>(icon.RenderTransform);
        var angles = new HashSet<double>();
        for (var i = 0; i < 40 && angles.Count < 3; i++)
        {
            Pump();
            angles.Add(Math.Round(rotate.Angle));
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }

        Assert.True(angles.Count >= 3, $"значок не вращается: {string.Join(",", angles)}");
        Shoot(window, "loading");

        api.Engine = new EngineStatus("model-a", null, "ready");
        await UntilAsync(() => !spinner.IsVisible, "конец загрузки");

        await model.Settings.OpenAsync();
        Pump();
        var unload = window.FindControl<Button>("UnloadModelButton")!;
        Assert.True(unload.IsEffectivelyVisible && unload.IsEffectivelyEnabled);
        Assert.StartsWith("Выгрузить модель", (string)ToolTip.GetTip(unload)!, StringComparison.Ordinal);
        Assert.Equal("Загружена: model-a", window.FindControl<TextBlock>("LoadedModelText")!.Text);
        Shoot(window, "settings-model");
        unload.Command!.Execute(null);
        await UntilAsync(() => !unload.IsEffectivelyEnabled, "кнопка выключена после выгрузки");
        Assert.Equal("Модель не загружена", window.FindControl<TextBlock>("LoadedModelText")!.Text);
        Assert.Empty(Localizer.Current.Missing);
        Application.Current!.RequestedThemeVariant = ThemeVariant.Default;
        window.Close();
    }

    [AvaloniaFact]
    public void Settings_file_theme_is_read_before_service()
    {
        var dir = Directory.CreateTempSubdirectory("ramvi-theme-");
        try
        {
            Assert.Equal("system", AppSettingsFile.Read(dir.FullName).Theme);
            File.WriteAllText(Path.Combine(dir.FullName, AppSettingsFile.FileName), """{"theme": "dark"}""");
            Assert.Equal("dark", AppSettingsFile.Read(dir.FullName).Theme);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    /// <summary>
    /// Снимки окна в обеих темах. Фон снимка = токен background темы (тема реально применена).
    /// С RAMVI_SNAPSHOT_DIR снимки сохраняются в PNG для сравнения с иллюстрациями ТЗ.
    /// </summary>
    [AvaloniaFact]
    public async Task Snapshots_in_light_and_dark_themes()
    {
        var api = new FakeChatApi();
        api.AddChat("Перевод письма", "model-a");
        api.AddChat("План на неделю", "model-a");
        api.AddChat("Как работает локальная модель", "model-b",
            Msg("user", "Привет! Объясни, как работает локальная языковая модель."),
            Msg("assistant", "Большая модель может использовать GPU и RAM: движок сам выбирает наиболее выгодный режим.",
                reasoning: "Пользователь спрашивает про устройство."));
        var (window, model, _) = await OpenAsync(api, pickFiles: () => Task.FromResult<IReadOnlyList<DroppedFile>>([]));
        model.Input = "Сколько будет 2 + 2?";
        await model.AddFilesAsync([AttachmentReaderTests.File("заметка.txt", "текст")]);

        model.Messages[1].IsThinkingOpen = true;
        Shoot(window, "main");

        // Ответ печатается, затем модель упирается в память — «печатает…» и плашка с действиями
        var stream = api.Stream.Writer;
        model.Input = "Ещё вопрос";
        var sending = model.SendCommand.RunAsync();
        await stream.WriteAsync(new ChatEvent("text", Text: "Промежуточный ответ 10003"), TestContext.Current.CancellationToken);
        await UntilAsync(() => model.Messages.Count == 3, "кусок ответа");
        Shoot(window, "typing");
        var error = Error("engine.no_memory", ("model_id", "model-b"), ("detail", ""));
        await stream.WriteAsync(
            new ChatEvent("done", Finish: "error", Error: new ApiError(error.Code, error.Parameters.ToDictionary())),
            TestContext.Current.CancellationToken);
        stream.Complete();
        await sending;
        Shoot(window, "problem");

        await model.Settings.OpenAsync();
        Shoot(window, "settings");
        await model.Settings.CloseCommand.RunAsync();

        await model.Onboarding.OpenAsync(null);
        await UntilAsync(() => !model.Onboarding.IsChecking, "проверки");
        Shoot(window, "onboarding");

        Application.Current!.RequestedThemeVariant = ThemeVariant.Default;
        window.Close();
    }

    private static void Shoot(Window window, string name)
    {
        var folder = Environment.GetEnvironmentVariable("RAMVI_SNAPSHOT_DIR");
        foreach (var variant in Variants)
        {
            Application.Current!.RequestedThemeVariant = variant;
            Pump();
            using var frame = window.CaptureRenderedFrame();
            Assert.NotNull(frame);
            if (!string.IsNullOrEmpty(folder))
            {
                Directory.CreateDirectory(folder);
                frame.Save(Path.Combine(folder, $"{name}-{variant.Key}.png"), new PngBitmapEncoderOptions());
            }

            Assert.Equal(Token("background", variant), Pixel(frame, frame.PixelSize.Width - 2, frame.PixelSize.Height - 2));
        }
    }

    private static Color Pixel(WriteableBitmap bitmap, int x, int y)
    {
        using var buffer = bitmap.Lock();
        var bytes = new byte[4];
        System.Runtime.InteropServices.Marshal.Copy(buffer.Address + (y * buffer.RowBytes) + (x * 4), bytes, 0, 4);
        return buffer.Format == Avalonia.Platform.PixelFormat.Rgba8888
            ? Color.FromRgb(bytes[0], bytes[1], bytes[2])
            : Color.FromRgb(bytes[2], bytes[1], bytes[0]);
    }
}
