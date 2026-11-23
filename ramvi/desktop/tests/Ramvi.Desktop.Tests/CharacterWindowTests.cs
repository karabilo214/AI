using System.Text.Json;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>3.13/9.26/9.27: персонаж вне окна RAMVI.</summary>
public sealed class CharacterWindowTests
{
    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();
    private const string Status = """{"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[{"id":"friendly","name":"Friendly"}],"history":[{"user":"Fresh 7919","assistant":"Fresh reply 7919"}],"job":null}""";

    private static (FakeChatApi Api, List<string> Calls, Func<bool> Listening) Api()
    {
        var api = new FakeChatApi(); var calls = new List<string>(); var listening = false;
        api.Sections.Add(new SectionInfo("character", "section.character", "character", 30, "character", "character", "available", null));
        api.ModuleHandler = (_, command, body) =>
        {
            var op = body.GetProperty("operation").GetString()!; calls.Add(command + ":" + op);
            if (command == "voice" && op == "start") listening = true;
            if (command == "voice" && op == "stop") listening = false;
            return Task.FromResult(Json(command switch
            {
                "voice" when op == "capabilities" => """{"available":true}""",
                "voice" => listening ? """{"state":"listening","partial":"","reply":null}""" : """{"state":"inactive","partial":"","reply":null}""",
                "speech" when op == "status" => """{"state":"idle","enabled":false,"mode":"text","last":null}""",
                "speech" => """{"available":false}""",
                _ => Status
            }));
        };
        return (api, calls, () => listening);
    }

    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "ramvi-window-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void Settle(Window window)
    {
        window.UpdateLayout(); Dispatcher.UIThread.RunJobs(); AvaloniaHeadlessPlatform.ForceRenderTimerTick();
    }

    [AvaloniaFact]
    public async Task Switching_views_keeps_one_character_one_renderer_and_the_conversation()
    {
        Localizer.Current = Localizer.Load("ru");
        var (api, calls, _) = Api();
        var dir = TempDir();
        using var model = new MainViewModel(api, characterWindow: new CharacterWindowStore(dir));
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "character");
            await MainViewModelTests.Until(() => model.Character.Transcript.Contains("Fresh reply 7919", StringComparison.Ordinal), "history loaded");
            Settle(window);
            var embedded = window.GetVisualDescendants().OfType<CharacterView>().Single();
            await MainViewModelTests.Until(() => embedded.IsAnimationRunning, "embedded avatar draws");
            var character = model.Character; var opens = api.Calls.Count(c => c == "open character");
            var transcript = character.Transcript;
            model.Character.Draft = "Fresh draft 104729";

            foreach (var mode in new[] { "companion", "overlay" })
            {
                await model.CharacterWindow.ShowCommand.RunAsync(mode);
                Settle(window);
                var floating = window.CharacterWindowHost!.Window!;
                Assert.True(floating.IsVisible);
                Settle(floating);
                var avatar = floating.FindControl<CharacterView>("FloatingCharacter")!;
                await MainViewModelTests.Until(() => avatar.IsAnimationRunning, mode + " avatar draws");
                Assert.False(embedded.IsAnimationRunning); // одно рисующее представление
                Assert.Equal(mode == "companion", floating.FindControl<CharacterDialogueView>("FloatingDialogue")!.IsVisible);
                Assert.Equal(mode == "overlay", floating.WindowDecorations == WindowDecorations.None);
                Assert.Equal(mode != "overlay", floating.ShowActivated); // оверлей не забирает фокус
                Assert.True(window.FindControl<StackPanel>("CharacterOutside")!.IsEffectivelyVisible);
            }

            await model.CharacterWindow.ShowCommand.RunAsync("embedded");
            Settle(window);
            Assert.Null(window.CharacterWindowHost!.Window);
            await MainViewModelTests.Until(() => embedded.IsAnimationRunning, "avatar back in RAMVI");
            Assert.Same(character, model.Character);
            Assert.Equal(transcript, model.Character.Transcript);
            Assert.Equal("Fresh draft 104729", model.Character.Draft);
            Assert.Equal(opens, api.Calls.Count(c => c == "open character")); // модуль не перезапускался
            Assert.DoesNotContain("dialogue:cancel", calls); // текущий ответ не отменялся
            Assert.Equal("embedded", new CharacterWindowStore(dir).Read().Mode);
        }
        finally { window.Close(); Directory.Delete(dir, true); }
    }

    [AvaloniaFact]
    public async Task Hidden_avatar_never_hides_an_active_microphone_and_window_close_only_hides()
    {
        Localizer.Current = Localizer.Load("ru");
        var (api, _, listening) = Api();
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            await model.CharacterWindow.ShowCommand.RunAsync("overlay");
            await MainViewModelTests.Until(() => model.Character.Profiles.Count > 0, "character connected without opening its section");
            Assert.True(model.Sections.IsChatSelected);
            var floating = window.CharacterWindowHost!.Window!;
            floating.Close(); // крестик/Alt+F4 окна персонажа
            Settle(window);
            Assert.True(model.CharacterWindow.Hidden);
            Assert.Equal("overlay", model.CharacterWindow.Mode);
            Assert.Null(window.CharacterWindowHost.Window);

            await model.Character.Voice.ToggleCommand.RunAsync();
            await MainViewModelTests.Until(() => model.Character.Voice.IsActive, "microphone on");
            Assert.True(model.CharacterWindow.ListeningWhileHidden);
            Assert.True(model.Character.CanStopAll);
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "character");
            Settle(window);
            Assert.True(window.FindControl<Button>("CharacterOutsideStop")!.IsEffectivelyVisible);
            await model.Character.DisconnectCommand.RunAsync();
            await MainViewModelTests.Until(() => !model.Character.Voice.IsActive && !listening(), "Stop all stops the microphone");
            Assert.False(model.CharacterWindow.ListeningWhileHidden);

            await model.CharacterWindow.ToggleHiddenCommand.RunAsync();
            Settle(window);
            Assert.NotNull(window.CharacterWindowHost.Window);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Character_outside_keeps_talking_when_RAMVI_switches_to_chat()
    {
        // Вне окна RAMVI персонаж не замолкает, когда в главном окне открыли чат; в окне — замолкает.
        Localizer.Current = Localizer.Load("ru");
        var (api, calls, listening) = Api();
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "character");
            await MainViewModelTests.Until(() => model.Character.Transcript.Length > 0, "history loaded");
            await model.CharacterWindow.ShowCommand.RunAsync("companion");
            await model.Character.Voice.ToggleCommand.RunAsync();
            await MainViewModelTests.Until(() => model.Character.Voice.IsActive, "microphone on");
            var before = calls.Count;
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "chat");
            Settle(window);
            Assert.True(listening()); // слушает дальше
            Assert.DoesNotContain(calls.Skip(before), c => c is "speech:stop" or "dialogue:cancel" or "voice:stop");
            await model.CharacterWindow.ShowCommand.RunAsync("embedded");
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "character");
            Settle(window);
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "chat");
            await MainViewModelTests.Until(() => !listening(), "embedded character stops listening");
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Narrow_companion_window_puts_mode_buttons_under_full_width_input()
    {
        Localizer.Current = Localizer.Load("ru");
        var (api, _, _) = Api();
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            await model.CharacterWindow.ShowCommand.RunAsync("companion");
            Settle(window);
            var floating = window.CharacterWindowHost!.Window!;
            Settle(floating);
            var dialogue = floating.FindControl<CharacterDialogueView>("FloatingDialogue")!;
            var composer = dialogue.FindControl<Border>("CharacterComposer")!;
            var buttons = dialogue.FindControl<StackPanel>("CharacterModeButtons")!;
            Assert.True(dialogue.Bounds.Width < CharacterDialogueView.NarrowWidth);
            // Поле ввода — почти во всю ширину окна, кнопки «Только текст» — под ним, а не сбоку.
            Assert.True(composer.Bounds.Width > dialogue.Bounds.Width * 0.8, $"{composer.Bounds.Width} из {dialogue.Bounds.Width}");
            Assert.Equal(1, Grid.GetRow(buttons));
            Assert.True(buttons.Bounds.Top >= composer.Bounds.Bottom);
            // Микрофон — только кнопка у персонажа; в поле ввода второго нет, надписи о нём тоже.
            Assert.False(dialogue.FindControl<Button>("CharacterMicrophone")!.IsEffectivelyVisible);
            Assert.False(dialogue.FindControl<TextBlock>("CharacterVoiceStatus")!.IsEffectivelyVisible);
            Assert.True(floating.FindControl<Button>("FloatingMicrophone")!.IsEffectivelyVisible);
            // Фон отдельного окна — тот же, что у окна RAMVI (тема), а не системный белый.
            Assert.NotNull(floating.Background);
            Assert.Equal(window.Background?.ToString(), floating.Background!.ToString());
            await model.CharacterWindow.ShowCommand.RunAsync("overlay");
            Settle(floating);
            Assert.Equal(Avalonia.Media.Brushes.Transparent, floating.Background);
            await model.CharacterWindow.ShowCommand.RunAsync("companion");
            Settle(floating);
            Assert.Equal(window.Background?.ToString(), floating.Background?.ToString());
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Closing_RAMVI_window_hides_it_only_when_background_was_chosen_outside()
    {
        Localizer.Current = Localizer.Load("ru");
        var (api, _, _) = Api();
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        var closed = false; window.Closed += (_, _) => closed = true;
        try
        {
            await model.InitializeAsync();
            model.CharacterWindow.Background = true; // в окне RAMVI фон не действует
            Assert.False(model.CharacterWindow.KeepsRunningWithoutMainWindow);
            await model.CharacterWindow.ShowCommand.RunAsync("companion");
            Assert.True(model.CharacterWindow.KeepsRunningWithoutMainWindow);
            window.Position = new PixelPoint(123, 77);
            var before = window.Position;
            window.Close();
            Settle(window);
            Assert.False(closed); Assert.False(window.IsVisible);
            Assert.NotNull(window.CharacterWindowHost!.Window); // персонаж остаётся
            window.ShowFromTray(); // «Открыть RAMVI» в трее — на прежнем месте
            Assert.True(window.IsVisible); Assert.Equal(before, window.Position);
            window.Close(); Settle(window); Assert.False(window.IsVisible);
            model.CharacterWindow.Background = false;
            window.Show(); window.Close();
            Assert.True(closed);
            Assert.Null(window.CharacterWindowHost);
        }
        finally { if (!closed) window.Close(); }
    }

    [AvaloniaFact]
    public async Task Start_in_overlay_connects_character_and_draws_without_opening_its_section()
    {
        Localizer.Current = Localizer.Load("ru");
        var (api, _, _) = Api();
        var dir = TempDir();
        new CharacterWindowStore(dir).Write(new CharacterWindowState("overlay"));
        using var model = new MainViewModel(api, characterWindow: new CharacterWindowStore(dir));
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            Assert.True(model.Sections.IsChatSelected);
            var floating = window.CharacterWindowHost!.Window!;
            var avatar = floating.FindControl<CharacterView>("FloatingCharacter")!;
            await MainViewModelTests.Until(() => model.Sections.CharacterSection?.CanPresent == true, "module active");
            Settle(floating);
            await MainViewModelTests.Until(() => avatar.IsAnimationRunning, "overlay avatar draws after start");
        }
        finally { window.Close(); Directory.Delete(dir, true); }
    }

    [AvaloniaFact]
    public async Task Saved_overlay_without_Character_module_opens_no_empty_window()
    {
        Localizer.Current = Localizer.Load("ru");
        var dir = TempDir();
        new CharacterWindowStore(dir).Write(new CharacterWindowState("overlay"));
        using var model = new MainViewModel(new FakeChatApi(), characterWindow: new CharacterWindowStore(dir));
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            Settle(window);
            Assert.False(model.CharacterWindow.Available);
            Assert.Null(window.CharacterWindowHost!.Window);
        }
        finally { window.Close(); Directory.Delete(dir, true); }
    }

    [Theory]
    [InlineData(0, 0, 1920, 1040, 1.0)]   // Windows: рабочая область уже без панели задач
    [InlineData(0, 50, 3024, 1740, 2.0)]  // Retina (пиксели): строка меню сверху, Dock снизу
    [InlineData(1920, 0, 2560, 1400, 1.5)] // второй монитор правее основного
    public void Default_place_is_bottom_right_but_above_the_tray(int x, int y, int width, int height, double scale)
    {
        var area = new PixelRect(x, y, width, height);
        var size = new PixelSize((int)(300 * scale), (int)(380 * scale));
        var place = CharacterFloatingWindow.DefaultPosition(area, size, scale);
        var right = area.Right - (place.X + size.Width);
        var bottom = area.Bottom - (place.Y + size.Height);
        Assert.True(right >= 0 && right <= 48 * scale, $"right gap {right}");
        Assert.True(bottom >= 96 * scale, $"bottom gap {bottom}"); // трей и уведомления открыты
        Assert.True(place.X > area.X + area.Width / 2 && place.Y > area.Y + area.Height / 4); // правый низ, не середина
        Assert.True(area.Contains(place) && area.Contains(new PixelPoint(place.X + size.Width - 1, place.Y + size.Height - 1)));
    }

    [Theory]
    [InlineData(1700, false)] // правый край экрана 0..1920 — смотрит влево
    [InlineData(200, true)]   // левый край — смотрит вправо (как нарисован)
    [InlineData(2400, true)]  // второй монитор 1920..3840, его левая половина
    public void Character_faces_the_screen_center(int center, bool facesRight)
    {
        var area = center > 1920 ? new PixelRect(1920, 0, 1920, 1040) : new PixelRect(0, 0, 1920, 1040);
        Assert.Equal(!facesRight, CharacterFloatingWindow.FacesLeft(center, area));
    }

    [AvaloniaFact]
    public async Task Microphone_is_always_next_to_the_character_and_shows_its_state()
    {
        Localizer.Current = Localizer.Load("ru");
        var (api, _, _) = Api();
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            await model.CharacterWindow.ShowCommand.RunAsync("overlay");
            var floating = window.CharacterWindowHost!.Window!;
            Settle(floating);
            var microphone = floating.FindControl<Button>("FloatingMicrophone")!;
            var slash = floating.FindControl<Avalonia.Controls.Shapes.Path>("FloatingMicrophoneSlash")!;
            Assert.True(microphone.IsEffectivelyVisible); Assert.True(slash.IsVisible);
            Assert.Equal(1, microphone.Opacity); // не приглушается вместе с полосой кнопок
            microphone.Command!.Execute(null);
            await MainViewModelTests.Until(() => model.Character.Voice.IsActive, "microphone on from the overlay");
            Assert.True(microphone.IsEffectivelyVisible); Assert.False(slash.IsVisible);
            await model.Character.DisconnectCommand.RunAsync();
            await MainViewModelTests.Until(() => !model.Character.Voice.IsActive, "microphone off");
            Assert.True(microphone.IsEffectivelyVisible);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Mirrored_character_keeps_the_R_badge_readable()
    {
        Localizer.Current = Localizer.Load("ru");
        var (api, _, _) = Api();
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            await model.CharacterWindow.ShowCommand.RunAsync("overlay");
            var floating = window.CharacterWindowHost!.Window!;
            var avatar = floating.FindControl<CharacterView>("FloatingCharacter")!;
            await MainViewModelTests.Until(() => avatar.IsAnimationRunning, "avatar draws");
            foreach (var mirrored in new[] { false, true })
            {
                avatar.Mirrored = mirrored;
                Settle(floating);
                using var frame = floating.CaptureRenderedFrame();
                Assert.NotNull(frame);
                if (Environment.GetEnvironmentVariable("RAMVI_MIRROR_SHOT") is { } dir)
                    frame!.Save(Path.Combine(dir, mirrored ? "mirrored.png" : "normal.png"), new Avalonia.Media.Imaging.PngBitmapEncoderOptions());
            }
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public async Task Size_changes_keep_the_bottom_right_corner_and_are_saved()
    {
        Localizer.Current = Localizer.Load("ru");
        var (api, _, _) = Api();
        var dir = TempDir();
        using var model = new MainViewModel(api, characterWindow: new CharacterWindowStore(dir));
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            await model.CharacterWindow.ShowCommand.RunAsync("overlay");
            var floating = window.CharacterWindowHost!.Window!;
            Settle(floating);
            Assert.Equal(1, model.CharacterWindow.Size);
            var corner = floating.Position + new PixelVector((int)floating.Width, (int)floating.Height);
            model.CharacterWindow.Size = 1.5;
            Settle(floating);
            Assert.Equal(450, floating.Width); Assert.Equal(570, floating.Height);
            Assert.Equal(corner, floating.Position + new PixelVector((int)floating.Width, (int)floating.Height));
            model.CharacterWindow.Size = 9; // ограничено
            Assert.Equal(CharacterWindowViewModel.MaxSize, model.CharacterWindow.Size);
            model.CharacterWindow.Size = 0.8;
            await model.CharacterWindow.ShowCommand.RunAsync("companion"); // у отдельного окна свой минимум
            Settle(window);
            Assert.Equal(0.8, new CharacterWindowStore(dir).Read().Width / CharacterWindowViewModel.BaseWidth, 2);
        }
        finally { window.Close(); Directory.Delete(dir, true); }
    }

    [AvaloniaFact]
    public async Task Speaker_next_to_the_character_mutes_and_unmutes_the_voice()
    {
        Localizer.Current = Localizer.Load("ru");
        const string catalog = """{"available":true,"settings":{"enabled":true,"provider":"system","voice":"local","remote_text_enabled":false},"providers":[{"id":"system","name":"System","external":false,"emotion_control":"none","pricing":{"source":"OS"},"voices":[{"id":"local","name":"Local","languages":["ru"]}]}]}""";
        var api = new FakeChatApi(); var calls = new List<string>(); var configured = new List<bool>();
        api.Sections.Add(new SectionInfo("character", "section.character", "character", 30, "character", "character", "available", null));
        api.ModuleHandler = (_, command, body) =>
        {
            var op = body.GetProperty("operation").GetString()!; calls.Add(command + ":" + op);
            if (command == "speech" && op == "configure")
                configured.Add(body.GetProperty("arguments").GetProperty("settings").GetProperty("enabled").GetBoolean());
            return Task.FromResult(Json(command switch
            {
                "voice" => """{"available":true,"state":"inactive","partial":"","reply":null}""",
                "speech" when op == "catalog" => catalog,
                "speech" when op == "policy" => """{"settings":{"mode":"sentences","buffer":2,"reactions":true}}""",
                "speech" when op == "status" => """{"state":"idle","enabled":true,"mode":"speech","last":null}""",
                "speech" => "{}",
                _ => Status
            }));
        };
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            await model.CharacterWindow.ShowCommand.RunAsync("overlay");
            await MainViewModelTests.Until(() => model.Character.Speech.Providers.Count > 0 && model.Character.Speech.Enabled, "speech loaded");
            var floating = window.CharacterWindowHost!.Window!;
            Settle(floating);
            var speaker = floating.FindControl<Button>("FloatingSpeaker")!;
            var slash = floating.FindControl<Avalonia.Controls.Shapes.Path>("FloatingSpeakerSlash")!;
            var microphone = floating.FindControl<Button>("FloatingMicrophone")!;
            Assert.True(speaker.IsEffectivelyVisible); Assert.False(slash.IsVisible);
            Assert.NotEqual(microphone.HorizontalAlignment, speaker.HorizontalAlignment); // по разные стороны
            Assert.True(speaker.Command!.CanExecute(null));
            await model.Character.Speech.ToggleMuteCommand.RunAsync();
            Assert.False(model.Character.Speech.Enabled); Assert.True(slash.IsVisible);
            Assert.Contains("speech:stop", calls); Assert.False(configured[^1]); // звук остановлен, озвучка выключена
            await model.Character.Speech.ToggleMuteCommand.RunAsync();
            Assert.True(model.Character.Speech.Enabled); Assert.False(slash.IsVisible);
            Assert.True(configured[^1]);
            Assert.DoesNotContain("dialogue:cancel", calls); // разговор не трогается
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void Store_keeps_the_choice_and_click_through_is_never_restored()
    {
        Localizer.Current = Localizer.Load("ru");
        var dir = TempDir();
        try
        {
            var store = new CharacterWindowStore(dir);
            Assert.Equal(new CharacterWindowState(), store.Read()); // нет файла
            File.WriteAllText(store.Path, "{broken");
            Assert.Equal(new CharacterWindowState(), store.Read()); // испорченный файл
            File.WriteAllText(store.Path, """{"mode":"elsewhere","opacity":7,"x":"no"}""");
            Assert.Equal(new CharacterWindowState(), store.Read());
            var (api, _, _) = Api();
            using (var model = new MainViewModel(api, characterWindow: store))
            {
                model.CharacterWindow.Mode = "overlay";
                model.CharacterWindow.Opacity = 0.1; // ниже предела — окно не исчезает совсем
                model.CharacterWindow.Background = true;
                model.CharacterWindow.ClickThrough = true;
                Assert.Equal(WindowPlatform.SupportsClickThrough, model.CharacterWindow.ClickThrough);
                model.CharacterWindow.RememberBounds(10, 20, 300, 400);
            }
            var state = store.Read();
            Assert.Equal(new CharacterWindowState("overlay", true, 0.3, 10, 20, 300, 400), state);
            using var again = new MainViewModel(api, characterWindow: store);
            Assert.True(again.CharacterWindow.IsOverlay);
            Assert.False(again.CharacterWindow.ClickThrough); // после перезапуска окно управляемо
            again.CharacterWindow.ForgetPosition();
            Assert.Null(store.Read().X);
        }
        finally { Directory.Delete(dir, true); }
    }

    [AvaloniaFact]
    public async Task Window_lost_off_screen_after_monitor_change_comes_back()
    {
        Localizer.Current = Localizer.Load("ru");
        var (api, _, _) = Api();
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 700 }; window.Show();
        try
        {
            await model.InitializeAsync();
            model.CharacterWindow.RememberBounds(-90_000, -90_000, 300, 400);
            await model.CharacterWindow.ShowCommand.RunAsync("overlay");
            var floating = window.CharacterWindowHost!.Window!;
            Settle(floating);
            floating.KeepOnScreenForTest();
            Assert.True(floating.IsOnScreen());
            Assert.NotEqual(-90_000, model.CharacterWindow.State.X ?? 0); // запомнено уже видимое место
        }
        finally { window.Close(); }
    }
}
