using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Character;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class CharacterTests
{
    private static string PackPath => Path.Combine(ServiceOptions.FromEnvironment().RamviRoot, "character", "ramvi_character", "assets", "pack.json");
    [Fact]
    public void Gestures_have_fixed_speeds_and_idle_never_uses_baked_frames()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        foreach (var (motion, duration) in new[] { ("nod", 517.5), ("shake", 575.0), ("blink", 600.0) })
        {
            player.Present("sad", motion);
            Assert.Equal(motion == "blink" ? 1 : 2, player.Speed);
            player.Tick(duration - 1); Assert.Equal(motion, player.Animation);
            player.Tick(1); Assert.Equal("idle", player.Animation);
            Assert.Equal("sad", player.Emotion); Assert.Equal(1, player.Speed);
        }
        for (var i = 0; i < 24; i++)
        {
            Assert.All(player.Layers(), layer => Assert.Equal("sprites/expressions/sad.png", layer.File));
            player.Tick(115);
        }
        player.Present("sad", "talk"); player.Tick(95 * 3);
        player.Present("cheerful", "talk"); Assert.Equal(3, player.Frame);
        Assert.Contains(player.Layers(), x => x.File == "parts/eyes/happy.png");
        Assert.Throws<ArgumentException>(() => player.Present("exec", "talk"));
        Assert.Equal("cheerful", player.Emotion);
        player.Stop(); Assert.Equal("idle", player.Animation);
        Assert.Throws<InvalidDataException>(() => pack.FilePath("../outside.png"));
    }

    [AvaloniaFact]
    public async Task Section_renders_lazily_and_releases_images_and_timer_when_hidden()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        api.Sections.Add(new SectionInfo("character", "section.character", "character", 30, "character", "character", "available", null));
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        try
        {
            await model.InitializeAsync();
            var view = window.GetVisualDescendants().OfType<CharacterView>().Single();
            var surface = view.FindControl<CharacterCanvas>("CharacterSurface")!;
            Assert.Null(surface.Player); Assert.False(view.IsAnimationRunning);
            model.Input = "Черновик чата";
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "character");
            await MainViewModelTests.Until(() => surface.Player is not null && view.IsAnimationRunning, "character rendering");
            Assert.True(model.Sections.IsCharacterSelected);
            Assert.False(model.Sections.IsPlaceholderSelected);
            Assert.Equal("", view.FindControl<TextBlock>("CharacterError")!.Text);
            var emotions = view.FindControl<ComboBox>("CharacterEmotion")!;
            emotions.SelectedIndex = emotions.Items.ToList().FindIndex(item => item?.ToString() == Localizer.Current["character.emotion.cheerful"]);
            surface.Player!.Present("cheerful", "nod");
            Dispatcher.UIThread.RunJobs();
            AvaloniaHeadlessPlatform.ForceRenderTimerTick();
            if (Environment.GetEnvironmentVariable("RAMVI_CHARACTER_SCREENSHOT") is { } screenshot)
            {
                using var image = window.CaptureRenderedFrame();
                image!.Save(screenshot, new PngBitmapEncoderOptions());
            }
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "chat");
            await MainViewModelTests.Until(() => surface.Player is null && !view.IsAnimationRunning, "character released");
            Assert.Equal("Черновик чата", model.Input);
        }
        finally { window.Close(); }
    }
}
