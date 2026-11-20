using System.Text.Json;
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

    [Theory]
    [InlineData("version", "2.0.0")]
    [InlineData("version", "broken")]
    [InlineData("version", "1.0")]
    [InlineData("version", "1.0.0.0")]
    [InlineData("body", "../outside.png")]
    [InlineData("body", "missing.png")]
    public void Invalid_pack_is_rejected_before_rendering(string field, string value)
    {
        var root = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(PackPath))!;
        root[field] = value;
        var path = Path.Combine(Path.GetDirectoryName(PackPath)!, $"test-{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(path, root.ToJsonString());
            Assert.Throws<InvalidDataException>(() => new CharacterPack(path));
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(1.1)]
    public void Invalid_effect_is_rejected(double minimum)
    {
        var root = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(PackPath))!;
        root["effects"]!["pulse"]!["minimum"] = minimum;
        var path = Path.Combine(Path.GetDirectoryName(PackPath)!, $"test-{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(path, root.ToJsonString());
            Assert.Throws<InvalidDataException>(() => new CharacterPack(path));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Every_pose_and_animation_produces_existing_resources_and_turns_are_validated()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        foreach (var emotion in pack.Emotions)
            foreach (var motion in new[] { "idle", "blink", "talk", "nod", "shake" })
            {
                player.Present(emotion, motion);
                for (var frame = 0; frame < 25; frame++)
                {
                    Assert.All(player.Layers(), layer => Assert.True(File.Exists(pack.FilePath(layer.File))));
                    player.Tick(95);
                }
            }
        foreach (var turn in pack.Turns)
        {
            player.SetTurn(turn);
            Assert.All(player.Layers(), layer => Assert.True(File.Exists(pack.FilePath(layer.File))));
            if (turn != "front") Assert.Equal(pack.Turn(turn), Assert.Single(player.Layers()).File);
        }
        player.Present("neutral", "talk");
        player.Tick(double.MaxValue);
        Assert.InRange(player.Frame, 0, 7);
        player.Present("neutral", "blink");
        Assert.Equal("front", player.Turn);
        var previous = player.Turn;
        Assert.Throws<ArgumentException>(() => player.SetTurn("../other"));
        Assert.Equal(previous, player.Turn);
    }

    [Fact]
    public void Pack_swap_keeps_generation_fence_and_effects_are_bounded_and_cancelled()
    {
        using var first = new CharacterPack(PackPath);
        using var second = new CharacterPack(PackPath);
        var player = new CharacterPlayer(first);
        var directive = new CharacterDirective(3, "reply", "segment", "sad", "talk");
        Assert.True(player.Apply(directive));
        player.SetTurn("profile_left");
        player.SetEffect("pulse");
        player.Tick(600);
        Assert.All(player.Layers(), layer => Assert.Equal(0.7, layer.Opacity, 6));
        Assert.Throws<ArgumentException>(() => player.SetEffect("exec"));
        player.ReplacePack(second);
        Assert.Equal("sad", player.Emotion);
        Assert.Equal("idle", player.Animation);
        Assert.Equal("profile_left", player.Turn);
        Assert.False(player.Apply(directive));
        Assert.False(player.Apply(directive with { Epoch = 2, SegmentId = "late" }));
        Assert.True(player.Apply(new CharacterDirective(4, null, null, "neutral", "idle")));
        Assert.Equal("none", player.Effect);
        Assert.All(player.Layers(), layer => Assert.Equal(1, layer.Opacity));
    }

    [AvaloniaFact]
    public void Turns_and_effect_render_distinct_frames_and_renderer_can_be_restored()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        var surface = new CharacterCanvas { Pack = pack, Player = player };
        var window = new Window { Content = surface, Width = 384, Height = 384 };
        window.Show();
        try
        {
            var frameIndex = 0;
            string Frame()
            {
                surface.InvalidateVisual(); Dispatcher.UIThread.RunJobs();
                AvaloniaHeadlessPlatform.ForceRenderTimerTick();
                using var image = window.CaptureRenderedFrame();
                using var bytes = new MemoryStream();
                image!.Save(bytes, new PngBitmapEncoderOptions());
                if (Environment.GetEnvironmentVariable("RAMVI_CHARACTER_FRAMES") is { } directory)
                {
                    Directory.CreateDirectory(directory);
                    File.WriteAllBytes(Path.Combine(directory, $"{frameIndex++:00}.png"), bytes.ToArray());
                }
                return Convert.ToBase64String(bytes.ToArray());
            }
            var front = Frame();
            var poses = new HashSet<string>(StringComparer.Ordinal) { front };
            foreach (var turn in pack.Turns.Where(x => x != "front"))
            {
                player.SetTurn(turn);
                Assert.True(poses.Add(Frame()), "Each supported turn must visibly differ");
            }
            player.SetTurn("front"); player.SetEffect("pulse"); player.Tick(600);
            Assert.NotEqual(front, Frame());
            player.Stop();
            surface.Renderer = new ProbeRenderer();
            var blank = Frame();
            surface.Renderer = new CharacterBitmapRenderer();
            var restored = Frame();
            Assert.NotEqual(blank, restored);
            Assert.Equal(front, restored);
        }
        finally { surface.Release(); window.Close(); }
    }

    private sealed class ProbeRenderer : ICharacterRenderer
    {
        public int Frames { get; private set; }
        public int Releases { get; private set; }
        public void Draw(Avalonia.Media.DrawingContext context, CharacterPack pack, IReadOnlyList<CharacterLayer> layers, Avalonia.Rect bounds)
        {
            Assert.NotEmpty(layers); Frames++;
        }
        public void Release() => Releases++;
    }

    [AvaloniaFact]
    public async Task Section_renders_lazily_and_releases_images_and_timer_when_hidden()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi();
        api.ModuleHandler = (_, _, _) => Task.FromResult(JsonDocument.Parse("""
            {"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[{"id":"friendly","name":"Friendly"}],"history":[{"user":"Привет! Как идёт работа?","assistant":"Привет! Текстовый диалог готов к проверке."}],"job":null}
            """).RootElement.Clone());
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
            surface.Player!.Present("cheerful", "nod");
            Dispatcher.UIThread.RunJobs();
            AvaloniaHeadlessPlatform.ForceRenderTimerTick();
            if (Environment.GetEnvironmentVariable("RAMVI_CHARACTER_SCREENSHOT") is { } screenshot)
            {
                using var image = window.CaptureRenderedFrame();
                image!.Save(screenshot, new PngBitmapEncoderOptions());
            }
            var dialogueState = (model.Character.Profile, model.Character.Transcript, model.Character.Draft, model.Character.Model);
            var originalPlayer = surface.Player;
            var originalResponse = view.Response;
            var replacement = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(PackPath))!;
            replacement["version"] = "1.17.29";
            replacement["expressions"]!["cheerful"]!["file"] = "sprites/expressions/sad.png";
            var replacementPath = Path.Combine(Path.GetDirectoryName(PackPath)!, $"test-{Guid.NewGuid():N}.json");
            try
            {
                File.WriteAllText(replacementPath, replacement.ToJsonString());
                view.LoadPack(replacementPath);
                Assert.Equal("1.17.29", surface.Pack!.Version);
                surface.Player!.Present("cheerful", "idle");
                Assert.Equal("sprites/expressions/sad.png", Assert.Single(surface.Player.Layers()).File);
                replacement["version"] = "2.0.0";
                File.WriteAllText(replacementPath, replacement.ToJsonString());
                Assert.Throws<InvalidDataException>(() => view.LoadPack(replacementPath));
                Assert.Equal("1.17.29", surface.Pack.Version);
                view.LoadPack(PackPath);
            }
            finally { File.Delete(replacementPath); }
            Assert.Same(originalPlayer, surface.Player);
            Assert.Equal(originalResponse, view.Response);
            Assert.Equal("cheerful", surface.Player!.Emotion);
            var probe = new ProbeRenderer();
            surface.Renderer = probe;
            Dispatcher.UIThread.RunJobs();
            AvaloniaHeadlessPlatform.ForceRenderTimerTick();
            Assert.True(probe.Frames > 0);
            Assert.Same(originalPlayer, surface.Player);
            surface.Renderer = new CharacterBitmapRenderer();
            Assert.Equal(1, probe.Releases);
            Assert.Equal(dialogueState, (model.Character.Profile, model.Character.Transcript, model.Character.Draft, model.Character.Model));
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "chat");
            await MainViewModelTests.Until(() => surface.Player is null && !view.IsAnimationRunning, "character released");
            Assert.Equal("Черновик чата", model.Input);
        }
        finally { window.Close(); }
    }
}
