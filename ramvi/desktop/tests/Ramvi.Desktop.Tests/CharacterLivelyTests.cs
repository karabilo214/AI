using System.Text.Json;
using Avalonia.Headless.XUnit;
using Ramvi.Desktop.Character;
using Ramvi.Desktop.Services;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>Живой персонаж: лицо и тело выбирает модель — глаза с бровями, рот, жесты, раскадровка движений.</summary>
public sealed class CharacterLivelyTests
{
    private static string PackPath => Path.Combine(ServiceOptions.FromEnvironment().RamviRoot, "character", "ramvi_character", "assets", "pack.json");

    [Fact]
    public void Model_chosen_eyes_and_mouth_build_the_face_from_parts()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        Assert.True(player.Apply(new CharacterDirective(1, "r", "s1", "furious", "none", "furious", "wide")));
        var files = player.Layers().Select(l => l.File).ToArray();
        Assert.Equal(["parts/body/visor_without_features.png", "parts/eyes/furious.png", "parts/mouths/wide.png"], files);
        // Неизвестные части набора не ломают кадр — берётся лицо эмоции.
        Assert.True(player.Apply(new CharacterDirective(1, "r", "s2", "furious", "none", "../x", "exec")));
        Assert.Equal(["sprites/expressions/furious.png"], player.Layers().Select(l => l.File));
    }

    [Fact]
    public void Jump_goes_up_and_lands_back_to_idle()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        player.Present("cheerful", "jump");
        player.Tick(225); // четверть прыжка — в воздухе
        Assert.True(player.Layers()[0].Y < -60, $"Y={player.Layers()[0].Y}");
        player.Tick(700);
        Assert.Equal("idle", player.Animation);
        Assert.InRange(player.Layers()[0].Y, -5, 5);
    }

    [Fact]
    public void Turn_away_shows_the_profile_and_comes_back()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        player.Present("angry", "turn_away");
        player.Tick(100);
        Assert.Equal(["sprites/turns/three_quarter_left.png"], player.Layers().Select(l => l.File));
        player.Tick(1100); // середина — отвернулся в профиль
        Assert.Equal(["sprites/turns/profile_left.png"], player.Layers().Select(l => l.File));
        player.Tick(1300);
        Assert.Equal("idle", player.Animation);
        Assert.Equal(["sprites/expressions/angry.png"], player.Layers().Select(l => l.File));
    }

    [Fact]
    public void Shout_lunges_forward_with_an_open_mouth_and_laugh_uses_happy_eyes()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        player.Present("furious", "shout");
        player.Tick(400);
        var layers = player.Layers();
        Assert.True(layers[0].Scale > 1.12, $"scale={layers[0].Scale}");
        Assert.Contains(layers, l => l.File == "parts/mouths/wide.png");
        player.Present("cheerful", "laugh");
        player.Tick(100);
        Assert.Contains(player.Layers(), l => l.File == "parts/eyes/happy.png");
        Assert.True(player.Layers()[0].Y < 0);
    }

    [Fact]
    public void Model_timeline_moves_the_body_switches_eyes_and_starts_a_gesture()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        CharacterMove[] moves =
        [
            new(null, 0.5, null, 1.15, null, null, null, 500),    // подвинулся ближе к таблице
            new(null, null, null, null, null, "half", null, 700),  // присматривается
            new(null, null, null, null, 0.85, "surprised", "o", 300), // присел, удивился
            new("jump", null, null, null, 1.0, null, null, 900),   // подпрыгнул
        ];
        Assert.True(player.Apply(new CharacterDirective(2, "r", "s1", "neutral", "none", Moves: moves)));
        player.Tick(500);
        var near = player.Layers()[0];
        Assert.Equal(75, near.X, 1);         // x 0.5 → 75 точек вправо
        Assert.Equal(1.15, near.Scale, 2);   // ближе
        player.Tick(400);
        Assert.Contains(player.Layers(), l => l.File == "parts/eyes/half.png");
        player.Tick(600);                    // третий шаг: присел и удивился
        Assert.Contains(player.Layers(), l => l.File == "parts/eyes/surprised.png");
        Assert.True(player.Layers()[0].ScaleY < 0.9);
        player.Tick(50);                     // четвёртый шаг начал прыжок
        Assert.Equal("jump", player.Animation);
        player.Tick(3000);                   // раскадровка кончилась — покой
        var rest = player.Layers()[0];
        Assert.InRange(rest.X, -0.01, 0.01);
        Assert.Equal(1, rest.Scale, 3);
        Assert.Equal("idle", player.Animation);
    }

    [Fact]
    public void Sync_and_segments_read_the_pack_face_eyes_mouth_and_moves()
    {
        const string json = """
        {"audio_generation":4,"sync":{"generation":4,"response_id":"r","segment_id":"s1","emotion":"anger","expression":"furious",
         "eyes":"furious","mouth":"frown","animation":"shout",
         "moves":[{"x":3,"ms":200},{"eyes":"half","ms":10},{"scale":1.2,"ms":400,"do":"jump"},"bad"],
         "playback":{"clock":"portaudio_dac","stamp":0,"windows":[{"start_ms":0,"end_ms":100,"position_s":0,"amplitude":0.5}]}}}
        """;
        var playback = CharacterPlayback.Read(JsonDocument.Parse(json).RootElement, 1, "r", true);
        Assert.Equal("furious", playback.Emotion);  // лицо набора, а не эмоция модели
        Assert.Equal(("furious", "frown"), (playback.Eyes, playback.Mouth));
        Assert.Equal(2, playback.Moves.Count);       // шаг с ms=10 и «bad» отброшены
        Assert.Equal(1, playback.Moves[0].X);        // x=3 ограничен до 1
        Assert.Equal("jump", playback.Moves[1].Do);
    }

    [Fact]
    public void Face_changes_by_parts_during_the_reply_and_relaxes_only_after_it()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        Assert.True(player.Apply(new CharacterDirective(1, "r", "s1", "furious", "none", "furious", null)));
        player.Tick(CharacterPlayer.RelaxAfterMs + 500);  // пауза внутри ответа — лицо держится
        Assert.Equal("furious", player.Emotion);
        Assert.True(player.Apply(new CharacterDirective(1, "r", "s2", "surprise", "none")));
        Assert.Equal("surprise", player.Emotion);          // вторая часть — другое лицо
        Assert.Null(player.Eyes);
        Assert.True(player.Apply(new CharacterDirective(1, "r", CharacterPlayer.EndOfReply, "neutral", "idle")));
        player.Tick(CharacterPlayer.RelaxAfterMs - 100);
        Assert.Equal("surprise", player.Emotion);          // ответ кончился — ещё немного держит
        player.Tick(200);
        Assert.Equal("neutral", player.Emotion);
        Assert.Equal(["sprites/expressions/neutral.png"], player.Layers().Select(l => l.File));
        player.Present("angry", "shake");                  // жест идёт — не сбрасывает
        for (var i = 0; i < 4; i++) player.Tick(140);
        Assert.Equal("angry", player.Emotion);
    }
}
