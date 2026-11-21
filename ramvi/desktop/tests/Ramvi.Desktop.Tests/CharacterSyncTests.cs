using Avalonia.Headless;
using System.Text.Json;
using Ramvi.Desktop.Character;
using Ramvi.Desktop.Services;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class CharacterSyncTests
{
    private static string PackPath => Path.Combine(ServiceOptions.FromEnvironment().RamviRoot, "character", "ramvi_character", "assets", "pack.json");
    private static CharacterPlayback Packet(long epoch = 1, string response = "reply", long audio = 1, double amplitude = 0.15,
        string? kind = null, string? cue = null) => new(epoch, response, audio, true, "s1", "sad", "talk",
        [new(100, 140, 0, amplitude, kind, cue)]);

    [Fact]
    public void With_voice_the_face_waits_until_every_part_of_the_reply_is_spoken()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        player.Apply(new(1, "reply", null, "neutral", "idle"));
        player.SetPlayback(CharacterPlayback.Waiting(1, "reply", true));
        player.Apply(new(1, "reply", "s1", "sad", "talk"));
        player.Apply(new(1, "reply", "s2", "cheerful", "talk"));
        player.Apply(new(1, "reply", CharacterPlayer.EndOfReply, "neutral", "idle")); // текст кончился раньше голоса
        player.SetPlayback(Packet());
        player.Tick(100);
        Assert.Equal("sad", player.Emotion);                // звучит первая часть
        player.Tick(CharacterPlayer.RelaxAfterMs + 500);    // пауза перед второй частью
        Assert.Equal("sad", player.Emotion);                // вторая не прозвучала — лицо держится
        player.SetPlayback(new(1, "reply", 2, true, "s2", "cheerful", "talk", [new(100, 140, 0, 0.15, null, null)]));
        player.Tick(100);
        Assert.Equal("cheerful", player.Emotion);
        player.Tick(CharacterPlayer.RelaxAfterMs + 200);    // всё прозвучало — обычное лицо
        Assert.Equal("neutral", player.Emotion);
    }

    [Fact]
    public void Delayed_speech_keeps_idle_alive_and_changes_emotion_only_at_device_time()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        player.Apply(new(1, "reply", null, "neutral", "idle"));
        player.SetPlayback(CharacterPlayback.Waiting(1, "reply", true));
        player.Apply(new(1, "reply", "s1", "sad", "talk"));
        var before = player.Layers().Single().Y;
        player.Tick(115);
        Assert.NotEqual(before, player.Layers().Single().Y);
        Assert.Equal("neutral", player.Emotion); Assert.Equal("idle", player.Animation);
        player.Tick(45_000);
        Assert.Equal("idle", player.Animation);
        Assert.Single(player.Layers());
        player.SetPlayback(Packet());
        player.Tick(99);
        Assert.Equal("neutral", player.Emotion);
        player.Tick(1);
        Assert.Equal("sad", player.Emotion);
        Assert.Contains(player.Layers(), layer => layer.File == "parts/mouths/talk_a.png");
        player.Tick(40);
        Assert.Equal("idle", player.Animation);
        Assert.Single(player.Layers());
    }

    [Theory]
    [InlineData(0, null, null, "neutral")]
    [InlineData(0.03, null, null, "talk_e")]
    [InlineData(0.15, null, null, "talk_a")]
    [InlineData(0.4, null, null, "open")]
    [InlineData(0.4, "characters", "о", "o")]
    [InlineData(0.4, "phonemes", "aa", "talk_a")]
    [InlineData(0.4, "visemes", "talk_e", "talk_e")]
    [InlineData(0, "visemes", "open", "neutral")]
    [InlineData(0.4, "visemes", "../exec", "open")]
    public void Silence_amplitude_and_available_cues_choose_only_known_mouths(double amplitude, string? kind, string? cue, string expected)
    {
        Assert.Equal(expected, CharacterSyncController.Mouth(new(0, 20, 0, amplitude, kind, cue)));
    }

    [Fact]
    public void Cancellation_expiry_and_stale_audio_cannot_reopen_old_mouth()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        player.Apply(new(1, "reply", null, "neutral", "idle"));
        player.SetPlayback(Packet()); player.Tick(110);
        Assert.Equal(3, player.Layers().Count);
        player.Tick(1000); Assert.Single(player.Layers());
        player.Apply(new(2, null, null, "neutral", "idle"));
        player.SetPlayback(Packet()); player.Tick(110); Assert.Single(player.Layers());
        var controller = new CharacterSyncController();
        Assert.True(controller.Update(Packet(audio: 7)));
        Assert.False(controller.Update(Packet(audio: 6)));
        controller.Advance(110); Assert.NotNull(controller.Current());
        controller.Advance(300); Assert.Null(controller.Current());
        Assert.True(controller.Update(Packet(audio: 8) with { Stamp = 100 }));
        Assert.False(controller.Update(Packet(audio: 8) with { Stamp = 100 }));
        Assert.False(controller.Update(Packet(audio: 8) with { Stamp = 99 }));
        controller.Clear(); Assert.Null(controller.Current());
        Assert.True(controller.Update(Packet(audio: 9) with { ReceivedAt = System.Diagnostics.Stopwatch.GetTimestamp() - System.Diagnostics.Stopwatch.Frequency }));
        Assert.Null(controller.Current());
        Assert.False(controller.Update(Packet(audio: 8) with { Stamp = 100 }));
    }

    [Fact]
    public void Transport_delay_and_wrong_response_never_replay_old_samples()
    {
        using var json = JsonDocument.Parse("""
        {"audio_generation":3,"sync":{"generation":3,"response_id":"reply","segment_id":"s1","emotion":"sad","animation":"talk",
        "playback":{"clock":"portaudio_dac","windows":[{"start_ms":10,"end_ms":30,"position_s":0.02,"amplitude":0.2,"cue":null}]}}}
        """);
        Assert.Single(CharacterPlayback.Read(json.RootElement, 1, "reply", true).Windows);
        Assert.Empty(CharacterPlayback.Read(json.RootElement, 1, "reply", true, 40).Windows);
        Assert.Empty(CharacterPlayback.Read(json.RootElement, 1, "reply", true, 500).Windows);
        Assert.Empty(CharacterPlayback.Read(json.RootElement, 1, "other", true).Windows);
        Assert.Empty(CharacterPlayback.Read(json.RootElement, 1, "reply", false).Windows);
        using var invalid = JsonDocument.Parse("{\"sync\":{}}");
        Assert.Empty(CharacterPlayback.Read(invalid.RootElement, 1, "reply", true).Windows);
    }
    [Avalonia.Headless.XUnit.AvaloniaFact]
    public async Task Text_can_run_ahead_while_playback_keeps_the_spoken_segment_emotion()
    {
        var started = false;
        static JsonElement Json(string value) => JsonDocument.Parse(value).RootElement.Clone();
        var api = new FakeChatApi();
        const string status = """{"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[{"id":"friendly","name":"Friendly"}],"history":[],"job":null,"session_id":"session"}""";
        api.ModuleHandler = (_, command, body) =>
        {
            var operation = body.GetProperty("operation").GetString();
            if (command == "voice") return Task.FromResult(Json("""{"available":false}"""));
            if (command == "speech") return Task.FromResult(Json(operation switch
            {
                "catalog" => """{"available":true,"settings":{"enabled":true,"provider":"system","voice":"local","remote_text_enabled":false},"providers":[{"id":"system","name":"System","external":false,"emotion_control":"none","pricing":{"source":"OS"},"voices":[{"id":"local","name":"Local","languages":["ru"]}]}]}""",
                "status" when started => """{"state":"playing","enabled":true,"armed":true,"mode":"speech","generation":1,"response_id":"reply","session_id":"session","audio_generation":3,"sync":{"generation":3,"response_id":"reply","segment_id":"s1","emotion":"sad","animation":"talk","playback":{"clock":"portaudio_dac","windows":[{"start_ms":0,"end_ms":200,"position_s":0,"amplitude":0.2,"cue":null}]}}}""",
                "status" => """{"state":"idle","enabled":true,"armed":true,"mode":"speech","generation":0,"session_id":"session","audio_generation":2,"sync":null}""",
                _ => "{}"
            }));
            if (operation == "start") { started = true; return Task.FromResult(Json("""{"response_id":"reply","generation":1,"session_id":"session"}""")); }
            return Task.FromResult(Json(operation == "poll" ? """{"job":{"state":"completed","settled":true,"cursor":2,"generation":1,"session_id":"session","events":[{"type":"segment","response_id":"reply","id":"s1","text":"First 65539","emotion":"sad","animation":"talk"},{"type":"segment","response_id":"reply","id":"s2","text":"Next 999983","emotion":"cheerful","animation":"talk"}]}}""" : status));
        };
        using var vm = new Ramvi.Desktop.ViewModels.CharacterViewModel(() => api, new System.Collections.ObjectModel.ObservableCollection<Ramvi.Desktop.Api.ModelInfo>(api.Models));
        await vm.LoadAsync();
        vm.Draft = "Fresh request"; await vm.Send.RunAsync();
        await MainViewModelTests.Until(() => vm.Playback?.SegmentId == "s1", "device-clock segment");
        Assert.Equal("cheerful", vm.Presentation!.Emotion);
        Assert.Equal("sad", vm.Playback!.Emotion);
        Assert.Equal(("reply", CharacterPlayer.EndOfReply), (vm.ReplyEnd!.ResponseId, vm.ReplyEnd.SegmentId)); // ответ закончен
        Assert.Contains("Next 999983", vm.Transcript);
        await vm.StopAsync();
        await Task.Delay(100);
        Assert.Null(vm.Presentation!.ResponseId);
        Assert.Empty(vm.Playback!.Windows);
    }

    [Avalonia.Headless.XUnit.AvaloniaFact]
    public void Actual_render_changes_between_silence_and_voice()
    {
        using var pack = new CharacterPack(PackPath);
        var player = new CharacterPlayer(pack);
        player.Apply(new(1, "reply", null, "neutral", "idle"));
        var surface = new CharacterCanvas { Pack = pack, Player = player };
        var window = new Avalonia.Controls.Window { Content = surface, Width = 384, Height = 384 };
        window.Show();
        try
        {
            byte[] Render(double amplitude)
            {
                var packet = Packet(amplitude: amplitude) with { Windows = [new(0, 200, 0, amplitude, null, null)] };
                player.SetPlayback(packet); surface.InvalidateVisual();
                Avalonia.Threading.Dispatcher.UIThread.RunJobs();
                Avalonia.Headless.AvaloniaHeadlessPlatform.ForceRenderTimerTick();
                using var frame = window.CaptureRenderedFrame();
                using var bytes = new MemoryStream();
                frame!.Save(bytes, new Avalonia.Media.Imaging.PngBitmapEncoderOptions());
                if (Environment.GetEnvironmentVariable("RAMVI_SYNC_FRAMES") is {} directory)
                {
                    Directory.CreateDirectory(directory);
                    File.WriteAllBytes(Path.Combine(directory, amplitude == 0 ? "silence.png" : "voice.png"), bytes.ToArray());
                }
                return bytes.ToArray();
            }
            var silence = Render(0);
            var voice = Render(0.4);
            Assert.False(silence.SequenceEqual(voice));
            Assert.True(silence.SequenceEqual(Render(0)));
        }
        finally { surface.Release(); window.Close(); }
    }

    [Theory]
    [InlineData("null")]
    [InlineData("[]")]
    [InlineData("1")]
    [InlineData("{}")]
    public void Malformed_packet_closes_mouth(string text)
    {
        using var json = JsonDocument.Parse(text);
        Assert.Empty(CharacterPlayback.Read(json.RootElement, 1, "reply", true).Windows);
    }

    [Fact]
    public void Recorded_native_clock_packets_drive_real_player_layers()
    {
        var path = Path.Combine(ServiceOptions.FromEnvironment().RamviRoot, "docs", "reports", "character_checks", "sync", "mac.json");
        using var data = JsonDocument.Parse(File.ReadAllText(path));
        using var pack = new CharacterPack(PackPath);
        var opened = 0;
        foreach (var run in data.RootElement.GetProperty("runs").EnumerateArray())
        foreach (var sync in run.GetProperty("trace").EnumerateArray())
        {
            var response = sync.GetProperty("response_id").GetString()!;
            using var packet = JsonDocument.Parse(JsonSerializer.Serialize(new { audio_generation = sync.GetProperty("generation").GetInt64(), sync }));
            var playback = CharacterPlayback.Read(packet.RootElement, 1, response, true);
            Assert.NotEmpty(playback.Windows);
            var window = playback.Windows[0];
            var player = new CharacterPlayer(pack);
            player.Apply(new(1, response, null, "neutral", "idle"));
            player.SetPlayback(playback with { ReceivedAt = 0 });
            player.Tick((Math.Max(0, window.StartMs) + window.EndMs) / 2);
            Assert.Equal(run.GetProperty("emotion").GetString(), player.Emotion);
            Assert.Contains(player.Layers(), layer => layer.File == pack.Mouth(CharacterSyncController.Mouth(window)));
            if (window.Amplitude >= 0.012) opened++;
        }
        Assert.True(opened > 0, "A permanently closed mouth must fail the native trace replay");
        Assert.True(data.RootElement.GetProperty("workers_stopped").GetBoolean());
    }

}
