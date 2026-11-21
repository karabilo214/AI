using System.Text.Json;

namespace Ramvi.Desktop.Character;

public sealed record PlaybackWindow(double StartMs, double EndMs, double Position, double Amplitude, string? CueKind, string? Cue);

/// <summary>Short device-clock look-ahead, never a timer started by LLM completion.</summary>
public sealed record CharacterPlayback(long Epoch, string? ResponseId, long AudioGeneration, bool Enabled,
    string? SegmentId, string Emotion, string Animation, IReadOnlyList<PlaybackWindow> Windows, double Stamp = 0)
{
    public long ReceivedAt { get; init; }
    /// <summary>Глаза с бровями и рот, выбранные моделью для сегмента (части набора).</summary>
    public string? Eyes { get; init; }
    public string? Mouth { get; init; }
    /// <summary>Раскадровка движений сегмента от модели.</summary>
    public IReadOnlyList<CharacterMove> Moves { get; init; } = [];
    public static CharacterPlayback Waiting(long epoch, string? response, bool enabled) => new(epoch, response, -1, enabled, null, "neutral", "idle", []);
    public static CharacterPlayback Read(JsonElement packet, long epoch, string? response, bool enabled, double transportMs = 0)
    {
        var waiting = Waiting(epoch, response, enabled);
        if (packet.ValueKind != JsonValueKind.Object) return waiting;
        if (packet.ValueKind == JsonValueKind.Object && packet.TryGetProperty("audio_generation", out var fence) && fence.ValueKind == JsonValueKind.Number && fence.TryGetInt64(out var audioFence))
            waiting = waiting with { AudioGeneration = audioFence };
        if (!double.IsFinite(transportMs) || transportMs < 0 || transportMs > 300) return waiting;
        if (!enabled || response is null || !packet.TryGetProperty("sync", out var sync) || sync.ValueKind != JsonValueKind.Object)
            return waiting;
        try
        {
            var generation = packet.GetProperty("audio_generation").GetInt64();
            if (sync.GetProperty("generation").GetInt64() != generation || sync.GetProperty("response_id").GetString() != response)
                return waiting;
            var playback = sync.GetProperty("playback");
            if (playback.GetProperty("clock").GetString() != "portaudio_dac") return waiting;
            var stamp = playback.TryGetProperty("stamp", out var timestamp) ? timestamp.GetDouble() : 0;
            if (!double.IsFinite(stamp) || stamp < 0) return waiting;
            var windows = playback.GetProperty("windows");
            if (windows.GetArrayLength() > 16) return waiting;
            var items = new List<PlaybackWindow>();
            foreach (var item in windows.EnumerateArray())
            {
                var start = item.GetProperty("start_ms").GetDouble();
                var end = item.GetProperty("end_ms").GetDouble();
                var position = item.GetProperty("position_s").GetDouble();
                var amplitude = item.GetProperty("amplitude").GetDouble();
                if (!double.IsFinite(start) || !double.IsFinite(end) || !double.IsFinite(position) || !double.IsFinite(amplitude) ||
                    start < -300 || end > 600 || start >= end || position is < 0 or > 120 || amplitude is < 0 or > 1)
                    return waiting;
                string? kind = null, cue = null;
                if (item.TryGetProperty("cue", out var value) && value.ValueKind == JsonValueKind.Object)
                {
                    kind = value.GetProperty("kind").GetString(); cue = value.GetProperty("value").GetString();
                    if (kind is not ("characters" or "phonemes" or "visemes") || string.IsNullOrEmpty(cue) || cue.Length > 16) return waiting;
                }
                if (end <= transportMs) continue;
                items.Add(new(start - transportMs, end - transportMs, position, amplitude, kind, cue));
            }
            var segment = sync.GetProperty("segment_id").GetString();
            // Лицо набора (anger → furious) приходит в expression; emotion — для старых модулей.
            var emotion = sync.TryGetProperty("expression", out var expression) && expression.ValueKind == JsonValueKind.String
                ? expression.GetString() : sync.GetProperty("emotion").GetString();
            var animation = sync.GetProperty("animation").GetString();
            if (string.IsNullOrWhiteSpace(segment) || segment.Length > 64 || string.IsNullOrWhiteSpace(emotion) || emotion.Length > 64 || string.IsNullOrWhiteSpace(animation) || animation.Length > 64)
                return waiting;
            return new(epoch, response, generation, true, segment, emotion, animation, items, stamp)
            { ReceivedAt = System.Diagnostics.Stopwatch.GetTimestamp(), Eyes = Part(sync, "eyes"), Mouth = Part(sync, "mouth"), Moves = ParseMoves(sync) };
        }
        catch (Exception error) when (error is InvalidOperationException or KeyNotFoundException or FormatException or OverflowException)
        {
            return waiting;
        }
    }

    /// <summary>Шаги раскадровки (не больше 8); неверные поля отбрасываются, значения ограничиваются.</summary>
    internal static IReadOnlyList<CharacterMove> ParseMoves(JsonElement record)
    {
        if (!record.TryGetProperty("moves", out var moves) || moves.ValueKind != JsonValueKind.Array) return [];
        var result = new List<CharacterMove>();
        foreach (var move in moves.EnumerateArray().Take(8))
        {
            if (move.ValueKind != JsonValueKind.Object || !move.TryGetProperty("ms", out var ms) || !ms.TryGetDouble(out var duration) ||
                !double.IsFinite(duration) || duration is < 80 or > 3000)
                continue;
            static double? Number(JsonElement item, string name, double low, double high) =>
                item.TryGetProperty(name, out var value) && value.TryGetDouble(out var number) && double.IsFinite(number) ? Math.Clamp(number, low, high) : null;
            result.Add(new(Part(move, "do"), Number(move, "x", -1, 1), Number(move, "y", -1, 1), Number(move, "scale", 0.7, 1.4),
                Number(move, "squash", 0.7, 1.3), Part(move, "eyes"), Part(move, "mouth"), duration));
        }
        return result;
    }

    internal static string? Part(JsonElement record, string name) =>
        record.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String && value.GetString() is { Length: > 0 and <= 64 } text ? text : null;
}

/// <summary>Clock advances independently of idle; missing/stale packets close the mouth.</summary>
public sealed class CharacterSyncController
{
    private CharacterPlayback? _packet;
    private double _elapsed;
    private long _audioGeneration = -1;
    private long _epoch = -1;
    private double _stamp = -1;
    public bool Enabled => _packet?.Enabled == true;
    public bool Update(CharacterPlayback packet)
    {
        if (packet.Epoch < _epoch || packet.Epoch == _epoch && packet.AudioGeneration >= 0 && packet.AudioGeneration < _audioGeneration) return false;
        if (packet.Epoch > _epoch || packet.AudioGeneration > _audioGeneration) { _epoch = packet.Epoch; _audioGeneration = -1; _stamp = -1; }
        if (packet.Stamp > 0 && packet.Stamp <= _stamp) return false;
        if (packet.Stamp > 0) _stamp = packet.Stamp;
        if (packet.AudioGeneration >= 0) _audioGeneration = packet.AudioGeneration;
        _packet = packet;
        _elapsed = packet.ReceivedAt > 0 ? System.Diagnostics.Stopwatch.GetElapsedTime(packet.ReceivedAt).TotalMilliseconds : 0;
        return true;
    }
    public void Clear() { _packet = null; _elapsed = 0; }
    public void Advance(double milliseconds) => _elapsed += milliseconds;
    public (CharacterPlayback Packet, PlaybackWindow Window)? Current()
    {
        if (!Enabled || _elapsed > 300) return null;
        foreach (var window in _packet!.Windows)
            if (window.StartMs <= _elapsed && _elapsed < window.EndMs) return (_packet, window);
        return null;
    }
    public static string Mouth(PlaybackWindow window)
    {
        if (window.Amplitude < 0.012) return "neutral";
        var cue = window.Cue?.ToLowerInvariant();
        if (window.CueKind == "visemes" && cue is "neutral" or "talk_a" or "talk_e" or "open" or "o") return cue;
        if (window.CueKind is "characters" or "phonemes")
        {
            if (cue is "\u0430" or "\u044f" or "a" or "aa") return "talk_a";
            if (cue is "\u0435" or "\u044d" or "\u0438" or "e" or "i" or "ee") return "talk_e";
            if (cue is "\u043e" or "\u0443" or "\u0451" or "\u044e" or "o" or "u" or "oo") return "o";
            if (cue is "\u043c" or "\u0431" or "\u043f" or "m" or "b" or "p" or "sil" or " ") return "neutral";
        }
        return window.Amplitude < 0.08 ? "talk_e" : window.Amplitude < 0.2 ? "talk_a" : "open";
    }
}
