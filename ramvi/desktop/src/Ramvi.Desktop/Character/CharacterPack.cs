using System.Text.Json;

namespace Ramvi.Desktop.Character;

/// <summary>Renderer-specific reader of a replaceable Character pack; no providers or devices.</summary>
public sealed class CharacterPack : IDisposable
{
    private readonly JsonDocument _json;
    private readonly string _root;
    public CharacterPack(string path)
    {
        _root = Path.GetDirectoryName(Path.GetFullPath(path))!;
        _json = JsonDocument.Parse(File.ReadAllText(path));
        try { Validate(); }
        catch (Exception error) when (error is InvalidOperationException or KeyNotFoundException or FormatException or OverflowException or IOException or UnauthorizedAccessException or ArgumentException)
        {
            _json.Dispose();
            throw new InvalidDataException("Invalid character pack", error);
        }
    }
    private void Validate()
    {
        if (Root.GetProperty("schema").GetString() != "ramvi.character.pack/1" ||
            !System.Version.TryParse(Root.GetProperty("version").GetString(), out var version) || version.Major != 1 || version.Build < 0 || version.Revision >= 0 ||
            Root.GetProperty("canvas").GetArrayLength() != 2 ||
            Root.GetProperty("canvas")[0].GetInt32() != 768 || Root.GetProperty("canvas")[1].GetInt32() != 768)
            throw new InvalidDataException("Unsupported character pack version or canvas");
        if (Root.GetProperty("speech").GetString() != "tts" || Root.GetProperty("auto_blink").GetBoolean())
            throw new InvalidDataException("Unsupported speech or blinking policy");
        CheckResource(Body);
        foreach (var group in new[] { "eyes", "mouths", "turns" })
            foreach (var item in Root.GetProperty(group).EnumerateObject()) CheckResource(item.Value.GetString()!);
        _ = Face("neutral");
        _ = Turn("front");
        foreach (var face in Root.GetProperty("expressions").EnumerateObject())
        {
            CheckResource(face.Value.GetProperty("file").GetString()!);
            _ = Eye(face.Value.GetProperty("eyes").GetString()!);
            _ = Mouth(face.Value.GetProperty("mouth").GetString()!);
        }
        foreach (var name in new[] { "idle", "blink", "talk", "nod", "shake" }) _ = Motion(name);
        foreach (var item in Root.GetProperty("animations").EnumerateObject())
        {
            var motion = item.Value;
            if (item.Name is not ("idle" or "blink" or "talk" or "nod" or "shake") ||
                !motion.GetProperty("duration_ms").TryGetDouble(out var duration) || !double.IsFinite(duration) || duration <= 0 || duration > 60000 ||
                motion.GetProperty("speed").GetInt32() != (item.Name is "nod" or "shake" ? 2 : 1) ||
                motion.GetProperty("loop").GetBoolean() != (item.Name is "idle" or "talk"))
                throw new InvalidDataException("Unsupported animation policy");
            if (item.Name == "idle")
            {
                if (motion.GetProperty("kind").GetString() != "expression_bob" || motion.TryGetProperty("frames", out _) ||
                    motion.GetProperty("frame_count").GetInt32() is < 1 or > 256)
                    throw new InvalidDataException("Invalid idle animation");
            }
            else
            {
                var frames = motion.GetProperty("frames");
                if (motion.GetProperty("kind").GetString() != "frames" || frames.GetArrayLength() is < 2 or > 256 ||
                    item.Name == "talk" && frames.GetArrayLength() != 8 || item.Name == "blink" && frames.GetArrayLength() != 6)
                    throw new InvalidDataException("Invalid animation frames");
                foreach (var frame in frames.EnumerateArray()) CheckResource(frame.GetString()!);
            }
        }
        foreach (var eye in new[] { "closed", "half" }) _ = Eye(eye);
        foreach (var mouth in new[] { "neutral", "talk_a", "talk_e", "open", "o" }) _ = Mouth(mouth);
        foreach (var name in Effects.Where(x => x != "none"))
        {
            var effect = Effect(name);
            var minimum = effect.GetProperty("minimum").GetDouble();
            var duration = effect.GetProperty("duration_ms").GetDouble();
            if (effect.GetProperty("kind").GetString() != "opacity" || !double.IsFinite(minimum) || minimum is < 0.25 or > 1 ||
                !double.IsFinite(duration) || duration is < 200 or > 60000)
                throw new InvalidDataException("Invalid character effect");
        }
        if (Root.TryGetProperty("effects", out var effects) && effects.TryGetProperty("none", out _))
            throw new InvalidDataException("Reserved character effect name");
        var sounds = new HashSet<string>(StringComparer.Ordinal);
        foreach (var sound in Root.GetProperty("sounds").EnumerateArray())
        {
            var id = sound.GetProperty("id").GetString()!;
            if (string.IsNullOrWhiteSpace(id) || !sounds.Add(id) || id.StartsWith("character/talk_", StringComparison.Ordinal) || id is "motion/shake_head" or "character/beep_no")
                throw new InvalidDataException("Duplicate or excluded sound");
            CheckResource(sound.GetProperty("file").GetString()!);
        }
    }
    private void CheckResource(string relative)
    {
        if (!File.Exists(FilePath(relative))) throw new InvalidDataException("Missing character resource");
    }
    public IReadOnlyList<string> Effects => Root.TryGetProperty("effects", out var effects)
        ? effects.EnumerateObject().Select(x => x.Name).Prepend("none").ToArray() : ["none"];
    public JsonElement Effect(string name) => Root.GetProperty("effects").GetProperty(name);
    public string Version => Root.GetProperty("version").GetString()!;
    public IReadOnlyList<string> Turns => Root.GetProperty("turns").EnumerateObject().Select(x => x.Name).ToArray();
    public string Turn(string name) => Root.GetProperty("turns").GetProperty(name).GetString()!;

    private JsonElement Root => _json.RootElement;
    public IReadOnlyList<string> Emotions => Root.GetProperty("expressions").EnumerateObject().Select(x => x.Name).ToArray();
    public bool HasAnimation(string name) => Root.GetProperty("animations").TryGetProperty(name, out _);
    public JsonElement Motion(string name) => Root.GetProperty("animations").GetProperty(name);
    public JsonElement Face(string name) => Root.GetProperty("expressions").GetProperty(name);
    public string Body => Root.GetProperty("body").GetString()!;
    public string Eye(string name) => Root.GetProperty("eyes").GetProperty(name).GetString()!;
    public string Mouth(string name) => Root.GetProperty("mouths").GetProperty(name).GetString()!;
    public string FilePath(string relative)
    {
        if (string.IsNullOrWhiteSpace(relative) || Path.IsPathRooted(relative) || relative.Contains('\\') || relative.Contains(':'))
            throw new InvalidDataException("Invalid character resource path");
        var path = Path.GetFullPath(Path.Combine(_root, relative));
        if (!path.StartsWith(_root + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            throw new InvalidDataException("Character resource escapes its pack");
        var current = new FileInfo(path) as FileSystemInfo;
        while (current is not null && current.FullName != _root)
        {
            if (current.LinkTarget is not null) throw new InvalidDataException("Linked character resources are not supported");
            current = current is FileInfo file ? file.Directory : ((DirectoryInfo)current).Parent;
        }
        return path;
    }
    public void Dispose() => _json.Dispose();
}

public sealed record CharacterLayer(string File, double X = 0, double Y = 0, double Opacity = 1);

/// <summary>Animation clock: named directives only. Audio is deliberately outside this player.</summary>
public sealed class CharacterPlayer(CharacterPack pack)
{
    public void ReplacePack(CharacterPack replacement)
    {
        ArgumentNullException.ThrowIfNull(replacement);
        pack = replacement;
        if (!pack.Emotions.Contains(Emotion, StringComparer.Ordinal)) Emotion = "neutral";
        if (!pack.Turns.Contains(Turn, StringComparer.Ordinal)) Turn = "front";
        if (!pack.Effects.Contains(Effect, StringComparer.Ordinal)) Effect = "none";
        _elapsed = 0; _effectElapsed = 0;
        // Epoch, response and seen segment IDs survive presentation changes.
    }
    private readonly CharacterSyncController _sync = new();
    private string? _syncSegment, _syncMouth;
    public void SetPlayback(CharacterPlayback playback)
    {
        if (playback.Epoch != _epoch || playback.ResponseId != _response) return;
        var wasEnabled = _sync.Enabled;
        if (!_sync.Update(playback)) return;
        if (wasEnabled && !playback.Enabled) Stop();
        SyncFrame();
    }
    private void SyncFrame()
    {
        var current = _sync.Current();
        _syncMouth = null;
        if (current is not {} active || active.Packet.Epoch != _epoch || active.Packet.ResponseId != _response)
        {
            if (_sync.Enabled && Animation == "talk") { Animation = "idle"; _elapsed = 0; }
            return;
        }
        var key = $"{active.Packet.AudioGeneration}:{active.Packet.SegmentId}";
        if (key != _syncSegment)
        {
            var emotion = pack.Emotions.Contains(active.Packet.Emotion, StringComparer.Ordinal) ? active.Packet.Emotion : "neutral";
            var motion = pack.HasAnimation(active.Packet.Animation) ? active.Packet.Animation : "idle";
            Present(emotion, motion); _syncSegment = key;
        }
        Turn = "front";
        _syncMouth = CharacterSyncController.Mouth(active.Window);
    }
    private double _elapsed;
    private double _effectElapsed;
    public string Effect { get; private set; } = "none";
    public void SetEffect(string effect)
    {
        if (!pack.Effects.Contains(effect, StringComparer.Ordinal)) throw new ArgumentException("Unsupported character effect");
        Effect = effect; _effectElapsed = 0;
    }
    private double Opacity => Effect == "none" ? 1 :
        1 - (1 - pack.Effect(Effect).GetProperty("minimum").GetDouble()) *
        (1 - Math.Cos(2 * Math.PI * _effectElapsed / pack.Effect(Effect).GetProperty("duration_ms").GetDouble())) / 2;
    private long _epoch = -1;
    private string? _response;
    private readonly HashSet<string> _seen = [];
    public bool Apply(CharacterDirective directive)
    {
        if(directive.Epoch<_epoch) return false;
        if(directive.Epoch>_epoch) {
            _epoch=directive.Epoch; _response=directive.ResponseId; _seen.Clear(); Stop();
        }
        if(directive.ResponseId!=_response) return false;
        if(directive.SegmentId is null) { Stop(); return true; }
        if(_seen.Count>=8 || !_seen.Add(directive.SegmentId)) return false;
        var emotion=pack.Emotions.Contains(directive.Emotion,StringComparer.Ordinal) ? directive.Emotion : "neutral";
        var animation=directive.Animation=="none" ? "idle" : directive.Animation;
        if(!pack.HasAnimation(animation)) animation="idle";
        if (!_sync.Enabled) Present(emotion,animation == "talk" ? "idle" : animation);
        return true;
    }
    public string Emotion { get; private set; } = "neutral";
    public string Animation { get; private set; } = "idle";
    public string Turn { get; private set; } = "front";
    public void SetTurn(string turn)
    {
        if (!pack.Turns.Contains(turn, StringComparer.Ordinal)) throw new ArgumentException("Unsupported character turn");
        Turn = turn;
        if (turn != "front") { Animation = "idle"; _elapsed = 0; }
    }
    public double Speed => Animation is "nod" or "shake" ? 2 : 1;
    public int Frame => Math.Min(Count - 1, (int)(_elapsed / Motion.GetProperty("duration_ms").GetDouble()));
    private System.Text.Json.JsonElement Motion => pack.Motion(Animation);
    private int Count => Animation == "idle" ? Motion.GetProperty("frame_count").GetInt32() : Motion.GetProperty("frames").GetArrayLength();
    public void Present(string emotion, string animation)
    {
        if (!pack.Emotions.Contains(emotion, StringComparer.Ordinal) || !pack.HasAnimation(animation))
            throw new ArgumentException("Unsupported presentation directive");
        Emotion = emotion;
        if (animation != "idle") Turn = "front";
        if (animation != Animation) _elapsed = 0;
        Animation = animation;
    }
    public void Tick(double milliseconds)
    {
        if (!double.IsFinite(milliseconds) || milliseconds < 0) throw new ArgumentOutOfRangeException(nameof(milliseconds));
        _sync.Advance(milliseconds);
        SyncFrame();
        if (Effect != "none")
        {
            var duration = pack.Effect(Effect).GetProperty("duration_ms").GetDouble();
            _effectElapsed = (_effectElapsed + milliseconds % duration) % duration;
        }
        // Reduce long jumps before multiplication so finite input cannot overflow the clock.
        var cycle = Count * Motion.GetProperty("duration_ms").GetDouble();
        if (Animation is "idle" or "talk") milliseconds %= cycle / Speed;
        _elapsed += milliseconds * Speed;
        var length = Count * Motion.GetProperty("duration_ms").GetDouble();
        if (_elapsed < length) return;
        if (Animation is "idle" or "talk") _elapsed %= length;
        else { Animation = "idle"; _elapsed = 0; }
    }
    public void Stop() { Animation = "idle"; _elapsed = 0; _sync.Clear(); _syncSegment = null; _syncMouth = null; SetEffect("none"); }
    public IReadOnlyList<CharacterLayer> Layers() => BaseLayers().Select(layer => layer with { Opacity = Opacity }).ToArray();
    private IReadOnlyList<CharacterLayer> BaseLayers()
    {
        // Non-front poses are complete sprites: never overlay front-facing facial parts.
        if (Turn != "front") return [new(pack.Turn(Turn))];
        var face = pack.Face(Emotion);
        var frame = Frame;
        if (_syncMouth is not null)
        {
            var syncEyes = face.GetProperty("eyes").GetString()!;
            if (Animation == "blink") syncEyes = frame is 2 or 3 ? "closed" : frame is 1 or 4 ? "half" : syncEyes;
            var phase = Math.Sin(frame / (double)Math.Max(1, Count - 1) * Math.PI * (Animation == "shake" ? 4 : 2));
            var x = Animation == "shake" ? phase * 15 : 0;
            var y = Animation == "nod" ? phase * 14 : Animation == "idle" ? Math.Sin(frame / (double)Count * Math.PI * 2) * 4 : 0;
            return [new(pack.Body, x, y), new(pack.Eye(syncEyes), x, y), new(pack.Mouth(_syncMouth), x, y)];
        }
        if (Animation == "idle")
            return [new(face.GetProperty("file").GetString()!, 0, Math.Sin(frame / (double)Count * Math.PI * 2) * 4)];
        if (Emotion == "neutral") return [new(Motion.GetProperty("frames")[frame].GetString()!)];
        var eyes = face.GetProperty("eyes").GetString()!;
        var mouth = face.GetProperty("mouth").GetString()!;
        if (Animation == "blink") eyes = frame is 2 or 3 ? "closed" : frame is 1 or 4 ? "half" : eyes;
        if (Animation == "talk") mouth = new[] { "neutral", "talk_a", "talk_e", "open", "talk_a", "o", "talk_e", "neutral" }[frame];
        if (Animation is "blink" or "talk") return [new(pack.Body), new(pack.Eye(eyes)), new(pack.Mouth(mouth))];
        var offset = Math.Sin(frame / (double)(Count - 1) * Math.PI * (Animation == "shake" ? 4 : 2));
        return [new(face.GetProperty("file").GetString()!, Animation == "shake" ? offset * 15 : 0, Animation == "nod" ? offset * 14 : 0)];
    }
}
