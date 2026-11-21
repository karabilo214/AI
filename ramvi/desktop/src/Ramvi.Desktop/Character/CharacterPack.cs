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
            if (Gestures.Contains(item.Name, StringComparer.Ordinal))
            {
                // Живой жест из частей набора: рисует проигрыватель, кадров нет.
                if (motion.GetProperty("kind").GetString() != "gesture" || motion.TryGetProperty("frames", out _) ||
                    !motion.GetProperty("duration_ms").TryGetDouble(out var length) || length is < 200 or > 4000 ||
                    motion.GetProperty("speed").GetInt32() != 1 || motion.GetProperty("loop").GetBoolean())
                    throw new InvalidDataException("Invalid character gesture");
                continue;
            }
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
    /// <summary>Живые жесты: подпрыгнуть, отвернуться, накричать, рассмеяться.</summary>
    public static readonly string[] Gestures = ["jump", "turn_away", "shout", "laugh"];
    public bool IsGesture(string name) => Gestures.Contains(name, StringComparer.Ordinal) && HasAnimation(name);
    public bool HasEye(string name) => Root.GetProperty("eyes").TryGetProperty(name, out _);
    public bool HasMouth(string name) => Root.GetProperty("mouths").TryGetProperty(name, out _);
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


/// <summary>Слой кадра: сдвиг в точках набора (768), масштаб от нижнего центра, сжатие по высоте.</summary>
public sealed record CharacterLayer(string File, double X = 0, double Y = 0, double Opacity = 1, double Scale = 1, double ScaleY = 1);

/// <summary>Шаг раскадровки от модели: к какой позе прийти за Ms, какие глаза/рот, какой жест начать.</summary>
public sealed record CharacterMove(string? Do, double? X, double? Y, double? Scale, double? Squash, string? Eyes, string? Mouth, double Ms);

internal readonly record struct Pose(double X, double Y, double Scale, double Squash)
{
    public static readonly Pose Rest = new(0, 0, 1, 1);
    public Pose Lerp(Pose to, double t) => new(X + (to.X - X) * t, Y + (to.Y - Y) * t, Scale + (to.Scale - Scale) * t, Squash + (to.Squash - Squash) * t);
}

/// <summary>Animation clock: named directives only. Audio is deliberately outside this player.</summary>
public sealed class CharacterPlayer(CharacterPack pack)
{
    private const double Reach = 150;    // x/y = ±1 — сдвиг на 150 точек набора
    private const double Settle = 350;   // после раскадровки — плавно в покой

    public void ReplacePack(CharacterPack replacement)
    {
        ArgumentNullException.ThrowIfNull(replacement);
        pack = replacement;
        if (!pack.Emotions.Contains(Emotion, StringComparer.Ordinal)) Emotion = "neutral";
        if (!pack.Turns.Contains(Turn, StringComparer.Ordinal)) Turn = "front";
        if (!pack.Effects.Contains(Effect, StringComparer.Ordinal)) Effect = "none";
        _elapsed = 0; _effectElapsed = 0; _moves = []; _eyes = _mouth = null;
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
            Present(emotion, motion, active.Packet.Eyes, active.Packet.Mouth, active.Packet.Moves);
            _syncSegment = key;
            if (active.Packet.SegmentId is { } spoken) _spoken.Add(spoken);
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
            _epoch=directive.Epoch; _response=directive.ResponseId; _seen.Clear(); _spoken.Clear(); _replyDone=false; Stop();
        }
        if(directive.ResponseId!=_response) return false;
        if(directive.SegmentId is null) { Stop(); return true; }
        if(directive.SegmentId==EndOfReply) { _replyDone=true; return true; }
        if(_seen.Count>=8 || !_seen.Add(directive.SegmentId)) return false;
        var emotion=pack.Emotions.Contains(directive.Emotion,StringComparer.Ordinal) ? directive.Emotion : "neutral";
        var animation=directive.Animation=="none" ? "idle" : directive.Animation;
        if(!pack.HasAnimation(animation)) animation="idle";
        if (!_sync.Enabled) Present(emotion,animation == "talk" ? "idle" : animation, directive.Eyes, directive.Mouth, directive.Moves);
        return true;
    }
    public string Emotion { get; private set; } = "neutral";
    public string Animation { get; private set; } = "idle";
    public string Turn { get; private set; } = "front";
    /// <summary>Глаза с бровями и рот, выбранные моделью для сегмента (null — по лицу эмоции).</summary>
    public string? Eyes => _eyes;
    public string? MouthOverride => _mouth;
    public void SetTurn(string turn)
    {
        if (!pack.Turns.Contains(turn, StringComparer.Ordinal)) throw new ArgumentException("Unsupported character turn");
        Turn = turn;
        if (turn != "front") { Animation = "idle"; _elapsed = 0; }
    }
    public double Speed => Animation is "nod" or "shake" ? 2 : 1;
    public int Frame => Math.Min(Count - 1, (int)(_elapsed / Motion.GetProperty("duration_ms").GetDouble()));
    private System.Text.Json.JsonElement Motion => pack.Motion(Animation);
    private bool Gesture => pack.IsGesture(Animation);
    private int Count => Gesture ? 1 : Animation == "idle" ? Motion.GetProperty("frame_count").GetInt32() : Motion.GetProperty("frames").GetArrayLength();
    /// <summary>Доля пройденного жеста 0..1.</summary>
    private double Progress => Gesture ? Math.Clamp(_elapsed / Motion.GetProperty("duration_ms").GetDouble(), 0, 1) : 0;

    private string? _eyes, _mouth;
    private CharacterMove[] _moves = [];
    private double _moveElapsed;
    private int _moveStep = -1;

    public void Present(string emotion, string animation, string? eyes = null, string? mouth = null, IReadOnlyList<CharacterMove>? moves = null)
    {
        if (!pack.Emotions.Contains(emotion, StringComparer.Ordinal) || !pack.HasAnimation(animation))
            throw new ArgumentException("Unsupported presentation directive");
        Emotion = emotion;
        if (animation != "idle") Turn = "front";
        if (animation != Animation) _elapsed = 0;
        Animation = animation;
        _eyes = eyes is not null && pack.HasEye(eyes) ? eyes : null;
        _mouth = mouth is not null && pack.HasMouth(mouth) ? mouth : null;
        _moves = (moves ?? []).Where(m => double.IsFinite(m.Ms) && m.Ms > 0).Take(8).ToArray();
        _moveElapsed = 0; _moveStep = -1; _quiet = 0;
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
        AdvanceMoves(milliseconds);
        Relax(milliseconds);
        // Reduce long jumps before multiplication so finite input cannot overflow the clock.
        var cycle = Count * Motion.GetProperty("duration_ms").GetDouble();
        if (Animation is "idle" or "talk") milliseconds %= cycle / Speed;
        _elapsed += milliseconds * Speed;
        var length = Count * Motion.GetProperty("duration_ms").GetDouble();
        if (_elapsed < length) return;
        if (Animation is "idle" or "talk") _elapsed %= length;
        else { Animation = "idle"; _elapsed = 0; }
    }
    /// <summary>Сколько лицо держит эмоцию после последней реплики, прежде чем вернуться в обычное.</summary>
    public const double RelaxAfterMs = 2500;
    /// <summary>Служебный сегмент «ответ модели закончен» (от экрана, не от модели).</summary>
    public const string EndOfReply = "~end";
    private double _quiet;
    private bool _replyDone = true;
    private readonly HashSet<string> _spoken = [];
    /// <summary>Ответ целиком позади: модель закончила, и при голосе каждая его часть уже прозвучала.
    /// Внутри ответа лицо меняется по частям и не сбрасывается в паузах между ними.</summary>
    private bool ReplyOver => _replyDone && (!_sync.Enabled || _seen.All(_spoken.Contains));
    /// <summary>Тишина (не говорит, жест и раскадровка кончились) дольше <see cref="RelaxAfterMs"/> —
    /// лицо в обычное: эмоция прошлой реплики не висит до следующей.</summary>
    private void Relax(double milliseconds)
    {
        var busy = _syncMouth is not null || _moves.Length > 0 || Animation != "idle" || !ReplyOver;
        _quiet = busy ? 0 : _quiet + milliseconds;
        if (_quiet < RelaxAfterMs || (Emotion == "neutral" && _eyes is null && _mouth is null)) return;
        Emotion = "neutral"; _eyes = _mouth = null;
    }
    private void AdvanceMoves(double milliseconds)
    {
        if (_moves.Length == 0) return;
        _moveElapsed = Math.Min(_moveElapsed + milliseconds, _moves.Sum(m => m.Ms) + Settle);
        var step = StepAt(_moveElapsed).Index;
        if (step != _moveStep && step < _moves.Length)
        {
            _moveStep = step;
            // Жест шага стартует вместе с шагом: «присел — подпрыгнул».
            if (_moves[step].Do is { } gesture && gesture != "none" && pack.HasAnimation(gesture) && gesture is not ("talk" or "blink" or "idle"))
            { Animation = gesture; _elapsed = 0; Turn = "front"; }
        }
        if (_moveElapsed >= _moves.Sum(m => m.Ms) + Settle) { _moves = []; _moveStep = -1; }
    }
    private (int Index, double T) StepAt(double time)
    {
        var start = 0.0;
        for (var i = 0; i < _moves.Length; i++)
        {
            if (time < start + _moves[i].Ms) return (i, (time - start) / _moves[i].Ms);
            start += _moves[i].Ms;
        }
        return (_moves.Length, Math.Min(1, (time - start) / Settle));
    }
    private static double Ease(double t) => t * t * (3 - 2 * t);
    /// <summary>Поза по раскадровке: шаги без поля держат прежнее значение; после — в покой.</summary>
    internal Pose CurrentPose()
    {
        if (_moves.Length == 0) return Pose.Rest;
        var (index, t) = StepAt(_moveElapsed);
        var previous = Pose.Rest;
        for (var i = 0; i < Math.Min(index, _moves.Length); i++) previous = Target(previous, _moves[i]);
        if (index >= _moves.Length) return previous.Lerp(Pose.Rest, Ease(t));
        return previous.Lerp(Target(previous, _moves[index]), Ease(t));
    }
    private static Pose Target(Pose from, CharacterMove move) => new(
        Math.Clamp(move.X ?? from.X, -1, 1), Math.Clamp(move.Y ?? from.Y, -1, 1),
        Math.Clamp(move.Scale ?? from.Scale, 0.7, 1.4), Math.Clamp(move.Squash ?? from.Squash, 0.7, 1.3));
    /// <summary>Глаза/рот последнего шага, где они заданы; иначе — сегмента.</summary>
    private (string? Eyes, string? Mouth) StepFace()
    {
        string? eyes = _eyes, mouth = _mouth;
        if (_moves.Length == 0) return (eyes, mouth);
        var index = Math.Min(StepAt(_moveElapsed).Index, _moves.Length - 1);
        for (var i = 0; i <= index; i++)
        {
            if (_moves[i].Eyes is { } e && pack.HasEye(e)) eyes = e;
            if (_moves[i].Mouth is { } m && pack.HasMouth(m)) mouth = m;
        }
        return (eyes, mouth);
    }
    public void Stop() { Animation = "idle"; _elapsed = 0; _sync.Clear(); _syncSegment = null; _syncMouth = null; _moves = []; _eyes = _mouth = null; SetEffect("none"); }
    public IReadOnlyList<CharacterLayer> Layers()
    {
        var pose = CurrentPose();
        var (gx, gy, gs, gsy) = GestureMotion();
        return BaseLayers().Select(layer => layer with
        {
            Opacity = Opacity,
            X = layer.X + pose.X * Reach + gx,
            Y = layer.Y + pose.Y * Reach + gy,
            Scale = layer.Scale * pose.Scale * gs,
            ScaleY = layer.ScaleY * pose.Squash * gsy,
        }).ToArray();
    }
    /// <summary>Движение жеста поверх позы: прыжок, рывок с дрожью, смех, отворот (покачивание).</summary>
    private (double X, double Y, double Scale, double ScaleY) GestureMotion()
    {
        if (!Gesture) return (0, 0, 1, 1);
        var t = Progress;
        return Animation switch
        {
            // Два прыжка, второй ниже; на приземлении — лёгкое сжатие.
            "jump" => (0, -Math.Abs(Math.Sin(t * 2 * Math.PI)) * 95 * (1 - 0.45 * t), 1, 1 - 0.1 * Math.Max(0, Math.Cos(t * 4 * Math.PI)) * (1 - t)),
            // Рывок к зрителю и дрожь, потом назад.
            "shout" => (t < 0.85 ? Math.Sin(t * 90) * 6 : 0, 0, 1 + 0.16 * Math.Sin(Math.Min(1, t / 0.25) * Math.PI / 2) * (t < 0.8 ? 1 : (1 - t) / 0.2), 1),
            "laugh" => (0, -Math.Abs(Math.Sin(t * 6 * Math.PI)) * 18, 1, 1 + 0.04 * Math.Sin(t * 12 * Math.PI)),
            "turn_away" => (0, Math.Sin(t * 4 * Math.PI) * 3, 1, 1),
            _ => (0, 0, 1, 1),
        };
    }
    private IReadOnlyList<CharacterLayer> BaseLayers()
    {
        if (Animation == "turn_away")
        {
            // Отвернулся: три четверти → профиль (держит) → три четверти → снова лицом.
            var t = Progress;
            var turn = t < 0.1 || (t >= 0.8 && t < 0.9) ? "three_quarter_left" : t < 0.8 ? "profile_left" : "front";
            if (turn != "front" && pack.Turns.Contains(turn, StringComparer.Ordinal)) return [new(pack.Turn(turn))];
        }
        // Non-front poses are complete sprites: never overlay front-facing facial parts.
        if (Turn != "front") return [new(pack.Turn(Turn))];
        var face = pack.Face(Emotion);
        var frame = Frame;
        var (stepEyes, stepMouth) = StepFace();
        var faceEyes = stepEyes ?? (Animation == "laugh" ? (pack.HasEye("happy") ? "happy" : null) : null) ?? face.GetProperty("eyes").GetString()!;
        var faceMouth = stepMouth ?? Animation switch
        {
            "shout" when pack.HasMouth("wide") => "wide",
            "laugh" when pack.HasMouth("grin") => Progress * 12 % 2 < 1 ? "grin" : pack.HasMouth("wide") ? "wide" : "grin",
            _ => null,
        } ?? face.GetProperty("mouth").GetString()!;
        if (_syncMouth is not null)
        {
            var syncEyes = stepEyes ?? face.GetProperty("eyes").GetString()!;
            if (Animation == "blink") syncEyes = frame is 2 or 3 ? "closed" : frame is 1 or 4 ? "half" : syncEyes;
            var phase = Math.Sin(frame / (double)Math.Max(1, Count - 1) * Math.PI * (Animation == "shake" ? 4 : 2));
            var x = Animation == "shake" ? phase * 15 : 0;
            var y = Animation == "nod" ? phase * 14 : Animation == "idle" ? Math.Sin(frame / (double)Count * Math.PI * 2) * 4 : 0;
            return [new(pack.Body, x, y), new(pack.Eye(syncEyes), x, y), new(pack.Mouth(_syncMouth), x, y)];
        }
        // Модель собрала лицо из частей или жест меняет глаза/рот — слои вместо готового лица.
        var composed = stepEyes is not null || stepMouth is not null || Animation is "shout" or "laugh";
        if (Gesture)
            return composed ? [new(pack.Body), new(pack.Eye(faceEyes)), new(pack.Mouth(faceMouth))] : [new(face.GetProperty("file").GetString()!)];
        if (Animation == "idle")
        {
            var bob = Math.Sin(frame / (double)Count * Math.PI * 2) * 4;
            return composed ? [new(pack.Body, 0, bob), new(pack.Eye(faceEyes), 0, bob), new(pack.Mouth(faceMouth), 0, bob)]
                : [new(face.GetProperty("file").GetString()!, 0, bob)];
        }
        if (Emotion == "neutral" && !composed) return [new(Motion.GetProperty("frames")[frame].GetString()!)];
        var eyes = faceEyes;
        var mouth = faceMouth;
        if (Animation == "blink") eyes = frame is 2 or 3 ? "closed" : frame is 1 or 4 ? "half" : eyes;
        if (Animation == "talk") mouth = new[] { "neutral", "talk_a", "talk_e", "open", "talk_a", "o", "talk_e", "neutral" }[frame];
        if (Animation is "blink" or "talk") return [new(pack.Body), new(pack.Eye(eyes)), new(pack.Mouth(mouth))];
        var offset = Math.Sin(frame / (double)(Count - 1) * Math.PI * (Animation == "shake" ? 4 : 2));
        if (composed)
        {
            var (sx, sy) = (Animation == "shake" ? offset * 15 : 0, Animation == "nod" ? offset * 14 : 0);
            return [new(pack.Body, sx, sy), new(pack.Eye(eyes), sx, sy), new(pack.Mouth(mouth), sx, sy)];
        }
        return [new(face.GetProperty("file").GetString()!, Animation == "shake" ? offset * 15 : 0, Animation == "nod" ? offset * 14 : 0)];
    }
}
