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
        if (Root.GetProperty("schema").GetString() != "ramvi.character.pack/1")
        {
            _json.Dispose();
            throw new InvalidDataException("Unsupported character pack schema");
        }
    }
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
        var path = Path.GetFullPath(Path.Combine(_root, relative));
        if (!path.StartsWith(_root + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            throw new InvalidDataException("Character resource escapes its pack");
        return path;
    }
    public void Dispose() => _json.Dispose();
}

public sealed record CharacterLayer(string File, double X = 0, double Y = 0);

/// <summary>Animation clock: named directives only. Audio is deliberately outside this player.</summary>
public sealed class CharacterPlayer(CharacterPack pack)
{
    private double _elapsed;
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
        Present(emotion,animation); return true;
    }
    public string Emotion { get; private set; } = "neutral";
    public string Animation { get; private set; } = "idle";
    public double Speed => Animation is "nod" or "shake" ? 2 : 1;
    public int Frame => Math.Min(Count - 1, (int)(_elapsed / Motion.GetProperty("duration_ms").GetDouble()));
    private System.Text.Json.JsonElement Motion => pack.Motion(Animation);
    private int Count => Animation == "idle" ? Motion.GetProperty("frame_count").GetInt32() : Motion.GetProperty("frames").GetArrayLength();
    public void Present(string emotion, string animation)
    {
        if (!pack.Emotions.Contains(emotion, StringComparer.Ordinal) || !pack.HasAnimation(animation))
            throw new ArgumentException("Unsupported presentation directive");
        Emotion = emotion;
        if (animation != Animation) _elapsed = 0;
        Animation = animation;
    }
    public void Tick(double milliseconds)
    {
        if (!double.IsFinite(milliseconds) || milliseconds < 0) throw new ArgumentOutOfRangeException(nameof(milliseconds));
        _elapsed += milliseconds * Speed;
        var length = Count * Motion.GetProperty("duration_ms").GetDouble();
        if (_elapsed < length) return;
        if (Animation is "idle" or "talk") _elapsed %= length;
        else { Animation = "idle"; _elapsed = 0; }
    }
    public void Stop() { Animation = "idle"; _elapsed = 0; }
    public IReadOnlyList<CharacterLayer> Layers()
    {
        var face = pack.Face(Emotion);
        var frame = Frame;
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
