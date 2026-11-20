using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Imaging;

namespace Ramvi.Desktop.Character;

/// <summary>Presentation-only adapter. Replacing it cannot access dialogue or audio state.</summary>
public interface ICharacterRenderer
{
    void Draw(DrawingContext context, CharacterPack pack, IReadOnlyList<CharacterLayer> layers, Rect bounds);
    void Release();
}

/// <summary>Avalonia bitmap adapter; uses the host's Skia backend without a dedicated GPU.</summary>
public sealed class CharacterBitmapRenderer : ICharacterRenderer
{
    private readonly Dictionary<string, Bitmap> _images = new(StringComparer.Ordinal);
    internal int CachedImageCount => _images.Count;
    public void Release()
    {
        foreach (var image in _images.Values) image.Dispose();
        _images.Clear();
    }
    public void Draw(DrawingContext context, CharacterPack pack, IReadOnlyList<CharacterLayer> layers, Rect bounds)
    {
        var size = Math.Min(bounds.Width, bounds.Height);
        var x = (bounds.Width - size) / 2;
        var y = (bounds.Height - size) / 2;
        foreach (var layer in layers)
        {
            var path = pack.FilePath(layer.File);
            // Full paths prevent stale images when two packs use the same resource names.
            if (!_images.TryGetValue(path, out var bitmap))
            {
                using var stream = File.OpenRead(path);
                bitmap = Bitmap.DecodeToWidth(stream, 384);
                _images[path] = bitmap;
            }
            using var opacity = context.PushOpacity(layer.Opacity);
            context.DrawImage(bitmap, new Rect(x + layer.X * size / 768, y + layer.Y * size / 768, size, size));
        }
    }
}

/// <summary>Surface lifecycle, separate from pack data, animation clock and renderer.</summary>
public sealed class CharacterCanvas : Control
{
    private ICharacterRenderer _renderer = new CharacterBitmapRenderer();
    public ICharacterRenderer Renderer
    {
        get => _renderer;
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            if (ReferenceEquals(_renderer, value)) return;
            _renderer.Release();
            _renderer = value;
            InvalidateVisual();
        }
    }
    public event EventHandler? ResourceFailed;
    private CharacterPack? _pack;
    public CharacterPack? Pack
    {
        get => _pack;
        set
        {
            if (ReferenceEquals(_pack, value)) return;
            _renderer.Release();
            _pack = value;
            InvalidateVisual();
        }
    }
    public CharacterPlayer? Player { get; set; }
    public void Release()
    {
        _renderer.Release();
        Pack = null; Player = null;
        InvalidateVisual();
    }
    public override void Render(DrawingContext context)
    {
        base.Render(context);
        if (Pack is null || Player is null) return;
        try { _renderer.Draw(context, Pack, Player.Layers(), Bounds); }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or InvalidOperationException or KeyNotFoundException)
        {
            Release(); ResourceFailed?.Invoke(this, EventArgs.Empty);
        }
    }
}
