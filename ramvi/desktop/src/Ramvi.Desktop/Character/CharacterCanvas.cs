using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Imaging;

namespace Ramvi.Desktop.Character;

/// <summary>Presentation-only adapter. Replacing it cannot access dialogue or audio state.</summary>
public interface ICharacterRenderer
{
    void Draw(DrawingContext context, CharacterPack pack, IReadOnlyList<CharacterLayer> layers, Rect bounds);
    /// <summary>9.26: с отражением по горизонтали; по умолчанию — без него.</summary>
    void Draw(DrawingContext context, CharacterPack pack, IReadOnlyList<CharacterLayer> layers, Rect bounds, bool mirrored) =>
        Draw(context, pack, layers, bounds);
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
    public void Draw(DrawingContext context, CharacterPack pack, IReadOnlyList<CharacterLayer> layers, Rect bounds) =>
        Draw(context, pack, layers, bounds, false);

    public void Draw(DrawingContext context, CharacterPack pack, IReadOnlyList<CharacterLayer> layers, Rect bounds, bool mirrored)
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
            // Масштаб — от нижнего центра (ближе/дальше, присел/вытянулся), сдвиг — в точках набора.
            var width = size * layer.Scale;
            var height = size * layer.Scale * layer.ScaleY;
            var left = x + (size - width) / 2 + layer.X * size / 768;
            var top = y + size - height + layer.Y * size / 768;
            var dest = new Rect(left, top, width, height);
            if (!mirrored) { context.DrawImage(bitmap, dest); continue; }
            var axis = 2 * (x + size / 2); // отражение вокруг середины квадрата персонажа: x → axis − x
            using (context.PushTransform(new Matrix(-1, 0, 0, 1, axis, 0))) context.DrawImage(bitmap, dest);
            if (pack.MirrorKeep(layer.File) is not { } keep) continue;
            // Надпись перерисовывается неотражённой на своём отражённом месте.
            var k = 768.0;
            var spot = new Rect(left + keep.X / k * width, top + keep.Y / k * height, keep.Width / k * width, keep.Height / k * height);
            var pixels = bitmap.PixelSize.Width / k;
            var source = new Rect(keep.X * pixels, keep.Y * pixels, keep.Width * pixels, keep.Height * pixels);
            context.DrawImage(bitmap, source, new Rect(axis - spot.Right, spot.Y, spot.Width, spot.Height));
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

    /// <summary>9.26: отразить по горизонтали — персонаж смотрит к центру экрана.</summary>
    public bool Mirrored
    {
        get => _mirrored;
        set { if (_mirrored != value) { _mirrored = value; InvalidateVisual(); } }
    }
    private bool _mirrored;
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
        try
        {
            _renderer.Draw(context, Pack, Player.Layers(), Bounds, _mirrored);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or InvalidOperationException or KeyNotFoundException)
        {
            Release(); ResourceFailed?.Invoke(this, EventArgs.Empty);
        }
    }
}
