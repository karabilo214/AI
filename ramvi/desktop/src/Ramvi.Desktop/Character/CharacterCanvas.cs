using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Imaging;

namespace Ramvi.Desktop.Character;

/// <summary>Small bitmap cache, released when the Character surface is hidden or detached.</summary>
public sealed class CharacterCanvas : Control
{
    private readonly Dictionary<string, Bitmap> _images = new(StringComparer.Ordinal);
    public event EventHandler? ResourceFailed;
    public CharacterPack? Pack { get; set; }
    public CharacterPlayer? Player { get; set; }
    public void Release()
    {
        foreach (var image in _images.Values) image.Dispose();
        _images.Clear();
        Pack = null; Player = null;
        InvalidateVisual();
    }
    public override void Render(DrawingContext context)
    {
        base.Render(context);
        if (Pack is null || Player is null) return;
        try { DrawFrame(context); }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or InvalidOperationException or KeyNotFoundException)
        {
            Release(); ResourceFailed?.Invoke(this, EventArgs.Empty);
        }
    }
    private void DrawFrame(DrawingContext context)
    {
        if (Pack is null || Player is null) return;
        var size = Math.Min(Bounds.Width, Bounds.Height);
        var x = (Bounds.Width - size) / 2;
        var y = (Bounds.Height - size) / 2;
        foreach (var layer in Player.Layers())
        {
            if (!_images.TryGetValue(layer.File, out var bitmap))
            {
                // Decode only the requested pose, at display size rather than full 768px.
                using var stream = File.OpenRead(Pack.FilePath(layer.File));
                bitmap = Bitmap.DecodeToWidth(stream, 384);
                _images[layer.File] = bitmap;
            }
            context.DrawImage(bitmap, new Rect(x + layer.X * size / 768, y + layer.Y * size / 768, size, size));
        }
    }
}
