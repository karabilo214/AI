using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ramvi.Desktop.Services;

/// <summary>
/// 3.13/9.26: где показан персонаж и как выглядит его окно. Это настройка оболочки, а не
/// модуля Character: новая форма показа не меняет ни Character Engine, ни его данные.
/// </summary>
public sealed record CharacterWindowState(
    string Mode = "embedded",
    bool Background = false,
    double Opacity = 1,
    double? X = null,
    double? Y = null,
    double Width = 300,
    double Height = 380)
{
    public static readonly IReadOnlyList<string> Modes = ["embedded", "companion", "overlay"];

    /// <summary>Испорченные или чужие значения заменяются безопасными, а не роняют окно.</summary>
    public CharacterWindowState Normalized() => this with
    {
        Mode = Modes.Contains(Mode) ? Mode : "embedded",
        Opacity = double.IsFinite(Opacity) ? Math.Clamp(Opacity, 0.3, 1) : 1,
        X = X is { } x && double.IsFinite(x) ? x : null,
        Y = Y is { } y && double.IsFinite(y) ? y : null,
        Width = double.IsFinite(Width) ? Math.Clamp(Width, 200, 4000) : 300,
        Height = double.IsFinite(Height) ? Math.Clamp(Height, 200, 4000) : 380,
    };
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(CharacterWindowState))]
internal sealed partial class CharacterWindowJson : JsonSerializerContext;

/// <summary>Файл `desktop_character.json` в папке данных; запись атомарная.</summary>
public sealed class CharacterWindowStore(string folder)
{
    public const string FileName = "desktop_character.json";
    public string Path { get; } = System.IO.Path.Combine(folder, FileName);

    public CharacterWindowState Read()
    {
        try
        {
            using var stream = File.OpenRead(Path);
            return (JsonSerializer.Deserialize(stream, CharacterWindowJson.Default.CharacterWindowState) ?? new()).Normalized();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException
                                          or NotSupportedException)
        {
            return new();
        }
    }

    public void Write(CharacterWindowState state)
    {
        try
        {
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Path)!);
            var temp = Path + ".tmp";
            File.WriteAllText(temp, JsonSerializer.Serialize(state.Normalized(), CharacterWindowJson.Default.CharacterWindowState));
            File.Move(temp, Path, overwrite: true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Не сохранилось — окно продолжает работать с текущими значениями.
        }
    }
}
