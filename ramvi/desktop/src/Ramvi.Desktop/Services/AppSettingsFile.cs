using System.Text.Json;

namespace Ramvi.Desktop.Services;

/// <summary>
/// То, что оболочке нужно из `settings.json` папки приложения до запуска сервиса (3.7):
/// папка данных, язык и тема окна. Пишет файл только сервис (`ramvi_service.app_config`, одна проверка).
/// Нет файла или он испорчен — значения по умолчанию: окно должно открыться в любом случае.
/// </summary>
public sealed record AppSettingsFile(string DataDir, string Language, string Theme = "system")
{
    public const string FileName = "settings.json";

    public static AppSettingsFile Default { get; } = new("data", "ru");

    public static AppSettingsFile Read(string appDir)
    {
        try
        {
            using var stream = File.OpenRead(Path.Combine(appDir, FileName));
            using var document = JsonDocument.Parse(stream);
            var root = document.RootElement;
            return new AppSettingsFile(
                Text(root, "data_dir") ?? Default.DataDir,
                Text(root, "language") ?? Default.Language,
                Text(root, "theme") ?? Default.Theme);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException
                                          or InvalidOperationException)
        {
            return Default;
        }
    }

    /// <summary>Абсолютный путь папки данных; относительный — от папки приложения (PORT-01).</summary>
    public string DataDirPath(string appDir) =>
        Path.IsPathRooted(DataDir) ? DataDir : Path.GetFullPath(Path.Combine(appDir, DataDir));

    private static string? Text(JsonElement root, string name) =>
        root.ValueKind == JsonValueKind.Object && root.TryGetProperty(name, out var value)
        && value.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(value.GetString())
            ? value.GetString()
            : null;
}
