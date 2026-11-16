using System.Text.Json;
using System.Text.Json.Nodes;

namespace Ramvi.Desktop.Services;

/// <summary>
/// Где лежат данные переносимого комплекта (PORT-01, пункт 3.9).
/// Обычно — `data_dir` из `settings.json` папки приложения. Если в папку приложения писать
/// нельзя (USB только на чтение), явно выбранная папка запоминается в профиле пользователя:
/// `%LOCALAPPDATA%/RAMVI/locations.json` — «папка приложения → папка данных».
/// </summary>
public static class AppLocations
{
    /// <summary>Файл соответствий в профиле; тесты подменяют его.</summary>
    public static string OverrideFile { get; set; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "RAMVI", "locations.json");

    public static string DataDir(string appDir)
    {
        if (!IsWritable(appDir) && ReadOverride(appDir) is { } chosen)
        {
            return chosen;
        }

        return AppSettingsFile.Read(appDir).DataDirPath(appDir);
    }

    /// <summary>
    /// Запомнить явно выбранную папку данных (сервис ещё не запущен, поэтому пишет оболочка):
    /// в `settings.json`, если папка приложения доступна на запись (внутри неё — относительный
    /// путь), иначе — в профиль пользователя.
    /// </summary>
    public static void SaveDataDir(string appDir, string dataDir)
    {
        var full = Path.GetFullPath(dataDir);
        if (IsWritable(appDir))
        {
            var path = Path.Combine(appDir, AppSettingsFile.FileName);
            var json = ReadObject(path);
            var relative = Path.GetRelativePath(Path.GetFullPath(appDir), full);
            json["data_dir"] = relative.StartsWith("..", StringComparison.Ordinal) || Path.IsPathRooted(relative)
                ? full
                : relative.Replace('\\', '/');
            WriteAtomic(path, json);
            return;
        }

        var overrides = ReadObject(OverrideFile);
        overrides[Key(appDir)] = full;
        WriteAtomic(OverrideFile, overrides);
    }

    public static string? ReadOverride(string appDir) =>
        ReadObject(OverrideFile)[Key(appDir)] is JsonValue value && value.TryGetValue(out string? text) ? text : null;

    /// <summary>Можно ли создать файл в папке (папка создаётся, если её нет).</summary>
    public static bool IsWritable(string dir)
    {
        try
        {
            Directory.CreateDirectory(dir);
            var probe = Path.Combine(dir, $".write-test-{Environment.ProcessId}");
            File.WriteAllBytes(probe, []);
            File.Delete(probe);
            return true;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    private static string Key(string appDir) =>
        Path.TrimEndingDirectorySeparator(Path.GetFullPath(appDir));

    private static JsonObject ReadObject(string path)
    {
        try
        {
            return JsonNode.Parse(File.ReadAllText(path)) as JsonObject ?? [];
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException)
        {
            return [];
        }
    }

    private static void WriteAtomic(string path, JsonObject value)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temp = path + $".{Environment.ProcessId}.tmp";
        File.WriteAllText(temp, value.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        File.Move(temp, path, overwrite: true);
    }
}
