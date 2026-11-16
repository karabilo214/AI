using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Ramvi.Desktop.Localization;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed partial class LocalizerTests
{
    private static string SourceDir([CallerFilePath] string path = "") =>
        Path.GetFullPath(Path.Combine(Path.GetDirectoryName(path)!, "../../src/Ramvi.Desktop"));

    [Fact]
    public void Ru_and_en_have_same_keys_and_parameters()
    {
        var ru = Localizer.ReadCatalog("ru");
        var en = Localizer.ReadCatalog("en");

        Assert.Equal(ru.Keys.Order(StringComparer.Ordinal), en.Keys.Order(StringComparer.Ordinal));
        foreach (var key in ru.Keys)
        {
            Assert.True(
                Params(ru[key]).SetEquals(Params(en[key])),
                $"{key}: параметры ru и en различаются");
            Assert.False(string.IsNullOrWhiteSpace(ru[key]), key);
            Assert.False(string.IsNullOrWhiteSpace(en[key]), key);
        }
    }

    [Fact]
    public void Every_key_used_in_shell_exists()
    {
        var catalog = Localizer.ReadCatalog("ru");
        var used = new HashSet<string>(StringComparer.Ordinal);
        foreach (var file in Directory.EnumerateFiles(SourceDir(), "*.*", SearchOption.AllDirectories)
                     .Where(f => f.EndsWith(".cs", StringComparison.Ordinal) || f.EndsWith(".axaml", StringComparison.Ordinal))
                     .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)))
        {
            foreach (Match match in UsedKey().Matches(File.ReadAllText(file)))
            {
                if (!match.Groups["key"].Value.EndsWith(".py", StringComparison.Ordinal)) // имя файла сервиса
                {
                    used.Add(match.Groups["key"].Value);
                }
            }
        }

        Assert.True(used.Count > 20, $"нашли мало ключей: {used.Count}");
        Assert.Equal([], used.Where(k => !catalog.ContainsKey(k)).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void Missing_key_is_shown_as_key_and_recorded()
    {
        var localizer = new Localizer("ru", new Dictionary<string, string> { ["a.b"] = "Файл {name}: {kb} КБ" });

        Assert.Equal("Файл x.txt: 3 КБ", localizer.Format("a.b", ("name", "x.txt"), ("kb", 3)));
        Assert.Equal("нет.ключа", localizer["нет.ключа"]);
        Assert.Equal(["нет.ключа"], localizer.Missing);
    }

    [Fact]
    public void Unknown_language_falls_back_to_english()
    {
        Assert.Equal("en", Localizer.Load("de").Language);
    }

    private static HashSet<string> Params(string text) =>
        [.. ParamName().Matches(text).Select(m => m.Value)];

    [GeneratedRegex(@"\{[a-z_]+\}")]
    private static partial Regex ParamName();

    // L["x.y"], L.Format("x.y", …), {l:T x.y}, а также ключи-константы вида "chat.too_many_files".
    [GeneratedRegex(@"(?:\{l:T |"")(?<key>(?:app|chat|status|ui|api|engine)\.[a-z_]+)(?:\}|"")")]
    private static partial Regex UsedKey();
}
