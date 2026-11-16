using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json;

namespace Ramvi.Desktop.Localization;

/// <summary>
/// Тексты интерфейса по ключу из общего каталога `ramvi/i18n/&lt;язык&gt;.json` (3.14).
/// Нет перевода — показывается сам ключ, и ключ попадает в <see cref="Missing"/>.
/// </summary>
public sealed class Localizer
{
    public static readonly IReadOnlyList<string> Languages = ["ru", "en"];

    private readonly Dictionary<string, string> _texts;
    private readonly HashSet<string> _missing = [];

    public Localizer(string language, IReadOnlyDictionary<string, string> texts)
    {
        Language = language;
        _texts = new Dictionary<string, string>(texts, StringComparer.Ordinal);
    }

    /// <summary>Текущий каталог для разметки (<see cref="T"/>).</summary>
    public static Localizer Current { get; set; } = Load("ru");

    public string Language { get; }

    public IReadOnlyCollection<string> Missing => _missing;

    public IReadOnlyCollection<string> Keys => _texts.Keys;

    public string this[string key] => Format(key, (IReadOnlyDictionary<string, object?>?)null);

    public static Localizer Load(string language)
    {
        if (!Languages.Contains(language))
        {
            language = "en";
        }

        return new Localizer(language, ReadCatalog(language));
    }

    public static IReadOnlyDictionary<string, string> ReadCatalog(string language)
    {
        var assembly = typeof(Localizer).Assembly;
        using var stream = assembly.GetManifestResourceStream($"i18n.{language}.json")
            ?? throw new InvalidOperationException($"нет каталога переводов {language}");
        var values = JsonSerializer.Deserialize(stream, CatalogJson.Default.DictionaryStringString);
        return values ?? throw new InvalidOperationException($"пустой каталог {language}");
    }

    /// <summary>Текст по ключу с параметрами вида {name}.</summary>
    public string Format(string key, IReadOnlyDictionary<string, object?>? parameters)
    {
        if (!_texts.TryGetValue(key, out var template))
        {
            _missing.Add(key);
            return key;
        }

        if (parameters is null || parameters.Count == 0)
        {
            return template;
        }

        var result = new StringBuilder(template);
        foreach (var (name, value) in parameters)
        {
            result.Replace("{" + name + "}", ValueText(value));
        }

        return result.ToString();
    }

    public string Format(string key, params (string Name, object? Value)[] parameters) =>
        Format(key, parameters.ToDictionary(p => p.Name, p => p.Value, StringComparer.Ordinal));

    private static string ValueText(object? value) => value switch
    {
        null => "",
        JsonElement { ValueKind: JsonValueKind.String } element => element.GetString() ?? "",
        JsonElement element => element.GetRawText(),
        IFormattable formattable => formattable.ToString(null, CultureInfo.CurrentCulture),
        _ => value.ToString() ?? "",
    };
}

[System.Text.Json.Serialization.JsonSerializable(typeof(Dictionary<string, string>))]
internal sealed partial class CatalogJson : System.Text.Json.Serialization.JsonSerializerContext;

/// <summary>Разметка: <c>Text="{l:T chat.send}"</c>.</summary>
public sealed class T : Avalonia.Markup.Xaml.MarkupExtension
{
    public T()
    {
    }

    public T(string key) => Key = key;

    public string Key { get; set; } = "";

    public override object ProvideValue(IServiceProvider serviceProvider) => Localizer.Current[Key];
}
