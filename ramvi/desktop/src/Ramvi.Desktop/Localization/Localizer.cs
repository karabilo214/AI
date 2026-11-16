using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Avalonia;

namespace Ramvi.Desktop.Localization;

/// <summary>
/// Тексты интерфейса по ключу из общего каталога `ramvi/i18n/&lt;язык&gt;.json` (3.14).
/// Нет перевода — показывается сам ключ, и ключ попадает в <see cref="Missing"/>.
/// Переводы модулей приходят от сервиса (<see cref="SetModuleTexts"/>). Смена
/// <see cref="Current"/> сразу меняет все тексты разметки (<see cref="T"/>) — без перезапуска.
/// </summary>
public sealed class Localizer
{
    public static readonly IReadOnlyList<string> Languages = ["ru", "en"];

    private static Localizer _current = Load("ru");
    private static IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> _moduleTexts =
        new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal);

    private readonly Dictionary<string, string> _texts;
    private readonly HashSet<string> _missing = [];

    public Localizer(string language, IReadOnlyDictionary<string, string> texts)
    {
        Language = language;
        _texts = new Dictionary<string, string>(texts, StringComparer.Ordinal);
    }

    /// <summary>Сменился язык или переводы модулей: тексты нужно показать заново.</summary>
    public static event EventHandler? Changed;

    /// <summary>Текущий каталог для разметки (<see cref="T"/>).</summary>
    public static Localizer Current
    {
        get => _current;
        set
        {
            _current = value;
            Changed?.Invoke(null, EventArgs.Empty);
        }
    }

    /// <summary>Переводы модулей (язык → ключ → текст) от сервиса; текущий каталог пересобирается.</summary>
    public static void SetModuleTexts(IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> texts)
    {
        _moduleTexts = texts;
        Current = Load(_current.Language);
    }

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

        var texts = new Dictionary<string, string>(ReadCatalog(language), StringComparer.Ordinal);
        if (_moduleTexts is not null && _moduleTexts.TryGetValue(language, out var module))
        {
            foreach (var (key, text) in module)
            {
                texts.TryAdd(key, text); // ключ ядра модуль не переопределяет (это проверяет и сервис)
            }
        }

        return new Localizer(language, texts);
    }

    public static IReadOnlyDictionary<string, string> ReadCatalog(string language)
    {
        var assembly = typeof(Localizer).Assembly;
        using var stream = assembly.GetManifestResourceStream($"i18n.{language}.json")
            ?? throw new InvalidOperationException($"no translation catalog {language}");
        var values = JsonSerializer.Deserialize(stream, CatalogJson.Default.DictionaryStringString);
        return values ?? throw new InvalidOperationException($"empty translation catalog {language}");
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

    /// <summary>Значение параметра; вложенное сообщение <c>{code, params}</c> переводится тем же языком.</summary>
    private string ValueText(object? value) => value switch
    {
        null => "",
        JsonElement { ValueKind: JsonValueKind.String } element => element.GetString() ?? "",
        JsonElement { ValueKind: JsonValueKind.Object } element when element.TryGetProperty("code", out var code)
            && code.ValueKind == JsonValueKind.String => Format(code.GetString()!, Nested(element)),
        JsonElement { ValueKind: JsonValueKind.Array } element =>
            string.Join("; ", element.EnumerateArray().Select(item => ValueText(item))),
        JsonElement element => element.GetRawText(),
        IFormattable formattable => formattable.ToString(null, CultureInfo.CurrentCulture),
        _ => value.ToString() ?? "",
    };

    private static Dictionary<string, object?> Nested(JsonElement message)
    {
        var result = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (message.TryGetProperty("params", out var values) && values.ValueKind == JsonValueKind.Object)
        {
            foreach (var item in values.EnumerateObject())
            {
                result[item.Name] = item.Value;
            }
        }

        return result;
    }
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

    /// <summary>Привязка, а не готовая строка: при смене языка текст обновляется сам.</summary>
    public override object ProvideValue(IServiceProvider serviceProvider) => new LocalizedText(Key).ToBinding();
}

/// <summary>
/// Текст по ключу как поток: текущее значение и новое при каждой смене языка.
/// Подписки слабые: закрытое окно не удерживается в памяти статическим событием.
/// Новое значение приходит в поток, где элемент подписался (поток его окна), кто бы ни сменил язык.
/// </summary>
public sealed class LocalizedText(string key) : IObservable<string>
{
    private static readonly List<Entry> Observers = [];

    static LocalizedText() => Localizer.Changed += (_, _) => Publish();

    public IDisposable Subscribe(IObserver<string> observer)
    {
        var entry = new Entry(
            new WeakReference<IObserver<string>>(observer), key,
            Environment.CurrentManagedThreadId, SynchronizationContext.Current);
        lock (Observers)
        {
            Observers.Add(entry);
        }

        observer.OnNext(Localizer.Current[key]);
        return new Unsubscribe(entry);
    }

    private static void Publish()
    {
        List<(IObserver<string> Observer, string Text, Entry Entry)> targets = [];
        lock (Observers)
        {
            Observers.RemoveAll(e => !e.Observer.TryGetTarget(out _));
            foreach (var entry in Observers)
            {
                if (entry.Observer.TryGetTarget(out var observer))
                {
                    targets.Add((observer, Localizer.Current[entry.Key], entry));
                }
            }
        }

        foreach (var (observer, text, entry) in targets)
        {
            if (entry.Context is null || entry.ThreadId == Environment.CurrentManagedThreadId)
            {
                observer.OnNext(text);
            }
            else
            {
                entry.Context.Post(_ => observer.OnNext(text), null);
            }
        }
    }

    private sealed record Entry(
        WeakReference<IObserver<string>> Observer, string Key, int ThreadId, SynchronizationContext? Context);

    private sealed class Unsubscribe(Entry entry) : IDisposable
    {
        public void Dispose()
        {
            lock (Observers)
            {
                Observers.Remove(entry);
            }
        }
    }
}
