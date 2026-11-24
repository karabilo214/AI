using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using Avalonia.Input;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.Services;

/// <summary>
/// B.13: горячие клавиши окна — действие → клавиша. Стандартные: Ctrl+M — микрофон,
/// Esc — остановить (ответ, запись, озвучку) или закрыть настройки, Ctrl+N — новый чат,
/// Ctrl+, — настройки. Пользователь переназначает их во вкладке «Горячие клавиши»; файл
/// `desktop_hotkeys.json` в папке данных (настройка оболочки, как окно персонажа).
/// </summary>
public sealed class Hotkeys : INotifyPropertyChanged
{
    public const string FileName = "desktop_hotkeys.json";
    public const string Microphone = "microphone";
    public const string Stop = "stop";
    public const string NewChat = "new_chat";
    public const string Settings = "settings";

    public static readonly IReadOnlyList<string> Actions = [Microphone, Stop, NewChat, Settings];

    public static readonly IReadOnlyDictionary<string, KeyGesture> Defaults = new Dictionary<string, KeyGesture>
    {
        [Microphone] = new(Key.M, KeyModifiers.Control),
        [Stop] = new(Key.Escape),
        [NewChat] = new(Key.N, KeyModifiers.Control),
        [Settings] = new(Key.OemComma, KeyModifiers.Control),
    };

    /// <summary>
    /// Клавиши окна — один экземпляр (на него привязаны подсказки в разметке); папку данных
    /// задаёт <see cref="Use"/>. Без папки (тесты, окно до выбора папки) — стандартные, без файла.
    /// </summary>
    public static Hotkeys Current { get; } = new();

    private readonly Dictionary<string, KeyGesture> _keys = new(Defaults);
    private string? _path;

    private readonly List<(PropertyChangedEventHandler Handler, int Thread, SynchronizationContext? Context)> _handlers = [];

    private Hotkeys()
    {
    }

    /// <summary>
    /// Подписчик получает уведомление в своём потоке (потоке своего окна), кто бы ни менял
    /// клавиши, — как <see cref="LocalizedText"/>.
    /// </summary>
    public event PropertyChangedEventHandler? PropertyChanged
    {
        add
        {
            if (value is null) return;
            lock (_handlers) _handlers.Add((value, Environment.CurrentManagedThreadId, SynchronizationContext.Current));
        }
        remove
        {
            lock (_handlers) _handlers.RemoveAll(h => h.Handler == value);
        }
    }

    /// <summary>Сменился язык — подсказки заново (вызывает окно после смены языка).</summary>
    public void Refresh() => Raise();

    /// <summary>Прочитать клавиши из папки данных (null — стандартные); испорченное — стандартное.</summary>
    public void Use(string? folder)
    {
        _path = folder is null ? null : Path.Combine(folder, FileName);
        foreach (var (action, gesture) in Defaults)
        {
            _keys[action] = gesture;
        }

        foreach (var (action, text) in Read())
        {
            if (Defaults.ContainsKey(action) && Parse(text) is { } gesture && Allowed(gesture)
                && Conflict(action, gesture) is null)
            {
                _keys[action] = gesture;
            }
        }

        Raise();
    }

    private static Localizer L => Localizer.Current;

    // Подсказки с текущей клавишей (привязаны в разметке через x:Static).
    public string NewChatHint => L.Format("chat.new_hint", ("key", Text(NewChat)));

    public string StopHint => L.Format("chat.stop_hint", ("key", Text(Stop)));

    public string CloseHint => L.Format("settings.close_hint", ("key", Text(Stop)));

    public string VoiceStartHint => L.Format("voice.start_hint", ("key", Text(Microphone)));

    public string VoiceStopHint => L.Format("voice.stop_hint", ("key", Text(Microphone)));

    public string VoiceCancelHint => L.Format("voice.cancel_hint", ("key", Text(Stop)));

    private void Raise()
    {
        List<(PropertyChangedEventHandler Handler, int Thread, SynchronizationContext? Context)> targets;
        lock (_handlers) targets = [.. _handlers];
        var args = new PropertyChangedEventArgs(null);
        foreach (var (handler, thread, context) in targets)
        {
            if (thread == Environment.CurrentManagedThreadId || context is null) handler(this, args);
            else context.Post(_ => handler(this, args), null);
        }
    }

    public KeyGesture this[string action] => _keys[action];

    /// <summary>Текст клавиши для подсказок: «Ctrl+M», «Esc», «Ctrl+,».</summary>
    public string Text(string action) => Display(_keys[action]);

    public static string Display(KeyGesture gesture)
    {
        var parts = new List<string>();
        if (gesture.KeyModifiers.HasFlag(KeyModifiers.Control)) parts.Add("Ctrl");
        if (gesture.KeyModifiers.HasFlag(KeyModifiers.Meta)) parts.Add(OperatingSystem.IsMacOS() ? "⌘" : "Win");
        if (gesture.KeyModifiers.HasFlag(KeyModifiers.Alt)) parts.Add(OperatingSystem.IsMacOS() ? "Option" : "Alt");
        if (gesture.KeyModifiers.HasFlag(KeyModifiers.Shift)) parts.Add("Shift");
        parts.Add(gesture.Key switch
        {
            Key.Escape => "Esc",
            Key.OemComma => ",",
            Key.OemPeriod => ".",
            Key.OemMinus => "-",
            Key.OemPlus => "=",
            Key.Space => "Space",
            >= Key.D0 and <= Key.D9 => ((int)(gesture.Key - Key.D0)).ToString(System.Globalization.CultureInfo.InvariantCulture),
            _ => gesture.Key.ToString(),
        });
        return string.Join("+", parts);
    }

    /// <summary>
    /// Нажатие — это клавиша действия. Стандартный Ctrl у «новый чат» и «настройки» на Mac
    /// срабатывает и с ⌘ (как раньше); остальное — точное совпадение.
    /// </summary>
    public bool Matches(string action, Key key, KeyModifiers modifiers)
    {
        var gesture = _keys[action];
        if (gesture.Key != key) return false;
        if (gesture.KeyModifiers == modifiers) return true;
        return action is NewChat or Settings && gesture.KeyModifiers == KeyModifiers.Control
            && modifiers == KeyModifiers.Meta;
    }

    /// <summary>
    /// Клавиша без Ctrl/Alt/⌘ мешала бы печатать: разрешены только Esc и F1–F24.
    /// Отдельные клавиши-модификаторы — не клавиша.
    /// </summary>
    public static bool Allowed(KeyGesture gesture) =>
        gesture.Key is not (Key.None or Key.LeftCtrl or Key.RightCtrl or Key.LeftShift or Key.RightShift
            or Key.LeftAlt or Key.RightAlt or Key.LWin or Key.RWin)
        && ((gesture.KeyModifiers & (KeyModifiers.Control | KeyModifiers.Alt | KeyModifiers.Meta)) != 0
            || gesture.Key is Key.Escape or >= Key.F1 and <= Key.F24);

    /// <summary>Действие, у которого уже есть эта клавиша (кроме самого действия), или null.</summary>
    public string? Conflict(string action, KeyGesture gesture) =>
        _keys.FirstOrDefault(p => p.Key != action && Same(p.Value, gesture)).Key;

    /// <summary>Назначить клавишу; конфликт или недопустимая клавиша — не сохраняется (false).</summary>
    public bool Set(string action, KeyGesture gesture)
    {
        if (!Defaults.ContainsKey(action) || !Allowed(gesture) || Conflict(action, gesture) is not null)
        {
            return false;
        }

        _keys[action] = gesture;
        Save();
        return true;
    }

    public void Reset()
    {
        foreach (var (action, gesture) in Defaults)
        {
            _keys[action] = gesture;
        }

        Save();
    }

    private static bool Same(KeyGesture a, KeyGesture b) => a.Key == b.Key && a.KeyModifiers == b.KeyModifiers;

    private static KeyGesture? Parse(string text)
    {
        try
        {
            return KeyGesture.Parse(text);
        }
        catch (Exception error) when (error is ArgumentException or FormatException)
        {
            return null;
        }
    }

    private Dictionary<string, string> Read()
    {
        if (_path is null) return [];
        try
        {
            using var stream = File.OpenRead(_path);
            return JsonSerializer.Deserialize(stream, HotkeysJson.Default.DictionaryStringString) ?? [];
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException
                                          or NotSupportedException)
        {
            return [];
        }
    }

    private void Save()
    {
        Raise();
        if (_path is null) return;
        var data = _keys.ToDictionary(p => p.Key, p => p.Value.ToString());
        try
        {
            var temp = _path + ".tmp";
            File.WriteAllText(temp, JsonSerializer.Serialize(data, HotkeysJson.Default.DictionaryStringString));
            File.Move(temp, _path, overwrite: true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Не сохранилось — клавиша работает до перезапуска; окно не падает.
        }
    }
}

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(Dictionary<string, string>))]
internal sealed partial class HotkeysJson : JsonSerializerContext;
