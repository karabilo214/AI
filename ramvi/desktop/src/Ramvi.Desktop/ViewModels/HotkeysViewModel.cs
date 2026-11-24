using System.Collections.ObjectModel;
using Avalonia.Input;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Строка вкладки «Горячие клавиши»: действие и его клавиша.</summary>
public sealed class HotkeyRow : ObservableObject
{
    private readonly HotkeysViewModel _owner;
    private bool _capturing;

    public HotkeyRow(HotkeysViewModel owner, string action)
    {
        _owner = owner;
        Action = action;
        ChangeCommand = new Command(() =>
        {
            _owner.Begin(this);
            return Task.CompletedTask;
        });
    }

    public string Action { get; }

    public string Title => Localizer.Current[$"hotkeys.action.{Action}"];

    public string KeyText => IsCapturing ? Localizer.Current["hotkeys.waiting"] : Hotkeys.Current.Text(Action);

    public bool IsCapturing
    {
        get => _capturing;
        internal set
        {
            if (Set(ref _capturing, value))
            {
                Raise(nameof(KeyText));
            }
        }
    }

    public Command ChangeCommand { get; }

    internal void Refresh()
    {
        Raise(nameof(KeyText));
        Raise(nameof(Title));
    }
}

/// <summary>
/// B.13: вкладка «Горячие клавиши» — список действий, переназначение (следующее нажатие в окне),
/// проверка конфликтов, сброс к стандартным. Сохраняет <see cref="Hotkeys.Current"/>.
/// </summary>
public sealed class HotkeysViewModel : ObservableObject
{
    private HotkeyRow? _capturing;
    private string? _error;

    public HotkeysViewModel()
    {
        Rows = [.. Hotkeys.Actions.Select(a => new HotkeyRow(this, a))];
        ResetCommand = new Command(() =>
        {
            Cancel();
            Hotkeys.Current.Reset();
            Error = null;
            Refresh();
            return Task.CompletedTask;
        });
        CancelCommand = new Command(() =>
        {
            Cancel();
            return Task.CompletedTask;
        }, () => IsCapturing);
        Hotkeys.Current.PropertyChanged += (_, _) => Refresh();
    }

    private static Localizer L => Localizer.Current;

    public ObservableCollection<HotkeyRow> Rows { get; }

    public Command ResetCommand { get; }

    public Command CancelCommand { get; }

    public bool IsCapturing => _capturing is not null;

    public string? Error
    {
        get => _error;
        private set
        {
            if (Set(ref _error, value))
            {
                Raise(nameof(HasError));
            }
        }
    }

    public bool HasError => _error is not null;

    internal void Begin(HotkeyRow row)
    {
        Cancel();
        _capturing = row;
        row.IsCapturing = true;
        Error = null;
        Raise(nameof(IsCapturing));
        CancelCommand.Refresh();
    }

    public void Cancel()
    {
        if (_capturing is null) return;
        _capturing.IsCapturing = false;
        _capturing = null;
        Raise(nameof(IsCapturing));
        CancelCommand.Refresh();
    }

    /// <summary>
    /// Нажатие в окне во время назначения: true — нажатие забрано (не выполнять действие).
    /// Одна клавиша-модификатор — ждём остальное; недопустимая — подсказка, ждём дальше;
    /// конфликт — не сохраняется.
    /// </summary>
    public bool Capture(Key key, KeyModifiers modifiers)
    {
        if (_capturing is not { } row) return false;
        if (key is Key.LeftCtrl or Key.RightCtrl or Key.LeftShift or Key.RightShift or Key.LeftAlt
            or Key.RightAlt or Key.LWin or Key.RWin or Key.None)
        {
            return true;
        }

        var gesture = new KeyGesture(key, modifiers);
        var text = Hotkeys.Display(gesture);
        if (!Hotkeys.Allowed(gesture))
        {
            Error = L.Format("hotkeys.need_modifier", ("key", text));
            return true;
        }

        if (Hotkeys.Current.Conflict(row.Action, gesture) is { } other)
        {
            Error = L.Format("hotkeys.conflict", ("key", text), ("action", L[$"hotkeys.action.{other}"]));
            Cancel();
            return true;
        }

        Hotkeys.Current.Set(row.Action, gesture);
        Error = null;
        Cancel();
        Refresh();
        return true;
    }

    private void Refresh()
    {
        foreach (var row in Rows)
        {
            row.Refresh();
        }
    }
}
