using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Platform;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;

namespace Ramvi.Desktop.Views;

/// <summary>
/// 3.13/9.27: значок в трее. Через него возвращают управление без клика по прозрачному окну
/// (пропуск кликов), показывают скрытого персонажа, открывают спрятанное окно RAMVI,
/// останавливают всё и выходят полностью. Значок виден, пока персонаж вне окна RAMVI или
/// окно RAMVI спрятано; подсказка говорит, что микрофон включён.
/// </summary>
public sealed class TrayController : IDisposable
{
    private readonly MainViewModel _model;
    private readonly Window _main;
    private readonly TrayIcon _tray;
    private readonly NativeMenuItem _open, _hide, _clickThrough, _stop, _exit;

    public TrayController(MainViewModel model, Window main, Action exit)
    {
        _model = model; _main = main;
        _open = new NativeMenuItem { Command = new Command(() => { ShowMain(); return Task.CompletedTask; }) };
        _hide = new NativeMenuItem { Command = model.CharacterWindow.ToggleHiddenCommand };
        _clickThrough = new NativeMenuItem { Command = model.CharacterWindow.ToggleClickThroughCommand };
        _stop = new NativeMenuItem { Command = model.Character.DisconnectCommand };
        _exit = new NativeMenuItem { Command = new Command(() => { exit(); return Task.CompletedTask; }) };
        var menu = new NativeMenu();
        foreach (var item in new[] { _open, _hide, _clickThrough, _stop }) menu.Add(item);
        menu.Add(new NativeMenuItemSeparator());
        menu.Add(_exit);
        _tray = new TrayIcon
        {
            Icon = new WindowIcon(AssetLoader.Open(new Uri("avares://RAMVI/Assets/ramvi-emblem.png"))),
            Menu = menu,
            Command = _open.Command,
            ToolTipText = "RAMVI", // Avalonia 12 падает на пустой подсказке при создании значка
        };
        // Avalonia 12 создаёт значок в системе только после регистрации на приложении;
        // без неё объект TrayIcon есть, а значка в трее/строке меню нет.
        if (Application.Current is { } app) TrayIcon.SetIcons(app, [_tray]);
        model.CharacterWindow.PropertyChanged += Changed;
        model.Character.PropertyChanged += Changed;
        model.Character.Voice.PropertyChanged += Changed;
        main.PropertyChanged += (_, e) => { if (e.Property == Visual.IsVisibleProperty) Update(); };
        Update();
    }

    private void Changed(object? sender, PropertyChangedEventArgs e) => Update();

    private void ShowMain()
    {
        if (_main is MainWindow main) { main.ShowFromTray(); return; }
        _main.Show();
        if (_main.WindowState == WindowState.Minimized) _main.WindowState = WindowState.Normal;
        _main.Activate();
    }

    private void Update()
    {
        var text = Localizer.Current;
        var window = _model.CharacterWindow;
        _open.Header = text["character.tray.open"];
        _hide.Header = window.HideText;
        _hide.IsVisible = window.IsFloating;
        _clickThrough.Header = window.ClickThroughText;
        _clickThrough.IsVisible = window.IsOverlay && !window.Hidden;
        _stop.Header = text["character.disconnect"];
        _stop.IsVisible = _model.Character.CanStopAll;
        _exit.Header = text["character.tray.exit"];
        _tray.ToolTipText = _model.Character.Voice.IsActive ? text["character.tray.listening"] : "RAMVI";
        _tray.IsVisible = window.IsFloating || !_main.IsVisible;
    }

    public void Dispose()
    {
        _model.CharacterWindow.PropertyChanged -= Changed;
        _model.Character.PropertyChanged -= Changed;
        _model.Character.Voice.PropertyChanged -= Changed;
        _tray.Dispose();
    }
}
