using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Threading;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;

namespace Ramvi.Desktop.Views;

/// <summary>
/// 9.26/9.27: окно персонажа вне RAMVI — отдельное окно (Companion) или прозрачный оверлей
/// поверх рабочего стола. Окно — только адаптер показа: закрытие кнопкой окна скрывает аватар,
/// а не выключает модуль; полный выход — из трея или главного окна.
/// </summary>
public sealed partial class CharacterFloatingWindow : Window
{
    private CharacterWindowViewModel? _window;
    private bool _closingByHost, _opened;

    public CharacterFloatingWindow()
    {
        InitializeComponent();
        // Перетаскивать можно и за полосу кнопок, и за самого персонажа.
        Toolbar.PointerPressed += Drag;
        FloatingCharacter.PointerPressed += Drag;
        // Колесо над персонажем меняет его размер (вверх — больше).
        FloatingCharacter.PointerWheelChanged += (_, e) =>
        {
            if (_window is null || e.Delta.Y == 0) return;
            _window.Size += e.Delta.Y > 0 ? 0.1 : -0.1;
            e.Handled = true;
        };
        PositionChanged += (_, _) => { Remember(); FaceCenter(); };
        Resized += (_, _) => Remember();
        PointerEntered += (_, _) => { if (_window?.IsOverlay == true) Toolbar.Opacity = 1; };
        PointerExited += (_, _) => { if (_window?.IsOverlay == true) Toolbar.Opacity = 0.15; };
        // Место ставится после открытия: до показа macOS пересчитывает Y по ещё не заданной
        // высоте окна, и персонаж оказывался вверху экрана.
        Opened += (_, _) => { ApplyPosition(); _opened = true; KeepOnScreen(); Remember(); ApplyClickThrough(); FaceCenter(); Opacity = _window?.Opacity ?? 1; };
        Screens.Changed += (_, _) => Dispatcher.UIThread.Post(KeepOnScreen);
        Closing += (_, e) =>
        {
            if (_closingByHost || _window is null) return;
            // Крестик окна — «скрыть персонажа»: разговор, память и модуль продолжают жить.
            e.Cancel = true;
            _window.Hidden = true;
        };
    }

    /// <summary>
    /// Персонаж смотрит к центру экрана: картинка нарисована смотрящей вправо, поэтому окно
    /// в правой половине экрана отражается. Микрофон — с той же стороны, куда он смотрит.
    /// </summary>
    internal void FaceCenter()
    {
        var screen = Screens.ScreenFromWindow(this) ?? Screens.Primary;
        if (screen is null) return;
        var center = Position.X + PixelSize.FromSize(Bounds.Size, screen.Scaling).Width / 2;
        var right = FacesLeft(center, screen.WorkingArea);
        FloatingCharacter.Mirrored = right;
        FloatingMicrophone.HorizontalAlignment = right ? Avalonia.Layout.HorizontalAlignment.Left : Avalonia.Layout.HorizontalAlignment.Right;
        FloatingSpeaker.HorizontalAlignment = right ? Avalonia.Layout.HorizontalAlignment.Right : Avalonia.Layout.HorizontalAlignment.Left;
    }

    /// <summary>Окно правее середины рабочей области — смотреть влево.</summary>
    internal static bool FacesLeft(int windowCenterX, PixelRect area) => windowCenterX > area.X + area.Width / 2;

    private void Drag(object? sender, PointerPressedEventArgs e)
    {
        if (e.GetCurrentPoint(this).Properties.IsLeftButtonPressed && e.Source is not Button) BeginMoveDrag(e);
    }

    /// <summary>
    /// Место по умолчанию — правый нижний угол рабочей области основного экрана (она уже без
    /// панели задач и Dock), с запасом снизу, чтобы не закрыть трей, часы и уведомления.
    /// </summary>
    internal static PixelPoint DefaultPosition(PixelRect area, PixelSize size, double scale)
    {
        var right = (int)(24 * scale);
        var bottom = Math.Max((int)(96 * scale), area.Height / 10);
        return new PixelPoint(
            Math.Max(area.X, area.Right - size.Width - right),
            Math.Max(area.Y, area.Bottom - size.Height - bottom));
    }

    private void PlaceDefault()
    {
        var screen = Screens.Primary ?? (Screens.All.Count > 0 ? Screens.All[0] : null);
        if (screen is null) { WindowStartupLocation = WindowStartupLocation.CenterScreen; return; }
        WindowStartupLocation = WindowStartupLocation.Manual;
        var size = PixelSize.FromSize(new Size(Width, Height), screen.Scaling);
        Position = DefaultPosition(screen.WorkingArea, size, screen.Scaling);
    }

    /// <summary>Настроить окно под режим. Вызывает хозяин окна при каждом показе.</summary>
    public void Attach(MainViewModel model)
    {
        DataContext = model;
        if (_window is not null) _window.PropertyChanged -= WindowChanged;
        _window = model.CharacterWindow;
        _window.PropertyChanged += WindowChanged;
        ApplyMode();
    }

    /// <summary>Хозяин закрывает окно по-настоящему (возврат в RAMVI или выход).</summary>
    public void CloseByHost()
    {
        _closingByHost = true;
        if (_window is not null) _window.PropertyChanged -= WindowChanged;
        Close();
    }

    private void WindowChanged(object? sender, PropertyChangedEventArgs e)
    {
        switch (e.PropertyName)
        {
            case nameof(CharacterWindowViewModel.Mode): ApplyMode(); break;
            case nameof(CharacterWindowViewModel.Opacity): Opacity = _window!.Opacity; break;
            case nameof(CharacterWindowViewModel.Size): Resize(); break;
            case nameof(CharacterWindowViewModel.ClickThrough): ApplyClickThrough(); break;
        }
    }

    private void ApplyMode()
    {
        if (_window is null) return;
        var state = _window.State;
        var overlay = _window.IsOverlay;
        // Оверлей не забирает фокус у приложения, с которым работают, и не виден в панели задач.
        WindowDecorations = overlay ? WindowDecorations.None : WindowDecorations.Full;
        TransparencyLevelHint = overlay ? [WindowTransparencyLevel.Transparent] : [];
        Background = overlay ? Brushes.Transparent : null;
        ShowInTaskbar = !overlay;
        ShowActivated = !overlay;
        Toolbar.Opacity = overlay ? 0.15 : 1;
        Opacity = _opened ? state.Opacity : 0; // невидимо, пока не встало на место
        // В отдельном окне под персонажем ещё и диалог: окно не меньше, чем нужно обоим.
        Width = overlay ? state.Width : Math.Max(state.Width, 380);
        Height = overlay ? state.Height : Math.Max(state.Height, 680);
        WindowStartupLocation = WindowStartupLocation.Manual;
        if (_opened) { ApplyPosition(); Remember(); }
        if (!overlay && _window.ClickThrough) _window.ClickThrough = false;
    }

    /// <summary>Новый размер; правый нижний угол остаётся на месте — персонаж не уезжает к трею.</summary>
    private void Resize()
    {
        if (_window is null) return;
        var state = _window.State;
        var scale = (Screens.ScreenFromWindow(this) ?? Screens.Primary)?.Scaling ?? 1;
        var old = PixelSize.FromSize(Bounds.Size, scale);
        Width = _window.IsOverlay ? state.Width : Math.Max(state.Width, 380);
        Height = _window.IsOverlay ? state.Height : Math.Max(state.Height, 680);
        if (!_opened) return;
        var next = PixelSize.FromSize(new Size(Width, Height), scale);
        Position = new PixelPoint(Position.X + old.Width - next.Width, Position.Y + old.Height - next.Height);
        KeepOnScreen();
        FaceCenter();
    }

    private void ApplyPosition()
    {
        var state = _window?.State;
        if (state?.X is { } x && state.Y is { } y) Position = new PixelPoint((int)x, (int)y);
        else PlaceDefault();
    }

    private void ApplyClickThrough()
    {
        if (_window is null || !IsVisible) return;
        if (!WindowPlatform.SetClickThrough(this, _window.ClickThrough) && _window.ClickThrough) _window.ClickThroughFailed();
    }

    private void Remember()
    {
        // До открытия окно стоит на временном месте — его не запоминать.
        if (_window is null || !_opened || !IsVisible || WindowState != WindowState.Normal) return;
        // Запоминается только место: размер задают ползунок и колесо (свойство Size).
        var state = _window.State;
        _window.RememberBounds(Position.X, Position.Y, state.Width, state.Height);
    }

    /// <summary>
    /// Сменили монитор или масштаб, и окно оказалось за пределами всех экранов: вернуть его
    /// на основной экран, иначе невидимое окно было бы невозможно достать.
    /// </summary>
    private void KeepOnScreen()
    {
        if (_window is null || Screens.All.Count == 0) return;
        var scale = (Screens.ScreenFromWindow(this) ?? Screens.Primary ?? Screens.All[0]).Scaling;
        var rect = new PixelRect(Position, PixelSize.FromSize(Bounds.Size, scale));
        var visible = Screens.All.Any(s => s.WorkingArea.Intersect(rect) is { Width: >= 48, Height: >= 48 });
        if (visible) return;
        var screen = Screens.Primary ?? Screens.All[0];
        Position = DefaultPosition(screen.WorkingArea, PixelSize.FromSize(Bounds.Size, screen.Scaling), screen.Scaling);
        _window.ForgetPosition();
    }

    /// <summary>Есть ли окно на каком-нибудь экране (для проверок 9.27).</summary>
    internal bool IsOnScreen()
    {
        var scale = (Screens.ScreenFromWindow(this) ?? Screens.Primary)?.Scaling ?? 1;
        var rect = new PixelRect(Position, PixelSize.FromSize(Bounds.Size, scale));
        return Screens.All.Count == 0 || Screens.All.Any(s => s.WorkingArea.Intersect(rect) is { Width: >= 48, Height: >= 48 });
    }

    internal void KeepOnScreenForTest() => KeepOnScreen();
}
