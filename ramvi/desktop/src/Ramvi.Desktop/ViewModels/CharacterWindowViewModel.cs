using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;

namespace Ramvi.Desktop.ViewModels;

/// <summary>
/// 3.13/9.26: где показан персонаж — в окне RAMVI (Embedded), в отдельном окне (Companion)
/// или поверх рабочего стола (Overlay). Меняется только представление: тот же
/// <see cref="CharacterViewModel"/>, тот же модуль в сервисе, тот же ответ и один звук.
/// Главное окно, видимость аватара и жизнь модуля — три отдельных состояния.
/// </summary>
public sealed class CharacterWindowViewModel : ObservableObject
{
    private readonly CharacterWindowStore? _store;
    private readonly CharacterViewModel _character;
    private readonly Func<Task> _openCharacter;
    private CharacterWindowState _state;
    private bool _hidden, _clickThrough;

    public CharacterWindowViewModel(CharacterViewModel character, Func<Task> openCharacter, CharacterWindowStore? store = null)
    {
        _character = character;
        _openCharacter = openCharacter;
        _store = store;
        _state = store?.Read() ?? new();
        ShowCommand = new Command(p => SetModeAsync(p as string ?? "embedded"));
        ToggleHiddenCommand = new Command(() => { Hidden = !Hidden; return Task.CompletedTask; }, () => IsFloating);
        ToggleClickThroughCommand = new Command(() => { ClickThrough = !ClickThrough; return Task.CompletedTask; }, () => CanClickThrough);
        _character.Voice.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(CharacterVoiceViewModel.IsActive)) Raise(nameof(ListeningWhileHidden));
        };
    }

    public Command ShowCommand { get; }
    public Command ToggleHiddenCommand { get; }
    public Command ToggleClickThroughCommand { get; }

    public CharacterWindowState State => _state;

    /// <summary>embedded, companion или overlay.</summary>
    public string Mode
    {
        get => _state.Mode;
        set => _ = SetModeAsync(value);
    }

    public bool IsEmbedded => Mode == "embedded";
    public bool IsCompanion => Mode == "companion";
    public bool IsOverlay => Mode == "overlay";
    public bool IsFloating => !IsEmbedded;

    /// <summary>Окно персонажа вне RAMVI сейчас на экране (только если модуль Character есть).</summary>
    public bool WindowVisible => IsFloating && !Hidden && Available;

    /// <summary>Модуль Character установлен; без него окно вне RAMVI не открывается пустым.</summary>
    public bool Available
    {
        get => _available;
        set { if (Set(ref _available, value)) Raise(nameof(WindowVisible)); }
    }
    private bool _available;

    /// <summary>Аватар скрыт пользователем; модуль и разговор при этом продолжают жить.</summary>
    public bool Hidden
    {
        get => _hidden;
        set
        {
            if (!Set(ref _hidden, value)) return;
            if (value) ClickThrough = false;
            Raise(nameof(WindowVisible)); Raise(nameof(ListeningWhileHidden)); Raise(nameof(HideText));
            ToggleClickThroughCommand.Refresh();
        }
    }

    /// <summary>
    /// Скрытие не превращает микрофон в невидимое прослушивание: пока он включён, это видно
    /// в окне RAMVI и в меню трея, и там же есть «Остановить всё».
    /// </summary>
    public bool ListeningWhileHidden => (Hidden || IsEmbedded) && _character.Voice.IsActive;

    /// <summary>Работать после закрытия окна RAMVI — только по явному выбору и только вне окна.</summary>
    public bool Background
    {
        get => _state.Background;
        set { if (_state.Background != value) Save(_state with { Background = value }, nameof(Background), nameof(KeepsRunningWithoutMainWindow)); }
    }

    public bool KeepsRunningWithoutMainWindow => IsFloating && Background;

    public double Opacity
    {
        get => _state.Opacity;
        set
        {
            var next = Math.Clamp(value, 0.3, 1);
            if (Math.Abs(next - _state.Opacity) > 0.001) Save(_state with { Opacity = next }, nameof(Opacity));
        }
    }

    /// <summary>Базовый размер окна персонажа на экране (масштаб 1).</summary>
    public const double BaseWidth = 300, BaseHeight = 380, MinSize = 0.5, MaxSize = 2.5;

    /// <summary>9.26: размер персонажа вне окна RAMVI — ползунок и колесо мыши над ним.</summary>
    public double Size
    {
        get => Math.Round(_state.Width / BaseWidth, 2);
        set
        {
            var next = Math.Clamp(value, MinSize, MaxSize);
            if (Math.Abs(next - Size) < 0.005) return;
            Save(_state with { Width = Math.Round(BaseWidth * next), Height = Math.Round(BaseHeight * next) }, nameof(Size));
        }
    }

    public bool CanClickThrough => IsOverlay && !Hidden && WindowPlatform.SupportsClickThrough;

    /// <summary>
    /// Временный пропуск кликов: окно не мешает работать под ним. Не сохраняется — после
    /// перезапуска окно снова управляемо; вернуть управление можно из трея и окна RAMVI.
    /// </summary>
    public bool ClickThrough
    {
        get => _clickThrough;
        set
        {
            if (value && !CanClickThrough) value = false;
            if (Set(ref _clickThrough, value)) Raise(nameof(ClickThroughText));
        }
    }

    public string ClickThroughText => Localizer.Current[ClickThrough ? "character.window.click_through_off" : "character.window.click_through_on"];
    public string HideText => Localizer.Current[Hidden ? "character.window.show" : "character.window.hide"];

    /// <summary>Окно сообщило, что ОС не включила пропуск кликов.</summary>
    public void ClickThroughFailed() { _clickThrough = false; Raise(nameof(ClickThrough)); Raise(nameof(ClickThroughText)); }

    /// <summary>Положение и размер окна вне RAMVI (сохраняются при перемещении).</summary>
    public void RememberBounds(double x, double y, double width, double height) =>
        Save(_state with { X = x, Y = y, Width = width, Height = height });

    /// <summary>Окно ушло за пределы всех мониторов (сменили монитор/DPI): вернуть на основной.</summary>
    public void ForgetPosition() => Save(_state with { X = null, Y = null });

    /// <summary>Запуск: персонаж был вне окна RAMVI — подключить модуль, не открывая раздел.</summary>
    public Task RestoreAsync()
    {
        Available = true;
        return IsFloating ? _openCharacter() : Task.CompletedTask;
    }

    private async Task SetModeAsync(string mode)
    {
        if (!CharacterWindowState.Modes.Contains(mode) || mode == Mode) return;
        Save(_state with { Mode = mode }, nameof(Mode), nameof(IsEmbedded), nameof(IsCompanion), nameof(IsOverlay),
            nameof(IsFloating), nameof(KeepsRunningWithoutMainWindow), nameof(ListeningWhileHidden));
        _hidden = false; _clickThrough = false;
        Raise(nameof(Hidden)); Raise(nameof(WindowVisible)); Raise(nameof(ClickThrough)); Raise(nameof(ClickThroughText));
        Raise(nameof(CanClickThrough)); Raise(nameof(HideText));
        ToggleHiddenCommand.Refresh(); ToggleClickThroughCommand.Refresh();
        // Модуль подключается один раз; смена представления не перечитывает его состояние,
        // иначе недописанный текст и текущий ответ сбросились бы.
        if (IsFloating && !_character.Initialized) await _openCharacter().ConfigureAwait(true);
    }

    private void Save(CharacterWindowState next, params string[] changed)
    {
        _state = next.Normalized();
        _store?.Write(_state);
        foreach (var name in changed) Raise(name);
    }
}
