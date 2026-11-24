using Avalonia;
using System.Collections.Specialized;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;

namespace Ramvi.Desktop.Views;

public sealed partial class MainWindow : Window, IDisposable
{
    /// <summary>Расстояние до низа, при котором окно считается «внизу» и следует за ответом.</summary>
    private const double FollowSlack = 48;

    /// <summary>Следовать за новым текстом; false — пользователь прокрутил вверх и читает.</summary>
    private bool _follow = true;

    /// <summary>
    /// B.3: размер окна, чтобы окно с рамкой целиком влезло в рабочую область экрана (без панели
    /// задач). Не влезает — уменьшается до рабочей области, но не меньше минимального.
    /// </summary>
    public static Size FitInto(Size client, Size frame, Size working, Size minimum)
    {
        var extraW = Math.Max(0, frame.Width - client.Width);
        var extraH = Math.Max(0, frame.Height - client.Height);
        return new Size(
            Math.Max(minimum.Width, Math.Min(client.Width, working.Width - extraW)),
            Math.Max(minimum.Height, Math.Min(client.Height, working.Height - extraH)));
    }

    private void FitToScreen()
    {
        var screen = Screens.ScreenFromWindow(this) ?? Screens.Primary;
        if (screen is null) return;
        var scale = screen.Scaling;
        var area = screen.WorkingArea;
        var working = new Size(area.Width / scale, area.Height / scale);
        var client = new Size(Width, Height);
        var frame = FrameSize ?? client;
        var fitted = FitInto(client, frame, working, new Size(MinWidth, MinHeight));
        if (fitted == client && Position.X >= area.X && Position.Y >= area.Y
            && Position.Y + frame.Height * scale <= area.Bottom) return;
        Width = fitted.Width;
        Height = fitted.Height;
        var outer = new Size(
            fitted.Width + frame.Width - client.Width, fitted.Height + frame.Height - client.Height);
        Position = new PixelPoint(
            area.X + Math.Max(0, (int)((area.Width - outer.Width * scale) / 2)),
            area.Y + Math.Max(0, (int)((area.Height - outer.Height * scale) / 2)));
    }

    public MainWindow()
    {
        InitializeComponent();
        TitleCaption.IsVisible = !OperatingSystem.IsWindows(); // B.2: одно название в полосе заголовка
        Closed += (_, _) => { Dispose(); Model?.Dispose(); };
        Closing += (_, e) =>
        {
            // 3.13: закрыть окно RAMVI ≠ выйти. Если пользователь сам включил работу в фоне и
            // персонаж вне окна, окно только прячется; полный выход — из трея или при выключении ОС.
            if (e.CloseReason is WindowCloseReason.WindowClosing or WindowCloseReason.Undefined
                && Model?.CharacterWindow.KeepsRunningWithoutMainWindow == true)
            {
                e.Cancel = true;
                _hiddenAt = Position; // при возврате из трея окно встаёт на прежнее место
                Hide();
            }
        };
        Opened += (_, _) => FitToScreen();
        MessagesScroll.ScrollChanged += OnMessagesScrolled;
        InputBox.AddHandler(KeyDownEvent, OnInputKeyDown, RoutingStrategies.Tunnel);
        AddHandler(KeyDownEvent, OnWindowKeyDown, RoutingStrategies.Tunnel);
        ChatList.KeyDown += OnChatListKeyDown;
        AddHandler(DragDrop.DropEvent, OnDrop);
        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        DataContextChanged += (_, _) =>
        {
            if (DataContext is MainViewModel model)
            {
                _characterWindow?.Dispose();
                _characterWindow = new CharacterWindowHost(model);
                model.Messages.CollectionChanged += OnMessagesChanged;
                model.Settings.PropertyChanged += (_, e) =>
                {
                    if (e.PropertyName == nameof(SettingsViewModel.IsOpen))
                    {
                        FocusLater(model.Settings.IsOpen ? SettingsClose : InputBox);
                    }
                };
                model.Onboarding.PropertyChanged += (_, e) =>
                {
                    if (e.PropertyName is nameof(OnboardingViewModel.IsOpen) or nameof(OnboardingViewModel.Step))
                    {
                        FocusLater(!model.Onboarding.IsOpen ? InputBox
                            : model.Onboarding.Step == 1 ? NextButton
                            : StartButton.IsEffectivelyEnabled ? StartButton : SkipButton2);
                    }
                };
            }
        };
    }

    private MainViewModel? Model => DataContext as MainViewModel;
    private CharacterWindowHost? _characterWindow;
    private PixelPoint? _hiddenAt;

    /// <summary>Показать спрятанное окно RAMVI на прежнем месте (трей, «Открыть RAMVI»).</summary>
    public void ShowFromTray()
    {
        Show();
        if (_hiddenAt is { } at) Position = at;
        if (WindowState == WindowState.Minimized) WindowState = WindowState.Normal;
        Activate();
    }

    public void Dispose()
    {
        _characterWindow?.Dispose();
        _characterWindow = null;
    }

    /// <summary>9.26: окно персонажа вне RAMVI (для проверок).</summary>
    internal CharacterWindowHost? CharacterWindowHost => _characterWindow;

    public async Task CopyAsync(string text)
    {
        if (Clipboard is { } clipboard)
        {
            await clipboard.SetTextAsync(text).ConfigureAwait(true);
        }
    }

    /// <summary>
    /// Выбрать файл модели; путь на диске или null (отмена, не локальный файл).
    /// Без фильтра по расширению: у Ollama модели лежат в `blobs` как `sha256-…` без `.gguf`,
    /// и фильтр их скрывал. Формат проверяет сервис по заголовку файла.
    /// </summary>
    public async Task<string?> PickModelFileAsync()
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = Localization.Localizer.Current["onboarding.pick_title"],
            AllowMultiple = false,
        }).ConfigureAwait(true);
        return files.Count == 1 ? files[0].TryGetLocalPath() : null;
    }

    /// <summary>«Прикрепить файл»: выбранные файлы (пустой список — отмена).</summary>
    public async Task<IReadOnlyList<DroppedFile>> PickFilesAsync()
    {
        var picked = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = Localization.Localizer.Current["chat.attach"],
            AllowMultiple = true,
        }).ConfigureAwait(true);
        return await ToDroppedAsync(picked).ConfigureAwait(true);
    }

    /// <summary>Выбрать папку (настройки); путь на диске или null.</summary>
    public async Task<string?> PickFolderAsync(string? start)
    {
        var options = new FolderPickerOpenOptions { AllowMultiple = false };
        if (!string.IsNullOrEmpty(start) && Directory.Exists(start))
        {
            options.SuggestedStartLocation = await StorageProvider.TryGetFolderFromPathAsync(start).ConfigureAwait(true);
        }

        var folders = await StorageProvider.OpenFolderPickerAsync(options).ConfigureAwait(true);
        return folders.Count == 1 ? folders[0].TryGetLocalPath() : null;
    }

    /// <summary>Открыть папку в файловом менеджере (действие «Открыть папку»).</summary>
    public async Task OpenFolderAsync(string path)
    {
        if (Launcher is { } launcher)
        {
            await launcher.LaunchDirectoryInfoAsync(new DirectoryInfo(path)).ConfigureAwait(true);
        }
    }

    /// <summary>
    /// Горячие клавиши окна. 5.3 (VOI-02): Ctrl+M — начать или остановить голосовую запись,
    /// Esc во время записи — отменить её. 8.5 (UI-04, основной путь без мыши): Ctrl+N (⌘N) —
    /// новый диалог, Ctrl+, (⌘,) — настройки, Esc — закрыть настройки, иначе остановить ответ;
    /// после этих действий фокус возвращается в поле ввода.
    /// </summary>
    private void OnWindowKeyDown(object? sender, KeyEventArgs e)
    {
        if (Model is not { } model)
        {
            return;
        }

        // B.13: идёт назначение клавиши во вкладке «Горячие клавиши» — нажатие и есть клавиша.
        if (model.Settings.Hotkeys.Capture(e.Key, e.KeyModifiers))
        {
            e.Handled = true;
            return;
        }

        var keys = Hotkeys.Current;
        var action = Hotkeys.Actions.FirstOrDefault(a => keys.Matches(a, e.Key, e.KeyModifiers));
        var (command, toInput) = action switch
        {
            Hotkeys.Microphone when model.Sections.IsCharacterSelected => (model.Character.Voice.ToggleCommand, false),
            Hotkeys.Microphone => (model.Voice.ToggleCommand, false),
            Hotkeys.Stop when model.Sections.IsCharacterSelected && model.Character.Stop.CanExecute(null) => (model.Character.Stop, true),
            Hotkeys.Stop when model.Character.Voice.IsActive => (model.Character.Voice.EndCommand, false),
            Hotkeys.Stop when model.Voice.IsRecording => (model.Voice.CancelCommand, false),
            Hotkeys.Stop when model.Settings.IsOpen => (model.Settings.CloseCommand, true),
            Hotkeys.Stop when model.IsGenerating => (model.StopCommand, true),
            Hotkeys.NewChat when !model.Settings.IsOpen && !model.Onboarding.IsOpen => (model.NewChatCommand, true),
            Hotkeys.Settings when !model.Onboarding.IsOpen => (model.OpenSettingsCommand, false),
            _ => ((System.Windows.Input.ICommand?)null, false),
        };
        if (command is null)
        {
            return;
        }

        e.Handled = true;
        if (command.CanExecute(null))
        {
            command.Execute(null);
            if (toInput)
            {
                FocusInput();
            }
        }
    }

    /// <summary>8.5: Delete в списке диалогов — удалить выбранный (как крестик мышью).</summary>
    private void OnChatListKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Delete || Model is not { SelectedChat: { } chat } model)
        {
            return;
        }

        e.Handled = true;
        if (model.DeleteChatCommand.CanExecute(chat))
        {
            model.DeleteChatCommand.Execute(chat);
        }
    }

    /// <summary>Фокус в поле ввода чата (после запуска, нового диалога, закрытия настроек).</summary>
    public void FocusInput() => FocusLater(Model?.Onboarding.IsOpen == true ? NextButton : InputBox);

    /// <summary>8.5: фокус переходит в открывшееся окно (настройки, мастер) и обратно в поле ввода.</summary>
    private static void FocusLater(Control control) => Avalonia.Threading.Dispatcher.UIThread.Post(
        () => control.Focus(NavigationMethod.Tab), Avalonia.Threading.DispatcherPriority.Input);

    /// <summary>Enter — отправить, Shift+Enter — новая строка.</summary>
    private void OnInputKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Enter || e.KeyModifiers.HasFlag(KeyModifiers.Shift) || Model is not { } model)
        {
            return;
        }

        e.Handled = true;
        if (model.SendCommand.CanExecute(null))
        {
            model.SendCommand.Execute(null);
        }
    }

    private void OnDragOver(object? sender, DragEventArgs e) =>
        e.DragEffects = e.DataTransfer.Contains(DataFormat.File) ? DragDropEffects.Copy : DragDropEffects.None;

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        if (Model is not { } model || e.DataTransfer.TryGetFiles() is not { } items)
        {
            return;
        }

        await model.AddFilesAsync(await ToDroppedAsync(items.OfType<IStorageFile>()).ConfigureAwait(true))
            .ConfigureAwait(true);
    }

    private static async Task<IReadOnlyList<DroppedFile>> ToDroppedAsync(IEnumerable<IStorageItem> items)
    {
        var files = new List<DroppedFile>();
        foreach (var file in items.OfType<IStorageFile>())
        {
            var properties = await file.GetBasicPropertiesAsync().ConfigureAwait(true);
            files.Add(new DroppedFile(file.Name, (long)(properties.Size ?? 0), file.OpenReadAsync));
        }

        return files;
    }

    /// <summary>Новое сообщение или другой диалог — снова следовать за низом.</summary>
    private void OnMessagesChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (e.Action is NotifyCollectionChangedAction.Add or NotifyCollectionChangedAction.Reset)
        {
            _follow = true;
            MessagesScroll.ScrollToEnd();
        }
    }

    /// <summary>
    /// (1) Ответ растёт внутри уже добавленного сообщения — раньше окно прокручивалось только при
    /// добавлении сообщения, и печатающийся текст уходил за нижнюю границу. Теперь при росте
    /// содержимого окно держится у низа, если пользователь сам не прокрутил вверх.
    /// </summary>
    private void OnMessagesScrolled(object? sender, ScrollChangedEventArgs e)
    {
        if ((e.ExtentDelta.Y != 0 || e.ViewportDelta.Y != 0) && _follow)
        {
            MessagesScroll.ScrollToEnd();
        }
        else if (e.OffsetDelta.Y != 0)
        {
            _follow = IsAtBottom;
        }
    }

    private bool IsAtBottom =>
        MessagesScroll.Offset.Y + MessagesScroll.Viewport.Height >= MessagesScroll.Extent.Height - FollowSlack;
}
