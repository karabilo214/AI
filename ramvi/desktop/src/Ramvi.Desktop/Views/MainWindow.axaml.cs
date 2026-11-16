using System.Collections.Specialized;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;

namespace Ramvi.Desktop.Views;

public sealed partial class MainWindow : Window
{
    /// <summary>Расстояние до низа, при котором окно считается «внизу» и следует за ответом.</summary>
    private const double FollowSlack = 48;

    /// <summary>Следовать за новым текстом; false — пользователь прокрутил вверх и читает.</summary>
    private bool _follow = true;

    public MainWindow()
    {
        InitializeComponent();
        MessagesScroll.ScrollChanged += OnMessagesScrolled;
        InputBox.AddHandler(KeyDownEvent, OnInputKeyDown, RoutingStrategies.Tunnel);
        AddHandler(DragDrop.DropEvent, OnDrop);
        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        DataContextChanged += (_, _) =>
        {
            if (DataContext is MainViewModel model)
            {
                model.Messages.CollectionChanged += OnMessagesChanged;
            }
        };
    }

    private MainViewModel? Model => DataContext as MainViewModel;

    public async Task CopyAsync(string text)
    {
        if (Clipboard is { } clipboard)
        {
            await clipboard.SetTextAsync(text).ConfigureAwait(true);
        }
    }

    /// <summary>Выбрать файл модели GGUF; путь на диске или null (отмена, не локальный файл).</summary>
    public async Task<string?> PickModelFileAsync()
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = Localization.Localizer.Current["onboarding.pick_title"],
            AllowMultiple = false,
            FileTypeFilter = [new FilePickerFileType("GGUF") { Patterns = ["*.gguf"] }],
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
