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
    public MainWindow()
    {
        InitializeComponent();
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

        var files = new List<DroppedFile>();
        foreach (var file in items.OfType<IStorageFile>())
        {
            var properties = await file.GetBasicPropertiesAsync().ConfigureAwait(true);
            files.Add(new DroppedFile(file.Name, (long)(properties.Size ?? 0), file.OpenReadAsync));
        }

        await model.AddFilesAsync(files).ConfigureAwait(true);
    }

    private void OnMessagesChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (e.Action == NotifyCollectionChangedAction.Add)
        {
            MessagesScroll.ScrollToEnd();
        }
    }
}
