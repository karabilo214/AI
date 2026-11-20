using Avalonia.Controls;
using Avalonia.Platform.Storage;
using Ramvi.Desktop.Character;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;

namespace Ramvi.Desktop.Views;

public sealed partial class CharacterSettingsView : UserControl
{
    public CharacterSettingsView()
    {
        InitializeComponent();
        CharacterDefaultPack.Click += (_, _) =>
        {
            if (DataContext is CharacterViewModel model) model.PackPath = "";
            PackError.Text = ""; PackError.IsVisible = false;
        };
        // Вкладку открыли — показать память и историю на сейчас, а не на момент загрузки.
        AttachedToVisualTree += async (_, _) =>
        {
            if (DataContext is CharacterViewModel model) await model.Memory.RefreshAsync().ConfigureAwait(true);
        };
        MemoryExport.Click += async (_, _) =>
        {
            var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
            if (storage?.CanSave != true || DataContext is not CharacterViewModel model) return;
            var json = await model.Memory.ExportAsync().ConfigureAwait(true);
            if (json is null) return;
            var file = await storage.SaveFilePickerAsync(new FilePickerSaveOptions
            {
                Title = Localizer.Current["character.memory_export"], SuggestedFileName = "ramvi-character-memory.json",
                FileTypeChoices = [new FilePickerFileType("JSON") { Patterns = ["*.json"] }]
            }).ConfigureAwait(true);
            if (file is null) return;
            await using var stream = await file.OpenWriteAsync().ConfigureAwait(true);
            await using var writer = new StreamWriter(stream, new System.Text.UTF8Encoding(false));
            await writer.WriteAsync(json).ConfigureAwait(true);
        };
        CharacterChoosePack.Click += async (_, _) =>
        {
            var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
            if (storage?.CanOpen != true || DataContext is not CharacterViewModel model) return;
            var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = Localizer.Current["character.choose_pack"], AllowMultiple = false,
                FileTypeFilter = [new FilePickerFileType(Localizer.Current["character.pack_filter"]) { Patterns = ["*.json"] }]
            }).ConfigureAwait(true);
            if (files.Count == 0 || files[0].TryGetLocalPath() is not {} path) return;
            try
            {
                using var pack = new CharacterPack(path);
                model.PackPath = Path.GetFullPath(path);
                PackError.Text = ""; PackError.IsVisible = false;
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException or System.Text.Json.JsonException or InvalidOperationException or KeyNotFoundException or ArgumentException)
            { PackError.Text = Localizer.Current["character.pack_error"]; PackError.IsVisible = true; }
        };
    }
}
