using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;

namespace Ramvi.Desktop.Tests;

/// <summary>3.7: экран настроек и чтение settings.json оболочкой до запуска сервиса.</summary>
public sealed class SettingsTests : IDisposable
{
    private static readonly Localizer Ru = Localizer.Load("ru");
    private readonly string _dir = Directory.CreateTempSubdirectory("ramvi-settings-").FullName;

    public SettingsTests() => Localizer.Current = Ru;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static async Task<(FakeChatApi Api, MainViewModel Model, List<string> Calls)> Open()
    {
        var api = new FakeChatApi();
        var calls = new List<string>();
        var model = new MainViewModel(
            api,
            pickFolder: start =>
            {
                calls.Add($"pick {start}");
                return Task.FromResult<string?>("/picked/folder");
            },
            restartApp: () =>
            {
                calls.Add("restart");
                return Task.CompletedTask;
            });
        await model.InitializeAsync();
        await model.OpenSettingsCommand.RunAsync();
        return (api, model, calls);
    }

    [Fact]
    public async Task Open_shows_saved_values_and_languages()
    {
        var (_, model, _) = await Open();
        var settings = model.Settings;

        Assert.True(settings.IsOpen);
        Assert.Equal(("data", "models", "11435", false), (settings.DataDir, settings.ModelsDir, settings.ApiPort, settings.NetworkAccess));
        Assert.Equal(["Русский", "English"], settings.Languages.Select(l => l.Label));
        Assert.Equal("ru", settings.Language?.Code);
        Assert.Equal(Ru.Format("settings.data_dir_note", ("path", "/app/data")), settings.DataDirNote);
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task Save_sends_only_changed_fields()
    {
        var (api, model, _) = await Open();
        var settings = model.Settings;
        settings.ApiPort = "12000";
        settings.NetworkAccess = true;

        await settings.SaveCommand.RunAsync();

        Assert.Equal([new SettingsUpdate(ApiPort: 12000, NetworkAccess: true)], api.SettingsUpdates);
        Assert.Equal(Ru["settings.saved"], settings.Message);
        Assert.False(settings.RestartNeeded);
    }

    [Fact]
    public async Task File_limits_are_saved_with_no_limit_as_zero_and_apply_at_once()
    {
        var (api, model, _) = await Open();
        var settings = model.Settings;
        Assert.Equal("5", settings.AttachFiles.Text); Assert.False(settings.AttachFiles.Unlimited);
        settings.AttachFiles.Text = "abc";
        await settings.SaveCommand.RunAsync();
        Assert.Equal(Ru["settings.attach_not_number"], settings.Error); Assert.Empty(api.SettingsUpdates);
        settings.AttachFiles.Text = "9";
        settings.AttachFileKb.Unlimited = true;
        settings.AttachTotalKb.Text = "4096";
        await settings.SaveCommand.RunAsync();
        var update = Assert.Single(api.SettingsUpdates);
        Assert.Equal(9, update.AttachMaxFiles); Assert.Equal(0, update.AttachMaxFileKb); Assert.Equal(4096, update.AttachMaxTotalKb);
        // Чат и агент сразу получают новые лимиты (без перезапуска).
        Assert.Equal(9, model.Limits.MaxFiles); Assert.Equal(0, model.Limits.MaxFileBytes);
        Assert.Equal(4096 * 1024, model.Agent.MessageLimits.MaxTotalBytes);
    }

    [Fact]
    public async Task Chat_cache_sizes_are_saved_and_off_is_zero()
    {
        // B.7: кэш неактивных чатов — RAM и диск в МБ, «Не хранить» = 0.
        var (api, model, _) = await Open();
        var settings = model.Settings;
        Assert.Equal("2048", settings.KvRam.Text); Assert.Equal("8192", settings.KvDisk.Text);
        settings.KvRam.Text = "-5";
        await settings.SaveCommand.RunAsync();
        Assert.Equal(Ru["settings.kv_not_number"], settings.Error); Assert.Empty(api.SettingsUpdates);
        settings.KvRam.Text = "1024";
        settings.KvDisk.Unlimited = true;
        await settings.SaveCommand.RunAsync();
        var update = Assert.Single(api.SettingsUpdates);
        Assert.Equal(1024, update.KvCacheRamMb); Assert.Equal(0, update.KvCacheDiskMb);
        Assert.Null(update.AttachMaxFiles);
    }

    [Fact]
    public async Task Nothing_changed_sends_nothing()
    {
        var (api, model, _) = await Open();

        await model.Settings.SaveCommand.RunAsync();

        Assert.Empty(api.SettingsUpdates);
        Assert.Equal(Ru["settings.saved"], model.Settings.Message);
    }

    [Fact]
    public async Task Language_change_applies_at_once_without_restart()
    {
        var (api, model, calls) = await Open();
        var settings = model.Settings;
        settings.Language = settings.Languages.Single(l => l.Code == "en");
        try
        {
            await settings.SaveCommand.RunAsync();

            Assert.Equal([new SettingsUpdate(Language: "en")], api.SettingsUpdates);
            Assert.False(settings.RestartNeeded); // 3.14: язык — без перезапуска
            Assert.Equal("en", Localizer.Current.Language);
            Assert.Equal("Settings saved", settings.Message);
            Assert.Equal("Ready", model.Status);
            Assert.Equal(["Same as system", "Light", "Dark", "High contrast"], settings.Themes.Select(t => t.Label));
            Assert.Equal("en", settings.Language?.Code);
            Assert.Empty(calls);
        }
        finally
        {
            Localizer.Current = Ru;
        }
    }

    [Fact]
    public async Task Browse_fills_folder_and_cancel_keeps_it()
    {
        var (api, model, calls) = await Open();

        await model.Settings.BrowseDataDirCommand.RunAsync();

        Assert.Equal("/picked/folder", model.Settings.DataDir);
        Assert.Equal(["pick data"], calls);
        await model.Settings.SaveCommand.RunAsync();
        Assert.Equal([new SettingsUpdate(DataDir: "/picked/folder")], api.SettingsUpdates);
        Assert.True(model.Settings.RestartNeeded);
    }

    [Fact]
    public async Task Port_not_a_number_is_rejected_locally()
    {
        var (api, model, _) = await Open();
        model.Settings.ApiPort = "12a";

        await model.Settings.SaveCommand.RunAsync();

        Assert.Empty(api.SettingsUpdates);
        Assert.Equal(Ru["settings.port_not_number"], model.Settings.Error);
    }

    [Fact]
    public async Task Service_error_is_shown()
    {
        var (api, model, _) = await Open();
        api.SettingsError = Error("settings.folder_no_access", ("field", "models_dir"), ("path", "/x"), ("detail", "denied"));
        model.Settings.ModelsDir = "/x";

        await model.Settings.SaveCommand.RunAsync();

        Assert.Equal(Ru.Format("settings.folder_no_access", ("path", "/x"), ("detail", "denied")), model.Settings.Error);
        Assert.False(model.Settings.RestartNeeded);
    }

    [Fact]
    public void Settings_button_needs_connection()
    {
        var model = new MainViewModel(new FakeChatApi());
        Assert.False(model.OpenSettingsCommand.CanExecute(null));
    }

    [Fact]
    public void Settings_file_gives_data_dir_and_language_before_service()
    {
        Assert.Equal(AppSettingsFile.Default, AppSettingsFile.Read(_dir)); // нет файла
        File.WriteAllText(Path.Combine(_dir, "settings.json"), """{"data_dir": "store/data2", "language": "en", "api_port": 12000}""");

        var file = AppSettingsFile.Read(_dir);

        Assert.Equal(("store/data2", "en"), (file.DataDir, file.Language));
        Assert.Equal(Path.GetFullPath(Path.Combine(_dir, "store", "data2")), file.DataDirPath(_dir));
        var absolute = Path.Combine(_dir, "abs");
        Assert.Equal(absolute, new AppSettingsFile(absolute, "ru").DataDirPath(_dir));
    }

    [Theory]
    [InlineData("{broken")]
    [InlineData("[1, 2]")]
    [InlineData("""{"data_dir": "", "language": 5}""")]
    public void Damaged_settings_file_falls_back_to_defaults(string content)
    {
        File.WriteAllText(Path.Combine(_dir, "settings.json"), content);

        Assert.Equal(AppSettingsFile.Default, AppSettingsFile.Read(_dir));
    }

    [Fact]
    public async Task Slow_models_drive_shows_a_warning()
    {
        var api = new FakeChatApi { ModelsDirWarning = "settings.models_dir_network" };
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        await model.OpenSettingsCommand.RunAsync();
        Assert.True(model.Settings.HasModelsDirWarning);
        Assert.Equal(Ru["settings.models_dir_network"], model.Settings.ModelsDirWarning);
        var local = new MainViewModel(new FakeChatApi());
        await local.InitializeAsync();
        await local.OpenSettingsCommand.RunAsync();
        Assert.False(local.Settings.HasModelsDirWarning);
        Assert.Empty(Ru.Missing);
    }
}
