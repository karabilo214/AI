using System.Text.Json;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>8.3: запуск с обновлением (применение, отметка, откат) и раздел «Обновления».</summary>
public sealed class UpdatesTests
{
    private static readonly Localizer Ru = Localizer.Load("ru");

    public UpdatesTests() => Localizer.Current = Ru;

    private static readonly ServiceOptions Options = new("python", "/app/ramvi", "/app/data", [], "/app");

    private static (List<string> Calls, Func<string, Task<JsonElement?>> Run) Updater(string status, string apply = "{\"applied\": \"0.9.0\"}")
    {
        var calls = new List<string>();
        return (calls, action =>
        {
            calls.Add(action);
            var json = action switch { "status" => status, "apply" => apply, _ => "{}" };
            return Task.FromResult<JsonElement?>(JsonDocument.Parse(json).RootElement.Clone());
        });
    }

    [Fact]
    public async Task Nothing_staged_just_starts()
    {
        var (calls, run) = Updater("{\"current\": \"0.8.0\"}");
        var started = 0;
        var host = await ServiceHost.WithUpdatesAsync(Options, TimeSpan.FromSeconds(1), run, (_, _) => Task.FromResult(++started));
        Assert.Equal((1, 1), (host, started));
        Assert.Equal(["status"], calls);
    }

    [Fact]
    public async Task Staged_update_is_applied_and_marked_good()
    {
        var (calls, run) = Updater("{\"current\": \"0.8.0\", \"staged\": \"0.9.0\"}");
        var host = await ServiceHost.WithUpdatesAsync(Options, TimeSpan.FromSeconds(1), run, (_, _) => Task.FromResult("new"));
        Assert.Equal("new", host);
        Assert.Equal(["status", "apply", "mark-good"], calls);
    }

    [Fact]
    public async Task New_version_that_does_not_start_is_rolled_back()
    {
        var (calls, run) = Updater("{\"current\": \"0.8.0\", \"staged\": \"0.9.0\"}");
        var attempts = 0;
        var host = await ServiceHost.WithUpdatesAsync(Options, TimeSpan.FromSeconds(1), run, (_, _) =>
            ++attempts == 1 ? Task.FromException<string>(new InvalidOperationException("broken build")) : Task.FromResult("old"));
        Assert.Equal(("old", 2), (host, attempts));
        Assert.Equal(["status", "apply", "rollback"], calls);
    }

    [Fact]
    public async Task Unconfirmed_previous_update_is_checked_again()
    {
        var (calls, run) = Updater("{\"current\": \"0.9.0\", \"applied\": \"0.9.0\", \"health\": \"pending\"}");
        await ServiceHost.WithUpdatesAsync(Options, TimeSpan.FromSeconds(1), run, (_, _) => Task.FromResult(1));
        Assert.Equal(["status", "mark-good"], calls);
    }

    [Fact]
    public async Task Failure_without_update_is_not_hidden()
    {
        var (calls, run) = Updater("{\"current\": \"0.8.0\"}");
        await Assert.ThrowsAsync<InvalidOperationException>(() => ServiceHost.WithUpdatesAsync<int>(
            Options, TimeSpan.FromSeconds(1), run, (_, _) => throw new InvalidOperationException("no python")));
        Assert.Equal(["status"], calls);
        var noApp = Options with { AppDir = null };
        var (none, run2) = Updater("{}");
        await ServiceHost.WithUpdatesAsync(noApp, TimeSpan.FromSeconds(1), run2, (_, _) => Task.FromResult(1));
        Assert.Empty(none);
    }

    [Fact]
    public async Task Settings_show_status_install_and_restart()
    {
        var api = new FakeChatApi { Update = new UpdateStatus("0.8.0") };
        var restarted = 0;
        var model = new MainViewModel(
            api,
            pickModelFile: () => Task.FromResult<string?>("/downloads/RAMVI-update-0.9.0.zip"),
            restartApp: () => { restarted++; return Task.CompletedTask; });
        await model.InitializeAsync();
        await model.OpenSettingsCommand.RunAsync();
        var settings = model.Settings;
        Assert.True(settings.HasUpdates);
        Assert.Equal(Ru.Format("settings.update_current", ("version", "0.8.0")), settings.UpdateText);
        Assert.False(settings.RestartToUpdateCommand.CanExecute(null));
        await settings.InstallUpdateCommand.RunAsync();
        Assert.Equal(["/downloads/RAMVI-update-0.9.0.zip"], api.StagedPaths);
        Assert.Equal(Ru.Format("settings.update_staged", ("version", "0.9.0"), ("current", "0.8.0")), settings.UpdateText);
        await settings.RestartToUpdateCommand.RunAsync();
        Assert.Equal(1, restarted);
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task Rolled_back_update_is_explained()
    {
        var api = new FakeChatApi { Update = new UpdateStatus("0.8.0", RolledBack: "0.9.0", Health: "rolled_back") };
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        await model.OpenSettingsCommand.RunAsync();
        Assert.Equal(Ru.Format("settings.update_rolled_back", ("version", "0.9.0"), ("current", "0.8.0")), model.Settings.UpdateText);
        var none = new MainViewModel(new FakeChatApi());
        await none.InitializeAsync();
        await none.OpenSettingsCommand.RunAsync();
        Assert.False(none.Settings.HasUpdates); // сервис без папки приложения — раздела нет
    }
}
