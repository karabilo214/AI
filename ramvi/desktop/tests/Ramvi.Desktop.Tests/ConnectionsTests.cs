using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;

namespace Ramvi.Desktop.Tests;

/// <summary>7.1–7.4: вкладка «Подключения» — адрес для IDE, ключи, отзыв, сеть.</summary>
public sealed class ConnectionsTests
{
    private static readonly Localizer Ru = Localizer.Load("ru");

    public ConnectionsTests() => Localizer.Current = Ru;

    private static async Task<(FakeChatApi Api, MainViewModel Model, List<string> Copied)> Open()
    {
        var api = new FakeChatApi();
        api.Clients.Add(Client("aaaaaaaa0001", "VS Code"));
        api.Clients.Add(Client("aaaaaaaa0002", "Old tool", "revoked"));
        var copied = new List<string>();
        var model = new MainViewModel(api, copy: text =>
        {
            copied.Add(text);
            return Task.CompletedTask;
        });
        await model.InitializeAsync();
        await model.Settings.OpenAsync("connections");
        await model.Settings.Connections.LoadAsync();
        return (api, model, copied);
    }

    [Fact]
    public async Task Tab_shows_address_queue_and_clients_without_keys()
    {
        var (_, model, copied) = await Open();
        var connections = model.Settings.Connections;
        Assert.True(model.Settings.IsConnectionsTab && model.Settings.HasFooter);
        Assert.Equal(Ru.Format("connections.api_running", ("url", "http://127.0.0.1:11435")), connections.Status);
        Assert.Equal("http://127.0.0.1:11435/v1", connections.OpenAiUrl);
        Assert.Equal(["VS Code", "Old tool"], connections.Clients.Select(c => c.Name));
        var first = connections.Clients[0];
        Assert.Equal("ramvi-…0001", first.KeyHint);
        Assert.Equal(Ru["connections.state.active"], first.State);
        Assert.Contains("120", first.Usage, StringComparison.Ordinal);
        Assert.Equal(Ru["connections.never_seen"], first.LastSeen);
        Assert.False(connections.Clients[1].IsActive);
        await connections.CopyUrlCommand.RunAsync();
        Assert.Equal(["http://127.0.0.1:11435/v1"], copied);
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task New_key_is_shown_once_and_then_hidden()
    {
        var (api, model, copied) = await Open();
        var connections = model.Settings.Connections;
        Assert.False(connections.CreateCommand.CanExecute(null)); // без названия нельзя
        connections.NewName = "  Cursor ";
        connections.NewPermissions.First(p => p.Code == "remote").Enabled = true;
        await connections.CreateCommand.RunAsync();
        Assert.Equal("Cursor", api.CreatedKeys[0].Name);
        Assert.Equal(["chat", "models", "remote"], api.CreatedKeys[0].Permissions);
        Assert.True(connections.HasNewKey);
        var key = connections.NewKey!;
        Assert.StartsWith("ramvi-SECRETKEY", key, StringComparison.Ordinal);
        Assert.Equal("", connections.NewName);
        Assert.Equal(3, connections.Clients.Count);
        Assert.DoesNotContain(connections.Clients, c => c.KeyHint == key || c.Usage.Contains(key, StringComparison.Ordinal));
        await connections.CopyKeyCommand.RunAsync();
        Assert.Equal([key], copied);
        await connections.DismissKeyCommand.RunAsync();
        Assert.False(connections.HasNewKey);
        Assert.Null(connections.NewKey);
        await connections.LoadAsync();
        Assert.False(connections.HasNewKey); // при повторном открытии ключа нет
    }

    [Fact]
    public async Task Cloud_models_are_a_separate_permission_off_by_default()
    {
        var (api, model, _) = await Open();
        var connections = model.Settings.Connections;
        var cloud = connections.NewPermissions.Single(p => p.Code == "cloud");
        Assert.False(cloud.Enabled); // платные запросы — только по явному выбору
        Assert.Equal(Ru["connections.permission.cloud"], cloud.Label);
        connections.NewName = "IDE";
        cloud.Enabled = true;
        await connections.CreateCommand.RunAsync();
        Assert.Equal(["chat", "models", "cloud"], api.CreatedKeys[0].Permissions);
    }

    [Fact]
    public async Task Key_needs_at_least_one_permission()
    {
        var (api, model, _) = await Open();
        var connections = model.Settings.Connections;
        connections.NewName = "Tool";
        foreach (var permission in connections.NewPermissions)
        {
            permission.Enabled = false;
        }

        await connections.CreateCommand.RunAsync();
        Assert.Empty(api.CreatedKeys);
        Assert.Equal(Ru["connections.no_permissions"], connections.Error);
    }

    [Fact]
    public async Task Revoke_asks_for_confirmation_first()
    {
        var (api, model, _) = await Open();
        var connections = model.Settings.Connections;
        var row = connections.Clients[0];
        Assert.False(connections.Clients[1].RevokeCommand.CanExecute(null)); // уже отозван
        await row.RevokeCommand.RunAsync();
        Assert.Empty(api.Revoked);
        Assert.Equal(Ru["connections.revoke_confirm"], row.RevokeLabel);
        await row.RevokeCommand.RunAsync();
        Assert.Equal([("aaaaaaaa0001", true)], api.Revoked);
        Assert.Equal(Ru.Format("connections.revoked", ("name", "VS Code"), ("stopped", 1)), connections.Message);
        Assert.Equal(Ru["connections.state.revoked"], connections.Clients.First(c => c.Name == "VS Code").State);
    }

    [Fact]
    public async Task Network_settings_are_saved_and_busy_port_is_reported()
    {
        var (api, model, _) = await Open();
        var settings = model.Settings;
        Assert.Equal("192.168.0.0/16, 10.0.0.0/8, 172.16.0.0/12", settings.NetworkAllow);
        settings.NetworkAccess = true;
        settings.NetworkAllow = " 192.168.1.0/24 ";
        await settings.SaveCommand.RunAsync();
        var update = api.SettingsUpdates[^1];
        Assert.Equal((true, "192.168.1.0/24"), (update.NetworkAccess, update.NetworkAllow));
        api.SaveApiError = "api.port_busy";
        settings.ApiPort = "12345";
        await settings.SaveCommand.RunAsync();
        Assert.Equal(Ru.Format("api.port_busy", ("port", 12345)), settings.Error);
    }

    [Fact]
    public async Task Stopped_api_is_shown()
    {
        var api = new FakeChatApi { ApiUrl = null };
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        await model.Settings.Connections.LoadAsync();
        Assert.Equal(Ru["connections.api_stopped"], model.Settings.Connections.Status);
        Assert.False(model.Settings.Connections.HasUrl);
    }

    [AvaloniaFact]
    public async Task Window_shows_the_new_key_card_and_client_list()
    {
        var (_, model, _) = await Open();
        var window = new MainWindow { DataContext = model };
        window.Show();
        Dispatcher.UIThread.RunJobs();
        Assert.True(window.FindControl<ScrollViewer>("ConnectionsTab")!.IsVisible);
        Assert.Equal(2, window.FindControl<ItemsControl>("ClientList")!.ItemCount);
        Assert.False(window.FindControl<Border>("NewKeyCard")!.IsVisible);
        model.Settings.Connections.NewName = "Zed";
        await model.Settings.Connections.CreateCommand.RunAsync();
        Dispatcher.UIThread.RunJobs();
        Assert.True(window.FindControl<Border>("NewKeyCard")!.IsVisible);
        var texts = window.GetVisualDescendants().OfType<TextBlock>().Where(t => t.IsEffectivelyVisible)
            .Select(t => t.Text).ToList();
        Assert.Contains(Ru["connections.key_once"], texts);
        Assert.Empty(Ru.Missing);
        window.Close();
    }
}
