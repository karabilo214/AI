using System.Net;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using System.Text;
using System.Text.Json;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>7.8: экран «Облачные модели» и облачная модель в общем выборе модели.</summary>
public sealed class CloudModelsTests
{
    private static readonly Localizer Ru = Localizer.Load("ru");

    public CloudModelsTests() => Localizer.Current = Ru;

    private sealed class Handler(Func<HttpRequestMessage, string> reply) : HttpMessageHandler
    {
        public List<(HttpMethod Method, string Path, string Body)> Requests { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancel)
        {
            var body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(cancel);
            Requests.Add((request.Method, request.RequestUri!.AbsolutePath, body));
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(reply(request), Encoding.UTF8, "application/json") };
        }
    }

    [Fact]
    public async Task Client_speaks_the_cloud_routes_and_never_sends_an_unchanged_key()
    {
        var connection = """{"id":"claude","kind":"anthropic","name":"Claude","url":"https://api.anthropic.com","has_key":true,"models":["claude-x"]}""";
        var handler = new Handler(request => request.RequestUri!.AbsolutePath switch
        {
            "/ui/v1/cloud" => """{"kinds":[{"kind":"openai","url":"https://api.openai.com/v1"}],"connections":[""" + connection + "]}",
            "/ui/v1/cloud/connections" => """{"connection":""" + connection + "}",
            "/ui/v1/cloud/models" => """{"models":["claude-x","claude-y"]}""",
            "/ui/v1/cloud/connections/claude/models" => """{"id":"claude","models":["claude-x"]}""",
            "/ui/v1/models" => """{"models":[{"id":"cloud:claude:claude-x","architecture":"cloud","quant":null,"bytes":null,"provider":"cloud","compat":{"status":"cloud"},"cloud":{"connection":"claude","name":"Claude","kind":"anthropic","host":"api.anthropic.com","model":"claude-x"}}]}""",
            _ => "{}",
        });
        using var client = new ChatApiClient(new Uri("http://127.0.0.1:5000"), "secret", handler);
        var cancel = TestContext.Current.CancellationToken;

        var view = await client.CloudAsync(cancel);
        Assert.True(view.Connections[0].HasKey);
        await client.SaveCloudAsync(new CloudSaveRequest("claude", "anthropic", "Claude", "https://api.anthropic.com", null), cancel);
        await client.SaveCloudAsync(new CloudSaveRequest(null, "anthropic", "Claude", "https://api.anthropic.com", "sk-new"), cancel);
        Assert.Equal(["claude-x", "claude-y"], await client.CloudModelsAsync(new CloudModelsRequest("claude"), cancel));
        await client.EnableCloudModelAsync("claude", "claude-x", true, cancel);
        await client.DeleteCloudAsync("claude", cancel);
        var models = await client.ModelsAsync(cancel);

        var saves = handler.Requests.Where(r => r.Path == "/ui/v1/cloud/connections").Select(r => JsonDocument.Parse(r.Body).RootElement).ToList();
        Assert.False(saves[0].TryGetProperty("key", out _)); // ключ не трогается
        Assert.Equal("sk-new", saves[1].GetProperty("key").GetString());
        Assert.False(saves[1].TryGetProperty("id", out _));
        Assert.Equal("""{"id":"claude"}""", handler.Requests.Single(r => r.Path == "/ui/v1/cloud/models").Body);
        Assert.Equal("""{"model":"claude-x","enabled":true}""", handler.Requests.Single(r => r.Path.EndsWith("/models", StringComparison.Ordinal) && r.Path.Contains("connections", StringComparison.Ordinal)).Body);
        Assert.Contains(handler.Requests, r => r.Method == HttpMethod.Delete && r.Path == "/ui/v1/cloud/connections/claude");
        Assert.True(models[0].IsCloud);
        Assert.Equal("claude-x · Claude", models[0].Label);
        Assert.Equal("api.anthropic.com", models[0].Cloud!.Host);
    }

    [Fact]
    public async Task New_connection_is_checked_saved_without_showing_the_key_and_models_are_ticked()
    {
        var api = new FakeChatApi();
        var changed = 0;
        var model = new CloudModelsViewModel(() => api, () => { changed++; return Task.CompletedTask; });
        await model.LoadAsync();
        Assert.Equal("https://api.openai.com/v1", model.Url);
        Assert.False(model.SaveCommand.CanExecute(null)); // без названия нечего сохранять

        model.Kind = model.Kinds.Single(k => k.Id == "anthropic");
        Assert.Equal("https://api.anthropic.com", model.Url); // адрес по умолчанию сменился вместе с сервисом
        model.Name = "Мой Claude";
        model.Key = "sk-secret";
        await model.CheckCommand.RunAsync();
        Assert.Equal(new CloudModelsRequest(null, "anthropic", "https://api.anthropic.com", "sk-secret"), api.CloudChecks[^1]);
        Assert.Equal(Ru.Format("cloud.check_ok", ("count", 3)), model.Message);

        await model.SaveCommand.RunAsync();
        Assert.Equal("sk-secret", api.CloudSaves[^1].Key);
        Assert.Equal("", model.Key); // ключ после сохранения не показывается
        var row = Assert.Single(model.Connections);
        Assert.Equal(Ru["cloud.key_set"], row.Key);
        Assert.Equal("api.anthropic.com", row.Host);
        Assert.Same(row, model.Selected);
        Assert.Equal(["gpt-b", "gpt-a", "gpt-codex"], model.Models.Select(m => m.Name));

        model.Models.Single(m => m.Name == "gpt-codex").Enabled = true;
        await MainViewModelTests.Until(() => changed == 1, "общий выбор модели перечитан");
        Assert.Contains(api.Models, m => m.Id == "cloud:мой-claude:gpt-codex" && m.IsCloud);
        Assert.Equal("gpt-codex", model.Models[2].Name); // строка не прыгает под мышью
        await model.Connections[0].ModelsCommand.RunAsync();
        Assert.Equal("gpt-codex", model.Models[0].Name); // при новом открытии включённые — первыми
        Assert.True(model.Models[0].Enabled);
        Assert.Equal(Ru.Format("cloud.models_enabled", ("models", "gpt-codex")), model.Connections[0].Models);

        model.Filter = "gpt-a";
        Assert.Equal(["gpt-a"], model.Models.Select(m => m.Name));
    }

    [Fact]
    public async Task Edit_keeps_the_key_unless_a_new_one_is_typed_and_errors_are_explained()
    {
        var api = new FakeChatApi();
        api.CloudConnections.Add(new CloudConnection("router", "openai", "Router", "https://openrouter.ai/api/v1", true, ["gpt-a"]));
        api.CloudKeys["router"] = "old";
        var model = new CloudModelsViewModel(() => api);
        await model.LoadAsync();

        await model.Connections[0].EditCommand.RunAsync();
        Assert.True(model.IsEditing);
        Assert.Equal("compatible", model.Kind.Id); // свой адрес — «совместимый с OpenAI»
        Assert.Equal("https://openrouter.ai/api/v1", model.Url);
        await model.CheckCommand.RunAsync();
        Assert.Equal(new CloudModelsRequest("router"), api.CloudChecks[^1]); // проверка сохранённого ключа
        model.Key = "sk-new";
        await model.CheckCommand.RunAsync();
        Assert.Equal(new CloudModelsRequest("router", Key: "sk-new"), api.CloudChecks[^1]); // новый ключ до сохранения
        model.Key = "";
        await model.SaveCommand.RunAsync();
        Assert.Null(api.CloudSaves[^1].Key);
        Assert.Equal("old", api.CloudKeys["router"]);
        Assert.False(model.IsEditing);
        api.CloudFail = new ApiException("cloud.key_rejected", new Dictionary<string, JsonElement>
        {
            ["host"] = JsonDocument.Parse("\"openrouter.ai\"").RootElement,
            ["status"] = JsonDocument.Parse("401").RootElement,
        });
        await model.Connections[0].ModelsCommand.RunAsync();
        Assert.Contains("openrouter.ai", model.Error, StringComparison.Ordinal);
        Assert.Equal(Ru.Format("cloud.key_rejected", new Dictionary<string, object?>(StringComparer.Ordinal) { ["host"] = "openrouter.ai", ["status"] = 401 }), model.Error);

        api.CloudFail = null;
        await model.Connections[0].ModelsCommand.RunAsync();
        api.CloudFail = new ApiException("api.internal", null);
        var option = model.Models.Single(m => m.Name == "gpt-b");
        option.Enabled = true;
        await MainViewModelTests.Until(() => !option.Enabled, "галочка вернулась после ошибки");
        Assert.True(model.HasError);
    }

    [Fact]
    public async Task Delete_asks_twice_and_removes_the_models_from_the_picker()
    {
        var api = new FakeChatApi();
        api.CloudConnections.Add(new CloudConnection("c", "anthropic", "Claude", "https://api.anthropic.com", true, []));
        var changed = 0;
        var model = new CloudModelsViewModel(() => api, () => { changed++; return Task.CompletedTask; });
        await model.LoadAsync();
        await api.EnableCloudModelAsync("c", "claude-x", true, TestContext.Current.CancellationToken);

        var row = model.Connections[0];
        await row.DeleteCommand.RunAsync();
        Assert.Equal(Ru["cloud.delete_confirm"], row.DeleteLabel);
        Assert.Single(api.CloudConnections);
        await row.DeleteCommand.RunAsync();
        Assert.Empty(api.CloudConnections);
        Assert.Empty(model.Connections);
        Assert.DoesNotContain(api.Models, m => m.IsCloud);
        Assert.Equal(1, changed);
    }

    [Fact]
    public async Task Picking_a_cloud_model_shows_where_requests_go_and_library_hides_it()
    {
        var api = new FakeChatApi();
        api.CloudConnections.Add(new CloudConnection("c", "anthropic", "Claude", "https://api.anthropic.com", true, []));
        await api.EnableCloudModelAsync("c", "claude-x", true, TestContext.Current.CancellationToken);
        api.AddChat("чат", "model-a");
        var main = new MainViewModel(api);
        await main.InitializeAsync();
        Assert.False(main.IsCloudModel);
        Assert.Equal("", main.CloudNotice);

        var raised = new List<string?>();
        main.PropertyChanged += (_, e) => raised.Add(e.PropertyName);
        main.SelectedModel = main.Models.Single(m => m.IsCloud);
        Assert.Contains(nameof(MainViewModel.CloudNotice), raised); // надпись обновляется на экране
        Assert.Contains(nameof(MainViewModel.IsCloudModel), raised);
        Assert.True(main.IsCloudModel);
        Assert.Equal(Ru.Format("cloud.sending", ("host", "api.anthropic.com")), main.CloudNotice);
        Assert.Same(main.SelectedModel, main.Agent.Model);
        Assert.Same(main.SelectedModel, main.Character.Model); // персонаж и агент — та же облачная модель
        Localizer.Current = Localizer.Load("en");
        try
        {
            Assert.Equal("requests go to api.anthropic.com", main.CloudNotice);
        }
        finally
        {
            Localizer.Current = Ru;
        }

        main.SelectedModel = main.Models.Single(m => m.Id == "model-b");
        Assert.False(main.IsCloudModel);

        await main.ModelsScreen.LoadAsync();
        Assert.DoesNotContain(main.ModelsScreen.Cards, c => c.Name.StartsWith("cloud:", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Opening_the_connections_tab_loads_the_cloud_connections()
    {
        var api = new FakeChatApi();
        api.CloudConnections.Add(new CloudConnection("c", "gemini", "Gemini", "https://generativelanguage.googleapis.com", false, []));
        var main = new MainViewModel(api);
        await main.InitializeAsync();
        await main.Settings.OpenAsync("connections");
        await MainViewModelTests.Until(() => main.Settings.Cloud.Connections.Count == 1, "подключения показаны");
        Assert.Equal(Ru["cloud.key_missing"], main.Settings.Cloud.Connections[0].Key);
    }

    [AvaloniaFact]
    public async Task Window_shows_cloud_connections_and_where_requests_go()
    {
        var api = new FakeChatApi();
        api.CloudConnections.Add(new CloudConnection("c", "anthropic", "Claude", "https://api.anthropic.com", true, []));
        await api.EnableCloudModelAsync("c", "claude-sonnet-5-5", true, TestContext.Current.CancellationToken);
        api.AddChat("чат", "cloud:c:claude-sonnet-5-5");
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        var window = new MainWindow { DataContext = model };
        window.Show();
        Dispatcher.UIThread.RunJobs();
        var notice = window.FindControl<TextBlock>("CloudNotice")!;
        Assert.True(notice.IsEffectivelyVisible);
        Assert.Equal(Ru.Format("cloud.sending", ("host", "api.anthropic.com")), notice.Text);
        var shot = Environment.GetEnvironmentVariable("RAMVI_CLOUD_UI_SCREENSHOT");
        if (shot is not null)
        {
            window.UpdateLayout(); Dispatcher.UIThread.RunJobs(); AvaloniaHeadlessPlatform.ForceRenderTimerTick();
            using var frame = window.CaptureRenderedFrame(); frame!.Save(Path.ChangeExtension(shot, "chat.png"), new Avalonia.Media.Imaging.PngBitmapEncoderOptions());
        }

        await model.Settings.OpenAsync("connections");
        await MainViewModelTests.Until(() => model.Settings.Cloud.Connections.Count == 1, "подключения показаны");
        await model.Settings.Cloud.Connections[0].ModelsCommand.RunAsync();
        Dispatcher.UIThread.RunJobs();
        var view = window.GetVisualDescendants().OfType<CloudModelsView>().Single();
        Assert.True(view.IsEffectivelyVisible);
        Assert.Equal(1, view.FindControl<ItemsControl>("CloudConnections")!.ItemCount);
        Assert.Equal(4, view.FindControl<ItemsControl>("CloudModelList")!.ItemCount);
        var texts = view.GetVisualDescendants().OfType<TextBlock>().Where(t => t.IsEffectivelyVisible).Select(t => t.Text).ToList();
        Assert.Contains(Ru["cloud.title"], texts);
        Assert.Contains(Ru.Format("cloud.models_of", ("name", "Claude")), texts);
        Assert.Empty(Ru.Missing);
        if (shot is not null)
        {
            window.UpdateLayout(); Dispatcher.UIThread.RunJobs(); AvaloniaHeadlessPlatform.ForceRenderTimerTick();
            using var frame = window.CaptureRenderedFrame(); frame!.Save(Path.ChangeExtension(shot, "settings.png"), new Avalonia.Media.Imaging.PngBitmapEncoderOptions());
        }

        window.Close();
    }
}
