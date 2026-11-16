using System.Net;
using System.Text;
using Ramvi.Desktop.Api;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class ChatApiClientTests
{
    private sealed class Handler(Func<HttpRequestMessage, HttpResponseMessage> reply) : HttpMessageHandler
    {
        public List<(HttpMethod Method, string Path, string? Auth, string Body)> Requests { get; } = [];

        public List<long?> Lengths { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancel)
        {
            Lengths.Add(request.Content?.Headers.ContentLength);
            var body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(cancel);
            Requests.Add((request.Method, request.RequestUri!.AbsolutePath, request.Headers.Authorization?.ToString(), body));
            return reply(request);
        }
    }

    private static HttpResponseMessage Json(string json, HttpStatusCode code = HttpStatusCode.OK) =>
        new(code) { Content = new StringContent(json, Encoding.UTF8, "application/json") };

    private static ChatApiClient Client(Handler handler) => new(new Uri("http://127.0.0.1:5000"), "secret", handler);

    [Fact]
    public async Task Stream_events_are_parsed_and_request_is_authorized()
    {
        const string body =
            "event: user\ndata: {\"type\":\"user\",\"message\":{\"id\":\"m1\",\"chat_id\":\"c 1\",\"role\":\"user\",\"content\":\"hi\",\"created_at\":1}}\n\n" +
            ": keep-alive\n\n" +
            "data: {\"type\":\"reasoning\",\"text\":\"hm\"}\n\n" +
            "data: {\"type\":\"text\",\"text\":\"При\"}\n\n" +
            "data: {\"type\":\"done\",\"finish\":\"stop\",\"message\":{\"id\":\"m2\",\"chat_id\":\"c 1\",\"role\":\"assistant\",\"content\":\"При\",\"created_at\":2,\"reasoning\":\"hm\",\"finish\":\"stop\"}}\n\n";
        var handler = new Handler(_ => new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(body, Encoding.UTF8, "text/event-stream"),
        });
        using var client = Client(handler);

        var events = new List<ChatEvent>();
        await foreach (var item in client.SendAsync("c/1", "hi", [new Attachment("a.txt", "abc", 3)], TestContext.Current.CancellationToken))
        {
            events.Add(item);
        }

        Assert.Equal(["user", "reasoning", "text", "done"], events.Select(e => e.Type));
        Assert.Equal("hi", events[0].Message?.Content);
        Assert.Equal("При", events[2].Text);
        Assert.Equal("hm", events[3].Message?.Reasoning);
        Assert.Equal("stop", events[3].Finish);
        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, request.Method);
        Assert.Equal("/ui/v1/chats/c%2F1/send", request.Path);
        Assert.Equal("Bearer secret", request.Auth);
        Assert.Equal("{\"text\":\"hi\",\"attachments\":[{\"name\":\"a.txt\",\"text\":\"abc\"}]}", request.Body);
        Assert.True(handler.Lengths[0] > 0, "у тела потока нет Content-Length");
    }

    [Fact]
    public async Task Error_response_becomes_api_exception_with_params()
    {
        var handler = new Handler(_ => Json(
            "{\"error\":{\"code\":\"chat.too_many_files\",\"params\":{\"limit\":5}}}", HttpStatusCode.BadRequest));
        using var client = Client(handler);

        var error = await Assert.ThrowsAsync<ApiException>(async () =>
        {
            await foreach (var _ in client.SendAsync("c", "x", [], TestContext.Current.CancellationToken))
            {
            }
        });

        Assert.Equal("chat.too_many_files", error.Code);
        Assert.Equal(5, error.Parameters["limit"].GetInt32());
    }

    [Fact]
    public async Task Non_json_error_is_internal()
    {
        var handler = new Handler(_ => new HttpResponseMessage(HttpStatusCode.BadGateway) { Content = new StringContent("<html>") });
        using var client = Client(handler);

        var error = await Assert.ThrowsAsync<ApiException>(() => client.ChatsAsync(TestContext.Current.CancellationToken));

        Assert.Equal("api.internal", error.Code);
    }

    [Fact]
    public async Task Status_and_chats_use_snake_case()
    {
        var handler = new Handler(request => request.RequestUri!.AbsolutePath switch
        {
            "/ui/v1/status" => Json(
                "{\"engine\":{\"model_id\":\"m\",\"loading\":null,\"state\":\"ready\"},\"limits\":{\"max_text_chars\":10,\"max_files\":2,\"max_file_bytes\":3,\"max_total_bytes\":4,\"max_name_chars\":5}}"),
            _ => Json("{\"chats\":[{\"id\":\"c\",\"title\":\"t\",\"model_id\":\"m\",\"created_at\":1,\"updated_at\":2,\"generating\":true}]}"),
        });
        using var client = Client(handler);

        var status = await client.StatusAsync(TestContext.Current.CancellationToken);
        var chats = await client.ChatsAsync(TestContext.Current.CancellationToken);

        Assert.Equal(new Limits(10, 2, 3, 4, 5), status.Limits);
        Assert.Equal("ready", status.Engine.State);
        Assert.Equal(new ChatInfo("c", "t", "m", 1, 2, true), Assert.Single(chats));
    }

    [Fact]
    public async Task Import_and_settings_requests_have_snake_case_bodies()
    {
        var handler = new Handler(request => request.RequestUri!.AbsolutePath switch
        {
            "/ui/v1/models/import" => Json("{\"model\":{\"id\":\"m\",\"architecture\":\"qwen3\",\"quant\":null,\"bytes\":1}}"),
            _ => Json("{\"done\":true,\"default_model\":\"m\"}"),
        });
        using var client = Client(handler);

        var model = await client.ImportModelAsync(@"C:\Users\Владимир\.ollama\models\blobs\sha256-7f40", TestContext.Current.CancellationToken);
        await client.CompleteOnboardingAsync(null, TestContext.Current.CancellationToken);

        Assert.Equal("m", model.Id);
        Assert.Equal("/ui/v1/models/import", handler.Requests[0].Path);
        using var body = System.Text.Json.JsonDocument.Parse(handler.Requests[0].Body);
        Assert.Equal(@"C:\Users\Владимир\.ollama\models\blobs\sha256-7f40", body.RootElement.GetProperty("path").GetString());
        Assert.Equal("{}", handler.Requests[1].Body); // пропуск мастера: model_id не отправляется
        // Python http.server не читает тело без Content-Length (chunked): длина обязана быть известна.
        Assert.All(handler.Lengths, length => Assert.True(length > 0, "у тела нет Content-Length"));
    }
}
