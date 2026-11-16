using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace Ramvi.Desktop.Api;

/// <summary>Ошибка сервиса: ключ сообщения и параметры (перевод — в интерфейсе).</summary>
public sealed class ApiException(string code, IReadOnlyDictionary<string, JsonElement>? parameters)
    : Exception(code)
{
    public string Code { get; } = code;

    public IReadOnlyDictionary<string, JsonElement> Parameters { get; } =
        parameters ?? new Dictionary<string, JsonElement>();
}

public interface IChatApi
{
    Task<ServiceStatus> StatusAsync(CancellationToken cancel = default);

    Task<IReadOnlyList<ModelInfo>> ModelsAsync(CancellationToken cancel = default);

    Task<IReadOnlyList<ChatInfo>> ChatsAsync(CancellationToken cancel = default);

    Task<ChatInfo> CreateChatAsync(string? modelId, CancellationToken cancel = default);

    Task<IReadOnlyList<MessageInfo>> MessagesAsync(string chatId, CancellationToken cancel = default);

    Task DeleteChatAsync(string chatId, CancellationToken cancel = default);

    Task ClearAsync(string chatId, CancellationToken cancel = default);

    Task<ChatInfo> SetModelAsync(string chatId, string modelId, CancellationToken cancel = default);

    Task<bool> StopAsync(string chatId, CancellationToken cancel = default);

    IAsyncEnumerable<ChatEvent> SendAsync(
        string chatId, string text, IReadOnlyList<Attachment> attachments, CancellationToken cancel = default);

    IAsyncEnumerable<ChatEvent> RegenerateAsync(string chatId, CancellationToken cancel = default);

    /// <summary>3.6: проверка компьютера для мастера первого запуска.</summary>
    Task<SystemReport> SystemAsync(CancellationToken cancel = default);

    Task<OnboardingState> OnboardingAsync(CancellationToken cancel = default);

    /// <summary>Мастер пройден: с моделью или пропуском (null).</summary>
    Task<OnboardingState> CompleteOnboardingAsync(string? modelId, CancellationToken cancel = default);

    /// <summary>Подключить файл GGUF без копирования.</summary>
    Task<ModelInfo> ImportModelAsync(string path, CancellationToken cancel = default);

    /// <summary>3.7: настройки приложения.</summary>
    Task<SettingsInfo> SettingsAsync(CancellationToken cancel = default);

    Task<SettingsInfo> SaveSettingsAsync(SettingsUpdate update, CancellationToken cancel = default);
}

/// <summary>HTTP-клиент `ramvi_service.ui_api`: только loopback, ключ в заголовке.</summary>
public sealed class ChatApiClient : IChatApi, IDisposable
{
    private readonly HttpClient _http;

    public ChatApiClient(Uri baseAddress, string token, HttpMessageHandler? handler = null)
    {
        _http = handler is null
            ? new HttpClient(new SocketsHttpHandler { UseProxy = false, AllowAutoRedirect = false })
            : new HttpClient(handler);
        _http.BaseAddress = new Uri(baseAddress, "/ui/v1/");
        _http.Timeout = Timeout.InfiniteTimeSpan; // поток ответа может идти долго
        _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
    }

    public void Dispose() => _http.Dispose();

    public Task<ServiceStatus> StatusAsync(CancellationToken cancel = default) =>
        GetAsync("status", ApiJson.Default.ServiceStatus, cancel);

    public async Task<IReadOnlyList<ModelInfo>> ModelsAsync(CancellationToken cancel = default) =>
        (await GetAsync("models", ApiJson.Default.ModelsResponse, cancel).ConfigureAwait(false)).Models;

    public async Task<IReadOnlyList<ChatInfo>> ChatsAsync(CancellationToken cancel = default) =>
        (await GetAsync("chats", ApiJson.Default.ChatsResponse, cancel).ConfigureAwait(false)).Chats;

    public async Task<ChatInfo> CreateChatAsync(string? modelId, CancellationToken cancel = default) =>
        (await PostAsync("chats", new ModelRequest(modelId), ApiJson.Default.ModelRequest,
            ApiJson.Default.ChatResponse, cancel).ConfigureAwait(false)).Chat;

    public async Task<IReadOnlyList<MessageInfo>> MessagesAsync(string chatId, CancellationToken cancel = default) =>
        (await GetAsync($"chats/{Id(chatId)}/messages", ApiJson.Default.MessagesResponse, cancel)
            .ConfigureAwait(false)).Messages;

    public async Task DeleteChatAsync(string chatId, CancellationToken cancel = default)
    {
        using var response = await _http.DeleteAsync(new Uri($"chats/{Id(chatId)}", UriKind.Relative), cancel)
            .ConfigureAwait(false);
        await EnsureAsync(response, cancel).ConfigureAwait(false);
    }

    public async Task ClearAsync(string chatId, CancellationToken cancel = default)
    {
        using var response = await PostRawAsync($"chats/{Id(chatId)}/clear", null, cancel).ConfigureAwait(false);
        await EnsureAsync(response, cancel).ConfigureAwait(false);
    }

    public async Task<ChatInfo> SetModelAsync(string chatId, string modelId, CancellationToken cancel = default) =>
        (await PostAsync($"chats/{Id(chatId)}/model", new ModelRequest(modelId), ApiJson.Default.ModelRequest,
            ApiJson.Default.ChatResponse, cancel).ConfigureAwait(false)).Chat;

    public async Task<bool> StopAsync(string chatId, CancellationToken cancel = default)
    {
        using var response = await PostRawAsync($"chats/{Id(chatId)}/stop", null, cancel).ConfigureAwait(false);
        await EnsureAsync(response, cancel).ConfigureAwait(false);
        var value = await response.Content.ReadFromJsonAsync(ApiJson.Default.StopResponse, cancel).ConfigureAwait(false);
        return value?.Stopped ?? false;
    }

    public IAsyncEnumerable<ChatEvent> SendAsync(
        string chatId, string text, IReadOnlyList<Attachment> attachments, CancellationToken cancel = default)
    {
        var body = new SendRequest(text, [.. attachments.Select(a => new SendAttachment(a.Name, a.Text))]);
        return StreamAsync($"chats/{Id(chatId)}/send", Body(body, ApiJson.Default.SendRequest), cancel);
    }

    public IAsyncEnumerable<ChatEvent> RegenerateAsync(string chatId, CancellationToken cancel = default) =>
        StreamAsync($"chats/{Id(chatId)}/regenerate", null, cancel);

    public Task<SystemReport> SystemAsync(CancellationToken cancel = default) =>
        GetAsync("system", ApiJson.Default.SystemReport, cancel);

    public Task<OnboardingState> OnboardingAsync(CancellationToken cancel = default) =>
        GetAsync("onboarding", ApiJson.Default.OnboardingState, cancel);

    public Task<OnboardingState> CompleteOnboardingAsync(string? modelId, CancellationToken cancel = default) =>
        PostAsync("onboarding", new ModelRequest(modelId), ApiJson.Default.ModelRequest,
            ApiJson.Default.OnboardingState, cancel);

    public Task<SettingsInfo> SettingsAsync(CancellationToken cancel = default) =>
        GetAsync("settings", ApiJson.Default.SettingsInfo, cancel);

    public Task<SettingsInfo> SaveSettingsAsync(SettingsUpdate update, CancellationToken cancel = default) =>
        PostAsync("settings", update, ApiJson.Default.SettingsUpdate, ApiJson.Default.SettingsInfo, cancel);

    public async Task<ModelInfo> ImportModelAsync(string path, CancellationToken cancel = default) =>
        (await PostAsync("models/import", new ImportRequest(path), ApiJson.Default.ImportRequest,
            ApiJson.Default.ImportResponse, cancel).ConfigureAwait(false)).Model;

    private async IAsyncEnumerable<ChatEvent> StreamAsync(
        string path, HttpContent? content, [EnumeratorCancellation] CancellationToken cancel)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, new Uri(path, UriKind.Relative))
        {
            Content = content ?? new StringContent("{}", System.Text.Encoding.UTF8, "application/json"),
        };
        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancel)
            .ConfigureAwait(false);
        await EnsureAsync(response, cancel).ConfigureAwait(false);
        using var stream = await response.Content.ReadAsStreamAsync(cancel).ConfigureAwait(false);
        using var reader = new StreamReader(stream);
        while (await reader.ReadLineAsync(cancel).ConfigureAwait(false) is { } line)
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal))
            {
                continue;
            }

            var item = JsonSerializer.Deserialize(line.AsSpan(6), ApiJson.Default.ChatEvent);
            if (item is not null)
            {
                yield return item;
            }
        }
    }

    private async Task<T> GetAsync<T>(string path, JsonTypeInfo<T> type, CancellationToken cancel)
    {
        using var response = await _http.GetAsync(new Uri(path, UriKind.Relative), cancel).ConfigureAwait(false);
        await EnsureAsync(response, cancel).ConfigureAwait(false);
        return await response.Content.ReadFromJsonAsync(type, cancel).ConfigureAwait(false)
            ?? throw new ApiException("api.internal", null);
    }

    private async Task<TResult> PostAsync<TBody, TResult>(
        string path, TBody body, JsonTypeInfo<TBody> bodyType, JsonTypeInfo<TResult> type, CancellationToken cancel)
    {
        using var response = await PostRawAsync(path, Body(body, bodyType), cancel).ConfigureAwait(false);
        await EnsureAsync(response, cancel).ConfigureAwait(false);
        return await response.Content.ReadFromJsonAsync(type, cancel).ConfigureAwait(false)
            ?? throw new ApiException("api.internal", null);
    }

    private Task<HttpResponseMessage> PostRawAsync(string path, HttpContent? content, CancellationToken cancel) =>
        _http.PostAsync(
            new Uri(path, UriKind.Relative),
            content ?? new StringContent("{}", System.Text.Encoding.UTF8, "application/json"),
            cancel);

    private static async Task EnsureAsync(HttpResponseMessage response, CancellationToken cancel)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        ErrorResponse? error = null;
        try
        {
            error = await response.Content.ReadFromJsonAsync(ApiJson.Default.ErrorResponse, cancel).ConfigureAwait(false);
        }
        catch (JsonException)
        {
        }

        throw new ApiException(error?.Error.Code ?? "api.internal", error?.Error.Params);
    }

    private static string Id(string chatId) => Uri.EscapeDataString(chatId);

    /// <summary>
    /// Тело JSON с известной длиной. JsonContent отправляет без Content-Length (chunked),
    /// а http.server сервиса такое тело не читает и видит пустой запрос.
    /// </summary>
    private static ByteArrayContent Body<T>(T value, JsonTypeInfo<T> type)
    {
        var content = new ByteArrayContent(JsonSerializer.SerializeToUtf8Bytes(value, type));
        content.Headers.ContentType = new MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };
        return content;
    }
}
