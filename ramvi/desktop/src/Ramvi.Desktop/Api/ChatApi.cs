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
    Task<JsonElement> AgentAsync(string path, JsonElement? body = null, CancellationToken cancel = default) =>
        throw new NotSupportedException();

    Task<ServiceStatus> StatusAsync(CancellationToken cancel = default);

    /// <summary>8.3: состояние обновлений комплекта.</summary>
    Task<UpdateStatus> UpdatesAsync(CancellationToken cancel = default) => throw new NotSupportedException();

    /// <summary>8.3: подготовить обновление из файла (применит оболочка при перезапуске).</summary>
    Task<UpdateStatus> StageUpdateAsync(string path, CancellationToken cancel = default) =>
        throw new NotSupportedException();

    /// <summary>7.2: клиенты внешнего API со счётчиками, адрес и очередь.</summary>
    Task<ConnectionsInfo> ConnectionsAsync(CancellationToken cancel = default) =>
        throw new NotSupportedException();

    /// <summary>7.2: новый ключ — показывается один раз.</summary>
    Task<NewKeyResponse> CreateKeyAsync(string name, IReadOnlyList<string> permissions, CancellationToken cancel = default) =>
        throw new NotSupportedException();

    /// <summary>7.2: отозвать ключ; <paramref name="cancelRunning"/> — остановить и идущие запросы.</summary>
    Task<RevokeResponse> RevokeKeyAsync(string id, bool cancelRunning, CancellationToken cancel = default) =>
        throw new NotSupportedException();

    Task<IReadOnlyList<ModelInfo>> ModelsAsync(CancellationToken cancel = default);

    /// <summary>Просмотреть папку моделей из настроек и подключить найденные модели на месте.</summary>
    Task<ScanResult> ScanModelsAsync(CancellationToken cancel = default);

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

    /// <summary>3.17: начать загрузку выбранной модели сразу, не дожидаясь сообщения.</summary>
    Task<LoadResponse> LoadModelAsync(string modelId, CancellationToken cancel = default);

    /// <summary>3.17: «Выгрузить модель» — ответы останавливаются, память освобождается.</summary>
    Task<UnloadResponse> UnloadModelAsync(CancellationToken cancel = default);

    /// <summary>3.14: переводы установленных модулей (язык → ключ → текст).</summary>
    Task<IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>> ModuleTextsAsync(
        CancellationToken cancel = default);

    /// <summary>3.11: разделы интерфейса.</summary>
    Task<IReadOnlyList<SectionInfo>> SectionsAsync(CancellationToken cancel = default);

    /// <summary>Открыть раздел: модуль загружается при первом открытии.</summary>
    Task<SectionInfo> OpenSectionAsync(string sectionId, CancellationToken cancel = default);

    /// <summary>4.2: каталог — последняя сохранённая копия (без сети).</summary>
    Task<CatalogView> CatalogAsync(CancellationToken cancel = default);

    /// <summary>4.2: обновить каталог у провайдеров.</summary>
    Task<CatalogView> RefreshCatalogAsync(CancellationToken cancel = default);

    /// <summary>4.3: загрузки моделей.</summary>
    Task<IReadOnlyList<DownloadInfo>> DownloadsAsync(CancellationToken cancel = default);

    Task<DownloadInfo> StartDownloadAsync(string entryId, CancellationToken cancel = default);

    /// <summary>action: pause / resume / cancel.</summary>
    Task<DownloadInfo> DownloadActionAsync(string downloadId, string action, CancellationToken cancel = default);

    /// <summary>4.6: выбор модели для профиля задачи.</summary>
    Task<Recommendation> RecommendationsAsync(CancellationToken cancel = default);

    /// <summary>4.11: модель по умолчанию для новых чатов.</summary>
    Task SetDefaultModelAsync(string modelId, CancellationToken cancel = default);

    /// <summary>4.11: забыть модель; файл удаляется, только если его скачал RAMVI и это выбрано.</summary>
    Task<DeleteModelResult> DeleteModelAsync(string modelId, bool deleteFile, CancellationToken cancel = default);

    /// <summary>5.2: состояние голосового ввода и модели распознавания.</summary>
    Task<VoiceStatus> VoiceStatusAsync(CancellationToken cancel = default);

    /// <summary>5.2: микрофоны системы.</summary>
    Task<IReadOnlyList<VoiceDevice>> VoiceDevicesAsync(CancellationToken cancel = default);

    /// <summary>5.2: начать запись (микрофон из настроек, если не указан).</summary>
    Task<VoiceStatus> StartVoiceAsync(string? device = null, CancellationToken cancel = default);

    /// <summary>5.2: остановить запись и распознать.</summary>
    Task<VoiceResult> StopVoiceAsync(CancellationToken cancel = default);

    /// <summary>5.2: прекратить запись без распознавания.</summary>
    Task<VoiceStatus> CancelVoiceAsync(CancellationToken cancel = default);

    /// <summary>5.2: скачать модель распознавания (по умолчанию — выбранную в настройках).</summary>
    Task<VoiceStatus> DownloadVoiceModelAsync(string? model = null, CancellationToken cancel = default);

    /// <summary>5.2: прервать скачивание модели распознавания.</summary>
    Task<VoiceStatus> CancelVoiceDownloadAsync(CancellationToken cancel = default);

    /// <summary>5.4: скачать ускоритель распознавания на видеокарте (Windows + NVIDIA).</summary>
    Task<VoiceStatus> DownloadVoiceAccelAsync(CancellationToken cancel = default);

    /// <summary>5.7: сохранить ключ API в хранилище системы; пустая строка — удалить.</summary>
    Task<VoiceStatus> SetVoiceApiKeyAsync(string key, CancellationToken cancel = default);

    /// <summary>5.7: «Проверить» API из формы (адрес и модель ещё могут быть не сохранены).</summary>
    Task<VoiceCheck> CheckVoiceApiAsync(string? url, string? model, CancellationToken cancel = default);
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

    public Task<ScanResult> ScanModelsAsync(CancellationToken cancel = default) =>
        PostAsync("models/scan", new ModelRequest(null), ApiJson.Default.ModelRequest, ApiJson.Default.ScanResult, cancel);

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

    public Task<LoadResponse> LoadModelAsync(string modelId, CancellationToken cancel = default) =>
        PostAsync("engine/load", new ModelRequest(modelId), ApiJson.Default.ModelRequest,
            ApiJson.Default.LoadResponse, cancel);

    public Task<UnloadResponse> UnloadModelAsync(CancellationToken cancel = default) =>
        PostAsync("engine/unload", new ModelRequest(null), ApiJson.Default.ModelRequest,
            ApiJson.Default.UnloadResponse, cancel);

    public async Task<IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>> ModuleTextsAsync(
        CancellationToken cancel = default)
    {
        var response = await GetAsync("i18n", ApiJson.Default.ModuleTextsResponse, cancel).ConfigureAwait(false);
        return response.Modules.ToDictionary(
            p => p.Key, p => (IReadOnlyDictionary<string, string>)p.Value, StringComparer.Ordinal);
    }

    public Task<SystemReport> SystemAsync(CancellationToken cancel = default) =>
        GetAsync("system", ApiJson.Default.SystemReport, cancel);

    public Task<OnboardingState> OnboardingAsync(CancellationToken cancel = default) =>
        GetAsync("onboarding", ApiJson.Default.OnboardingState, cancel);

    public Task<OnboardingState> CompleteOnboardingAsync(string? modelId, CancellationToken cancel = default) =>
        PostAsync("onboarding", new ModelRequest(modelId), ApiJson.Default.ModelRequest,
            ApiJson.Default.OnboardingState, cancel);

    public async Task<IReadOnlyList<SectionInfo>> SectionsAsync(CancellationToken cancel = default) =>
        (await GetAsync("sections", ApiJson.Default.SectionsResponse, cancel).ConfigureAwait(false)).Sections;

    public async Task<SectionInfo> OpenSectionAsync(string sectionId, CancellationToken cancel = default) =>
        (await PostAsync($"sections/{Id(sectionId)}/open", new ModelRequest(null), ApiJson.Default.ModelRequest,
            ApiJson.Default.SectionResponse, cancel).ConfigureAwait(false)).Section;

    public Task<SettingsInfo> SettingsAsync(CancellationToken cancel = default) =>
        GetAsync("settings", ApiJson.Default.SettingsInfo, cancel);

    public Task<UpdateStatus> UpdatesAsync(CancellationToken cancel = default) =>
        GetAsync("updates", ApiJson.Default.UpdateStatus, cancel);

    public Task<UpdateStatus> StageUpdateAsync(string path, CancellationToken cancel = default) =>
        PostAsync("updates/stage", new PathRequest(path), ApiJson.Default.PathRequest, ApiJson.Default.UpdateStatus, cancel);

    public Task<ConnectionsInfo> ConnectionsAsync(CancellationToken cancel = default) =>
        GetAsync("connections", ApiJson.Default.ConnectionsInfo, cancel);

    public Task<NewKeyResponse> CreateKeyAsync(
        string name, IReadOnlyList<string> permissions, CancellationToken cancel = default) =>
        PostAsync("connections/keys", new NewKeyRequest(name, permissions), ApiJson.Default.NewKeyRequest,
            ApiJson.Default.NewKeyResponse, cancel);

    public Task<RevokeResponse> RevokeKeyAsync(string id, bool cancelRunning, CancellationToken cancel = default) =>
        PostAsync($"connections/keys/{Id(id)}/revoke", new RevokeRequest(cancelRunning),
            ApiJson.Default.RevokeRequest, ApiJson.Default.RevokeResponse, cancel);

    public Task<SettingsInfo> SaveSettingsAsync(SettingsUpdate update, CancellationToken cancel = default) =>
        PostAsync("settings", update, ApiJson.Default.SettingsUpdate, ApiJson.Default.SettingsInfo, cancel);

    public async Task<ModelInfo> ImportModelAsync(string path, CancellationToken cancel = default) =>
        (await PostAsync("models/import", new ImportRequest(path), ApiJson.Default.ImportRequest,
            ApiJson.Default.ImportResponse, cancel).ConfigureAwait(false)).Model;

    public Task<CatalogView> CatalogAsync(CancellationToken cancel = default) =>
        GetAsync("catalog", ApiJson.Default.CatalogView, cancel);

    public Task<CatalogView> RefreshCatalogAsync(CancellationToken cancel = default) =>
        PostAsync("catalog/refresh", new ModelRequest(null), ApiJson.Default.ModelRequest,
            ApiJson.Default.CatalogView, cancel);

    public async Task<IReadOnlyList<DownloadInfo>> DownloadsAsync(CancellationToken cancel = default) =>
        (await GetAsync("downloads", ApiJson.Default.DownloadsResponse, cancel).ConfigureAwait(false)).Downloads;

    public async Task<DownloadInfo> StartDownloadAsync(string entryId, CancellationToken cancel = default) =>
        (await PostAsync("downloads", new EntryRequest(entryId), ApiJson.Default.EntryRequest,
            ApiJson.Default.DownloadResponse, cancel).ConfigureAwait(false)).Download;

    public async Task<DownloadInfo> DownloadActionAsync(
        string downloadId, string action, CancellationToken cancel = default) =>
        (await PostAsync($"downloads/{Id(downloadId)}/{Id(action)}", new ModelRequest(null),
            ApiJson.Default.ModelRequest, ApiJson.Default.DownloadResponse, cancel).ConfigureAwait(false)).Download;

    public Task<Recommendation> RecommendationsAsync(CancellationToken cancel = default) =>
        GetAsync("recommendations", ApiJson.Default.Recommendation, cancel);

    public async Task SetDefaultModelAsync(string modelId, CancellationToken cancel = default) =>
        await PostAsync($"models/{Id(modelId)}/default", new ModelRequest(null), ApiJson.Default.ModelRequest,
            ApiJson.Default.DefaultModelResponse, cancel).ConfigureAwait(false);

    public Task<DeleteModelResult> DeleteModelAsync(string modelId, bool deleteFile, CancellationToken cancel = default) =>
        PostAsync($"models/{Id(modelId)}/delete", new DeleteModelRequest(deleteFile),
            ApiJson.Default.DeleteModelRequest, ApiJson.Default.DeleteModelResult, cancel);

    public Task<VoiceStatus> VoiceStatusAsync(CancellationToken cancel = default) =>
        GetAsync("voice", ApiJson.Default.VoiceStatus, cancel);

    public async Task<IReadOnlyList<VoiceDevice>> VoiceDevicesAsync(CancellationToken cancel = default) =>
        (await GetAsync("voice/devices", ApiJson.Default.VoiceDevicesResponse, cancel).ConfigureAwait(false)).Devices;

    public Task<VoiceStatus> StartVoiceAsync(string? device = null, CancellationToken cancel = default) =>
        PostAsync("voice/start", new DeviceRequest(device), ApiJson.Default.DeviceRequest,
            ApiJson.Default.VoiceStatus, cancel);

    public Task<VoiceResult> StopVoiceAsync(CancellationToken cancel = default) =>
        PostAsync("voice/stop", new ModelRequest(null), ApiJson.Default.ModelRequest, ApiJson.Default.VoiceResult, cancel);

    public Task<VoiceStatus> CancelVoiceAsync(CancellationToken cancel = default) =>
        PostAsync("voice/cancel", new ModelRequest(null), ApiJson.Default.ModelRequest, ApiJson.Default.VoiceStatus, cancel);

    public Task<VoiceStatus> DownloadVoiceModelAsync(string? model = null, CancellationToken cancel = default) =>
        PostAsync("voice/model/download", new VoiceModelRequest(model), ApiJson.Default.VoiceModelRequest,
            ApiJson.Default.VoiceStatus, cancel);

    public Task<VoiceStatus> CancelVoiceDownloadAsync(CancellationToken cancel = default) =>
        PostAsync("voice/model/cancel", new ModelRequest(null), ApiJson.Default.ModelRequest,
            ApiJson.Default.VoiceStatus, cancel);

    public Task<VoiceStatus> DownloadVoiceAccelAsync(CancellationToken cancel = default) =>
        PostAsync("voice/accel/download", new ModelRequest(null), ApiJson.Default.ModelRequest,
            ApiJson.Default.VoiceStatus, cancel);

    public Task<VoiceStatus> SetVoiceApiKeyAsync(string key, CancellationToken cancel = default) =>
        PostAsync("voice/api/key", new KeyRequest(key), ApiJson.Default.KeyRequest, ApiJson.Default.VoiceStatus, cancel);

    public Task<VoiceCheck> CheckVoiceApiAsync(string? url, string? model, CancellationToken cancel = default) =>
        PostAsync("voice/api/check", new ApiCheckRequest(url, model), ApiJson.Default.ApiCheckRequest,
            ApiJson.Default.VoiceCheck, cancel);

    public Task<JsonElement> AgentAsync(string path, JsonElement? body = null, CancellationToken cancel = default) =>
        body is { } payload
            ? PostAsync("agent/" + path, payload, ApiJson.Default.JsonElement, ApiJson.Default.JsonElement, cancel)
            : GetAsync("agent/" + path, ApiJson.Default.JsonElement, cancel);

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
