using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ramvi.Desktop.Api;

// Контракт HTTP API сервиса `ramvi_service.ui_api` (3.4). Имена полей — snake_case.

public sealed record ChatInfo(
    string Id,
    string Title,
    string? ModelId,
    double CreatedAt,
    double UpdatedAt,
    bool Generating = false);

public sealed record Attachment(string Name, string Text, int Bytes = 0);

public sealed record MessageInfo(
    string Id,
    string ChatId,
    string Role,
    string Content,
    double CreatedAt,
    string Reasoning = "",
    string? Finish = null,
    IReadOnlyList<Attachment>? Attachments = null);

public sealed record ModelInfo(
    string Id,
    string Architecture,
    string? Quant,
    long? Bytes,
    Dictionary<string, JsonElement>? Capabilities = null,
    CompatInfo? Compat = null,
    string? Provider = null,
    CloudModelInfo? Cloud = null)
{
    /// <summary>7.8: модель считает облачный сервис — запросы уходят туда.</summary>
    [JsonIgnore]
    public bool IsCloud => Provider == "cloud";

    /// <summary>Имя в списке выбора: у облачной — модель и подключение.</summary>
    [JsonIgnore]
    public string Label => Cloud is { } cloud ? $"{cloud.Model} · {cloud.Name}" : Id;
}

/// <summary>7.8: куда уходят запросы облачной модели.</summary>
public sealed record CloudModelInfo(string Connection, string Name, string Kind, string Host, string Model);

// --- 7.7/7.8: облачные модели по API -------------------------------------------------------

public sealed record CloudKind(string Kind, string Url);

/// <summary>Подключение: ключ не возвращается — только HasKey.</summary>
public sealed record CloudConnection(string Id, string Kind, string Name, string Url, bool HasKey, IReadOnlyList<string> Models);

/// <summary>Вход через ChatGPT (Codex CLI): установлен ли codex, вошёл ли, идёт ли вход.</summary>
public sealed record ChatGptState(bool Installed, bool LoggedIn, bool LoginRunning = false, bool LoginFailed = false);

public sealed record CloudView(IReadOnlyList<CloudKind> Kinds, IReadOnlyList<CloudConnection> Connections, CloudSpending? Spending = null);

// --- 7.10: стоимость и лимиты ---

/// <summary>Цена за миллион токенов (USD), источник и дата, когда её вписали.</summary>
public sealed record CloudPrice(double Input, double Output, string Source, string CheckedOn);

public sealed record CloudLimits(double? Daily, double? Monthly);

/// <summary>Один облачный ответ: токены и стоимость (null — цена не задана).</summary>
public sealed record CloudUsageRow(double At, string Model, string Source, long PromptTokens, long CompletionTokens, double? Cost, string State);

public sealed record CloudSpending(
    string Currency,
    CloudLimits Limits,
    double Today,
    double Month,
    int UnknownToday,
    int UnknownMonth,
    Dictionary<string, CloudPrice> Prices,
    IReadOnlyList<CloudUsageRow> Recent);

public sealed record CloudLimitsRequest(double? Daily, double? Monthly);

public sealed record CloudLimitsResponse(CloudSpending Spending);

/// <summary>Input/Output null — убрать цену.</summary>
public sealed record CloudPriceRequest(string Model, double? Input, double? Output, string? Source);

public sealed record CloudPriceResponse(string Model, CloudPrice? Price);

/// <summary>Key: строка — заменить, "" — удалить, null — оставить прежний.</summary>
public sealed record CloudSaveRequest(string? Id, string Kind, string Name, string Url, string? Key);

public sealed record CloudConnectionResponse(CloudConnection Connection);

/// <summary>Сохранённое подключение (Id) или значения формы (Kind, Url, Key).</summary>
public sealed record CloudModelsRequest(string? Id, string? Kind = null, string? Url = null, string? Key = null);

public sealed record CloudModelsResponse(IReadOnlyList<string> Models);

public sealed record CloudEnableRequest(string Model, bool Enabled);

public sealed record CloudEnableResponse(string Id, IReadOnlyList<string> Models);

// --- 4.11: каталог, загрузки, совместимость, рекомендации ---------------------------------

/// <summary>План слоёв модели на этом компьютере (4.4).</summary>
public sealed record PlanInfo(int Layers, int Resident, int Gpu, int Stream, int Cpu, int Hyb, long OffGpuBytes);

/// <summary>Архитектура, размер (млрд параметров) и квант из заголовка GGUF.</summary>
public sealed record CompatModel(string Key, double? SizeB, string? Quant);

/// <summary>Подходит ли модель задаче профиля (4.5).</summary>
public sealed record TaskFit(bool Suits, IReadOnlyList<JsonElement>? Reasons);

/// <summary>4.7: замер контрольного прогона на этом компьютере (фактические ресурсы).</summary>
public sealed record ControlInfo(
    bool Stable, double? TokS = null, double? TtftS = null, long? PeakRam = null, long? PeakVram = null,
    int? Resident = null);

/// <summary>Статус совместимости, причины (сообщения), скорость (сообщение) и план.</summary>
public sealed record CompatInfo(
    string Status,
    IReadOnlyList<JsonElement>? Reasons = null,
    JsonElement? Speed = null,
    PlanInfo? Plan = null,
    CompatModel? Model = null,
    TaskFit? Task = null,
    ControlInfo? Control = null);

public sealed record CatalogEntryInfo(
    string Id,
    string Provider,
    string Name,
    string Source,
    long Bytes,
    string Architecture,
    string Quantization,
    string License,
    int? ContextLength = null,
    string? Parameters = null,
    CompatInfo? Compat = null);

public sealed record CatalogProviderInfo(double? FetchedAt, ApiError? Error, int Shown, int Hidden);

public sealed record CatalogView(
    Dictionary<string, CatalogProviderInfo> Providers, IReadOnlyList<CatalogEntryInfo> Entries, bool Stale);

/// <summary>Загрузка модели (4.3): queued/running/verifying/done/paused/failed/cancelled.</summary>
public sealed record DownloadInfo(
    string Id, string EntryId, string State, long BytesDone, long? TotalBytes, string File, ApiError? Error = null,
    string? ModelId = null);

public sealed record RecommendedItem(string Id, string Name, bool Installed, JsonElement? Reason = null);

/// <summary>4.6: лучшая модель, причина (сообщение) и альтернативы.</summary>
public sealed record Recommendation(
    string Priority, RecommendedItem? Best, JsonElement Reason, IReadOnlyList<RecommendedItem> Alternatives);

public sealed record DeleteModelResult(bool Deleted, bool FileDeleted);

internal sealed record DownloadsResponse(IReadOnlyList<DownloadInfo> Downloads);

internal sealed record DownloadResponse(DownloadInfo Download);

internal sealed record EntryRequest(string EntryId);

internal sealed record DeleteModelRequest(bool DeleteFile);

internal sealed record DefaultModelResponse(string DefaultModel);

public sealed record Limits(
    int MaxTextChars,
    int MaxFiles,
    int MaxFileBytes,
    int MaxTotalBytes,
    int MaxNameChars);

/// <summary>Движок: загруженная модель, модель в загрузке и причина сбоя загрузки (3.5, 3.17).</summary>
public sealed record EngineStatus(string? ModelId, string? Loading, string State, ApiError? Problem = null);

public sealed record ServiceStatus(EngineStatus Engine, Limits Limits);

public sealed record ApiError(string Code, Dictionary<string, JsonElement>? Params);

/// <summary>Событие потока ответа: user, regenerate, status, reasoning, text, done.</summary>
public sealed record ChatEvent(
    string Type,
    string? Text = null,
    MessageInfo? Message = null,
    string? State = null,
    string? ModelId = null,
    string? Finish = null,
    ApiError? Error = null,
    IReadOnlyList<string>? Removed = null);

/// <summary>Одна проверка мастера: статус ok / warn / fail / unknown и текст по ключу.</summary>
public sealed record SystemCheck(string Id, string Status, string Code, Dictionary<string, JsonElement>? Params);

/// <summary>CloudOnly (B.12): локальные модели здесь не работают (Mac) — только облачные.</summary>
public sealed record SystemReport(IReadOnlyList<SystemCheck> Checks, bool CloudOnly = false);

public sealed record OnboardingState(bool Done, string? DefaultModel);

/// <summary>3.7: настройки приложения (как в `ramvi_service.settings.Settings`).</summary>
public sealed record AppSettings(
    string DataDir, string ModelsDir, string Language, int ApiPort, bool NetworkAccess, string Theme = "system",
    string VoiceDevice = "", string VoiceLanguage = "auto", string VoiceModel = "large-v3-turbo-q5_0",
    string VoiceBackend = "local", string VoiceApiUrl = "", string VoiceApiModel = "whisper-1",
    bool VoiceAutoSend = true, string NetworkAllow = "192.168.0.0/16, 10.0.0.0/8, 172.16.0.0/12",
    int AttachMaxFiles = 5, int AttachMaxFileKb = 128, int AttachMaxTotalKb = 256,
    int KvCacheRamMb = 2048, int KvCacheDiskMb = 8192);

public sealed record SettingsFolders(string DataDir, string ModelsDir);

public sealed record SettingsInfo(
    AppSettings Settings,
    SettingsFolders Resolved,
    string CurrentDataDir,
    IReadOnlyList<string> Languages,
    IReadOnlyList<string>? RestartRequired = null,
    string? ApiError = null,
    string? ModelsDirWarning = null);

/// <summary>Только изменённые поля; null не отправляется.</summary>
public sealed record SettingsUpdate(
    string? DataDir = null, string? ModelsDir = null, string? Language = null, int? ApiPort = null,
    bool? NetworkAccess = null, string? Theme = null, string? VoiceDevice = null, string? VoiceLanguage = null,
    string? VoiceModel = null, string? VoiceBackend = null, string? VoiceApiUrl = null,
    string? VoiceApiModel = null, bool? VoiceAutoSend = null, string? NetworkAllow = null,
    int? AttachMaxFiles = null, int? AttachMaxFileKb = null, int? AttachMaxTotalKb = null,
    int? KvCacheRamMb = null, int? KvCacheDiskMb = null);

/// <summary>8.3: обновления — текущая версия, подготовленная, итог прошлого обновления.</summary>
public sealed record UpdateStatus(
    string Current, string? Staged = null, string? Applied = null, string? RolledBack = null,
    string? Health = null, string? Notes = null);

public sealed record PathRequest(string Path);

/// <summary>7.1–7.4: внешний API — адрес для программ на этом компьютере, порт, работает ли.</summary>
public sealed record ApiState(string? Url, int Port, bool Running);

/// <summary>7.3: очередь клиентов внешнего API.</summary>
public sealed record QueueState(int MaxConcurrent, int PerClientQueue, int MaxClients, int ActiveClients);

/// <summary>7.2: клиент внешнего API — без ключа: подсказка, права, счётчики.</summary>
public sealed record ClientInfo(
    string Id, string Name, IReadOnlyList<string> Permissions, double Created, double? LastSeen, string State,
    double? Revoked, string KeyHint, long Requests, long PromptTokens, long CompletionTokens,
    double BusySeconds, long Errors, int Running);

public sealed record ConnectionsInfo(ApiState Api, IReadOnlyList<ClientInfo> Clients, QueueState? Queue = null);

public sealed record NewKeyRequest(string Name, IReadOnlyList<string> Permissions);

/// <summary>Ключ целиком — только в этом ответе.</summary>
public sealed record NewKeyResponse(ClientInfo Client, string Key);

public sealed record RevokeRequest(bool CancelRunning);

public sealed record RevokeResponse(string Revoked, int Stopped);

/// <summary>5.2: модель распознавания речи — скачана ли, размер, держится ли в памяти.</summary>
public sealed record VoiceModelState(string Id, bool Installed, long Size, bool Loaded = false);

public sealed record VoiceModelOption(string Id, long Size, bool Installed = false);

/// <summary>5.2: скачивание модели распознавания (idle, running, done, failed).</summary>
public sealed record VoiceDownload(string State, string? Model = null, long? Done = null, long? Total = null,
    ApiError? Error = null);

/// <summary>5.2: состояние голосового ввода: idle, recording, transcribing.</summary>
public sealed record VoiceStatus(
    string State,
    double Seconds,
    double Level,
    bool LimitReached,
    double MaxSeconds,
    string Language,
    IReadOnlyList<string> Languages,
    string Device,
    VoiceModelState Model,
    IReadOnlyList<VoiceModelOption> Models,
    VoiceDownload Download,
    string? Backend = null,
    VoiceLast? Last = null,
    VoiceAccel? Accel = null,
    VoiceApi? Api = null,
    bool AutoSend = true);

/// <summary>5.7: распознавание через сторонний API; ключ — только «задан/нет».</summary>
public sealed record VoiceApi(bool Enabled, string Url, string Host, string Model, bool KeySet);

/// <summary>5.7: «Проверить» — сервис ответил.</summary>
public sealed record VoiceCheck(bool Ok, string Host, double Seconds);

internal sealed record KeyRequest(string Key);

internal sealed record ApiCheckRequest(string? Url, string? Model);

/// <summary>5.4: где прошло последнее распознавание: gpu или cpu.</summary>
public sealed record VoiceLast(string Backend, string Model, double ProcS);

/// <summary>5.4: ускоритель на видеокарте — предлагается ли (Windows + NVIDIA) и скачан ли.</summary>
public sealed record VoiceAccel(string Id, bool Offered, bool Installed, long Size);

public sealed record VoiceDevice(string Name, bool Default);

internal sealed record VoiceDevicesResponse(IReadOnlyList<VoiceDevice> Devices);

/// <summary>5.2: распознанный текст — в поле ввода, не в чат.</summary>
public sealed record VoiceResult(
    string Text, string? Language, double AudioS, double ProcS, string? Backend = null, string? Model = null);

internal sealed record DeviceRequest(string? Device);

internal sealed record VoiceModelRequest(string? Model);

/// <summary>3.11: раздел интерфейса (встроенный или модуля) и его состояние.</summary>
public sealed record SectionInfo(
    string Id, string TitleKey, string Icon, int Order, string View, string? ModuleId, string State, string? Error);

internal sealed record SectionsResponse(IReadOnlyList<SectionInfo> Sections);

internal sealed record SectionResponse(SectionInfo Section);

internal sealed record ImportRequest(string Path);

internal sealed record ImportResponse(ModelInfo Model);

/// <summary>Пропущенный при просмотре папки файл: путь и код причины.</summary>
public sealed record ScanSkip(string Path, string Code);

/// <summary>«Обновить список»: папка моделей просмотрена, новые модели подключены на месте.</summary>
public sealed record ScanResult(
    string Folder, IReadOnlyList<string> Added, IReadOnlyList<ScanSkip> Skipped, IReadOnlyList<ModelInfo> Models);

internal sealed record ChatsResponse(IReadOnlyList<ChatInfo> Chats);

internal sealed record ChatResponse(ChatInfo Chat);

internal sealed record MessagesResponse(IReadOnlyList<MessageInfo> Messages);

internal sealed record ModelsResponse(IReadOnlyList<ModelInfo> Models);

internal sealed record ErrorResponse(ApiError Error);

internal sealed record SendRequest(string Text, IReadOnlyList<SendAttachment> Attachments);

internal sealed record SendAttachment(string Name, string Text);

internal sealed record ModelRequest(string? ModelId);

internal sealed record StopResponse(bool Stopped);

/// <summary>3.14: переводы модулей — язык → ключ → текст.</summary>
internal sealed record ModuleTextsResponse(Dictionary<string, Dictionary<string, string>> Modules);

/// <summary>3.17: ответ на «загрузить модель»: started=false — уже загружена или грузится.</summary>
public sealed record LoadResponse(bool Started, EngineStatus Engine);

/// <summary>3.17: ответ на «Выгрузить модель»: unloaded=false — модель не была загружена.</summary>
public sealed record UnloadResponse(bool Unloaded, EngineStatus Engine);

[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(JsonElement))]
[JsonSerializable(typeof(ChatsResponse))]
[JsonSerializable(typeof(ChatResponse))]
[JsonSerializable(typeof(MessagesResponse))]
[JsonSerializable(typeof(ModelsResponse))]
[JsonSerializable(typeof(ServiceStatus))]
[JsonSerializable(typeof(ErrorResponse))]
[JsonSerializable(typeof(ChatEvent))]
[JsonSerializable(typeof(SendRequest))]
[JsonSerializable(typeof(ModelRequest))]
[JsonSerializable(typeof(StopResponse))]
[JsonSerializable(typeof(ModuleTextsResponse))]
[JsonSerializable(typeof(LoadResponse))]
[JsonSerializable(typeof(UnloadResponse))]
[JsonSerializable(typeof(SystemReport))]
[JsonSerializable(typeof(OnboardingState))]
[JsonSerializable(typeof(ImportRequest))]
[JsonSerializable(typeof(ImportResponse))]
[JsonSerializable(typeof(ScanResult))]
[JsonSerializable(typeof(SettingsInfo))]
[JsonSerializable(typeof(SectionsResponse))]
[JsonSerializable(typeof(SectionResponse))]
[JsonSerializable(typeof(SettingsUpdate))]
[JsonSerializable(typeof(CatalogView))]
[JsonSerializable(typeof(DownloadsResponse))]
[JsonSerializable(typeof(DownloadResponse))]
[JsonSerializable(typeof(EntryRequest))]
[JsonSerializable(typeof(Recommendation))]
[JsonSerializable(typeof(DeleteModelRequest))]
[JsonSerializable(typeof(DeleteModelResult))]
[JsonSerializable(typeof(DefaultModelResponse))]
[JsonSerializable(typeof(VoiceStatus))]
[JsonSerializable(typeof(VoiceDevicesResponse))]
[JsonSerializable(typeof(VoiceResult))]
[JsonSerializable(typeof(DeviceRequest))]
[JsonSerializable(typeof(VoiceModelRequest))]
[JsonSerializable(typeof(VoiceCheck))]
[JsonSerializable(typeof(KeyRequest))]
[JsonSerializable(typeof(ApiCheckRequest))]
[JsonSerializable(typeof(ConnectionsInfo))]
[JsonSerializable(typeof(UpdateStatus))]
[JsonSerializable(typeof(PathRequest))]
[JsonSerializable(typeof(NewKeyRequest))]
[JsonSerializable(typeof(NewKeyResponse))]
[JsonSerializable(typeof(RevokeRequest))]
[JsonSerializable(typeof(RevokeResponse))]
[JsonSerializable(typeof(CloudView))]
[JsonSerializable(typeof(ChatGptState))]
[JsonSerializable(typeof(CloudSaveRequest))]
[JsonSerializable(typeof(CloudConnectionResponse))]
[JsonSerializable(typeof(CloudModelsRequest))]
[JsonSerializable(typeof(CloudModelsResponse))]
[JsonSerializable(typeof(CloudEnableRequest))]
[JsonSerializable(typeof(CloudEnableResponse))]
[JsonSerializable(typeof(CloudLimitsRequest))]
[JsonSerializable(typeof(CloudLimitsResponse))]
[JsonSerializable(typeof(CloudPriceRequest))]
[JsonSerializable(typeof(CloudPriceResponse))]
internal sealed partial class ApiJson : JsonSerializerContext;
