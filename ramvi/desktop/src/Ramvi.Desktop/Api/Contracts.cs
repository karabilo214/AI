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

public sealed record ModelInfo(string Id, string Architecture, string? Quant, long? Bytes);

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

public sealed record SystemReport(IReadOnlyList<SystemCheck> Checks);

public sealed record OnboardingState(bool Done, string? DefaultModel);

/// <summary>3.7: настройки приложения (как в `ramvi_service.settings.Settings`).</summary>
public sealed record AppSettings(
    string DataDir, string ModelsDir, string Language, int ApiPort, bool NetworkAccess, string Theme = "system");

public sealed record SettingsFolders(string DataDir, string ModelsDir);

public sealed record SettingsInfo(
    AppSettings Settings,
    SettingsFolders Resolved,
    string CurrentDataDir,
    IReadOnlyList<string> Languages,
    IReadOnlyList<string>? RestartRequired = null);

/// <summary>Только изменённые поля; null не отправляется.</summary>
public sealed record SettingsUpdate(
    string? DataDir = null, string? ModelsDir = null, string? Language = null, int? ApiPort = null,
    bool? NetworkAccess = null, string? Theme = null);

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
internal sealed partial class ApiJson : JsonSerializerContext;
