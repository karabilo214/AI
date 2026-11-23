using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading.Channels;
using Ramvi.Desktop.Api;

namespace Ramvi.Desktop.Tests;

/// <summary>API в памяти: тест сам выдаёт события потока через <see cref="Stream"/>.</summary>
internal sealed class FakeChatApi : IChatApi
{
    private int _next;
    public Func<string, string, JsonElement, Task<JsonElement>>? ModuleHandler { get; set; }
    public Task<JsonElement> ModuleCommandAsync(string section, string command, JsonElement body, CancellationToken cancel = default) =>
        ModuleHandler is { } handle ? handle(section, command, body) : throw new NotSupportedException();
    public Func<string, JsonElement?, JsonElement>? AgentHandler { get; set; }
    public Task<JsonElement> AgentAsync(string path, JsonElement? body = null, CancellationToken cancel = default) =>
        Task.FromResult(AgentHandler is { } handle ? handle(path, body) : throw new NotSupportedException());

    public List<ChatInfo> Chats { get; } = [];

    public Dictionary<string, List<MessageInfo>> Messages { get; } = [];

    public List<ModelInfo> Models { get; } =
    [
        new("model-a", "qwen3", "q4", 1),
        new("model-b", "qwen3", "q8", 2),
    ];

    public Limits Limits { get; set; } = new(1000, 2, 100, 150, 255);

    // --- 7.8: облачные подключения в памяти ---
    public List<CloudConnection> CloudConnections { get; } = [];
    public Dictionary<string, string> CloudKeys { get; } = [];
    public List<string> RemoteModels { get; set; } = ["gpt-b", "gpt-a", "gpt-codex"];
    public List<CloudModelsRequest> CloudChecks { get; } = [];
    public List<CloudSaveRequest> CloudSaves { get; } = [];
    public ApiException? CloudFail { get; set; }

    public CloudSpending Spending { get; set; } = new("USD", new(null, null), 0, 0, 0, 0, [], []);
    public List<(double? Daily, double? Monthly)> LimitCalls { get; } = [];
    public List<(string Connection, CloudPriceRequest Price)> PriceCalls { get; } = [];

    public Task<CloudView> CloudAsync(CancellationToken cancel = default) =>
        Task.FromResult(new CloudView([new("openai", "https://api.openai.com/v1")], [.. CloudConnections], Spending));

    public Task<CloudSpending> SetCloudLimitsAsync(double? daily, double? monthly, CancellationToken cancel = default)
    {
        LimitCalls.Add((daily, monthly));
        Spending = Spending with { Limits = new CloudLimits(daily, monthly) };
        return Task.FromResult(Spending);
    }

    public Task<CloudPriceResponse> SetCloudPriceAsync(string connection, CloudPriceRequest price, CancellationToken cancel = default)
    {
        PriceCalls.Add((connection, price));
        var id = $"cloud:{connection}:{price.Model}";
        var prices = new Dictionary<string, CloudPrice>(Spending.Prices);
        if (price.Input is null) prices.Remove(id);
        else prices[id] = new CloudPrice(price.Input.Value, price.Output!.Value, price.Source!, "2026-11-20");
        Spending = Spending with { Prices = prices };
        return Task.FromResult(new CloudPriceResponse(id, prices.GetValueOrDefault(id)));
    }

    public Task<CloudConnection> SaveCloudAsync(CloudSaveRequest request, CancellationToken cancel = default)
    {
        CloudSaves.Add(request);
        var id = request.Id ?? request.Name.ToLowerInvariant().Replace(' ', '-');
        if (request.Key is { Length: > 0 } key) CloudKeys[id] = key;
        else if (request.Key is "") CloudKeys.Remove(id);
        var models = CloudConnections.FirstOrDefault(c => c.Id == id)?.Models ?? [];
        var saved = new CloudConnection(id, request.Kind, request.Name, request.Url, CloudKeys.ContainsKey(id), models);
        CloudConnections.RemoveAll(c => c.Id == id);
        CloudConnections.Add(saved);
        return Task.FromResult(saved);
    }

    public Task<IReadOnlyList<string>> CloudModelsAsync(CloudModelsRequest request, CancellationToken cancel = default)
    {
        CloudChecks.Add(request);
        if (CloudFail is { } fail) throw fail;
        return Task.FromResult<IReadOnlyList<string>>([.. RemoteModels]);
    }

    public Task<CloudEnableResponse> EnableCloudModelAsync(string connection, string model, bool enabled, CancellationToken cancel = default)
    {
        if (CloudFail is { } fail) throw fail;
        var row = CloudConnections.Single(c => c.Id == connection);
        var models = enabled ? [.. row.Models.Where(m => m != model), model] : row.Models.Where(m => m != model).ToList();
        CloudConnections[CloudConnections.IndexOf(row)] = row with { Models = models };
        var id = $"cloud:{connection}:{model}";
        Models.RemoveAll(m => m.Id == id);
        if (enabled) Models.Add(new ModelInfo(id, "cloud", null, null, Provider: "cloud", Cloud: new CloudModelInfo(connection, row.Name, row.Kind, new Uri(row.Url).Host, model)));
        return Task.FromResult(new CloudEnableResponse(connection, models));
    }

    public Task DeleteCloudAsync(string connection, CancellationToken cancel = default)
    {
        CloudConnections.RemoveAll(c => c.Id == connection);
        CloudKeys.Remove(connection);
        Models.RemoveAll(m => m.Cloud?.Connection == connection);
        return Task.CompletedTask;
    }

    public List<string> Calls { get; } = [];

    public ApiException? FailNext { get; set; }

    /// <summary>Поток текущего ответа: тест пишет события и закрывает канал.</summary>
    public Channel<ChatEvent> Stream { get; private set; } = Channel.CreateUnbounded<ChatEvent>();

    public string AddChat(string title, string? model = "model-a", params MessageInfo[] messages)
    {
        var id = $"{_next++:x16}";
        Chats.Insert(0, new ChatInfo(id, title, model, 0, 0));
        Messages[id] = [.. messages.Select(m => m with { ChatId = id })];
        return id;
    }

    public static MessageInfo Msg(string role, string content, string reasoning = "", string? finish = null) =>
        new(Guid.NewGuid().ToString("N")[..16], "", role, content, 0, reasoning, finish);

    /// <summary>Состояние движка, которое видит оболочка в /status (3.17).</summary>
    public EngineStatus Engine { get; set; } = new(null, null, "stopped");

    /// <summary>true — загрузка не заканчивается сама: тест сам меняет <see cref="Engine"/>.</summary>
    public bool LoadPending { get; set; }

    /// <summary>Загрузки и выгрузки моделей — отдельно от <see cref="Calls"/>, чтобы прежние проверки не менялись.</summary>
    public List<string> EngineCalls { get; } = [];

    public Task<ServiceStatus> StatusAsync(CancellationToken cancel = default) =>
        Task.FromResult(new ServiceStatus(Engine, Limits));

    public Task<LoadResponse> LoadModelAsync(string modelId, CancellationToken cancel = default)
    {
        EngineCalls.Add($"load {modelId}");
        if (Engine.ModelId == modelId && Engine.State == "ready")
        {
            return Task.FromResult(new LoadResponse(false, Engine));
        }

        Engine = LoadPending ? new EngineStatus(null, modelId, "starting") : new EngineStatus(modelId, null, "ready");
        return Task.FromResult(new LoadResponse(true, Engine));
    }

    /// <summary>Переводы модулей, которые «отдаёт» сервис (3.14).</summary>
    public Dictionary<string, IReadOnlyDictionary<string, string>> ModuleTexts { get; } = [];

    public Task<IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>> ModuleTextsAsync(
        CancellationToken cancel = default) =>
        Task.FromResult<IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>>(ModuleTexts);

    public Task<UnloadResponse> UnloadModelAsync(CancellationToken cancel = default)
    {
        EngineCalls.Add("unload");
        var loaded = Engine.ModelId is not null || Engine.Loading is not null;
        Engine = new EngineStatus(null, null, "stopped");
        return Task.FromResult(new UnloadResponse(loaded, Engine));
    }

    /// <summary>Что «найдёт» просмотр папки моделей: модели добавляются в список.</summary>
    public List<ModelInfo> FolderModels { get; } = [];

    public int Scans { get; private set; }

    public Task<ScanResult> ScanModelsAsync(CancellationToken cancel = default)
    {
        Scans++;
        var added = FolderModels.Where(m => Models.All(x => x.Id != m.Id)).ToList();
        Models.AddRange(added);
        return Task.FromResult(new ScanResult("/app/models", [.. added.Select(m => m.Id)], [], [.. Models]));
    }

    public Task<IReadOnlyList<ModelInfo>> ModelsAsync(CancellationToken cancel = default) =>
        Task.FromResult<IReadOnlyList<ModelInfo>>(Models);

    public Task<IReadOnlyList<ChatInfo>> ChatsAsync(CancellationToken cancel = default) =>
        Task.FromResult<IReadOnlyList<ChatInfo>>([.. Chats]);

    public Task<ChatInfo> CreateChatAsync(string? modelId, CancellationToken cancel = default)
    {
        Calls.Add($"create {modelId}");
        var id = AddChat("", modelId);
        return Task.FromResult(Chats[0]);
    }

    public Task<IReadOnlyList<MessageInfo>> MessagesAsync(string chatId, CancellationToken cancel = default) =>
        Task.FromResult<IReadOnlyList<MessageInfo>>([.. Messages[chatId]]);

    public Task DeleteChatAsync(string chatId, CancellationToken cancel = default)
    {
        Calls.Add($"delete {chatId}");
        Chats.RemoveAll(c => c.Id == chatId);
        return Task.CompletedTask;
    }

    public Task ClearAsync(string chatId, CancellationToken cancel = default)
    {
        Calls.Add($"clear {chatId}");
        Messages[chatId].Clear();
        return Task.CompletedTask;
    }

    public Task<ChatInfo> SetModelAsync(string chatId, string modelId, CancellationToken cancel = default)
    {
        Calls.Add($"model {chatId} {modelId}");
        if (FailNext is { } error)
        {
            FailNext = null;
            throw error;
        }

        var index = Chats.FindIndex(c => c.Id == chatId);
        Chats[index] = Chats[index] with { ModelId = modelId };
        return Task.FromResult(Chats[index]);
    }

    public Task<bool> StopAsync(string chatId, CancellationToken cancel = default)
    {
        Calls.Add($"stop {chatId}");
        return Task.FromResult(true);
    }

    public IAsyncEnumerable<ChatEvent> SendAsync(
        string chatId, string text, IReadOnlyList<Attachment> attachments, CancellationToken cancel = default)
    {
        Calls.Add($"send {chatId} {text} files={attachments.Count}");
        return Read(cancel);
    }

    public IAsyncEnumerable<ChatEvent> RegenerateAsync(string chatId, CancellationToken cancel = default)
    {
        Calls.Add($"regenerate {chatId}");
        return Read(cancel);
    }

    private async IAsyncEnumerable<ChatEvent> Read([EnumeratorCancellation] CancellationToken cancel)
    {
        if (FailNext is { } error)
        {
            FailNext = null;
            throw error;
        }

        var stream = Stream;
        Stream = Channel.CreateUnbounded<ChatEvent>();
        await foreach (var item in stream.Reader.ReadAllAsync(cancel))
        {
            yield return item;
        }
    }

    public List<SystemCheck> SystemChecks { get; } =
    [
        new("os", "ok", "system.os", Json(("name", "Windows 11"), ("arch", "AMD64"))),
        new("gpu", "fail", "system.gpu_none", null),
    ];

    public OnboardingState Onboarding { get; set; } = new(true, null);

    /// <summary>Что вернёт импорт по пути: модель или ошибку.</summary>
    public Dictionary<string, object> Imports { get; } = [];

    /// <summary>B.12: система без локальных моделей (Mac).</summary>
    public bool CloudOnly { get; set; }

    public Task<SystemReport> SystemAsync(CancellationToken cancel = default)
    {
        Calls.Add("system");
        return Task.FromResult(new SystemReport(SystemChecks, CloudOnly));
    }

    public Task<OnboardingState> OnboardingAsync(CancellationToken cancel = default) => Task.FromResult(Onboarding);

    public Task<OnboardingState> CompleteOnboardingAsync(string? modelId, CancellationToken cancel = default)
    {
        Calls.Add($"complete {modelId ?? "-"}");
        Onboarding = new OnboardingState(true, modelId ?? Onboarding.DefaultModel);
        return Task.FromResult(Onboarding);
    }

    public Task<ModelInfo> ImportModelAsync(string path, CancellationToken cancel = default)
    {
        Calls.Add($"import {path}");
        return Imports[path] switch
        {
            ModelInfo model => AddModel(model),
            ApiException error => Task.FromException<ModelInfo>(error),
            _ => throw new InvalidOperationException(),
        };
    }

    private Task<ModelInfo> AddModel(ModelInfo model)
    {
        if (!Models.Contains(model))
        {
            Models.Add(model);
        }

        return Task.FromResult(model);
    }

    public List<SectionInfo> Sections { get; } = [new("chat", "section.chat", "chat", 10, "chat", null, "ready", null)];

    public Dictionary<string, SectionInfo> OpenResults { get; } = [];

    public Task<IReadOnlyList<SectionInfo>> SectionsAsync(CancellationToken cancel = default) =>
        Task.FromResult<IReadOnlyList<SectionInfo>>([.. Sections]);

    public Task<SectionInfo> OpenSectionAsync(string sectionId, CancellationToken cancel = default)
    {
        Calls.Add($"open {sectionId}");
        var info = OpenResults.TryGetValue(sectionId, out var result)
            ? result
            : Sections.Single(s => s.Id == sectionId) with { State = "active" };
        return Task.FromResult(info);
    }

    public AppSettings AppSettings { get; set; } = new("data", "models", "ru", 11435, false);

    public List<SettingsUpdate> SettingsUpdates { get; } = [];

    /// <summary>7.2: клиенты внешнего API; ключи и отзывы, которые попросил экран.</summary>
    public List<ClientInfo> Clients { get; } = [];

    public List<(string Name, IReadOnlyList<string> Permissions)> CreatedKeys { get; } = [];

    public List<(string Id, bool CancelRunning)> Revoked { get; } = [];

    public string? ApiUrl { get; set; } = "http://127.0.0.1:11435";

    public string? SaveApiError { get; set; }

    public static ClientInfo Client(string id, string name, string state = "active", params string[] permissions) =>
        new(id, name, permissions.Length > 0 ? permissions : ["chat", "models"], 1, null, state, null,
            "ramvi-…" + id[^4..], 3, 120, 45, 2.5, 1, 0);

    public Task<ConnectionsInfo> ConnectionsAsync(CancellationToken cancel = default) =>
        Task.FromResult(new ConnectionsInfo(
            new ApiState(ApiUrl, 11435, ApiUrl is not null), [.. Clients], new QueueState(1, 8, 16, 0)));

    public Task<NewKeyResponse> CreateKeyAsync(
        string name, IReadOnlyList<string> permissions, CancellationToken cancel = default)
    {
        CreatedKeys.Add((name, permissions));
        var client = Client("k" + CreatedKeys.Count.ToString("D11", System.Globalization.CultureInfo.InvariantCulture), name, "active", [.. permissions]);
        Clients.Add(client);
        return Task.FromResult(new NewKeyResponse(client, "ramvi-SECRETKEY" + client.Id));
    }

    public Task<RevokeResponse> RevokeKeyAsync(string id, bool cancelRunning, CancellationToken cancel = default)
    {
        Revoked.Add((id, cancelRunning));
        var index = Clients.FindIndex(c => c.Id == id);
        Clients[index] = Clients[index] with { State = "revoked", Revoked = 2 };
        return Task.FromResult(new RevokeResponse(id, 1));
    }

    public ApiException? SettingsError { get; set; }

    public Task<SettingsInfo> SettingsAsync(CancellationToken cancel = default) => Task.FromResult(Info(null));

    public Task<SettingsInfo> SaveSettingsAsync(SettingsUpdate update, CancellationToken cancel = default)
    {
        SettingsUpdates.Add(update);
        if (SettingsError is { } error)
        {
            SettingsError = null;
            return Task.FromException<SettingsInfo>(error);
        }

        var before = AppSettings;
        AppSettings = new AppSettings(
            update.DataDir ?? before.DataDir, update.ModelsDir ?? before.ModelsDir, update.Language ?? before.Language,
            update.ApiPort ?? before.ApiPort, update.NetworkAccess ?? before.NetworkAccess,
            update.Theme ?? before.Theme, update.VoiceDevice ?? before.VoiceDevice,
            update.VoiceLanguage ?? before.VoiceLanguage, update.VoiceModel ?? before.VoiceModel,
            update.VoiceBackend ?? before.VoiceBackend, update.VoiceApiUrl ?? before.VoiceApiUrl,
            update.VoiceApiModel ?? before.VoiceApiModel, update.VoiceAutoSend ?? before.VoiceAutoSend,
            update.NetworkAllow ?? before.NetworkAllow,
            update.AttachMaxFiles ?? before.AttachMaxFiles, update.AttachMaxFileKb ?? before.AttachMaxFileKb,
            update.AttachMaxTotalKb ?? before.AttachMaxTotalKb);
        if (update.AttachMaxFiles is not null || update.AttachMaxFileKb is not null || update.AttachMaxTotalKb is not null)
            Limits = Limits with { MaxFiles = AppSettings.AttachMaxFiles, MaxFileBytes = AppSettings.AttachMaxFileKb * 1024, MaxTotalBytes = AppSettings.AttachMaxTotalKb * 1024 };
        var restart = new List<string>();
        if (AppSettings.DataDir != before.DataDir)
        {
            restart.Add("data_dir");
        }

        var info = Info(restart);
        return Task.FromResult(SaveApiError is { } apiError ? info with { ApiError = apiError } : info);
    }

    public string? ModelsDirWarning { get; set; }

    /// <summary>8.3: состояние обновлений; null — у сервиса нет обновлений (api.not_found).</summary>
    public UpdateStatus? Update { get; set; }

    public List<string> StagedPaths { get; } = [];

    public Task<UpdateStatus> UpdatesAsync(CancellationToken cancel = default) =>
        Update is { } update ? Task.FromResult(update) : Task.FromException<UpdateStatus>(new ApiException("api.not_found", null));

    public Task<UpdateStatus> StageUpdateAsync(string path, CancellationToken cancel = default)
    {
        StagedPaths.Add(path);
        Update = (Update ?? new UpdateStatus("0.8.0")) with { Staged = "0.9.0" };
        return Task.FromResult(Update);
    }

    private SettingsInfo Info(IReadOnlyList<string>? restart) =>
        new(AppSettings, new SettingsFolders("/app/" + AppSettings.DataDir, "/app/" + AppSettings.ModelsDir), "/app/data",
            ["ru", "en"], restart, ModelsDirWarning: ModelsDirWarning);

    // --- 4.11: каталог, загрузки, выбор, по умолчанию, удаление ------------------------------

    public List<CatalogEntryInfo> Catalog { get; } = [];

    public bool CatalogStale { get; set; }

    public int CatalogRefreshes { get; private set; }

    public List<DownloadInfo> DownloadList { get; } = [];

    public List<string> ModelCalls { get; } = [];

    public Recommendation Recommendation { get; set; } =
        new("balanced", null, Message("recommend.none", ("not_runnable", 0), ("not_for_task", 0), ("unknown", 0)), []);

    /// <summary>Сколько раз экран моделей читал каталог.</summary>
    public int CatalogLoads { get; private set; }

    public Task<CatalogView> CatalogAsync(CancellationToken cancel = default)
    {
        CatalogLoads++;
        return Task.FromResult(new CatalogView([], [.. Catalog], CatalogStale));
    }

    public Task<CatalogView> RefreshCatalogAsync(CancellationToken cancel = default)
    {
        CatalogRefreshes++;
        return CatalogAsync(cancel);
    }

    public Task<IReadOnlyList<DownloadInfo>> DownloadsAsync(CancellationToken cancel = default) =>
        Task.FromResult<IReadOnlyList<DownloadInfo>>([.. DownloadList]);

    public Task<DownloadInfo> StartDownloadAsync(string entryId, CancellationToken cancel = default)
    {
        ModelCalls.Add($"download {entryId}");
        var entry = Catalog.First(e => e.Id == entryId);
        var item = new DownloadInfo($"{DownloadList.Count:x16}", entryId, "running", 0, entry.Bytes, entry.Name);
        DownloadList.Add(item);
        return Task.FromResult(item);
    }

    public Task<DownloadInfo> DownloadActionAsync(string downloadId, string action, CancellationToken cancel = default)
    {
        ModelCalls.Add($"{action} {downloadId}");
        var index = DownloadList.FindIndex(d => d.Id == downloadId);
        var state = action switch { "pause" => "paused", "resume" => "running", _ => "cancelled" };
        DownloadList[index] = DownloadList[index] with { State = state };
        return Task.FromResult(DownloadList[index]);
    }

    /// <summary>Загрузка дошла до конца: модель появляется в списке установленных.</summary>
    public void FinishDownload(string downloadId, ModelInfo installed)
    {
        var index = DownloadList.FindIndex(d => d.Id == downloadId);
        DownloadList[index] = DownloadList[index] with
        {
            State = "done",
            BytesDone = DownloadList[index].TotalBytes ?? 0,
            ModelId = installed.Id,
        };
        Models.Add(installed);
    }

    public Task<Recommendation> RecommendationsAsync(CancellationToken cancel = default) =>
        Task.FromResult(Recommendation);

    public Task SetDefaultModelAsync(string modelId, CancellationToken cancel = default)
    {
        ModelCalls.Add($"default {modelId}");
        Onboarding = Onboarding with { DefaultModel = modelId };
        return Task.CompletedTask;
    }

    public Task<DeleteModelResult> DeleteModelAsync(string modelId, bool deleteFile, CancellationToken cancel = default)
    {
        ModelCalls.Add($"delete {modelId} file={deleteFile}");
        Models.RemoveAll(m => m.Id == modelId);
        return Task.FromResult(new DeleteModelResult(true, deleteFile));
    }

    // --- 5.2: голосовой ввод ------------------------------------------------------------------

    public List<string> VoiceCalls { get; } = [];

    public string VoiceState { get; set; } = "idle";

    public double VoiceSeconds { get; set; }

    public bool VoiceLimit { get; set; }

    public string VoiceText { get; set; } = "привет мир";

    /// <summary>Ошибка следующего вызова голоса (start/stop/cancel).</summary>
    public ApiException? VoiceError { get; set; }

    /// <summary>Пока не завершён, «Стоп» ждёт (распознавание идёт).</summary>
    public TaskCompletionSource? VoiceStopGate { get; set; }

    public List<VoiceDevice> VoiceDeviceList { get; } = [new("Mic A", true), new("Mic B", false)];

    public List<VoiceModelOption> VoiceModels { get; } =
        [new("large-v3-turbo-q5_0", 574_041_195), new("small-q5_1", 190_085_487, true)];

    public VoiceDownload VoiceDownloadState { get; set; } = new("idle");

    /// <summary>5.4: ускоритель на видеокарте; null — сервис его не прислал.</summary>
    public VoiceAccel? VoiceAccelState { get; set; } = new("cuda-12.4", false, false, 684_913_404);

    public VoiceLast? VoiceLastState { get; set; }

    /// <summary>5.7: ключ API в «хранилище» (тест видит, что ушло в сервис).</summary>
    public string? VoiceApiKey { get; set; }

    public VoiceApi VoiceApiState => new(
        AppSettings.VoiceBackend == "api", AppSettings.VoiceApiUrl,
        Uri.TryCreate(AppSettings.VoiceApiUrl, UriKind.Absolute, out var uri) ? uri.Host : "",
        AppSettings.VoiceApiModel, VoiceApiKey is not null);

    public List<(string? Url, string? Model)> VoiceChecks { get; } = [];

    public VoiceStatus Voice() => new(
        VoiceState, VoiceSeconds, 0.5, VoiceLimit, 120, AppSettings.VoiceLanguage, ["auto", "ru", "en", "uk"],
        AppSettings.VoiceDevice,
        new VoiceModelState(AppSettings.VoiceModel, VoiceModels.First(m => m.Id == AppSettings.VoiceModel).Installed,
            1),
        [.. VoiceModels], VoiceDownloadState, null, VoiceLastState, VoiceAccelState, VoiceApiState,
        AppSettings.VoiceAutoSend);

    private Task<T> VoiceCall<T>(string call, Func<T> result)
    {
        VoiceCalls.Add(call);
        if (VoiceError is { } error)
        {
            VoiceError = null;
            return Task.FromException<T>(error);
        }

        return Task.FromResult(result());
    }

    public Task<VoiceStatus> VoiceStatusAsync(CancellationToken cancel = default) => Task.FromResult(Voice());

    public Task<IReadOnlyList<VoiceDevice>> VoiceDevicesAsync(CancellationToken cancel = default) =>
        Task.FromResult<IReadOnlyList<VoiceDevice>>([.. VoiceDeviceList]);

    public Task<VoiceStatus> StartVoiceAsync(string? device = null, CancellationToken cancel = default) =>
        VoiceCall("start", () =>
        {
            VoiceState = "recording";
            return Voice();
        });

    public async Task<VoiceResult> StopVoiceAsync(CancellationToken cancel = default)
    {
        VoiceState = "transcribing";
        if (VoiceStopGate is { } gate)
        {
            await gate.Task.ConfigureAwait(true);
        }

        try
        {
            return await VoiceCall("stop", () => new VoiceResult(VoiceText, "ru", VoiceSeconds, 0.1)).ConfigureAwait(true);
        }
        finally
        {
            VoiceState = "idle";
        }
    }

    public Task<VoiceStatus> CancelVoiceAsync(CancellationToken cancel = default) =>
        VoiceCall("cancel", () =>
        {
            VoiceState = "idle";
            return Voice();
        });

    public Task<VoiceStatus> DownloadVoiceModelAsync(string? model = null, CancellationToken cancel = default) =>
        VoiceCall($"download {model}", () =>
        {
            VoiceDownloadState = new VoiceDownload("running", model, 0, 100);
            return Voice();
        });

    public Task<VoiceStatus> DownloadVoiceAccelAsync(CancellationToken cancel = default) =>
        VoiceCall("accel", () =>
        {
            VoiceDownloadState = new VoiceDownload("running", "cuda-12.4", 0, 684_913_404);
            return Voice();
        });

    public Task<VoiceStatus> SetVoiceApiKeyAsync(string key, CancellationToken cancel = default) =>
        VoiceCall("key", () =>
        {
            VoiceApiKey = key.Length > 0 ? key : null;
            return Voice();
        });

    public Task<VoiceCheck> CheckVoiceApiAsync(string? url, string? model, CancellationToken cancel = default) =>
        VoiceCall("check", () =>
        {
            VoiceChecks.Add((url, model));
            return new VoiceCheck(true, new Uri(url!).Host, 0.42);
        });

    public Task<VoiceStatus> CancelVoiceDownloadAsync(CancellationToken cancel = default) =>
        VoiceCall("download-cancel", () =>
        {
            VoiceDownloadState = new VoiceDownload("idle");
            return Voice();
        });

    /// <summary>Сообщение сервиса <c>{code, params}</c>, как его присылает API.</summary>
    public static JsonElement Message(string code, params (string Name, object Value)[] values)
    {
        var parameters = values.ToDictionary(v => v.Name, v => (object)v.Value);
        return JsonSerializer.SerializeToElement(
            new Dictionary<string, object> { ["code"] = code, ["params"] = parameters }, TestJson.Default.Object);
    }

    public static Dictionary<string, System.Text.Json.JsonElement> Json(params (string Name, object Value)[] values) =>
        values.ToDictionary(v => v.Name, v => JsonSerializer.SerializeToElement(v.Value, TestJson.Default.Object));

    public static ApiException Error(string code, params (string Name, object Value)[] parameters) =>
        new(code, parameters.ToDictionary(p => p.Name, p => JsonSerializer.SerializeToElement(p.Value, TestJson.Default.Object)));
}

[System.Text.Json.Serialization.JsonSerializable(typeof(object))]
[System.Text.Json.Serialization.JsonSerializable(typeof(Dictionary<string, object>))]
[System.Text.Json.Serialization.JsonSerializable(typeof(bool))]
[System.Text.Json.Serialization.JsonSerializable(typeof(JsonElement))]
[System.Text.Json.Serialization.JsonSerializable(typeof(int))]
[System.Text.Json.Serialization.JsonSerializable(typeof(string))]
internal sealed partial class TestJson : System.Text.Json.Serialization.JsonSerializerContext;
