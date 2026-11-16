using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading.Channels;
using Ramvi.Desktop.Api;

namespace Ramvi.Desktop.Tests;

/// <summary>API в памяти: тест сам выдаёт события потока через <see cref="Stream"/>.</summary>
internal sealed class FakeChatApi : IChatApi
{
    private int _next;

    public List<ChatInfo> Chats { get; } = [];

    public Dictionary<string, List<MessageInfo>> Messages { get; } = [];

    public List<ModelInfo> Models { get; } =
    [
        new("model-a", "qwen3", "q4", 1),
        new("model-b", "qwen3", "q8", 2),
    ];

    public Limits Limits { get; set; } = new(1000, 2, 100, 150, 255);

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

    public Task<ServiceStatus> StatusAsync(CancellationToken cancel = default) =>
        Task.FromResult(new ServiceStatus(new EngineStatus(null, null, "stopped"), Limits));

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

    public Task<SystemReport> SystemAsync(CancellationToken cancel = default)
    {
        Calls.Add("system");
        return Task.FromResult(new SystemReport(SystemChecks));
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

    public AppSettings AppSettings { get; set; } = new("data", "models", "ru", 11435, false);

    public List<SettingsUpdate> SettingsUpdates { get; } = [];

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
            update.ApiPort ?? before.ApiPort, update.NetworkAccess ?? before.NetworkAccess);
        var restart = new List<string>();
        if (AppSettings.DataDir != before.DataDir)
        {
            restart.Add("data_dir");
        }

        if (AppSettings.Language != before.Language)
        {
            restart.Add("language");
        }

        return Task.FromResult(Info(restart));
    }

    private SettingsInfo Info(IReadOnlyList<string>? restart) =>
        new(AppSettings, new SettingsFolders("/app/" + AppSettings.DataDir, "/app/" + AppSettings.ModelsDir), "/app/data",
            ["ru", "en"], restart);

    public static Dictionary<string, System.Text.Json.JsonElement> Json(params (string Name, object Value)[] values) =>
        values.ToDictionary(v => v.Name, v => JsonSerializer.SerializeToElement(v.Value, TestJson.Default.Object));

    public static ApiException Error(string code, params (string Name, object Value)[] parameters) =>
        new(code, parameters.ToDictionary(p => p.Name, p => JsonSerializer.SerializeToElement(p.Value, TestJson.Default.Object)));
}

[System.Text.Json.Serialization.JsonSerializable(typeof(object))]
[System.Text.Json.Serialization.JsonSerializable(typeof(int))]
[System.Text.Json.Serialization.JsonSerializable(typeof(string))]
internal sealed partial class TestJson : System.Text.Json.Serialization.JsonSerializerContext;
