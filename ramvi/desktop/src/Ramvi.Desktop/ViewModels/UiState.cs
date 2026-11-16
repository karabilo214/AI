using System.Windows.Input;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Состояния интерфейса UI-03 (пункт 3.5). У каждого есть действие восстановления.</summary>
public enum UiState
{
    Connecting,
    NoModel,
    Loading,
    Ready,
    Generating,
    Error,
    LowMemory,
    Incompatible,
    ApiBusy,
    NoAccess,
}

/// <summary>Кнопка восстановления на плашке состояния.</summary>
public sealed record StateAction(string Id, string Label, ICommand Command, object? Parameter = null);

/// <summary>Плашка над диалогом: что случилось и что можно сделать.</summary>
public sealed record StateBanner(UiState State, string Title, string Text, IReadOnlyList<StateAction> Actions)
{
    public bool IsProblem => State is not (UiState.Loading or UiState.Connecting);
}

/// <summary>Причина, полученная от сервиса или оболочки: код (ключ перевода) и параметры.</summary>
public sealed record Problem(string Code, IReadOnlyDictionary<string, object?> Params)
{
    public UiState State => StateFor(Code);

    public string? Path => Params.TryGetValue("path", out var value) ? TextOf(value) : null;

    /// <summary>Код ошибки сервиса → состояние интерфейса.</summary>
    public static UiState StateFor(string code) => code switch
    {
        "engine.no_memory" => UiState.LowMemory,
        "engine.incompatible" => UiState.Incompatible,
        "engine.no_access" or "engine.model_missing" or "storage.no_access" => UiState.NoAccess,
        "api.busy" or "chat.busy" or "chat.busy_timeout" => UiState.ApiBusy,
        "chat.no_model" or "engine.no_model" or "chat.model_not_found" or "engine.model_not_found" => UiState.NoModel,
        _ => UiState.Error,
    };

    private static string? TextOf(object? value) => value switch
    {
        System.Text.Json.JsonElement { ValueKind: System.Text.Json.JsonValueKind.String } element => element.GetString(),
        string text => text,
        _ => null,
    };
}
