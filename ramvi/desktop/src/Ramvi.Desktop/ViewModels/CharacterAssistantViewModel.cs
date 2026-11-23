using System.Text.Json;
using System.Text.Json.Nodes;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>
/// 9.28: помощник в приложениях. Пользователь выбирает окно (тот же выбор, что у агента) и сам
/// решает, что персонажу можно: читать выделенный текст и элементы окна, отправлять прочитанное
/// облачной модели, поручать агенту действовать в этом окне. Смена окна или «Отключить» —
/// немедленный отзыв; без системного адаптера (Mac) раздел честно говорит, что недоступен.
/// </summary>
public sealed class CharacterAssistantViewModel : ObservableObject
{
    private readonly Func<string, JsonObject, Task<JsonElement>> _agent;
    private readonly Func<string, JsonObject?, Task<JsonElement>> _call;
    private readonly CharacterAgentViewModel _delegation;
    private readonly TimeSpan _countdown;
    private JsonObject? _target;
    private string _title = "", _status = "", _error = "";
    private bool _readUi = true, _sendToCloud, _agentActs, _active;

    public CharacterAssistantViewModel(
        Func<string, JsonObject, Task<JsonElement>> agent,
        Func<string, JsonObject?, Task<JsonElement>> call,
        CharacterAgentViewModel delegation,
        TimeSpan? countdown = null)
    {
        _agent = agent; _call = call; _delegation = delegation;
        _countdown = countdown ?? TimeSpan.FromSeconds(5);
        PickWindowCommand = new Command(PickWindowAsync);
        ApplyCommand = new Command(ApplyAsync, () => _target is not null);
        OffCommand = new Command(OffAsync, () => _active);
    }

    public Command PickWindowCommand { get; }
    public Command ApplyCommand { get; }
    public Command OffCommand { get; }
    public string Title { get => _title; private set { if (Set(ref _title, value)) Raise(nameof(HasTarget)); } }
    public bool HasTarget => _title.Length > 0;
    public string Status { get => _status; private set => Set(ref _status, value); }
    public string Error { get => _error; private set => Set(ref _error, value); }
    /// <summary>Читать выделенный текст и элементы окна (UI Context).</summary>
    public bool ReadUi { get => _readUi; set => Set(ref _readUi, value); }
    /// <summary>Отправлять прочитанное облачной модели (отдельное разрешение).</summary>
    public bool SendToCloud { get => _sendToCloud; set => Set(ref _sendToCloud, value); }
    /// <summary>Поручения агенту выполняются в этом окне (Agent Control).</summary>
    public bool AgentActs { get => _agentActs; set => Set(ref _agentActs, value); }
    public bool Active { get => _active; private set { if (Set(ref _active, value)) OffCommand.Refresh(); } }

    private async Task PickWindowAsync()
    {
        try
        {
            Error = "";
            Status = Localizer.Current["agent.target_countdown"];
            await Task.Delay(_countdown).ConfigureAwait(true);
            var target = (await _agent("target", new JsonObject()).ConfigureAwait(true)).GetProperty("target");
            _target = new JsonObject
            {
                ["application"] = target.GetProperty("application").GetString(),
                ["window_id"] = target.GetProperty("window_id").GetString(),
                ["title"] = target.GetProperty("title").GetString(),
            };
            Title = _target["title"]!.ToString();
            Active = false; // новое окно — прежний доступ отозван, разрешить заново
            Status = Localizer.Current["character.assistant_selected"];
            ApplyCommand.Refresh();
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        {
            Status = "";
            Error = Localizer.Current["character.assistant_unavailable"];
        }
    }

    private async Task ApplyAsync()
    {
        if (_target is null) return;
        try
        {
            Error = "";
            // Поручения персонажа агенту идут от имени персонажа: право действовать в окне — у него,
            // гранты, выданные на экране агента, этот раздел не трогает. Действовать — значит и видеть.
            await _agent("target/module", new JsonObject
            {
                ["module"] = "character", ["ui"] = ReadUi || AgentActs, ["screen"] = false,
                ["egress"] = SendToCloud, ["control"] = AgentActs,
            }).ConfigureAwait(true);
            await _call("assistant", new JsonObject { ["target"] = ReadUi ? _target.DeepClone() : null }).ConfigureAwait(true);
            await _delegation.SetWindowAsync(AgentActs).ConfigureAwait(true);
            Active = ReadUi || AgentActs;
            Status = Localizer.Current[Active ? "character.assistant_on" : "character.assistant_off"];
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        {
            Error = Localizer.Current["character.assistant_unavailable"];
        }
    }

    private async Task OffAsync()
    {
        try
        {
            await _agent("target/module", new JsonObject { ["module"] = "character", ["ui"] = false, ["screen"] = false, ["egress"] = false, ["control"] = false }).ConfigureAwait(true);
            await _call("assistant", new JsonObject { ["target"] = null }).ConfigureAwait(true);
            await _delegation.SetWindowAsync(false).ConfigureAwait(true);
            Active = false;
            Status = Localizer.Current["character.assistant_off"];
        }
        catch (Exception error) when (CharacterViewModel.Handled(error))
        {
            Error = Localizer.Current["character.assistant_unavailable"];
        }
    }
}
