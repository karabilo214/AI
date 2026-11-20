using System.Collections.ObjectModel;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Сервис в форме подключения; «совместимый» — тот же протокол OpenAI со своим адресом.</summary>
public sealed class CloudKindOption(string id, string kind, string url) : ObservableObject
{
    public string Id { get; } = id;
    public string Kind { get; } = kind;
    public string Url { get; } = url;
    public string Label => Localizer.Current[$"cloud.kind.{Id}"];
}

/// <summary>Модель сервиса: галочка — модель видна в общем выборе модели Chat/Agent/Character.</summary>
public sealed class CloudModelOption(string name, bool enabled, Func<CloudModelOption, bool, Task> toggle) : ObservableObject
{
    private bool _enabled = enabled;

    public string Name { get; } = name;

    public bool Enabled
    {
        get => _enabled;
        set
        {
            if (Set(ref _enabled, value))
            {
                _ = toggle(this, value);
            }
        }
    }

    /// <summary>Вернуть галочку без запроса к сервису (запрос не удался).</summary>
    internal void Reset(bool value)
    {
        _enabled = value;
        Raise(nameof(Enabled));
    }
}

/// <summary>Строка подключения: сервис, адрес, есть ли ключ, включённые модели; удаление в два нажатия.</summary>
public sealed class CloudConnectionRow : ObservableObject
{
    private readonly Func<CloudConnectionRow, Task> _delete;
    private bool _confirming;

    public CloudConnectionRow(CloudConnection info, Func<CloudConnectionRow, Task> select, Func<CloudConnectionRow, Task> edit, Func<CloudConnectionRow, Task> delete)
    {
        Info = info;
        _delete = delete;
        ModelsCommand = new Command(() => select(this));
        EditCommand = new Command(() => edit(this));
        DeleteCommand = new Command(DeleteAsync);
    }

    private static Localizer L => Localizer.Current;

    public CloudConnection Info { get; }
    public string Name => Info.Name;
    public string Host => Uri.TryCreate(Info.Url, UriKind.Absolute, out var url) ? url.Host : Info.Url;
    public string Key => Info.HasKey ? L["cloud.key_set"] : L["cloud.key_missing"];
    public string Models => Info.Models.Count == 0
        ? L["cloud.no_models"]
        : L.Format("cloud.models_enabled", ("models", string.Join(", ", Info.Models)));
    public string DeleteLabel => _confirming ? L["cloud.delete_confirm"] : L["cloud.delete"];
    public Command ModelsCommand { get; }
    public Command EditCommand { get; }
    public Command DeleteCommand { get; }

    private async Task DeleteAsync()
    {
        if (!_confirming)
        {
            _confirming = true;
            Raise(nameof(DeleteLabel));
            return;
        }

        _confirming = false;
        Raise(nameof(DeleteLabel));
        await _delete(this).ConfigureAwait(true);
    }
}

/// <summary>
/// 7.8: «Облачные модели» во вкладке «Подключения» — сервис, адрес, ключ (после сохранения не
/// показывается), «Проверить», список моделей сервиса с галочками. Включённые модели появляются в
/// общем выборе модели; запросы к ним уходят в сервис (метка у выбора модели).
/// </summary>
public sealed class CloudModelsViewModel : ObservableObject
{
    private readonly Func<IChatApi> _api;
    private readonly Func<Task> _modelsChanged;
    private readonly List<CloudModelOption> _allModels = [];
    private CloudKindOption _kind;
    private string _name = "";
    private string _url = "";
    private string _key = "";
    private string? _editing;
    private string _filter = "";
    private CloudConnectionRow? _selected;
    private string? _error;
    private string? _message;

    public CloudModelsViewModel(Func<IChatApi> api, Func<Task>? modelsChanged = null)
    {
        _api = api;
        _modelsChanged = modelsChanged ?? (() => Task.CompletedTask);
        Kinds =
        [
            new("openai", "openai", "https://api.openai.com/v1"),
            new("anthropic", "anthropic", "https://api.anthropic.com"),
            new("gemini", "gemini", "https://generativelanguage.googleapis.com"),
            new("compatible", "openai", ""),
        ];
        _kind = Kinds[0];
        _url = _kind.Url;
        CheckCommand = new Command(CheckAsync, () => CanSave);
        SaveCommand = new Command(SaveAsync, () => CanSave);
        NewCommand = new Command(() => { Clear(); return Task.CompletedTask; });
    }

    private static Localizer L => Localizer.Current;

    public IReadOnlyList<CloudKindOption> Kinds { get; }
    public ObservableCollection<CloudConnectionRow> Connections { get; } = [];
    public ObservableCollection<CloudModelOption> Models { get; } = [];
    public Command CheckCommand { get; }
    public Command SaveCommand { get; }
    public Command NewCommand { get; }

    public CloudKindOption Kind
    {
        get => _kind;
        set
        {
            var previous = _kind;
            if (value is not null && Set(ref _kind, value))
            {
                if (Url.Length == 0 || Url == previous.Url)
                {
                    Url = value.Url; // адрес по умолчанию нового сервиса, если свой не вписан
                }

                Raise(nameof(IsCompatible));
            }
        }
    }

    public bool IsCompatible => _kind.Id == "compatible";
    public string Name { get => _name; set { if (Set(ref _name, value)) Refresh(); } }
    public string Url { get => _url; set { if (Set(ref _url, value)) Refresh(); } }

    /// <summary>Новый ключ; пустое поле при изменении подключения оставляет прежний.</summary>
    public string Key { get => _key; set => Set(ref _key, value); }

    public bool IsEditing => _editing is not null;
    public string FormTitle => _editing is null ? L["cloud.new"] : L.Format("cloud.edit", ("name", _name));
    public bool CanSave => _name.Trim().Length > 0 && _url.Trim().Length > 0;

    public CloudConnectionRow? Selected
    {
        get => _selected;
        private set
        {
            if (Set(ref _selected, value))
            {
                Raise(nameof(HasSelected));
                Raise(nameof(ModelsTitle));
            }
        }
    }

    public bool HasSelected => _selected is not null;
    public string ModelsTitle => _selected is null ? "" : L.Format("cloud.models_of", ("name", _selected.Name));

    public string Filter
    {
        get => _filter;
        set
        {
            if (Set(ref _filter, value))
            {
                ApplyFilter();
            }
        }
    }

    public string? Error { get => _error; private set { if (Set(ref _error, value)) Raise(nameof(HasError)); } }
    public bool HasError => _error is not null;
    public string? Message { get => _message; private set { if (Set(ref _message, value)) Raise(nameof(HasMessage)); } }
    public bool HasMessage => _message is not null;

    public async Task LoadAsync()
    {
        Error = null;
        try
        {
            Show(await _api().CloudAsync().ConfigureAwait(true));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException or NotSupportedException)
        {
            Error = Text(error);
        }
    }

    private void Show(CloudView view)
    {
        var selected = _selected?.Info.Id;
        Connections.Clear();
        foreach (var connection in view.Connections)
        {
            Connections.Add(new CloudConnectionRow(connection, SelectAsync, EditAsync, DeleteAsync));
        }

        Selected = Connections.FirstOrDefault(c => c.Info.Id == selected);
        if (Selected is null)
        {
            Models.Clear();
            _allModels.Clear();
        }
    }

    private CloudModelsRequest FormRequest() =>
        _editing is not null
            ? new CloudModelsRequest(_editing, Key: _key.Length > 0 ? _key.Trim() : null)
            : new CloudModelsRequest(null, _kind.Kind, _url.Trim(), _key.Trim());

    private async Task CheckAsync()
    {
        Error = Message = null;
        try
        {
            var models = await _api().CloudModelsAsync(FormRequest()).ConfigureAwait(true);
            Message = L.Format("cloud.check_ok", ("count", models.Count));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException or NotSupportedException)
        {
            Error = Text(error);
        }
    }

    private async Task SaveAsync()
    {
        Error = Message = null;
        try
        {
            // Ключ: новый — заменить; пустое поле у нового подключения — без ключа, у старого — оставить.
            string? key = _key.Trim().Length > 0 ? _key.Trim() : _editing is null ? "" : null;
            var saved = await _api().SaveCloudAsync(
                new CloudSaveRequest(_editing, _kind.Kind, _name.Trim(), _url.Trim(), key)).ConfigureAwait(true);
            Message = L.Format("cloud.saved", ("name", saved.Name));
            Show(await _api().CloudAsync().ConfigureAwait(true));
            Clear(); // форма пустая, ключ больше не показывается
            if (Connections.FirstOrDefault(c => c.Info.Id == saved.Id) is { } row)
            {
                await SelectAsync(row).ConfigureAwait(true);
            }
        }
        catch (Exception error) when (error is ApiException or HttpRequestException or NotSupportedException)
        {
            Error = Text(error);
        }
    }

    private Task EditAsync(CloudConnectionRow row)
    {
        _editing = row.Info.Id;
        _kind = Kinds.FirstOrDefault(k => k.Kind == row.Info.Kind && k.Url == row.Info.Url)
            ?? Kinds.First(k => k.Id == (row.Info.Kind == "openai" ? "compatible" : row.Info.Kind));
        _name = row.Info.Name;
        _url = row.Info.Url;
        Key = "";
        Raise(string.Empty);
        Refresh();
        return Task.CompletedTask;
    }

    private void Clear()
    {
        _editing = null;
        _kind = Kinds[0];
        _name = "";
        _url = _kind.Url;
        Key = "";
        Raise(string.Empty);
        Refresh();
    }

    private async Task SelectAsync(CloudConnectionRow row)
    {
        Error = null;
        Selected = row;
        Models.Clear();
        _allModels.Clear();
        try
        {
            var names = await _api().CloudModelsAsync(new CloudModelsRequest(row.Info.Id)).ConfigureAwait(true);
            var enabled = row.Info.Models.ToHashSet(StringComparer.Ordinal);
            foreach (var name in names.Concat(row.Info.Models.Where(m => !names.Contains(m))))
            {
                _allModels.Add(new CloudModelOption(name, enabled.Contains(name), ToggleAsync));
            }

            Filter = "";
            ApplyFilter();
        }
        catch (Exception error) when (error is ApiException or HttpRequestException or NotSupportedException)
        {
            Error = Text(error);
        }
    }

    private void ApplyFilter()
    {
        Models.Clear();
        var filter = _filter.Trim();
        // Включённые — первыми: их ищут, чтобы выключить.
        foreach (var model in _allModels
            .Where(m => filter.Length == 0 || m.Name.Contains(filter, StringComparison.OrdinalIgnoreCase))
            .OrderBy(m => !m.Enabled))
        {
            Models.Add(model);
        }
    }

    private async Task ToggleAsync(CloudModelOption model, bool enabled)
    {
        if (_selected is not { } row)
        {
            return;
        }

        Error = null;
        try
        {
            await _api().EnableCloudModelAsync(row.Info.Id, model.Name, enabled).ConfigureAwait(true);
            Show(await _api().CloudAsync().ConfigureAwait(true));
            await _modelsChanged().ConfigureAwait(true); // общий выбор модели перечитывается
        }
        catch (Exception error) when (error is ApiException or HttpRequestException or NotSupportedException)
        {
            model.Reset(!enabled);
            Error = Text(error);
        }
    }

    private async Task DeleteAsync(CloudConnectionRow row)
    {
        Error = Message = null;
        try
        {
            await _api().DeleteCloudAsync(row.Info.Id).ConfigureAwait(true);
            if (_editing == row.Info.Id)
            {
                Clear();
            }

            Show(await _api().CloudAsync().ConfigureAwait(true));
            await _modelsChanged().ConfigureAwait(true);
            Message = L.Format("cloud.deleted", ("name", row.Name));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException or NotSupportedException)
        {
            Error = Text(error);
        }
    }

    private void Refresh()
    {
        CheckCommand.Refresh();
        SaveCommand.Refresh();
        Raise(nameof(CanSave));
        Raise(nameof(FormTitle));
        Raise(nameof(IsEditing));
    }

    public override void Relocalize()
    {
        foreach (var kind in Kinds)
        {
            kind.Relocalize();
        }

        foreach (var row in Connections)
        {
            row.Relocalize();
        }

        base.Relocalize();
    }

    private static string Text(Exception error) => error switch
    {
        ApiException api => L.Format(
            api.Code, api.Parameters.ToDictionary(p => p.Key, p => (object?)p.Value, StringComparer.Ordinal)),
        NotSupportedException => L["api.not_found"],
        _ => L["ui.connection_lost"],
    };
}
