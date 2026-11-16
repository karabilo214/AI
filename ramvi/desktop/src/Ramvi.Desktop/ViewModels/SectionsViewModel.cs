using System.Collections.ObjectModel;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>Пункт панели разделов.</summary>
public sealed class SectionItem(SectionInfo info) : ObservableObject
{
    private SectionInfo _info = info;

    public string Id => _info.Id;

    public string View => _info.View;

    public string Title => Localizer.Current[_info.TitleKey];

    public SectionInfo Info => _info;

    public void Update(SectionInfo info)
    {
        _info = info;
        Raise(nameof(Info));
    }
}

/// <summary>
/// Вид раздела модуля (Agent, Character). Создаётся при первом открытии и хранится,
/// поэтому возврат в раздел не сбрасывает его. Собственные экраны Agent/Character
/// появятся в этапах 6 и 9; сейчас показывается состояние модуля.
/// </summary>
public sealed class ModuleSectionViewModel(SectionItem item) : ObservableObject
{
    public SectionItem Item { get; } = item;

    public string Title => Item.Title;

    public string State => Localizer.Current[$"section.state.{Item.Info.State}"];

    /// <summary>Причина ошибки модуля (техническая) в рамке перевода (3.14).</summary>
    public string? Error => string.IsNullOrEmpty(Item.Info.Error)
        ? null
        : Localizer.Current.Format("section.error", ("detail", Item.Info.Error));

    public bool HasError => !string.IsNullOrEmpty(Item.Info.Error);

    public string Placeholder => Localizer.Current[$"section.placeholder.{Item.View}"];

    public void Refresh()
    {
        Raise(nameof(State));
        Raise(nameof(Error));
        Raise(nameof(HasError));
    }
}

/// <summary>
/// Реестр разделов (CORE-UI-01, пункт 3.11): «Чат» встроен, разделы модулей приходят от
/// сервиса. Панель видна, только если разделов больше одного. Переключение не трогает
/// чат: он остаётся тем же объектом со всеми диалогами, вводом и вложениями.
/// </summary>
public sealed class SectionsViewModel : ObservableObject
{
    private readonly Func<IChatApi> _api;
    private readonly Dictionary<string, ModuleSectionViewModel> _views = new(StringComparer.Ordinal);
    private SectionItem? _selected;
    private ModuleSectionViewModel? _current;

    public SectionsViewModel(Func<IChatApi> api) => _api = api;

    public ObservableCollection<SectionItem> Items { get; } = [];

    public bool HasMany => Items.Count > 1;

    public bool IsChatSelected => _selected is null || _selected.View == "chat";

    /// <summary>Вид открытого раздела модуля (null, когда открыт чат).</summary>
    public ModuleSectionViewModel? Current
    {
        get => _current;
        private set => Set(ref _current, value);
    }

    /// <summary>Сколько видов разделов уже создано (для проверки ленивости).</summary>
    public int CreatedViews => _views.Count;

    public SectionItem? Selected
    {
        get => _selected;
        set
        {
            if (value is not null && Set(ref _selected, value))
            {
                Raise(nameof(IsChatSelected));
                _ = OpenAsync(value);
            }
        }
    }

    public async Task LoadAsync()
    {
        var sections = await _api().SectionsAsync().ConfigureAwait(true);
        var keep = _selected?.Id;
        Items.Clear();
        foreach (var info in sections.OrderBy(s => s.Order))
        {
            Items.Add(new SectionItem(info));
        }

        Raise(nameof(HasMany));
        _selected = Items.FirstOrDefault(i => i.Id == keep) ?? Items.FirstOrDefault(i => i.View == "chat");
        Raise(nameof(Selected));
        Raise(nameof(IsChatSelected));
    }

    private async Task OpenAsync(SectionItem item)
    {
        if (item.View == "chat")
        {
            Current = null;
            return;
        }

        if (!_views.TryGetValue(item.Id, out var view))
        {
            view = new ModuleSectionViewModel(item);
            _views[item.Id] = view;
            Raise(nameof(CreatedViews));
        }

        Current = view;
        try
        {
            item.Update(await _api().OpenSectionAsync(item.Id).ConfigureAwait(true));
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            item.Update(item.Info with { State = "error", Error = error.Message });
        }

        view.Refresh();
    }

    public override void Relocalize()
    {
        base.Relocalize();
        foreach (var item in Items)
        {
            item.Relocalize();
        }

        foreach (var view in _views.Values)
        {
            view.Relocalize();
        }
    }
}
