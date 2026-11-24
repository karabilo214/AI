using System.Text.Json;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;

namespace Ramvi.Desktop.Tests;

/// <summary>4.11: экран «Модели» — карточки, фильтры, загрузка, по умолчанию, удаление.</summary>
public sealed class ModelsTests
{
    private static readonly Localizer Ru = Localizer.Load("ru");
    internal static readonly SectionInfo ModelsSection = new("models", "section.models", "models", 15, "models", null, "ready", null);

    public ModelsTests() => Localizer.Current = Ru;

    private static CompatInfo Compat(string status, string key, int gpu = 36, int layers = 36, bool suits = true) =>
        new(status, [], Message("compat.speed_estimate", ("low", "48"), ("high", "57")),
            new PlanInfo(layers, gpu, gpu, 0, layers - gpu, 0, 0), new CompatModel(key, 8, "Q4_K_M"),
            new TaskFit(suits, []));

    private static CatalogEntryInfo Entry(string id, string name, string key, string status = "compat.recommended") =>
        new(id, "ollama", name, id, 5_000_000_000, "qwen3", "Q4_K_M", "apache-2.0", Compat: Compat(status, key));

    internal static FakeChatApi Api()
    {
        var api = new FakeChatApi();
        api.Models.Clear();
        api.Models.Add(new ModelInfo("qwen3-8b-q4_k_m", "qwen3", "Q4_K_M", 5_000_000_000,
            Json(("imported", false), ("catalog_id", "ollama:qwen3:8b")), Compat("compat.recommended", "qwen3-8b-q4_k_m")));
        api.Models.Add(new ModelInfo("my-local", "qwen3", "Q4_K_M", 2_000_000_000,
            Json(("imported", true)), Compat("compat.recommended", "qwen3-4b-q4_k_m")));
        api.Catalog.Add(Entry("ollama:qwen3:8b", "qwen3:8b Q4_K_M", "qwen3-8b-q4_k_m"));
        api.Catalog.Add(Entry("hf:4b", "Qwen3-4B Q4_K_M", "qwen3-4b-q4_k_m")); // тот же файл, что my-local
        api.Catalog.Add(Entry("ollama:qwen3:14b", "qwen3:14b Q4_K_M", "qwen3-14b-q4_k_m", "compat.slow"));
        api.Catalog.Add(Entry("ollama:qwen3:32b", "qwen3:32b Q4_K_M", "qwen3-32b-q4_k_m", "compat.no_ram"));
        api.Onboarding = new OnboardingState(true, "qwen3-8b-q4_k_m");
        return api;
    }

    private static async Task<(FakeChatApi Api, ModelsViewModel Screen, List<string> Changed)> Open(FakeChatApi? fake = null)
    {
        var api = fake ?? Api();
        var changed = new List<string>();
        var screen = new ModelsViewModel(() => api, () =>
        {
            changed.Add("models");
            return Task.CompletedTask;
        }, TimeSpan.FromMilliseconds(10));
        await screen.LoadAsync();
        return (api, screen, changed);
    }

    private static ModelCardViewModel Card(ModelsViewModel screen, string name) => screen.Cards.Single(c => c.Name == name);

    [Fact]
    public async Task Installed_and_catalog_copies_are_one_card_with_illustration_states()
    {
        var (_, screen, _) = await Open();

        Assert.Equal(
            ["my-local", "qwen3-8b-q4_k_m", "qwen3:14b Q4_K_M", "qwen3:32b Q4_K_M"],
            screen.Cards.Select(c => c.Name));
        var current = Card(screen, "qwen3-8b-q4_k_m");
        Assert.Equal("Установлена · Текущая", current.StateText);
        Assert.True(current.IsCurrentChip && current.OwnsFile && !current.CanDownload);
        Assert.Equal("Установлена", Card(screen, "my-local").StateText);
        var online = Card(screen, "qwen3:14b Q4_K_M");
        Assert.Equal("Доступна онлайн", online.StateText);
        Assert.True(online.CanDownload && online.IsSlow);
        Assert.Equal("Полностью в GPU · оценка ~48–57 токенов/с · 5,0 ГБ", current.Detail.Replace('.', ','));
    }

    [Fact]
    public async Task Missing_file_offers_locate_and_remove_without_deleting_files()
    {
        // B.5: модель перенесли на другой ПК — «Файл модели не найден», «Указать файл», «Убрать».
        var api = Api();
        api.Models[0] = api.Models[0] with { Compat = Compat("compat.missing", "qwen3-8b-q4_k_m") };
        var changed = new List<string>();
        var screen = new ModelsViewModel(() => api, () =>
        {
            changed.Add("models");
            return Task.CompletedTask;
        }, TimeSpan.FromMilliseconds(10), () => Task.FromResult<string?>("/new/disk/qwen3-8b.gguf"));
        await screen.LoadAsync();

        var card = Card(screen, "qwen3-8b-q4_k_m");
        Assert.True(card.IsMissing && card.IsUnfit && card.RelinkCommand.CanExecute(null));
        Assert.Equal("Файл модели не найден", card.CompatText);
        Assert.Equal("Убрать", card.DeleteLabel);
        Assert.False(Card(screen, "my-local").IsMissing);
        Assert.False(Card(screen, "my-local").RelinkCommand.CanExecute(null));

        await screen.RelinkAsync(card);
        Assert.Contains("relink qwen3-8b-q4_k_m /new/disk/qwen3-8b.gguf", api.ModelCalls);
        Assert.Equal(["models"], changed);

        card = Card(screen, "qwen3-8b-q4_k_m");
        card.DeleteCommand.Execute(null); // первое нажатие спрашивает
        card.DeleteCommand.Execute(null);
        Dispatcher.UIThread.RunJobs();
        Assert.Contains("delete qwen3-8b-q4_k_m file=False", api.ModelCalls); // файла нет — удалять нечего
    }

    [Fact]
    public async Task Measured_resources_from_control_run_are_shown()
    {
        var api = Api();
        var measured = api.Models[0] with
        {
            Compat = api.Models[0].Compat! with { Control = new ControlInfo(true, 55, 0.25, 2_000_000_000, 6_400_000_000, 36) },
        };
        api.Models[0] = measured;
        var (_, screen, _) = await Open(api);

        var card = Card(screen, "qwen3-8b-q4_k_m");
        Assert.True(card.HasMeasured);
        Assert.Equal("Замер на этом ПК: пик видеопамяти 6,4 ГБ, RAM 2,0 ГБ, первый токен 0,25 с",
            card.Measured!.Replace('.', ','));
        Assert.False(Card(screen, "my-local").HasMeasured);
    }

    [Fact]
    public async Task Filters_and_search()
    {
        var (_, screen, _) = await Open();

        screen.Filter = ModelsViewModel.Installed;
        Assert.Equal(["my-local", "qwen3-8b-q4_k_m"], screen.Cards.Select(c => c.Name));
        screen.Filter = ModelsViewModel.Suitable; // «Не хватает RAM» не подходит
        Assert.DoesNotContain(screen.Cards, c => c.Name.Contains("32b", StringComparison.Ordinal));
        Assert.Contains(screen.Cards, c => c.Name.Contains("14b", StringComparison.Ordinal));
        screen.Filter = ModelsViewModel.All;
        screen.Search = "14B";
        Assert.Equal(["qwen3:14b Q4_K_M"], screen.Cards.Select(c => c.Name));
        screen.Search = "nothing";
        Assert.True(screen.IsEmpty);
    }

    [Fact]
    public async Task Download_with_progress_pause_resume_and_installed_at_the_end()
    {
        var (api, screen, changed) = await Open();
        var card = Card(screen, "qwen3:14b Q4_K_M");

        await card.DownloadCommand.RunAsync();
        Assert.True(card.IsDownloading && card.HasProgress);
        Assert.Equal("Скачивается", card.StateText);
        var id = card.Download!.Id;
        api.DownloadList[0] = api.DownloadList[0] with { BytesDone = 2_500_000_000 };
        await MainViewModelTests.Until(() => card.Progress >= 50, "прогресс обновился");
        Assert.Equal("50% · 2,5 из 5,0 ГБ", card.ProgressText.Replace('.', ','));

        await card.PauseCommand.RunAsync();
        Assert.True(card.IsPaused && !card.IsDownloading);
        await card.ResumeCommand.RunAsync();
        Assert.True(card.IsDownloading);
        Assert.Equal([$"download ollama:qwen3:14b", $"pause {id}", $"resume {id}"], api.ModelCalls);

        api.FinishDownload(id, new ModelInfo("qwen3-14b-q4_k_m", "qwen3", "Q4_K_M", 9_000_000_000,
            Json(("imported", false), ("catalog_id", "ollama:qwen3:14b")), Compat("compat.slow", "qwen3-14b-q4_k_m", 29, 40)));
        await MainViewModelTests.Until(() => screen.Cards.Any(c => c.Name == "qwen3-14b-q4_k_m"), "модель установлена");
        Assert.DoesNotContain(screen.Cards, c => c.Name == "qwen3:14b Q4_K_M"); // карточка каталога ушла
        Assert.Contains("models", changed); // список моделей чата обновлён
        Assert.Contains("Гибрид GPU+RAM: 29 из 40", Card(screen, "qwen3-14b-q4_k_m").Detail, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Cancel_returns_card_to_download()
    {
        var (_, screen, _) = await Open();
        var card = Card(screen, "qwen3:14b Q4_K_M");
        await card.DownloadCommand.RunAsync();
        await card.CancelCommand.RunAsync();
        Assert.True(card.CanDownload && !card.HasProgress);
    }

    [Fact]
    public async Task Make_default_marks_card_current()
    {
        var (api, screen, changed) = await Open();
        await Card(screen, "my-local").MakeDefaultCommand.RunAsync();

        Assert.Contains("default my-local", api.ModelCalls);
        Assert.Equal("Установлена · Текущая", Card(screen, "my-local").StateText);
        Assert.Equal("Установлена", Card(screen, "qwen3-8b-q4_k_m").StateText);
        Assert.False(Card(screen, "my-local").MakeDefaultCommand.CanExecute(null));
        Assert.Contains("models", changed);
    }

    [Fact]
    public async Task Delete_asks_first_and_removes_file_only_when_ramvi_downloaded_it()
    {
        var (api, screen, _) = await Open();
        var local = Card(screen, "my-local");

        await local.DeleteCommand.RunAsync();
        Assert.Empty(api.ModelCalls); // первое нажатие только спрашивает
        Assert.Equal("Точно забыть?", local.DeleteLabel);
        await local.DeleteCommand.RunAsync();
        Assert.Equal(["delete my-local file=False"], api.ModelCalls); // файл пользователя не трогаем

        var own = Card(screen, "qwen3-8b-q4_k_m");
        Assert.Equal("Удалить модель и её файл (скачан RAMVI)", own.DeleteHint);
        await own.DeleteCommand.RunAsync();
        Assert.Equal("Точно удалить файл?", own.DeleteLabel);
        await own.DeleteCommand.RunAsync();
        Assert.Contains("delete qwen3-8b-q4_k_m file=True", api.ModelCalls);
        Assert.Contains(screen.Cards, c => c.Name == "qwen3:8b Q4_K_M" && c.CanDownload); // снова в каталоге
    }

    [Fact]
    public async Task Recommendation_and_stale_catalog_are_shown()
    {
        var api = Api();
        api.CatalogStale = true;
        api.Recommendation = new Recommendation(
            "balanced",
            new RecommendedItem("qwen3-8b-q4_k_m", "Qwen3-8B Q4_K_M", true),
            Message("recommend.reason.speed", ("speed", Message("compat.speed_estimate", ("low", "48"), ("high", "57"))),
                ("status", Message("compat.recommended.inline")), ("size", Message("recommend.size", ("size", "8"), ("quant", "Q4_K_M")))),
            []);
        var (_, screen, _) = await Open(api);

        Assert.True(screen.IsStale);
        Assert.Equal(
            "Подбор модели: Qwen3-8B Q4_K_M — быстрее остальных подходящих: оценка ~48–57 токенов/с; рекомендуется, 8B Q4_K_M",
            screen.Recommendation);
        await screen.RefreshCommand.RunAsync();
        Assert.Equal(1, api.CatalogRefreshes);
    }

    [Fact]
    public async Task Models_are_a_settings_tab_not_a_sidebar_section()
    {
        var api = Api();
        api.Sections.Add(ModelsSection); // сервис по-прежнему объявляет раздел «Модели»
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        Assert.DoesNotContain(model.Sections.Items, i => i.View == "models"); // в боковом меню его нет
        Assert.Empty(model.ModelsScreen.Cards); // экран не грузится, пока вкладку не открыли

        await model.Settings.OpenAsync();
        Assert.Equal(["general", "sound", "models", "connections"], model.Settings.Tabs.Select(t => t.Id));
        Assert.True(model.Settings.IsGeneralTab && model.Settings.HasFooter);
        Assert.Equal("qwen3-8b-q4_k_m", model.Settings.DefaultModel?.Id);
        await model.Settings.ManageModelsCommand.RunAsync(); // «Управлять моделями…» — вкладка «Модели»
        Assert.True(model.Settings.IsOpen && model.Settings.IsModelsTab && !model.Settings.HasFooter);
        await MainViewModelTests.Until(() => model.ModelsScreen.Cards.Count > 0, "экран загружен");

        await model.Settings.CloseCommand.RunAsync();
        var loads = api.CatalogLoads;
        await model.OpenModelsAsync(); // прямой переход — та же вкладка, данные перечитываются
        Assert.True(model.Settings.IsOpen && model.Settings.IsModelsTab);
        await MainViewModelTests.Until(() => api.CatalogLoads > loads, "перечитан каталог");
    }

    [Fact]
    public async Task Module_tab_is_added_once_and_relocalized()
    {
        var api = Api();
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        model.Settings.AddTab("agent", "section.agent");
        model.Settings.AddTab("agent", "section.agent");
        Assert.Equal(["general", "sound", "models", "connections", "module:agent"], model.Settings.Tabs.Select(t => t.Id));
        model.Settings.SelectedTab = model.Settings.Tabs[^1];
        // У агента свой экран настроек (6.13) — это не общая заглушка вкладки модуля.
        Assert.True(model.Settings.IsAgentTab && !model.Settings.IsModuleTab && !model.Settings.HasFooter);
        model.Settings.AddTab("character", "section.character");
        model.Settings.SelectedTab = model.Settings.Tabs[^1];
        // У персонажа тоже свой экран настроек (9.13).
        Assert.True(model.Settings.IsCharacterTab && !model.Settings.IsModuleTab && !model.Settings.IsAgentTab && !model.Settings.HasFooter);
        Assert.Equal("Общие", model.Settings.Tabs[0].Title);
    }

    [Fact]
    public async Task Default_model_is_saved_from_settings()
    {
        var api = Api();
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        await model.Settings.OpenAsync();
        model.Settings.DefaultModel = model.Models.Single(m => m.Id == "my-local");
        await model.Settings.SaveCommand.RunAsync();

        Assert.Contains("default my-local", api.ModelCalls);
        Assert.Equal("my-local", api.Onboarding.DefaultModel);
    }

    [AvaloniaFact]
    public async Task Window_shows_models_tab_with_cards_in_settings()
    {
        var api = Api();
        api.Sections.Add(ModelsSection);
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        await model.OpenModelsAsync();
        await MainViewModelTests.Until(() => model.ModelsScreen.Cards.Count == 4, "карточки");
        Dispatcher.UIThread.RunJobs();

        Assert.True(VoiceTests.Shown(window.FindControl<DockPanel>("ModelsPanel")!));
        Assert.False(VoiceTests.Shown(window.FindControl<ScrollViewer>("GeneralTab")!));
        Assert.False(VoiceTests.Shown(window.FindControl<Button>("SettingsSave")!)); // у «Моделей» нет «Сохранить»
        Assert.True(VoiceTests.Shown(window.FindControl<Button>("SettingsClose")!));
        model.Settings.SelectedTab = model.Settings.Tabs[0];
        Dispatcher.UIThread.RunJobs();
        Assert.True(VoiceTests.Shown(window.FindControl<Button>("SettingsSave")!));
        Assert.False(VoiceTests.Shown(window.FindControl<DockPanel>("ModelsPanel")!));
        var texts = window.GetVisualDescendants().OfType<TextBlock>().Select(t => t.Text).ToList();
        Assert.Contains("Установлена · Текущая", texts);
        Assert.Contains("Доступна онлайн", texts);
        Assert.DoesNotContain(window.FindControl<ListBox>("SectionList")!.Items.OfType<SectionItem>(), i => i.View == "models");
    }
}
