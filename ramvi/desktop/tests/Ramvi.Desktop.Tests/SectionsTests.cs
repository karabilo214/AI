using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;

namespace Ramvi.Desktop.Tests;

/// <summary>3.11: реестр разделов в оболочке — ленивые виды, независимость чата.</summary>
public sealed class SectionsTests
{
    private static readonly Localizer Ru = Localizer.Load("ru");
    private static readonly SectionInfo Agent = new("agent", "section.agent", "agent", 20, "agent", "agent", "available", null);
    private static readonly SectionInfo Character = new("character", "section.character", "character", 30, "character", "character", "available", null);

    public SectionsTests() => Localizer.Current = Ru;

    private static async Task<(FakeChatApi Api, MainViewModel Model)> Open(params SectionInfo[] modules)
    {
        var api = new FakeChatApi();
        api.Sections.AddRange(modules);
        api.AddChat("Диалог", "model-a", Msg("user", "вопрос"), Msg("assistant", "ответ"));
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        return (api, model);
    }

    private static Task Select(MainViewModel model, string id)
    {
        model.Sections.Selected = model.Sections.Items.Single(i => i.Id == id);
        return MainViewModelTests.Until(() => id == "chat" || model.Sections.Current?.Item.Id == id, "раздел открыт");
    }

    [Fact]
    public async Task Only_chat_hides_rail_character_may_be_absent()
    {
        var (api, model) = await Open();

        Assert.False(model.Sections.HasMany);
        Assert.True(model.Sections.IsChatSelected);
        Assert.Equal(["chat"], model.Sections.Items.Select(i => i.Id));
        Assert.DoesNotContain(api.Calls, c => c.StartsWith("open", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Module_view_is_created_only_when_opened()
    {
        var (api, model) = await Open(Agent, Character);

        Assert.True(model.Sections.HasMany);
        Assert.Equal(["Чат", "Агент", "Персонаж"], model.Sections.Items.Select(i => i.Title));
        Assert.Equal(0, model.Sections.CreatedViews);

        await Select(model, "agent");

        Assert.False(model.Sections.IsChatSelected);
        Assert.Equal(1, model.Sections.CreatedViews);
        Assert.Equal(Ru["section.state.active"], model.Sections.Current!.State);
        Assert.Equal(Ru["section.placeholder.agent"], model.Sections.Current.Placeholder);
        Assert.DoesNotContain("open character", api.Calls); // персонаж не загружался
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task Switching_sections_keeps_chat_and_module_view()
    {
        var (_, model) = await Open(Agent, Character);
        model.Input = "недописанный вопрос";
        var chat = model.SelectedChat;
        var messages = model.Messages.ToList();

        await Select(model, "agent");
        var agentView = model.Sections.Current;
        await Select(model, "chat");
        Assert.True(model.Sections.IsChatSelected);
        Assert.Null(model.Sections.Current);
        await Select(model, "character");
        await Select(model, "agent");

        Assert.Same(agentView, model.Sections.Current); // вид не пересоздан
        Assert.Equal(2, model.Sections.CreatedViews);
        await Select(model, "chat");
        Assert.Equal("недописанный вопрос", model.Input); // контекст чата не тронут
        Assert.Same(chat, model.SelectedChat);
        Assert.Equal(messages, model.Messages);
    }

    [Fact]
    public async Task Module_error_shows_in_its_section_and_chat_still_works()
    {
        var (api, model) = await Open(Character);
        api.OpenResults["character"] = Character with { State = "error", Error = "ModuleNotFoundError: character" };

        await Select(model, "character");
        await MainViewModelTests.Until(() => model.Sections.Current!.HasError, "ошибка показана");

        Assert.Equal(Ru["section.state.error"], model.Sections.Current!.State);
        Assert.Equal("Ошибка модуля: ModuleNotFoundError: character", model.Sections.Current.Error);
        await Select(model, "chat");
        model.Input = "работает?";
        api.Stream.Writer.Complete();
        await model.SendCommand.RunAsync();
        Assert.Contains(api.Calls, c => c.StartsWith("send", StringComparison.Ordinal));
    }
}
