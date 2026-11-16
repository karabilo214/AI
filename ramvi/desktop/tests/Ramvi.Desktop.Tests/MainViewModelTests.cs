using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;

namespace Ramvi.Desktop.Tests;

public sealed class MainViewModelTests
{
    private static readonly Localizer Ru = Localizer.Load("ru");

    public MainViewModelTests() => Localizer.Current = Ru;

    /// <summary>Ждать условия: поток ответа читается в другом потоке.</summary>
    internal static async Task Until(Func<bool> condition, string what)
    {
        var deadline = DateTime.UtcNow.AddSeconds(5);
        while (!condition())
        {
            Assert.True(DateTime.UtcNow < deadline, $"не дождались: {what}");
            await Task.Delay(5, TestContext.Current.CancellationToken);
        }
    }

    private static async Task<MainViewModel> Open(FakeChatApi api)
    {
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        return model;
    }

    [Fact]
    public async Task Initialize_opens_latest_chat_with_its_model_and_limits()
    {
        var api = new FakeChatApi();
        api.AddChat("старый", "model-a", Msg("user", "раз"));
        api.AddChat("новый", "model-b", Msg("user", "привет"), Msg("assistant", "здравствуй", reasoning: "думаю"));

        var model = await Open(api);

        Assert.Equal(["новый", "старый"], model.Chats.Select(c => c.Title));
        Assert.Equal("новый", model.SelectedChat?.Title);
        Assert.Equal("model-b", model.SelectedModel?.Id);
        Assert.Equal(["привет", "здравствуй"], model.Messages.Select(m => m.Content));
        Assert.True(model.Messages[1].HasReasoning);
        Assert.False(model.Messages[1].IsThinkingOpen); // thinking свёрнут по умолчанию
        Assert.Equal(2, model.Limits.MaxFiles);
        Assert.Equal(Ru["status.ready"], model.Status);
        Assert.Empty(api.Calls); // открытие ничего не меняет в сервисе
    }

    [Fact]
    public async Task Send_streams_reasoning_and_text_then_finishes()
    {
        var api = new FakeChatApi();
        var chat = api.AddChat("", "model-a");
        var model = await Open(api);
        var stream = api.Stream.Writer;
        model.Input = "привет";

        var sending = model.SendCommand.RunAsync();
        await Until(() => model.IsGenerating, "генерация началась");
        Assert.True(model.StopCommand.CanExecute(null));
        Assert.False(model.SendCommand.CanExecute(null));
        Assert.False(model.NewChatCommand.CanExecute(null));

        await stream.WriteAsync(new ChatEvent("user", Message: Msg("user", "привет") with { ChatId = chat }), TestContext.Current.CancellationToken);
        await stream.WriteAsync(new ChatEvent("reasoning", Text: "мысль "), TestContext.Current.CancellationToken);
        await stream.WriteAsync(new ChatEvent("text", Text: "Здрав"), TestContext.Current.CancellationToken);
        await Until(() => model.Messages.Count == 2 && model.Messages[1].Content == "Здрав", "первая часть ответа");
        var answer = model.Messages[1];
        Assert.True(answer.IsStreaming);
        Assert.Equal("мысль ", answer.Reasoning);
        Assert.False(answer.IsThinkingOpen);

        await stream.WriteAsync(new ChatEvent("text", Text: "ствуй"), TestContext.Current.CancellationToken);
        await Until(() => answer.Content == "Здравствуй", "вторая часть ответа");
        var saved = Msg("assistant", "Здравствуй", reasoning: "мысль ", finish: "stop");
        await stream.WriteAsync(new ChatEvent("done", Message: saved, Finish: "stop"), TestContext.Current.CancellationToken);
        stream.Complete();
        await sending;

        Assert.False(model.IsGenerating);
        Assert.False(answer.IsStreaming);
        Assert.Equal(saved.Id, answer.Id);
        Assert.False(answer.HasNote);
        Assert.Equal("", model.Input);
        Assert.Equal(Ru["status.ready"], model.Status);
        Assert.False(model.IsError);
        Assert.Equal([$"send {chat} привет files=0"], api.Calls);
        Assert.True(model.RegenerateCommand.CanExecute(null));
    }

    [Fact]
    public async Task Send_without_chat_creates_one_with_selected_model()
    {
        var api = new FakeChatApi();
        var model = await Open(api);
        Assert.Null(model.SelectedChat);
        Assert.Equal("model-a", model.SelectedModel?.Id);
        model.SelectedModel = model.Models[1];
        Assert.Empty(api.Calls); // без диалога выбор модели только запоминается
        model.Input = "текст";
        api.Stream.Writer.Complete();

        await model.SendCommand.RunAsync();

        Assert.Single(model.Chats);
        Assert.Equal(api.Chats[0].Id, model.SelectedChat?.Id);
        Assert.Equal(["create model-b", $"send {api.Chats[0].Id} текст files=0"], api.Calls);
    }

    [Fact]
    public async Task Stop_asks_service_and_marks_answer_stopped()
    {
        var api = new FakeChatApi();
        var chat = api.AddChat("", "model-a");
        var model = await Open(api);
        var stream = api.Stream.Writer;
        model.Input = "длинно";
        var sending = model.SendCommand.RunAsync();
        await stream.WriteAsync(new ChatEvent("text", Text: "нач"), TestContext.Current.CancellationToken);
        await Until(() => model.Messages.Count == 1, "ответ начался");

        await model.StopCommand.RunAsync();
        Assert.Contains($"stop {chat}", api.Calls);
        await stream.WriteAsync(new ChatEvent("done", Message: Msg("assistant", "нач", finish: "cancelled"), Finish: "cancelled"), TestContext.Current.CancellationToken);
        stream.Complete();
        await sending;

        Assert.Equal(Ru["chat.stopped"], model.Messages[0].Note);
        Assert.False(model.StopCommand.CanExecute(null));
    }

    [Fact]
    public async Task Regenerate_replaces_previous_answer()
    {
        var api = new FakeChatApi();
        var old = Msg("assistant", "старый ответ");
        var chat = api.AddChat("t", "model-a", Msg("user", "вопрос"), old);
        var model = await Open(api);
        var stream = api.Stream.Writer;
        await stream.WriteAsync(new ChatEvent("regenerate", Removed: [old.Id]), TestContext.Current.CancellationToken);
        await stream.WriteAsync(new ChatEvent("text", Text: "новый"), TestContext.Current.CancellationToken);
        await stream.WriteAsync(new ChatEvent("done", Message: Msg("assistant", "новый ответ", finish: "stop")), TestContext.Current.CancellationToken);
        stream.Complete();

        await model.RegenerateCommand.RunAsync();

        Assert.Equal(["вопрос", "новый ответ"], model.Messages.Select(m => m.Content));
        Assert.Equal([$"regenerate {chat}"], api.Calls);
    }

    [Fact]
    public async Task Error_in_done_is_shown_localized_on_answer_and_status()
    {
        var api = new FakeChatApi();
        api.AddChat("t", "model-a");
        var model = await Open(api);
        var error = Error("engine.start_failed", ("detail", "нет памяти"));
        var stream = api.Stream.Writer;
        await stream.WriteAsync(new ChatEvent("done", Error: new ApiError(error.Code, error.Parameters.ToDictionary())), TestContext.Current.CancellationToken);
        stream.Complete();
        model.Input = "x";

        await model.SendCommand.RunAsync();

        var text = Ru.Format("engine.start_failed", ("detail", "нет памяти"));
        Assert.DoesNotContain("{", text, StringComparison.Ordinal);
        Assert.Equal(text, model.Status);
        Assert.True(model.IsError);
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task Rejected_send_keeps_input_and_attachments()
    {
        var api = new FakeChatApi();
        api.AddChat("t", "model-a");
        var model = await Open(api);
        await model.AddFilesAsync([AttachmentReaderTests.File("a.txt", "abc")]);
        model.Input = "сохрани меня";
        api.FailNext = Error("chat.busy");

        await model.SendCommand.RunAsync();

        Assert.Equal("сохрани меня", model.Input);
        Assert.Single(model.Attachments);
        Assert.True(model.IsError);
        Assert.Equal(Ru["chat.busy"], model.Status);
        Assert.False(model.IsGenerating);
    }

    [Fact]
    public async Task Attachments_are_sent_and_cleared()
    {
        var api = new FakeChatApi();
        var chat = api.AddChat("t", "model-a");
        var model = await Open(api);
        await model.AddFilesAsync([AttachmentReaderTests.File("a.txt", "abc"), AttachmentReaderTests.File("b.md", "# b")]);
        Assert.True(model.SendCommand.CanExecute(null)); // можно отправить только файлы
        await api.Stream.Writer.WriteAsync(new ChatEvent("user", Message: Msg("user", "")), TestContext.Current.CancellationToken);
        api.Stream.Writer.Complete();

        await model.SendCommand.RunAsync();

        Assert.Equal([$"send {chat}  files=2"], api.Calls);
        Assert.Empty(model.Attachments);
    }

    [Fact]
    public async Task Too_many_files_keeps_accepted_and_shows_limit()
    {
        var api = new FakeChatApi();
        var model = await Open(api);

        await model.AddFilesAsync([
            AttachmentReaderTests.File("1.txt", "1"),
            AttachmentReaderTests.File("2.txt", "2"),
            AttachmentReaderTests.File("3.txt", "3"),
        ]);

        Assert.Equal(2, model.Attachments.Count);
        Assert.True(model.IsError);
        Assert.Equal(Ru.Format("chat.too_many_files", ("limit", 2)), model.Status);
        await model.RemoveAttachmentCommand.RunAsync(model.Attachments[0]);
        Assert.Single(model.Attachments);
    }

    [Fact]
    public async Task Clear_and_delete_chat()
    {
        var api = new FakeChatApi();
        var first = api.AddChat("первый", "model-a", Msg("user", "a"));
        var second = api.AddChat("второй", "model-a", Msg("user", "b"));
        var model = await Open(api);
        Assert.True(model.ClearCommand.CanExecute(null));

        await model.ClearCommand.RunAsync();
        Assert.Empty(model.Messages);
        Assert.False(model.ClearCommand.CanExecute(null));

        await model.DeleteChatCommand.RunAsync(model.SelectedChat);
        Assert.Equal([$"clear {second}", $"delete {second}"], api.Calls);
        Assert.Equal(first, model.SelectedChat?.Id);
        Assert.Equal(["a"], model.Messages.Select(m => m.Content));

        await model.DeleteChatCommand.RunAsync(model.SelectedChat);
        Assert.Null(model.SelectedChat);
        Assert.False(model.HasChats);
        Assert.False(model.DeleteChatCommand.CanExecute(null));
    }

    [Fact]
    public async Task Copy_passes_message_text()
    {
        var api = new FakeChatApi();
        api.AddChat("t", "model-a", Msg("assistant", "скопируй"));
        var copied = new List<string>();
        var model = new MainViewModel(api, text =>
        {
            copied.Add(text);
            return Task.CompletedTask;
        });
        await model.InitializeAsync();

        await model.CopyCommand.RunAsync(model.Messages[0]);

        Assert.Equal(["скопируй"], copied);
        Assert.Equal(Ru["chat.copied"], model.Status);
    }

    [Fact]
    public async Task Model_change_is_saved_and_rejected_change_rolls_back()
    {
        var api = new FakeChatApi();
        var chat = api.AddChat("t", "model-a");
        var model = await Open(api);

        model.SelectedModel = model.Models[1];
        await Until(() => model.SelectedChat?.ModelId == "model-b", "модель сохранена");
        Assert.Equal([$"model {chat} model-b"], api.Calls);

        api.FailNext = Error("chat.model_not_found", ("model_id", "model-a"));
        model.SelectedModel = model.Models[0];
        await Until(() => model.IsError, "ошибка показана");
        Assert.Equal("model-b", model.SelectedModel?.Id);
        Assert.Equal("model-b", model.SelectedChat?.ModelId);
    }

    [Fact]
    public async Task Answer_of_other_chat_does_not_appear_in_opened_chat()
    {
        var api = new FakeChatApi();
        var first = api.AddChat("первый", "model-a");
        var second = api.AddChat("второй", "model-a");
        var model = await Open(api);
        var stream = api.Stream.Writer;
        model.Input = "во втором";
        var sending = model.SendCommand.RunAsync();
        await Until(() => model.IsGenerating, "генерация началась");

        model.SelectedChat = model.Chats.Single(c => c.Id == first);
        await stream.WriteAsync(new ChatEvent("text", Text: "чужой"), TestContext.Current.CancellationToken);
        await stream.WriteAsync(new ChatEvent("done", Message: Msg("assistant", "чужой")), TestContext.Current.CancellationToken);
        stream.Complete();
        await sending;

        Assert.Empty(model.Messages);
        Assert.Equal([$"send {second} во втором files=0"], api.Calls);
    }

    [Fact]
    public async Task Broken_stream_marks_partial_answer()
    {
        var api = new FakeChatApi();
        api.AddChat("t", "model-a");
        var model = await Open(api);
        var stream = api.Stream.Writer;
        model.Input = "x";
        var sending = model.SendCommand.RunAsync();
        await stream.WriteAsync(new ChatEvent("text", Text: "обры"), TestContext.Current.CancellationToken);
        await Until(() => model.Messages.Count == 1, "ответ начался");

        stream.Complete(new HttpRequestException("обрыв"));
        await sending;

        Assert.False(model.IsGenerating);
        Assert.False(model.Messages[0].IsStreaming);
        Assert.Equal(Ru["ui.connection_lost"], model.Messages[0].Note);
        Assert.True(model.IsError);
    }
}
