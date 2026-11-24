using System.Text.Json;
using Avalonia.Controls;
using Ramvi.Desktop.Api;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.Threading;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>
/// B.13: горячие клавиши в настройках — переназначение, конфликт не сохраняется, сброс,
/// сохранение между запусками; новая клавиша работает в чате, у агента и у персонажа.
/// </summary>
public sealed class HotkeysTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("ramvi-hotkeys-").FullName;

    public HotkeysTests()
    {
        Localizer.Current = Localizer.Load("ru");
        Hotkeys.Current.Use(_folder);
    }

    public void Dispose()
    {
        Hotkeys.Current.Use(null); // остальным тестам — стандартные клавиши
        Directory.Delete(_folder, recursive: true);
    }

    private static HotkeyRow Row(HotkeysViewModel screen, string action) => screen.Rows.Single(r => r.Action == action);

    [Fact]
    public void Reassign_conflict_reset_and_restart()
    {
        var screen = new HotkeysViewModel();
        Assert.Equal(["Ctrl+M", "Esc", "Ctrl+N", "Ctrl+,"], screen.Rows.Select(r => r.KeyText));
        Assert.Equal("Новый чат (Ctrl+N)", Hotkeys.Current.NewChatHint);
        Assert.False(screen.Capture(Key.V, KeyModifiers.Control)); // не назначаем — нажатие не забрано

        Row(screen, Hotkeys.Microphone).ChangeCommand.Execute(null);
        Assert.Equal("Нажмите сочетание…", Row(screen, Hotkeys.Microphone).KeyText);
        Assert.True(screen.Capture(Key.LeftCtrl, KeyModifiers.Control)); // ждём остальное
        Assert.True(screen.Capture(Key.V, KeyModifiers.None)); // буква без Ctrl мешала бы печатать
        Assert.Equal("V без Ctrl, Alt или Win/⌘ мешала бы печатать — выберите сочетание", screen.Error);
        Assert.True(screen.IsCapturing);
        Assert.True(screen.Capture(Key.V, KeyModifiers.Control | KeyModifiers.Shift));
        Assert.False(screen.IsCapturing);
        Assert.Null(screen.Error);
        Assert.Equal("Ctrl+Shift+V", Row(screen, Hotkeys.Microphone).KeyText);
        Assert.Equal("Остановить запись и распознать (Ctrl+Shift+V)", Hotkeys.Current.VoiceStopHint);

        // Конфликт: та же клавиша у другого действия — не сохраняется.
        Row(screen, Hotkeys.NewChat).ChangeCommand.Execute(null);
        screen.Capture(Key.V, KeyModifiers.Control | KeyModifiers.Shift);
        Assert.Equal("Ctrl+Shift+V уже назначено действию «Микрофон: начать или остановить запись» — не сохранено", screen.Error);
        Assert.Equal("Ctrl+N", Row(screen, Hotkeys.NewChat).KeyText);

        // Перезапуск: клавиши из папки данных.
        Hotkeys.Current.Use(_folder);
        Assert.Equal("Ctrl+Shift+V", Hotkeys.Current.Text(Hotkeys.Microphone));
        Assert.True(Hotkeys.Current.Matches(Hotkeys.Microphone, Key.V, KeyModifiers.Control | KeyModifiers.Shift));
        Assert.False(Hotkeys.Current.Matches(Hotkeys.Microphone, Key.M, KeyModifiers.Control));

        screen.ResetCommand.Execute(null);
        Assert.Equal("Ctrl+M", Row(screen, Hotkeys.Microphone).KeyText);
        Hotkeys.Current.Use(_folder);
        Assert.Equal("Ctrl+M", Hotkeys.Current.Text(Hotkeys.Microphone));
    }

    [Fact]
    public void Broken_or_conflicting_file_falls_back_to_defaults()
    {
        File.WriteAllText(Path.Combine(_folder, Hotkeys.FileName),
            """{"microphone": "Ctrl+N", "stop": "nonsense", "settings": "Q", "new_chat": "Ctrl+N"}""");
        Hotkeys.Current.Use(_folder);
        // Ctrl+N уже у «нового чата» по умолчанию — микрофону не достаётся; мусор и голая буква — стандартные.
        Assert.Equal(["Ctrl+M", "Esc", "Ctrl+N", "Ctrl+,"], Hotkeys.Actions.Select(Hotkeys.Current.Text));
        File.WriteAllText(Path.Combine(_folder, Hotkeys.FileName), "{");
        Hotkeys.Current.Use(_folder);
        Assert.Equal("Ctrl+M", Hotkeys.Current.Text(Hotkeys.Microphone));
    }

    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();

    private const string Status = """{"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[{"id":"friendly","name":"Friendly"}],"history":[],"job":null}""";

    [AvaloniaFact]
    public async Task Reassigned_keys_work_in_chat_agent_and_character_and_old_ones_stop()
    {
        var api = new FakeChatApi();
        api.AddChat("Диалог", "model-a");
        var calls = new List<string>();
        var listening = false;
        api.ModuleHandler = (_, command, body) =>
        {
            var op = body.GetProperty("operation").GetString()!;
            calls.Add(command + ":" + op);
            if (command == "voice" && op == "start") listening = true;
            if (command == "voice" && op == "stop") listening = false;
            return Task.FromResult(Json(command switch
            {
                "voice" when op == "capabilities" => """{"available":true}""",
                "voice" => listening ? """{"state":"listening","partial":"","reply":null}""" : """{"state":"inactive","partial":"","reply":null}""",
                "speech" => "{}",
                _ => Status,
            }));
        };
        api.Sections.Add(new SectionInfo("agent", "section.agent", "agent", 20, "agent", "agent", "available", null));
        api.Sections.Add(new SectionInfo("character", "section.character", "character", 30, "character", "character", "available", null));
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 760 };
        window.Show();
        try
        {
            await model.InitializeAsync();
            Dispatcher.UIThread.RunJobs();

            async Task Press(PhysicalKey key, RawInputModifiers modifiers)
            {
                window.KeyPressQwerty(key, modifiers);
                for (var i = 0; i < 20; i++)
                {
                    Dispatcher.UIThread.RunJobs();
                    await Task.Delay(5, TestContext.Current.CancellationToken);
                }
            }

            // Назначение в окне: «Изменить» и нажатие.
            await model.Settings.OpenAsync("hotkeys");
            Assert.True(model.Settings.IsHotkeysTab);
            Row(model.Settings.Hotkeys, Hotkeys.Microphone).ChangeCommand.Execute(null);
            await Press(PhysicalKey.F2, RawInputModifiers.None);
            Assert.Equal("F2", Hotkeys.Current.Text(Hotkeys.Microphone));
            Row(model.Settings.Hotkeys, Hotkeys.NewChat).ChangeCommand.Execute(null);
            await Press(PhysicalKey.T, RawInputModifiers.Control | RawInputModifiers.Shift);
            Assert.Equal("Ctrl+Shift+T", Hotkeys.Current.Text(Hotkeys.NewChat));
            await Press(PhysicalKey.Escape, RawInputModifiers.None); // Esc закрывает настройки
            Assert.False(model.Settings.IsOpen);

            // Чат: старая клавиша не работает, новая — работает.
            await Press(PhysicalKey.M, RawInputModifiers.Control);
            Assert.Empty(api.VoiceCalls);
            await Press(PhysicalKey.F2, RawInputModifiers.None);
            Assert.Equal(["start"], api.VoiceCalls);
            await Press(PhysicalKey.Escape, RawInputModifiers.None);
            Assert.Equal(["start", "cancel"], api.VoiceCalls);
            var chats = model.Chats.Count;
            await Press(PhysicalKey.N, RawInputModifiers.Control);
            Assert.Equal(chats, model.Chats.Count);
            await Press(PhysicalKey.T, RawInputModifiers.Control | RawInputModifiers.Shift);
            await MainViewModelTests.Until(() => model.Chats.Count == chats + 1, "новый чат по новой клавише");

            // Агент: микрофон окна по новой клавише.
            model.Sections.Selected = model.Sections.Items.Single(s => s.View == "agent");
            await Press(PhysicalKey.M, RawInputModifiers.Control);
            Assert.Equal(2, api.VoiceCalls.Count);
            await Press(PhysicalKey.F2, RawInputModifiers.None);
            Assert.Equal("start", api.VoiceCalls[^1]);
            await Press(PhysicalKey.Escape, RawInputModifiers.None);

            // Персонаж: голосовая сессия персонажа по новой клавише, старая не работает.
            model.Sections.Selected = model.Sections.Items.Single(s => s.View == "character");
            Dispatcher.UIThread.RunJobs();
            Assert.True(model.Sections.IsCharacterSelected);
            await Press(PhysicalKey.M, RawInputModifiers.Control);
            Assert.DoesNotContain("voice:start", calls);
            await Press(PhysicalKey.F2, RawInputModifiers.None);
            await MainViewModelTests.Until(() => calls.Contains("voice:start"), "голос персонажа по новой клавише");

            Assert.Equal("Новый чат (Ctrl+Shift+T)", Hotkeys.Current.NewChatHint);
        }
        finally
        {
            window.Close();
        }
    }
}
