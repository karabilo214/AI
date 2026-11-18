using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;
using static Ramvi.Desktop.Tests.FakeChatApi;
using static Ramvi.Desktop.Tests.MainViewModelTests;

namespace Ramvi.Desktop.Tests;

/// <summary>5.2 (VOI-01): микрофон у поля ввода и раздел «Звук» в настройках.</summary>
public sealed class VoiceTests
{
    private static readonly Localizer Ru = Localizer.Load("ru");

    public VoiceTests() => Localizer.Current = Ru;

    private static (VoiceViewModel Voice, List<string> Inserted, List<Exception> Errors) Make(FakeChatApi api)
    {
        var inserted = new List<string>();
        var errors = new List<Exception>();
        var voice = new VoiceViewModel(() => api, inserted.Add, errors.Add, () => true, TimeSpan.FromMilliseconds(10));
        return (voice, inserted, errors);
    }

    [Fact]
    public async Task Two_clicks_record_then_put_text_into_input()
    {
        var api = new FakeChatApi();
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        model.Input = "Вопрос:";

        await model.Voice.ToggleCommand.RunAsync();
        Assert.True(model.Voice.IsRecording);
        Assert.Equal(Ru["voice.stop_hint"], model.Voice.Tip);
        await model.Voice.ToggleCommand.RunAsync();

        Assert.Equal(["start", "stop"], api.VoiceCalls);
        Assert.Equal("Вопрос: привет мир", model.Input); // в поле ввода, не в чат
        Assert.Empty(api.Calls);
        Assert.True(model.Voice.IsIdle);
        Assert.True(model.SendCommand.CanExecute(null));
    }

    [Fact]
    public async Task Transcribing_blocks_second_click_and_shows_state()
    {
        var api = new FakeChatApi { VoiceStopGate = new TaskCompletionSource() };
        var (voice, inserted, _) = Make(api);
        await voice.ToggleCommand.RunAsync();

        var stop = voice.ToggleCommand.RunAsync();
        await Until(() => voice.IsTranscribing, "распознавание началось");
        Assert.False(voice.ToggleCommand.CanExecute(null));
        Assert.Equal(Ru["voice.transcribing"], voice.Tip);
        api.VoiceStopGate.SetResult();
        await stop;

        Assert.Equal(["привет мир"], inserted);
        Assert.True(voice.IsIdle);
    }

    [Fact]
    public async Task Elapsed_time_follows_service_and_limit_stops_by_itself()
    {
        var api = new FakeChatApi();
        var (voice, inserted, _) = Make(api);
        await voice.ToggleCommand.RunAsync();

        api.VoiceSeconds = 65.4;
        await Until(() => voice.Elapsed == "1:05", "длительность записи");
        Assert.Equal(0.5, voice.Level);
        api.VoiceLimit = true;
        await Until(() => voice.IsIdle && inserted.Count == 1, "запись остановилась на пределе");
        Assert.Equal(["start", "stop"], api.VoiceCalls);
    }

    [Fact]
    public async Task Cancel_forgets_recording_without_text()
    {
        var api = new FakeChatApi();
        var (voice, inserted, _) = Make(api);
        await voice.ToggleCommand.RunAsync();
        Assert.True(voice.CancelCommand.CanExecute(null));

        await voice.CancelCommand.RunAsync();

        Assert.Equal(["start", "cancel"], api.VoiceCalls);
        Assert.Empty(inserted);
        Assert.True(voice.IsIdle);
        Assert.False(voice.CancelCommand.CanExecute(null));
    }

    [Fact]
    public async Task Errors_are_shown_and_chat_keeps_working()
    {
        var api = new FakeChatApi { VoiceError = Error("voice.model_missing", ("model", "large-v3-turbo-q5_0")) };
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        model.Input = "текст";

        await model.Voice.ToggleCommand.RunAsync();

        Assert.True(model.Voice.IsIdle);
        Assert.True(model.IsError);
        Assert.Equal(Ru.Format("voice.model_missing", ("model", "large-v3-turbo-q5_0")), model.Status);
        Assert.Equal("текст", model.Input);
        Assert.True(model.SendCommand.CanExecute(null)); // чат не сломан

        await model.Voice.ToggleCommand.RunAsync();
        api.VoiceError = Error("voice.no_signal");
        await model.Voice.ToggleCommand.RunAsync();
        Assert.Equal(Ru["voice.no_signal"], model.Status);
        Assert.True(model.Voice.IsIdle);

        await model.Voice.ToggleCommand.RunAsync();
        await model.Voice.ToggleCommand.RunAsync();
        Assert.False(model.IsError); // удачная запись убирает прошлую ошибку голоса
        Assert.Equal("текст привет мир", model.Input);
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task Sound_section_shows_and_saves_device_language_model()
    {
        var api = new FakeChatApi();
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        await model.OpenSettingsCommand.RunAsync();
        var sound = model.Settings.Sound;

        Assert.Equal(["Системный по умолчанию", "Mic A", "Mic B"], sound.Devices.Select(d => d.Label));
        Assert.Equal("", sound.Device?.Code);
        Assert.Equal(["auto", "ru", "en", "uk"], sound.Languages.Select(l => l.Code));
        Assert.Equal("auto", sound.Language?.Code);
        Assert.Equal("large-v3-turbo-q5_0", sound.Model?.Code);
        Assert.Equal(Ru.Format("voice.model_not_installed", ("size", "574")), sound.ModelStatus);
        Assert.True(sound.DownloadCommand.CanExecute(null));

        sound.Model = sound.Models.Single(m => m.Code == "small-q5_1");
        Assert.Equal(Ru["voice.model_installed"], sound.ModelStatus);
        Assert.False(sound.DownloadCommand.CanExecute(null));
        sound.Device = sound.Devices.Single(d => d.Code == "Mic B");
        sound.Language = sound.Languages.Single(l => l.Code == "uk");
        await model.Settings.SaveCommand.RunAsync();

        Assert.Equal(new SettingsUpdate(VoiceDevice: "Mic B", VoiceLanguage: "uk", VoiceModel: "small-q5_1"),
            api.SettingsUpdates.Single());
        Assert.Equal(("Mic B", "uk", "small-q5_1"), (sound.Device?.Code, sound.Language?.Code, sound.Model?.Code));
        Assert.Empty(Ru.Missing);
    }

    [Fact]
    public async Task Saved_microphone_that_is_gone_stays_visible()
    {
        var api = new FakeChatApi();
        api.AppSettings = api.AppSettings with { VoiceDevice = "USB Mic" };
        var model = new MainViewModel(api);
        await model.InitializeAsync();
        await model.OpenSettingsCommand.RunAsync();

        var sound = model.Settings.Sound;
        Assert.Equal("USB Mic", sound.Device?.Code);
        Assert.Equal(Ru.Format("voice.device_absent", ("device", "USB Mic")), sound.Device?.Label);
        await model.Settings.SaveCommand.RunAsync();
        Assert.Empty(api.SettingsUpdates); // ничего не изменилось — микрофон не сброшен
    }

    [Fact]
    public async Task Download_shows_progress_and_can_be_cancelled()
    {
        var api = new FakeChatApi();
        var sound = new SoundSettingsViewModel(() => api, TimeSpan.FromMilliseconds(10));
        await sound.LoadAsync(api.AppSettings);

        await sound.DownloadCommand.RunAsync();
        Assert.Equal(["download large-v3-turbo-q5_0"], api.VoiceCalls);
        Assert.True(sound.IsDownloading);
        Assert.Equal(Ru.Format("voice.model_downloading", ("percent", 0)), sound.ModelStatus);
        api.VoiceDownloadState = new VoiceDownload("running", "large-v3-turbo-q5_0", 35, 100);
        await Until(() => sound.ModelStatus == Ru.Format("voice.model_downloading", ("percent", 35)), "ход скачивания");

        await sound.CancelDownloadCommand.RunAsync();
        Assert.False(sound.IsDownloading);
        api.VoiceDownloadState = new VoiceDownload("failed", "large-v3-turbo-q5_0",
            Error: new ApiError("download.hash_mismatch", null));
        await sound.LoadAsync(api.AppSettings);
        Assert.Equal(Ru["download.hash_mismatch"], sound.ModelStatus);
    }

    [AvaloniaFact]
    public async Task Ctrl_M_toggles_recording_while_typing_and_Esc_cancels()
    {
        var api = new FakeChatApi();
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        Dispatcher.UIThread.RunJobs();
        var input = window.FindControl<TextBox>("InputBox")!;
        input.Focus();
        input.Text = "текст";

        async Task Press(PhysicalKey key, RawInputModifiers modifiers, int calls)
        {
            window.KeyPressQwerty(key, modifiers);
            for (var i = 0; i < 100 && api.VoiceCalls.Count < calls; i++)
            {
                Dispatcher.UIThread.RunJobs();
                await Task.Delay(5, TestContext.Current.CancellationToken);
            }
        }

        await Press(PhysicalKey.Escape, RawInputModifiers.None, 0);
        Assert.Empty(api.VoiceCalls); // Esc без записи ничего не делает
        await Press(PhysicalKey.M, RawInputModifiers.Control, 1);
        Assert.Equal(["start"], api.VoiceCalls);
        Assert.True(model.Voice.IsRecording);
        await Press(PhysicalKey.Escape, RawInputModifiers.None, 2);
        Assert.Equal(["start", "cancel"], api.VoiceCalls);
        await Press(PhysicalKey.M, RawInputModifiers.Control, 3);
        await Press(PhysicalKey.M, RawInputModifiers.Control, 4);
        Assert.Equal(["start", "cancel", "start", "stop"], api.VoiceCalls);
        await Until(() => model.Input == "текст привет мир", "текст в поле ввода");
        Assert.Equal("текст привет мир", input.Text);
        await Press(PhysicalKey.M, RawInputModifiers.None, 4);
        Assert.Equal(4, api.VoiceCalls.Count); // буква M без Ctrl — просто текст
        window.Close();
    }

    [AvaloniaFact]
    public async Task Window_has_mic_button_recording_indicator_and_sound_section()
    {
        var api = new FakeChatApi();
        var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model };
        window.Show();
        await model.InitializeAsync();
        Dispatcher.UIThread.RunJobs();

        var mic = window.FindControl<Button>("MicButton")!;
        var indicator = window.FindControl<StackPanel>("RecordingIndicator")!;
        Assert.True(mic.IsEffectivelyVisible && mic.IsEffectivelyEnabled);
        Assert.False(indicator.IsVisible);
        Assert.Equal(Ru["voice.start_hint"], ToolTip.GetTip(mic));

        await model.Voice.ToggleCommand.RunAsync();
        Dispatcher.UIThread.RunJobs();
        Assert.True(indicator.IsEffectivelyVisible);
        Assert.Contains("recording", mic.Classes);
        Assert.NotEmpty(indicator.GetVisualDescendants().OfType<Ellipse>());

        await model.Voice.CancelCommand.RunAsync();
        Dispatcher.UIThread.RunJobs();
        Assert.False(indicator.IsVisible);
        Assert.DoesNotContain("recording", mic.Classes);

        await model.OpenSettingsCommand.RunAsync();
        Dispatcher.UIThread.RunJobs();
        Assert.True(window.FindControl<ComboBox>("VoiceDeviceBox")!.IsEffectivelyVisible);
        Assert.Equal(3, window.FindControl<ComboBox>("VoiceDeviceBox")!.ItemCount);
        Assert.Equal("auto", (window.FindControl<ComboBox>("VoiceLanguageBox")!.SelectedItem as VoiceOption)?.Code);
        Assert.Empty(Localizer.Current.Missing);
        window.Close();
    }
}
