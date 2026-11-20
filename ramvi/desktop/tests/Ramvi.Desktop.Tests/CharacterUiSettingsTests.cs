using Avalonia;
using System.Text.Json;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.Input;
using Avalonia.VisualTree;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Character;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class CharacterUiSettingsTests
{
    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();
    private const string Status = """{"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[{"id":"friendly","name":"Friendly"}],"history":[],"job":null}""";
    private const string Catalog = """{"available":true,"settings":{"enabled":true,"provider":"system","voice":"local","remote_text_enabled":false},"providers":[{"id":"system","name":"System","external":false,"emotion_control":"none","pricing":{"source":"OS"},"voices":[{"id":"local","name":"Local","languages":["ru"]}]}]}""";

    [AvaloniaFact]
    public async Task Delivery_policy_roundtrips_and_labels_relocalize_without_changing_codes()
    {
        Localizer.Current = Localizer.Load("ru");
        var saved = "";
        using var speech = new CharacterSpeechViewModel((op, args) =>
        {
            if (op == "policy" && args is not null) saved = args.ToJsonString();
            return Task.FromResult(Json(op switch
            {
                "catalog" => Catalog,
                "policy" => """{"settings":{"mode":"complete","buffer":3,"reactions":false}}""",
                "status" => """{"state":"idle","last":null}""",
                _ => "{}"
            }));
        });
        await speech.LoadAsync();
        Assert.Equal("complete", speech.DeliveryMode!.Code); Assert.Equal("3", speech.Buffer!.Code); Assert.False(speech.Reactions);
        speech.Reactions = true; await speech.SaveCommand.RunAsync();
        Assert.Contains("complete", saved); Assert.Contains("true", saved);
        var label = speech.DeliveryMode.Label;
        Localizer.Current = Localizer.Load("en"); speech.Relocalize();
        Assert.Equal("complete", speech.DeliveryMode!.Code); Assert.NotEqual(label, speech.DeliveryMode.Label);
        Localizer.Current = Localizer.Load("ru");
    }

    [AvaloniaFact]
    public async Task Settings_live_in_settings_tab_and_small_dialogue_keeps_composer_visible()
    {
        Localizer.Current = Localizer.Load("ru");
        var api = new FakeChatApi(); var calls = new List<string>();
        var listening = false;
        string? sentModel = null;
        api.ModuleHandler = (_, command, body) =>
        {
            var op = body.GetProperty("operation").GetString()!; calls.Add(command + ":" + op);
            if (command == "voice" && op == "start") listening = true;
            if (command == "dialogue" && op == "start") sentModel = body.GetProperty("arguments").GetProperty("model").GetString();
            if (command == "voice" && op == "stop") listening = false;
            return Task.FromResult(Json(command switch
            {
                "voice" when op == "capabilities" => """{"available":true}""",
                "voice" => listening ? """{"state":"listening","partial":"","reply":null}""" : """{"state":"inactive","partial":"","reply":null}""",
                "speech" when op == "catalog" => Catalog,
                "speech" when op == "policy" => """{"settings":{"mode":"sentences","buffer":2,"reactions":true}}""",
                "speech" when op == "status" => """{"state":"idle","enabled":true,"armed":true,"mode":"speech","last":null}""",
                "speech" => "{}",
                "dialogue" when op == "start" => """{"response_id":"keyboard-reply"}""",
                "dialogue" when op == "poll" => """{"job":{"state":"completed","cursor":1,"events":[{"type":"segment","response_id":"keyboard-reply","text":"Fresh keyboard reply"}]}}""",
                _ => Status
            }));
        };
        api.Sections.Add(new SectionInfo("character", "section.character", "character", 30, "character", "character", "available", null));
        using var model = new MainViewModel(api);
        var window = new MainWindow { DataContext = model, Width = 1180, Height = 500 }; window.Show();
        try
        {
            await model.InitializeAsync();
            Assert.Empty(calls);
            var picker = window.FindControl<ComboBox>("ModelBox")!;
            picker.SelectedItem = model.Models.Single(m => m.Id == "model-b");
            Assert.Same(model.SelectedModel, model.Agent.Model);
            Assert.Same(model.SelectedModel, model.Character.Model);
            Assert.Empty(calls); // Choosing the global model does not activate Character.
            Assert.Null(window.FindControl<ComboBox>("AgentModel"));
            await model.Settings.OpenAsync("module:character");
            await MainViewModelTests.Until(() => model.Character.Profiles.Count > 0, "Character settings load from chat");
            Assert.True(model.Sections.IsChatSelected);
            Assert.Contains("open character", api.Calls);
            Assert.Equal("model-b", model.Character.Model!.Id); // Saved Character model-a cannot override it.
            Assert.Same(model.SelectedModel, model.Agent.Model);
            model.Settings.CloseCommand.Execute(null);
            model.Sections.Selected = model.Sections.Items.Single(x => x.View == "character");
            var character = window.GetVisualDescendants().OfType<CharacterView>().Single();
            var dialogue = window.GetVisualDescendants().OfType<CharacterDialogueView>().Single();
            await MainViewModelTests.Until(() => character.FindControl<CharacterCanvas>("CharacterSurface")!.Player is not null && model.Character.Speech.Providers.Count > 0, "Character UI ready");
            model.Character.Draft = string.Join('\n', Enumerable.Repeat("Fresh 524287", 20));
            var draft = model.Character.Draft;
            await model.Settings.OpenAsync("module:character");
            window.UpdateLayout(); Dispatcher.UIThread.RunJobs();
            var settings = window.GetVisualDescendants().OfType<CharacterSettingsView>().Single();
            Assert.True(model.Settings.IsCharacterTab);
            Assert.False(model.Settings.IsModuleTab);
            Assert.True(settings.IsEffectivelyVisible);
            Assert.Null(settings.FindControl<Expander>("CharacterInputSettings"));
            Assert.NotNull(settings.FindControl<StackPanel>("SpeechSettings"));
            Assert.Empty(settings.GetVisualDescendants().OfType<Expander>());
            Assert.Null(settings.FindControl<ComboBox>("CharacterModel"));
            Assert.Null(dialogue.FindControl<Expander>("SpeechSettings"));
            Assert.Null(dialogue.FindControl<Expander>("CharacterInputSettings"));
            Assert.Null(character.FindControl<CheckBox>("AvatarSwitch"));
            Assert.Null(character.FindControl<Expander>("CharacterPreview"));
            Assert.DoesNotContain(settings.GetVisualDescendants().OfType<Button>(), button => button.Command == model.Character.Speech.PreviewCommand);
            if (Environment.GetEnvironmentVariable("RAMVI_CHARACTER_UI_SCREENSHOT") is {} settingsPath)
            {
                using var frame = window.CaptureRenderedFrame(); frame!.Save(Path.ChangeExtension(settingsPath, "settings.png"), new Avalonia.Media.Imaging.PngBitmapEncoderOptions());
            }
            model.Settings.CloseCommand.Execute(null);
            model.Character.Diagnostics = true;
            window.UpdateLayout(); Dispatcher.UIThread.RunJobs(); AvaloniaHeadlessPlatform.ForceRenderTimerTick();
            var composer = dialogue.FindControl<Border>("CharacterComposer")!;
            var point = composer.TranslatePoint(new Avalonia.Point(0, composer.Bounds.Height), window);
            Assert.NotNull(point); Assert.InRange(point.Value.Y, 0, window.Bounds.Height);
            Assert.Contains("composer", composer.Classes);
            var input = dialogue.FindControl<TextBox>("CharacterInput")!;
            Assert.Contains("bare", input.Classes);
            Assert.Equal(40, input.MinHeight); Assert.Equal(200, input.MaxHeight);
            Assert.Equal(window.FindControl<TextBox>("InputBox")!.MinHeight, input.MinHeight);
            Assert.True(dialogue.FindControl<Button>("CharacterSend")!.IsVisible);
            Assert.False(dialogue.FindControl<Button>("CharacterStop")!.IsVisible);
            Assert.True(character.IsAnimationRunning);
            Assert.True(picker.IsEffectivelyVisible);
            Assert.True(dialogue.FindControl<Avalonia.Controls.Shapes.Path>("MicrophoneSlash")!.IsVisible);
            await model.Character.Voice.ToggleCommand.RunAsync();
            Assert.False(dialogue.FindControl<Avalonia.Controls.Shapes.Path>("MicrophoneSlash")!.IsVisible);
            await model.Character.Voice.ToggleCommand.RunAsync();
            await MainViewModelTests.Until(() => !model.Character.Voice.IsActive, "microphone switched off");
            Assert.True(dialogue.FindControl<Avalonia.Controls.Shapes.Path>("MicrophoneSlash")!.IsVisible);
            var capabilityReads = calls.Count(x => x == "voice:capabilities");
            await model.Settings.OpenAsync("sound");
            model.Settings.Sound.Language = model.Settings.Sound.Languages.Single(x => x.Code == "en");
            await model.Settings.SaveCommand.RunAsync();
            Assert.Equal("en", api.AppSettings.VoiceLanguage);
            Assert.True(calls.Count(x => x == "voice:capabilities") > capabilityReads);
            Assert.False(model.Character.Voice.IsActive);
            model.Settings.CloseCommand.Execute(null);
            await model.Character.TextModeCommand.RunAsync();
            Assert.False(model.Character.Speech.Enabled);
            Assert.Equal(draft, model.Character.Draft);
            Assert.Contains("speech:stop", calls);
            Assert.DoesNotContain("dialogue:cancel", calls);
            if (Environment.GetEnvironmentVariable("RAMVI_CHARACTER_UI_SCREENSHOT") is {} path)
            {
                using var frame = window.CaptureRenderedFrame(); frame!.Save(path, new Avalonia.Media.Imaging.PngBitmapEncoderOptions());
                window.Height = 840; model.Character.Draft = "Fresh 524287";
                window.UpdateLayout(); Dispatcher.UIThread.RunJobs(); AvaloniaHeadlessPlatform.ForceRenderTimerTick();
                using var normal = window.CaptureRenderedFrame(); normal!.Save(Path.ChangeExtension(path, "normal.png"), new Avalonia.Media.Imaging.PngBitmapEncoderOptions());
            }
            model.Character.Draft = "Fresh keyboard input";
            var starts = calls.Count(x => x == "dialogue:start");
            var newline = new KeyEventArgs { RoutedEvent = InputElement.KeyDownEvent, Key = Key.Enter, KeyModifiers = KeyModifiers.Shift };
            input.RaiseEvent(newline);
            Assert.Contains("\n", model.Character.Draft); Assert.Equal(starts, calls.Count(x => x == "dialogue:start"));
            input.RaiseEvent(new KeyEventArgs { RoutedEvent = InputElement.KeyDownEvent, Key = Key.Enter });
            await MainViewModelTests.Until(() => model.Character.Transcript.Contains("Fresh keyboard reply", StringComparison.Ordinal), "Enter sends Character text");
            Assert.Equal("model-b", sentModel);
            Assert.Contains("load model-b", api.EngineCalls);
            Assert.True(model.Character.Draft.Length == 0);
            await model.Character.DisconnectCommand.RunAsync();
            Assert.Contains("dialogue:cancel", calls);
        }
        finally { window.Close(); }
    }
}
