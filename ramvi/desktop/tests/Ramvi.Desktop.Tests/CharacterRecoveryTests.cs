using System.Collections.ObjectModel;
using System.Text.Json;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Avalonia.Headless.XUnit;
using Ramvi.Desktop.Views;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class CharacterRecoveryTests
{
    private static JsonElement Json(string value)=>JsonDocument.Parse(value).RootElement.Clone();
    private const string Catalog="""{"available":true,"settings":{"enabled":true,"provider":"system","voice":"local","remote_text_enabled":false},"providers":[{"id":"system","name":"System","external":false,"emotion_control":"none","pricing":{"source":"OS"},"voices":[{"id":"local","name":"Local","languages":["ru"]},{"id":"second","name":"Second","languages":["en"]}]}]}""";
    [AvaloniaFact]
    public async Task Voice_switch_waits_for_pause_before_apply_and_does_not_lose_selection()
    {
        var stopped=new TaskCompletionSource<JsonElement>();var applied=false;string? selected=null;
        using var vm=new CharacterSpeechViewModel((op,args)=> {
            if(op=="catalog") return Task.FromResult(Json(Catalog));
            if(op=="stop") return stopped.Task;
            if(op=="configure") { applied=true;selected=args!["settings"]!["voice"]!.GetValue<string>(); }
            return Task.FromResult(Json("""{"state":"idle","error":null,"last":null}"""));
        });
        await vm.LoadAsync();vm.Voice=vm.Voices.Single(v=>v.Id=="second");
        var apply=vm.SaveCommand.RunAsync();Assert.False(applied);
        stopped.SetResult(Json("{}"));await apply;
        Assert.True(applied);Assert.Equal("second",selected);Assert.Empty(vm.Error);
    }
    [AvaloniaFact]
    public async Task Missing_saved_voice_is_preserved_until_explicit_selection()
    {
        var catalog=Catalog.Replace("\"voice\":\"local\"","\"voice\":\"removed\"");var calls=new List<string>();
        using var vm=new CharacterSpeechViewModel((op,_)=> {calls.Add(op);return Task.FromResult(op=="catalog" ? Json(catalog):Json("""{"state":"idle","error":null}"""));});
        await vm.LoadAsync();Assert.Equal("removed",vm.Voice!.Id);
        await vm.RefreshCommand.RunAsync();Assert.Equal("removed",vm.Voice!.Id);Assert.DoesNotContain("configure",calls);
    }
    [AvaloniaFact]
    public async Task Speech_failure_and_recovery_do_not_discard_in_progress_text()
    {
        var pending=new TaskCompletionSource<JsonElement>();var audioFailed=true;var cancels=0;
        const string status="""{"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[],"history":[],"job":null}""";
        var api=new FakeChatApi();api.ModuleHandler=(_,command,body)=> {
            var op=body.GetProperty("operation").GetString();
            if(command=="voice") return Task.FromResult(Json("""{"available":false}"""));
            if(command=="speech") {
                if(op=="catalog") return Task.FromResult(Json(Catalog));
                if(op=="configure") audioFailed=false;
                return Task.FromResult(audioFailed ? Json("""{"state":"error","mode":"text","error":"speech.device_lost","last":null}""") : Json("""{"state":"idle","mode":"speech","error":null,"last":null}"""));
            }
            if(op=="cancel") cancels++;
            return op switch {
                "start"=>Task.FromResult(Json("""{"response_id":"ongoing"}""")),
                "poll"=>pending.Task,
                _=>Task.FromResult(Json(status))
            };
        };
        using var vm=new CharacterViewModel(()=>api,new ObservableCollection<ModelInfo>(api.Models));await vm.LoadAsync();
        vm.Draft="Fresh 104729";await vm.Send.RunAsync();Assert.True(vm.Busy);Assert.True(vm.Speech.TextOnly);Assert.NotEmpty(vm.Speech.Error);
        await vm.Speech.ResumeCommand.RunAsync();Assert.True(vm.Busy);Assert.Equal(1,cancels);Assert.Empty(vm.Speech.Error);
        pending.SetResult(Json("""{"job":{"state":"completed","settled":true,"cursor":1,"events":[{"type":"segment","response_id":"ongoing","id":"s1","text":"Continued 393241"}]}}"""));
        await MainViewModelTests.Until(()=>!vm.Busy,"text completion after TTS failure");
        Assert.Contains("Continued 393241",vm.Transcript);
    }
    [AvaloniaFact]
    public async Task Failure_banner_and_composer_remain_accessible_in_short_window()
    {
        const string status="""{"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[],"history":[],"job":null}""";
        var api=new FakeChatApi();api.ModuleHandler=(_,command,body)=>Task.FromResult(command switch {
            "voice"=>Json("""{"available":false}"""),
            "speech"=>body.GetProperty("operation").GetString()=="catalog" ? Json(Catalog) : Json("""{"state":"error","mode":"text","error":"speech.device_lost","last":null}"""),
            _=>Json(status)
        });
        api.Sections.Add(new SectionInfo("character","section.character","character",30,"character","character","available",null));
        using var model=new MainViewModel(api);var window=new MainWindow {DataContext=model,Width=1180,Height=500};window.Show();
        try {
            await model.InitializeAsync();model.Sections.Selected=model.Sections.Items.Single(x=>x.View=="character");
            await MainViewModelTests.Until(()=>model.Character.Speech.TextOnly,"text fallback banner");
            var dialogue=window.GetVisualDescendants().OfType<CharacterDialogueView>().Single();
            model.Character.Draft=string.Join("\n",Enumerable.Repeat("Fresh 104729",25));
            window.UpdateLayout();Dispatcher.UIThread.RunJobs();AvaloniaHeadlessPlatform.ForceRenderTimerTick();
            var input=dialogue.GetVisualDescendants().OfType<TextBox>().Single(x=>x.AcceptsReturn);
            var bottom=input.TranslatePoint(new Point(0,input.Bounds.Height),window)!.Value;
            Assert.True(bottom.Y<=window.ClientSize.Height);
            var banner=dialogue.FindControl<StackPanel>("SpeechRecovery")!;
            Assert.True(banner.IsEffectivelyVisible);
            var controls=dialogue.FindControl<ScrollViewer>("CharacterControls")!;
            var viewport=controls.TranslatePoint(new Point(),window)!.Value;
            var warning=banner.TranslatePoint(new Point(),window)!.Value;
            Assert.InRange(warning.Y,viewport.Y,viewport.Y+controls.Bounds.Height-16);
            if(Environment.GetEnvironmentVariable("RAMVI_RECOVERY_SCREENSHOT") is {} file) {
                using var image=window.CaptureRenderedFrame();image!.Save(file,new PngBitmapEncoderOptions());
            }
        } finally {window.Close();}
    }
    [AvaloniaFact]
    public async Task Microphone_stop_waits_for_late_start_and_never_follows_old_reply()
    {
        var started=new TaskCompletionSource<JsonElement>();var stopped=false;
        const string status="""{"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[],"history":[],"job":null}""";
        var api=new FakeChatApi();api.ModuleHandler=(_,command,body)=> {
            var op=body.GetProperty("operation").GetString();
            if(command=="speech") return Task.FromResult(Json("""{"available":false}"""));
            if(command=="voice") {
                if(op=="capabilities") return Task.FromResult(Json("""{"available":true}"""));
                if(op=="start") return started.Task;
                if(op=="stop") stopped=true;
                return Task.FromResult(Json("""{"state":"inactive","partial":"","error":null,"reply":{"response_id":"old","text":"must not send"}}"""));
            }
            return Task.FromResult(Json(status));
        };
        using var vm=new CharacterViewModel(()=>api,new ObservableCollection<ModelInfo>(api.Models));await vm.LoadAsync();
        var start=vm.Voice.ToggleCommand.RunAsync();var stop=vm.Voice.StopAsync();Assert.False(stopped);
        started.SetResult(Json("""{"state":"starting","error":null}"""));await start;await stop;
        Assert.True(stopped);Assert.False(vm.Voice.IsActive);Assert.DoesNotContain("must not send",vm.Transcript);
    }
}
