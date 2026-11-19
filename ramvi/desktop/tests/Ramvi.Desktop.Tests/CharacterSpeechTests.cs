using System.Text.Json;
using System.Text.Json.Nodes;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Views;
using Avalonia.Headless.XUnit;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class CharacterSpeechTests
{
    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();
    private const string Catalog = """
      {"available":true,"settings":{"enabled":false,"provider":"system","voice":"local","remote_text_enabled":false},"providers":[
       {"id":"system","name":"System","external":false,"emotion_control":"none","pricing":{"source":"OS"},"voices":[{"id":"local","name":"Local","languages":["ru"]}]},
       {"id":"openai","name":"OpenAI","external":true,"emotion_control":"prompt","pricing":{"source":"https://developers.openai.com/api/docs/models/gpt-4o-mini-tts","checked_on":"2026-11-19","text_input_per_million_tokens":0.6,"audio_output_per_million_tokens":12},"voices":[{"id":"coral","name":"coral","languages":["ru","en"]}]}
      ]}
      """;
    [AvaloniaFact]
    public async Task Local_default_remote_opt_in_separate_key_and_preview()
    {
        var calls = new List<(string Op, JsonObject? Args)>();
        using var vm = new CharacterSpeechViewModel((op,args) => {
            calls.Add((op,args?.DeepClone().AsObject()));
            return Task.FromResult(op switch {
                "catalog" => Json(Catalog),
                "status" => Json("""{"state":"idle","last":{"fallback":true},"error":null}"""),
                _ => Json("{}")
            });
        });
        await vm.LoadAsync();
        Assert.False(vm.Enabled); Assert.False(vm.RemoteText); Assert.False(vm.IsExternal);
        Assert.Equal("local",vm.Voice!.Id); Assert.NotEmpty(vm.Fallback);
        vm.Provider=vm.Providers[1]; Assert.True(vm.IsExternal); Assert.Equal("coral",vm.Voice!.Id);
        Assert.Contains("2026-11-19",vm.Price); Assert.StartsWith("https://",vm.Source);
        vm.Key="fake-key"; await vm.KeyCommand.RunAsync(); Assert.Equal("",vm.Key);
        Assert.Equal("openai",calls.Last(x=>x.Op=="key").Args!["provider"]!.GetValue<string>());
        vm.Enabled=true; vm.RemoteText=true; await vm.SaveCommand.RunAsync();
        var config=calls.Last(x=>x.Op=="configure").Args!["settings"]!;
        Assert.True(config["remote_text_enabled"]!.GetValue<bool>());
        Assert.DoesNotContain("fake-key",config.ToJsonString());
        await vm.PreviewCommand.RunAsync(); Assert.Contains(calls,x=>x.Op=="preview");
        await vm.StopAsync(); Assert.Equal("stop",calls.Last().Op);
    }
    [AvaloniaFact]
    public async Task Failed_key_save_clears_secret_and_shows_error()
    {
        using var vm=new CharacterSpeechViewModel((op,_)=>op=="key" ? throw new InvalidOperationException("secret backend unavailable") : Task.FromResult(op=="catalog" ? Json(Catalog) : Json("""{"state":"off"}""")));
        await vm.LoadAsync(); vm.Provider=vm.Providers[1]; vm.Key="fake-secret";
        await vm.KeyCommand.RunAsync(); Assert.Empty(vm.Key); Assert.NotEmpty(vm.Error);
    }
    [AvaloniaFact]
    public async Task Expanded_output_settings_keep_the_composer_inside_window()
    {
        Localizer.Current=Localizer.Load("ru");
        var api=new FakeChatApi();
        api.ModuleHandler=(_,command,body)=>Task.FromResult(command switch {
            "voice"=>Json("""{"available":false}"""),
            "speech"=>body.GetProperty("operation").GetString() switch {
                "catalog"=>Json(Catalog),
                "status"=>Json("""{"state":"idle","last":null}"""),
                _=>Json("{}")
            },
            _=>Json("""{"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[{"id":"friendly","name":"Дружелюбный"}],"history":[{"user":"Привет!","assistant":"Проверим голос."}],"job":null}""")
        });
        api.Sections.Add(new SectionInfo("character","section.character","character",30,"character","character","available",null));
        using var model=new MainViewModel(api);
        var window=new MainWindow {DataContext=model,Width=1360,Height=900};window.Show();
        try {
            await model.InitializeAsync();
            model.Sections.Selected=model.Sections.Items.Single(x=>x.View=="character");
            var dialogue=window.GetVisualDescendants().OfType<CharacterDialogueView>().Single();
            var vm=(CharacterViewModel)dialogue.DataContext!;
            await MainViewModelTests.Until(()=>vm.Speech.Providers.Count==2,"speech catalog");
            var panel=dialogue.FindControl<Expander>("SpeechSettings")!;panel.IsExpanded=true;
            window.UpdateLayout();
            Dispatcher.UIThread.RunJobs();AvaloniaHeadlessPlatform.ForceRenderTimerTick();
            Assert.True(panel.Bounds.Height>200);
            var input=dialogue.GetVisualDescendants().OfType<TextBox>().Single(x=>x.AcceptsReturn);
            Assert.True(input.Bounds.Height>=72);
            if(Environment.GetEnvironmentVariable("RAMVI_SPEECH_SCREENSHOT") is {} screenshot) {
                using var image=window.CaptureRenderedFrame();image!.Save(screenshot,new PngBitmapEncoderOptions());
            }
            await vm.Speech.StopAsync();
        } finally {window.Close();}
    }
}
