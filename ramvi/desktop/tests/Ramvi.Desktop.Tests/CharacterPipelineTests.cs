using System.Collections.ObjectModel;
using System.Text.Json;
using Avalonia.Headless.XUnit;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Character;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class CharacterPipelineTests
{
    private static JsonElement Json(string text)=>JsonDocument.Parse(text).RootElement.Clone();
    private const string Status="""{"character":{"model":"model-a","draft":"","profile_id":"friendly"},"profiles":[{"id":"friendly","name":"Friendly"}],"history":[],"job":null}""";
    private static string Packet(string id,string text,string state="completed")=>JsonSerializer.Serialize(new {job=new {state,cursor=1,settled=true,events=new[] {new {type="segment",response_id=id,id="s1",text,emotion="neutral",animation="nod"}}}});
    [AvaloniaFact]
    public async Task New_text_preempts_pending_poll_and_drops_old_text_and_animation()
    {
        var late=new TaskCompletionSource<JsonElement>();var starts=0;var cancels=0;
        var api=new FakeChatApi();
        api.ModuleHandler=(_,command,body)=> {
            if(command!="dialogue") return Task.FromResult(Json("""{"available":false}"""));
            var op=body.GetProperty("operation").GetString();
            if(op=="start") return Task.FromResult(Json($$"""{"response_id":"reply{{++starts}}"}"""));
            if(op=="cancel") cancels++;
            if(op=="poll") return body.GetProperty("arguments").GetProperty("response_id").GetString()=="reply1" ? late.Task : Task.FromResult(Json(Packet("reply2","Fresh 131071")));
            return Task.FromResult(Json(Status));
        };
        using var vm=new CharacterViewModel(()=>api,new ObservableCollection<ModelInfo>(api.Models));await vm.LoadAsync();
        vm.Draft="Old 783041";await vm.Send.RunAsync();Assert.True(vm.Busy);
        vm.Draft="New 131071";await vm.Send.RunAsync();
        Assert.Equal(2,starts);Assert.True(cancels>=2);Assert.Contains("Fresh 131071",vm.Transcript);
        var current=vm.Presentation;
        late.SetResult(Json(Packet("reply1","STALE 783041")));await Task.Yield();
        Assert.DoesNotContain("STALE",vm.Transcript);Assert.Equal(current,vm.Presentation);Assert.False(vm.Busy);
        await vm.Stop.RunAsync();Assert.Null(vm.Presentation!.ResponseId);
    }
    [AvaloniaFact]
    public async Task Stop_while_start_acknowledgement_is_late_never_polls_or_presents_it()
    {
        var pending=new TaskCompletionSource<JsonElement>();var polls=0;var cancels=0;
        var api=new FakeChatApi();api.ModuleHandler=(_,command,body)=> {
            if(command!="dialogue") return Task.FromResult(Json("""{"available":false}"""));
            var op=body.GetProperty("operation").GetString();
            if(op=="start") return pending.Task;
            if(op=="poll") polls++;
            if(op=="cancel") cancels++;
            return Task.FromResult(Json(Status));
        };
        using var vm=new CharacterViewModel(()=>api,new ObservableCollection<ModelInfo>(api.Models));await vm.LoadAsync();
        vm.Draft="Fresh 524287";var send=vm.Send.RunAsync();var stop=vm.Stop.RunAsync();
        pending.SetResult(Json("""{"response_id":"late"}"""));await send;await stop;
        Assert.Equal(0,polls);Assert.True(cancels>=2);Assert.False(vm.Busy);Assert.Null(vm.Presentation!.ResponseId);
    }
    [AvaloniaFact]
    public async Task Repeated_segment_poll_does_not_duplicate_transcript_or_restart_motion()
    {
        var api=new FakeChatApi();var polls=0;
        api.ModuleHandler=(_,command,body)=>Task.FromResult(command!="dialogue" ? Json("""{"available":false}""") : body.GetProperty("operation").GetString() switch {
            "start"=>Json("""{"response_id":"repeat"}"""),
            "poll"=>Json(Packet("repeat","Unique 262147",++polls==1 ? "running":"completed")),
            _=>Json(Status)
        });
        using var vm=new CharacterViewModel(()=>api,new ObservableCollection<ModelInfo>(api.Models));await vm.LoadAsync();
        vm.Draft="Count once";await vm.Send.RunAsync();
        var directive=vm.Presentation;
        await MainViewModelTests.Until(()=>!vm.Busy,"repeated poll finished");
        Assert.Equal(1,vm.Transcript.Split("Unique 262147").Length-1);Assert.Same(directive,vm.Presentation);
    }
    [Fact]
    public void Renderer_rejects_old_generation_and_replayed_segments()
    {
        using var pack=new CharacterPack(Path.Combine(ServiceOptions.FromEnvironment().RamviRoot,"character","ramvi_character","assets","pack.json"));
        var player=new CharacterPlayer(pack);
        Assert.True(player.Apply(new(1,"old","s1","neutral","talk")));Assert.Equal("talk",player.Animation);
        Assert.True(player.Apply(new(2,null,null,"neutral","idle")));Assert.Equal("idle",player.Animation);
        Assert.False(player.Apply(new(1,"old","s2","neutral","nod")));
        Assert.False(player.Apply(new(2,"old","s3","neutral","shake")));
        Assert.True(player.Apply(new(3,"new","s1","neutral","nod")));player.Tick(100);var frame=player.Frame;
        Assert.False(player.Apply(new(3,"new","s1","neutral","nod")));Assert.Equal(frame,player.Frame);
    }
    [AvaloniaFact]
    public async Task Module_restart_accepts_new_session_with_lower_generation()
    {
        var session="before";var generation=100;var reply="first";
        var api=new FakeChatApi();api.ModuleHandler=(_,command,body)=>Task.FromResult(command!="dialogue" ? Json("""{"available":false}""") : body.GetProperty("operation").GetString() switch {
            "start"=>Json(JsonSerializer.Serialize(new {response_id=reply,generation,session_id=session})),
            "poll"=>Json(Packet(reply,"Fresh "+session)),
            _=>Json(Status.Replace("\"character\":", "\"session_id\":\""+session+"\",\"character\":"))
        });
        using var vm=new CharacterViewModel(()=>api,new ObservableCollection<ModelInfo>(api.Models));await vm.LoadAsync();
        vm.Draft="First";await vm.Send.RunAsync();Assert.Contains("Fresh before",vm.Transcript);
        session="after";generation=1;reply="second";await vm.LoadAsync();
        vm.Draft="Second";await vm.Send.RunAsync();
        Assert.Contains("Fresh after",vm.Transcript);Assert.Equal("second",vm.Presentation!.ResponseId);
    }
    [AvaloniaFact]
    public async Task Nullable_operational_error_does_not_reject_voice_or_speech_commands()
    {
        const string catalog="""{"available":true,"settings":{"enabled":true,"provider":"system","voice":"local","remote_text_enabled":false},"providers":[{"id":"system","name":"System","external":false,"emotion_control":"none","pricing":{"source":"OS"},"voices":[{"id":"local","name":"Local","languages":["ru"]}]}]}""";
        var previews=0;var voice="listening";
        var api=new FakeChatApi();api.ModuleHandler=(_,command,body)=> {
            var op=body.GetProperty("operation").GetString();
            if(command=="speech") {
                if(op=="catalog") return Task.FromResult(Json(catalog));
                if(op=="preview") previews++;
                return Task.FromResult(Json("""{"state":"idle","error":null,"last":null}"""));
            }
            if(command=="voice") {
                if(op=="capabilities") return Task.FromResult(Json("""{"available":true}"""));
                if(op=="stop") voice="inactive";
                return Task.FromResult(Json(JsonSerializer.Serialize(new {state=voice,error=(string?)null,partial="",reply=(string?)null})));
            }
            return Task.FromResult(Json(Status));
        };
        using var vm=new CharacterViewModel(()=>api,new ObservableCollection<ModelInfo>(api.Models));await vm.LoadAsync();
        Assert.True(vm.Speech.PreviewCommand.CanExecute(null));
        await vm.Speech.PreviewCommand.RunAsync();Assert.Equal(1,previews);Assert.Empty(vm.Speech.Error);
        await vm.Voice.ToggleCommand.RunAsync();Assert.True(vm.Voice.IsActive);Assert.Empty(vm.Voice.Error);
        await vm.Voice.StopAsync();await vm.Speech.StopAsync();
    }
}
