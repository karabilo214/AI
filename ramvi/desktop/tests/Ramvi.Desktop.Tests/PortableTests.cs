using System.Text.Json;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>3.9 (PORT-01): раскладка комплекта, папка данных при папке приложения только на чтение.</summary>
[Collection("environment")] // меняет переменные окружения процесса
public sealed class PortableTests : IDisposable
{
    private static readonly string[] Variables = ["RAMVI_PYTHON", "RAMVI_ROOT", "RAMVI_DATA_DIR", "RAMVI_SERVICE_ARGS_JSON"];
    private readonly Dictionary<string, string?> _saved = [];
    private readonly string _dir = Directory.CreateTempSubdirectory("ramvi-portable-").FullName;
    private readonly string _overrides;

    public PortableTests()
    {
        foreach (var name in Variables)
        {
            _saved[name] = Environment.GetEnvironmentVariable(name);
            Environment.SetEnvironmentVariable(name, null);
        }

        _overrides = AppLocations.OverrideFile;
        AppLocations.OverrideFile = Path.Combine(_dir, "profile", "RAMVI", "locations.json");
        Localizer.Current = Localizer.Load("ru");
    }

    public void Dispose()
    {
        foreach (var (name, value) in _saved)
        {
            Environment.SetEnvironmentVariable(name, value);
        }

        AppLocations.OverrideFile = _overrides;
        if (!OperatingSystem.IsWindows())
        {
            foreach (var dir in Directory.EnumerateDirectories(_dir, "*", SearchOption.AllDirectories).Append(_dir))
            {
                File.SetUnixFileMode(dir, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            }
        }

        Directory.Delete(_dir, recursive: true);
    }

    private string Bundle()
    {
        var app = Path.Combine(_dir, "RAMVI");
        Directory.CreateDirectory(Path.Combine(app, "runtime", "python"));
        File.WriteAllText(Path.Combine(app, "runtime", "python", "python.exe"), "");
        Directory.CreateDirectory(Path.Combine(app, "runtime", "tokenizers", "Qwen3-8B"));
        Directory.CreateDirectory(Path.Combine(app, "ramvi", "service", "ramvi_service"));
        File.WriteAllText(Path.Combine(app, "ramvi", "service", "ramvi_service", "app.py"), "");
        return app;
    }

    [Fact]
    public void Bundle_layout_uses_embedded_python_code_and_tokenizer()
    {
        var app = Bundle();

        var options = ServiceOptions.Resolve(app, app);

        Assert.Equal(Path.Combine(app, ServiceOptions.BundledPython), options.Python);
        Assert.Equal(Path.Combine(app, "ramvi"), options.RamviRoot);
        Assert.Equal(Path.Combine(app, "data"), options.DataDir);
        Assert.Equal(app, options.AppDir);
        var env = options.Environment!;
        Assert.Equal(Path.GetFullPath(Path.Combine(app, ServiceOptions.BundledTokenizer)), env["RAMVI_TOKENIZER"]);
        Assert.Equal(("1", "1", "1"), (env["PYTHONNOUSERSITE"], env["HF_HUB_OFFLINE"], env["TRANSFORMERS_OFFLINE"]));
    }

    [Fact]
    public void Without_bundle_system_python_and_no_isolation()
    {
        var app = Path.Combine(_dir, "plain");
        Directory.CreateDirectory(app);

        var options = ServiceOptions.Resolve(app, app);

        Assert.Equal(OperatingSystem.IsWindows() ? "python.exe" : "python3", options.Python);
        Assert.Empty(options.Environment!);
    }

    [Fact]
    public void Chosen_data_dir_inside_app_is_saved_relative_in_settings()
    {
        var app = Bundle();
        File.WriteAllText(Path.Combine(app, "settings.json"), """{"language": "en", "api_port": 12000}""");

        AppLocations.SaveDataDir(app, Path.Combine(app, "store", "d"));

        using var saved = JsonDocument.Parse(File.ReadAllText(Path.Combine(app, "settings.json")));
        Assert.Equal("store/d", saved.RootElement.GetProperty("data_dir").GetString());
        Assert.Equal("en", saved.RootElement.GetProperty("language").GetString()); // прочие поля не тронуты
        Assert.Equal(12000, saved.RootElement.GetProperty("api_port").GetInt32());
        Assert.Equal(Path.Combine(app, "store", "d"), AppLocations.DataDir(app));
        Assert.False(File.Exists(AppLocations.OverrideFile));

        var outside = Path.Combine(_dir, "elsewhere");
        AppLocations.SaveDataDir(app, outside);
        Assert.Equal(outside, AppLocations.DataDir(app));
    }

    [Fact]
    public void Read_only_app_folder_remembers_choice_in_profile()
    {
        var app = Bundle();
        var chosen = Path.Combine(_dir, "usb-data");
        if (OperatingSystem.IsWindows())
        {
            Assert.Skip("права POSIX");
            return;
        }

        File.SetUnixFileMode(app, UnixFileMode.UserRead | UnixFileMode.UserExecute);
        Assert.False(AppLocations.IsWritable(app));

        AppLocations.SaveDataDir(app, chosen);

        Assert.False(File.Exists(Path.Combine(app, "settings.json")));
        Assert.Equal(chosen, AppLocations.ReadOverride(app));
        Assert.Equal(chosen, AppLocations.DataDir(app));
        Assert.Equal(chosen, ServiceOptions.Resolve(app, app).DataDir);
    }

    [Fact]
    public void Override_is_ignored_when_app_folder_is_writable()
    {
        var app = Bundle();
        Directory.CreateDirectory(Path.GetDirectoryName(AppLocations.OverrideFile)!);
        File.WriteAllText(AppLocations.OverrideFile,
            new System.Text.Json.Nodes.JsonObject { [Path.GetFullPath(app)] = "/old/choice" }.ToJsonString());
        Assert.Equal("/old/choice", AppLocations.ReadOverride(app)); // запись есть, но не применяется

        Assert.Equal(Path.Combine(app, "data"), AppLocations.DataDir(app));
    }

    [Fact]
    public async Task Data_folder_without_access_offers_choosing_another()
    {
        var api = new FakeChatApi();
        var attempts = 0;
        var chosen = new List<string>();
        var model = new MainViewModel(
            () =>
            {
                if (++attempts == 1)
                {
                    throw new ServiceStartException("storage.no_access", new Dictionary<string, object?>
                    {
                        ["path"] = "/usb/RAMVI/data",
                        ["detail"] = "read-only",
                    });
                }

                return Task.FromResult<IChatApi>(api);
            },
            pickFolder: _ => Task.FromResult<string?>("/home/me/ramvi-data"),
            chooseDataDir: path =>
            {
                chosen.Add(path);
                return Task.CompletedTask;
            });
        await model.InitializeAsync();
        Assert.Equal(UiState.NoAccess, model.State);
        Assert.Equal("choose_data_dir", model.Banner!.Actions[0].Id);
        Assert.Equal(Localizer.Current["action.choose_data_dir"], model.Banner.Actions[0].Label);

        await model.ChooseDataDirCommand.RunAsync();

        Assert.Equal(["/home/me/ramvi-data"], chosen);
        Assert.Equal(2, attempts); // сервис запущен заново уже с новой папкой
        Assert.Equal(UiState.Ready, model.State);
    }

    [Fact]
    public async Task Bundle_environment_reaches_service_and_pythonhome_is_dropped()
    {
        var app = Bundle();
        var package = Path.Combine(app, "ramvi", "service", "ramvi_service");
        File.WriteAllText(Path.Combine(package, "__init__.py"), "");
        File.WriteAllText(Path.Combine(package, "app.py"), """
            import json, os
            keys = ("RAMVI_TOKENIZER", "PYTHONNOUSERSITE", "HF_HUB_OFFLINE", "PYTHONHOME")
            print(json.dumps({"event": "error", "code": "env", "params": {k: os.environ.get(k, "-") for k in keys}}), flush=True)
            """);
        var python = Environment.GetEnvironmentVariable("RAMVI_TEST_PYTHON") ?? (OperatingSystem.IsWindows() ? "python.exe" : "python3");
        var options = ServiceOptions.Resolve(app, app) with { Python = python };
        Environment.SetEnvironmentVariable("PYTHONHOME", "/broken/home");
        try
        {
            var error = await Assert.ThrowsAsync<ServiceStartException>(
                () => ServiceHost.StartAsync(options, TimeSpan.FromSeconds(30)));
            Assert.Equal(options.Environment!["RAMVI_TOKENIZER"], error.Parameters["RAMVI_TOKENIZER"]?.ToString());
            Assert.Equal("1", error.Parameters["HF_HUB_OFFLINE"]?.ToString());
            Assert.Equal("-", error.Parameters["PYTHONHOME"]?.ToString());
        }
        finally
        {
            Environment.SetEnvironmentVariable("PYTHONHOME", null);
        }
    }
}
