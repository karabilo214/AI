using Ramvi.Desktop.Services;
using Xunit;

namespace Ramvi.Desktop.Tests;

/// <summary>Запуск настоящего процесса: подменный `ramvi_service.app` во временной папке.</summary>
public sealed class ServiceHostTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("ramvi-host-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private ServiceOptions Service(string body)
    {
        var package = Path.Combine(_root, "service", "ramvi_service");
        Directory.CreateDirectory(package);
        File.WriteAllText(Path.Combine(package, "__init__.py"), "");
        File.WriteAllText(Path.Combine(package, "app.py"), body);
        var python = Environment.GetEnvironmentVariable("RAMVI_TEST_PYTHON")
            ?? (OperatingSystem.IsWindows() ? "python.exe" : "python3");
        return new ServiceOptions(python, _root, Path.Combine(_root, "data"), []);
    }

    [Fact]
    public async Task Error_event_becomes_service_start_exception_with_params()
    {
        var options = Service("""
            import json, sys
            print(json.dumps({"event": "error", "code": "storage.no_access",
                              "params": {"path": sys.argv[-1], "detail": "Permission denied"}}), flush=True)
            sys.exit(3)
            """);

        var error = await Assert.ThrowsAsync<ServiceStartException>(
            () => ServiceHost.StartAsync(options, TimeSpan.FromSeconds(30)));

        Assert.Equal("storage.no_access", error.Code);
        Assert.Equal(Path.Combine(_root, "data"), error.Parameters["path"]?.ToString());
        Assert.Equal("Permission denied", error.Parameters["detail"]?.ToString());
    }

    [Fact]
    public async Task Crash_without_ready_reports_stderr()
    {
        var options = Service("import sys\nsys.stderr.write('ModuleNotFoundError: boom\\n')\nsys.exit(1)\n");

        var error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => ServiceHost.StartAsync(options, TimeSpan.FromSeconds(30)));

        Assert.Contains("ModuleNotFoundError: boom", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Ready_line_gives_address_and_stdin_close_stops_service()
    {
        var options = Service("""
            import json, sys
            print(json.dumps({"event": "ready", "port": 45678}), flush=True)
            sys.stdin.read()
            """);

        var host = await ServiceHost.StartAsync(options, TimeSpan.FromSeconds(30));
        Assert.Equal(new Uri("http://127.0.0.1:45678"), host.Address);
        Assert.Equal(64, host.Token.Length);
        await host.DisposeAsync(); // закрыть stdin — сервис выходит сам, без kill
    }
}

public sealed class RealServiceStartTests
{
    [Fact]
    public async Task Real_service_with_file_instead_of_data_dir_reports_no_access()
    {
        var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../.."));
        Assert.True(File.Exists(Path.Combine(root, "service", "ramvi_service", "app.py")), root);
        var python = Environment.GetEnvironmentVariable("RAMVI_TEST_PYTHON");
        Assert.SkipWhen(python is null, "нужен RAMVI_TEST_PYTHON с зависимостями сервиса");
        var file = Path.GetTempFileName();
        try
        {
            var options = new ServiceOptions(python!, root, Path.Combine(file, "data"), []);
            var started = DateTime.UtcNow;
            var error = await Assert.ThrowsAsync<ServiceStartException>(
                () => ServiceHost.StartAsync(options, TimeSpan.FromSeconds(30)));
            Assert.Equal("storage.no_access", error.Code);
            Assert.True(DateTime.UtcNow - started < TimeSpan.FromSeconds(10));
        }
        finally
        {
            File.Delete(file);
        }
    }
}
