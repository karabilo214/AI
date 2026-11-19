using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;

namespace Ramvi.Desktop.Services;

/// <summary>Как запускать сервис: интерпретатор Python, корень `ramvi/`, папка данных и папка приложения.</summary>
public sealed record ServiceOptions(
    string Python,
    string RamviRoot,
    string DataDir,
    IReadOnlyList<string> ExtraArgs,
    string? AppDir = null,
    IReadOnlyDictionary<string, string>? Environment = null)
{
    /// <summary>Встроенный Python комплекта (PORT-02): `runtime/python/python.exe`.</summary>
    public const string BundledPython = "runtime/python/python.exe";

    /// <summary>Токенизатор комплекта для GGUF-движка (вместо кеша Hugging Face).</summary>
    public const string BundledTokenizer = "runtime/tokenizers/Qwen3-8B";

    /// <summary>Папка приложения с `settings.json` (3.7): `RAMVI_APP_DIR` или папка exe.</summary>
    public static string AppDirectory =>
        System.Environment.GetEnvironmentVariable("RAMVI_APP_DIR") ?? AppContext.BaseDirectory;

    /// <summary>
    /// Для разработки: `RAMVI_PYTHON`, `RAMVI_ROOT`, `RAMVI_DATA_DIR`. В комплекте (3.9):
    /// встроенный Python, код сервиса в `ramvi/`, токенизатор в `runtime/tokenizers`.
    /// Папка данных — из `settings.json` папки приложения, а если писать в неё нельзя
    /// (USB только на чтение) — явно выбранная пользователем (<see cref="AppLocations"/>).
    /// </summary>
    public static ServiceOptions FromEnvironment() => Resolve(AppDirectory, AppContext.BaseDirectory);

    internal static ServiceOptions Resolve(string appDir, string exeDir)
    {
        static string? Env(string name) => System.Environment.GetEnvironmentVariable(name);
        var bundled = Path.Combine(appDir, BundledPython);
        var python = Env("RAMVI_PYTHON")
            ?? (File.Exists(bundled) ? bundled : OperatingSystem.IsWindows() ? "python.exe" : "python3");
        var root = Env("RAMVI_ROOT") ?? FindRoot(appDir) ?? FindRoot(exeDir) ?? exeDir;
        var data = Env("RAMVI_DATA_DIR") ?? AppLocations.DataDir(appDir);
        var extra = Env("RAMVI_SERVICE_ARGS_JSON") is { } json
            ? JsonSerializer.Deserialize(json, HostJson.Default.StringArray) ?? []
            : [];
        var environment = new Dictionary<string, string>(StringComparer.Ordinal);
        var tokenizer = Path.Combine(appDir, BundledTokenizer);
        if (Directory.Exists(tokenizer))
        {
            environment["RAMVI_TOKENIZER"] = Path.GetFullPath(tokenizer);
        }

        if (python == bundled)
        {
            // Комплект изолирован от Python пользователя и не ходит в сеть Hugging Face.
            environment["PYTHONNOUSERSITE"] = "1";
            environment["HF_HUB_OFFLINE"] = "1";
            environment["TRANSFORMERS_OFFLINE"] = "1";
        }

        return new ServiceOptions(python, root, data, extra, appDir, environment);
    }

    /// <summary>Корень кода сервиса: папка с `service/ramvi_service/app.py` или её `ramvi/`.</summary>
    private static string? FindRoot(string start)
    {
        for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
        {
            foreach (var candidate in new[] { dir.FullName, Path.Combine(dir.FullName, "ramvi") })
            {
                if (File.Exists(Path.Combine(candidate, "service", "ramvi_service", "app.py")))
                {
                    return candidate;
                }
            }
        }

        return null;
    }
}

/// <summary>
/// Процесс сервиса, принадлежащий окну. Ключ — случайный, через окружение.
/// stdin сервиса держится открытым: закрылось окно (даже аварийно) — сервис
/// видит конец потока и выходит сам вместе с движком (3.3).
/// </summary>
public sealed class ServiceHost : IAsyncDisposable
{
    private static readonly string[] ServiceArgs = ["-P", "-m", "ramvi_service.app", "--data-dir"];
    private static readonly string[] PythonFolders = ["service", "engine", "storage"];

    private readonly Process _process;

    private ServiceHost(Process process, Uri address, string token)
    {
        _process = process;
        Address = address;
        Token = token;
    }

    public Uri Address { get; }

    public string Token { get; }

    /// <summary>Процесс встроенного Python с окружением сервиса (код RAMVI, без чужого PYTHONHOME).</summary>
    private static ProcessStartInfo PythonInfo(ServiceOptions options, IEnumerable<string> args)
    {
        var info = new ProcessStartInfo(options.Python)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = options.RamviRoot,
        };
        foreach (var arg in args)
        {
            info.ArgumentList.Add(arg);
        }

        info.Environment["PYTHONUTF8"] = "1";
        info.Environment["PYTHONPATH"] = string.Join(
            Path.PathSeparator,
            PythonFolders.Select(p => Path.Combine(options.RamviRoot, p)));
        info.Environment.Remove("PYTHONHOME"); // чужой PYTHONHOME сломал бы встроенный Python
        foreach (var (name, value) in options.Environment ?? new Dictionary<string, string>())
        {
            info.Environment[name] = value;
        }

        return info;
    }

    /// <summary>
    /// 8.3: запуск с обновлением. Подготовленное обновление применяется до старта сервиса;
    /// новая версия не стартовала — откат и запуск прежней; стартовала — отметка «хорошо».
    /// <paramref name="run"/> — вызов `ramvi_service.updates` (подменяется в проверках).
    /// </summary>
    public static Task<ServiceHost> StartWithUpdatesAsync(ServiceOptions options, TimeSpan timeout) =>
        WithUpdatesAsync(options, timeout, action => RunUpdaterAsync(options, action), StartAsync);

    /// <summary>Логика запуска с обновлением отдельно от процессов — для проверок.</summary>
    internal static async Task<T> WithUpdatesAsync<T>(
        ServiceOptions options, TimeSpan timeout, Func<string, Task<JsonElement?>> run,
        Func<ServiceOptions, TimeSpan, Task<T>> start)
    {
        if (options.AppDir is null)
        {
            return await start(options, timeout).ConfigureAwait(false);
        }

        var status = await run("status").ConfigureAwait(false);
        var applied = false;
        if (status is { } s && s.TryGetProperty("staged", out var staged) && staged.ValueKind == JsonValueKind.String)
        {
            var result = await run("apply").ConfigureAwait(false);
            applied = result is { } r && r.TryGetProperty("applied", out _);
        }
        else if (status is { } p && p.TryGetProperty("health", out var health) && health.GetString() == "pending"
                 && p.TryGetProperty("applied", out _))
        {
            applied = true; // прошлый запуск после обновления не дошёл до отметки
        }

        try
        {
            var host = await start(options, timeout).ConfigureAwait(false);
            if (applied)
            {
                await run("mark-good").ConfigureAwait(false);
            }

            return host;
        }
        catch when (applied)
        {
            await run("rollback").ConfigureAwait(false);
            return await start(options, timeout).ConfigureAwait(false);
        }
    }

    /// <summary>Вызвать `python -m ramvi_service.updates &lt;action&gt;`; ответ — JSON последней строки.</summary>
    private static async Task<JsonElement?> RunUpdaterAsync(ServiceOptions options, string action)
    {
        if (options.AppDir is not { } appDir)
        {
            return null;
        }

        var info = PythonInfo(options, ["-P", "-m", "ramvi_service.updates", action, "--app-dir", appDir, "--data-dir", options.DataDir]);
        using var process = Process.Start(info);
        if (process is null)
        {
            return null;
        }

        using var wait = new CancellationTokenSource(TimeSpan.FromMinutes(10));
        try
        {
            var output = await process.StandardOutput.ReadToEndAsync(wait.Token).ConfigureAwait(false);
            await process.WaitForExitAsync(wait.Token).ConfigureAwait(false);
            var last = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).LastOrDefault();
            return last is null ? null : JsonDocument.Parse(last).RootElement.Clone();
        }
        catch (Exception error) when (error is OperationCanceledException or JsonException)
        {
            Kill(process);
            return null;
        }
    }

    public static async Task<ServiceHost> StartAsync(ServiceOptions options, TimeSpan timeout)
    {
        var token = Convert.ToHexString(RandomNumberGenerator.GetBytes(32));
        var args = ServiceArgs.Append(options.DataDir).Concat(options.ExtraArgs).ToList();
        if (options.AppDir is { } appDir)
        {
            args.Add("--app-dir");
            args.Add(appDir);
        }

        var info = PythonInfo(options, args);
        info.Environment["RAMVI_UI_TOKEN"] = token;
        info.Environment["RAMVI_EXIT_WITH_PARENT"] = "1";
        var process = Process.Start(info) ?? throw new InvalidOperationException("service did not start");
        var errors = new System.Text.StringBuilder();
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is not null && errors.Length < 4000)
            {
                errors.AppendLine(e.Data);
            }
        };
        process.BeginErrorReadLine();
        using var wait = new CancellationTokenSource(timeout);
        try
        {
            var line = await process.StandardOutput.ReadLineAsync(wait.Token).ConfigureAwait(false);
            using var ready = JsonDocument.Parse(line ?? "{}");
            if (ready.RootElement.TryGetProperty("event", out var kind) && kind.GetString() == "error")
            {
                throw ServiceStartException.From(ready.RootElement);
            }

            if (!ready.RootElement.TryGetProperty("port", out var port))
            {
                throw new InvalidOperationException(errors.ToString().Trim());
            }

            return new ServiceHost(process, new Uri($"http://127.0.0.1:{port.GetInt32()}"), token);
        }
        catch (Exception error) when (error is OperationCanceledException or JsonException)
        {
            Kill(process);
            throw new InvalidOperationException(errors.ToString().Trim(), error);
        }
        catch
        {
            Kill(process);
            throw;
        }
    }

    public async ValueTask DisposeAsync()
    {
        try
        {
            _process.StandardInput.Close(); // мягкий выход: сервис останавливает движок
            using var wait = new CancellationTokenSource(TimeSpan.FromSeconds(15));
            await _process.WaitForExitAsync(wait.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            Kill(_process);
        }
        finally
        {
            _process.Dispose();
        }
    }

    private static void Kill(Process process)
    {
        try
        {
            process.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException)
        {
        }
    }
}

/// <summary>Сервис не запустился по известной причине (например, нет доступа к папке данных).</summary>
public sealed class ServiceStartException(string code, IReadOnlyDictionary<string, object?> parameters)
    : Exception(code)
{
    public string Code { get; } = code;

    public IReadOnlyDictionary<string, object?> Parameters { get; } = parameters;

    internal static ServiceStartException From(JsonElement root)
    {
        var code = root.TryGetProperty("code", out var c) ? c.GetString() ?? "api.internal" : "api.internal";
        var values = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (root.TryGetProperty("params", out var p) && p.ValueKind == JsonValueKind.Object)
        {
            foreach (var item in p.EnumerateObject())
            {
                values[item.Name] = item.Value.Clone();
            }
        }

        return new ServiceStartException(code, values);
    }
}

[System.Text.Json.Serialization.JsonSerializable(typeof(string[]), TypeInfoPropertyName = "StringArray")]
internal sealed partial class HostJson : System.Text.Json.Serialization.JsonSerializerContext;
