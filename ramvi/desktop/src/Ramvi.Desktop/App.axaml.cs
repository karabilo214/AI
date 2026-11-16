using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;
using Ramvi.Desktop.Views;

namespace Ramvi.Desktop;

public sealed partial class App : Application, IDisposable
{
    private ServiceHost? _service;
    private ChatApiClient? _api;

    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public void Dispose() => _api?.Dispose();

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            Localizer.Current = Localizer.Load(Environment.GetEnvironmentVariable("RAMVI_LANGUAGE") ?? "ru");
            var window = new MainWindow();
            desktop.MainWindow = window;
            window.Opened += async (_, _) => await ConnectAsync(window).ConfigureAwait(true);
            desktop.ShutdownRequested += async (_, _) => await StopServiceAsync().ConfigureAwait(true);
        }

        base.OnFrameworkInitializationCompleted();
    }

    private async Task ConnectAsync(MainWindow window)
    {
        var model = new MainViewModel(StartServiceAsync, window.CopyAsync, window.OpenFolderAsync);
        window.DataContext = model;
        await model.InitializeAsync().ConfigureAwait(true);
    }

    /// <summary>Запустить (или перезапустить) сервис; вызывается и действием «Повторить».</summary>
    private async Task<IChatApi> StartServiceAsync()
    {
        await StopServiceAsync().ConfigureAwait(true);
        _service = await ServiceHost.StartAsync(ServiceOptions.FromEnvironment(), TimeSpan.FromSeconds(60))
            .ConfigureAwait(true);
        _api = new ChatApiClient(_service.Address, _service.Token);
        return _api;
    }

    private async Task StopServiceAsync()
    {
        _api?.Dispose();
        _api = null;
        if (_service is not null)
        {
            await _service.DisposeAsync().ConfigureAwait(true);
            _service = null;
        }
    }
}
