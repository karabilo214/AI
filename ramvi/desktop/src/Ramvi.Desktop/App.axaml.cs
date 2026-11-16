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
        try
        {
            _service = await ServiceHost.StartAsync(ServiceOptions.FromEnvironment(), TimeSpan.FromSeconds(60))
                .ConfigureAwait(true);
        }
        catch (Exception error) when (error is InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            window.ShowFatal(Localizer.Current.Format("status.service_failed", ("detail", error.Message)));
            return;
        }

        _api = new ChatApiClient(_service.Address, _service.Token);
        var model = new MainViewModel(_api, window.CopyAsync);
        window.DataContext = model;
        await model.InitializeAsync().ConfigureAwait(true);
    }

    private async Task StopServiceAsync()
    {
        _api?.Dispose();
        if (_service is not null)
        {
            await _service.DisposeAsync().ConfigureAwait(true);
            _service = null;
        }
    }
}
