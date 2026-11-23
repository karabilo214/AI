using System.ComponentModel;
using Ramvi.Desktop.ViewModels;

namespace Ramvi.Desktop.Views;

/// <summary>
/// 3.13: держит окно персонажа вне RAMVI. Окно есть, только пока выбран режим вне RAMVI и
/// аватар не скрыт; одно окно за раз — значит один рисующий персонаж, звук один (в сервисе).
/// </summary>
public sealed class CharacterWindowHost : IDisposable
{
    private readonly MainViewModel _model;
    private CharacterFloatingWindow? _window;

    public CharacterWindowHost(MainViewModel model)
    {
        _model = model;
        _model.CharacterWindow.PropertyChanged += Changed;
        Sync();
    }

    public CharacterFloatingWindow? Window => _window;

    private void Changed(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(CharacterWindowViewModel.WindowVisible) or nameof(CharacterWindowViewModel.Mode)) Sync();
    }

    private void Sync()
    {
        var show = _model.CharacterWindow.WindowVisible;
        if (show && _window is null)
        {
            _window = new CharacterFloatingWindow();
            _window.Attach(_model);
            _window.Show();
        }
        else if (!show && _window is not null)
        {
            var window = _window;
            _window = null;
            window.CloseByHost();
        }
    }

    public void Dispose()
    {
        _model.CharacterWindow.PropertyChanged -= Changed;
        _window?.CloseByHost();
        _window = null;
    }
}
