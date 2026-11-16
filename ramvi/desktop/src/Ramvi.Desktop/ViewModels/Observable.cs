using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;

namespace Ramvi.Desktop.ViewModels;

public abstract class ObservableObject : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    protected bool Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        Raise(name);
        return true;
    }

    protected void Raise([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    /// <summary>Сменился язык (3.14): все тексты этого объекта показать заново.</summary>
    public virtual void Relocalize() => Raise(string.Empty);
}

/// <summary>Команда; асинхронная версия не запускается повторно, пока выполняется.</summary>
public sealed class Command : ICommand
{
    private readonly Func<object?, Task> _run;
    private readonly Func<object?, bool>? _can;
    private bool _running;

    public Command(Func<object?, Task> run, Func<object?, bool>? can = null)
    {
        _run = run;
        _can = can;
    }

    public Command(Func<Task> run, Func<bool>? can = null)
        : this(_ => run(), can is null ? null : _ => can())
    {
    }

    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter) => !_running && (_can?.Invoke(parameter) ?? true);

    public async void Execute(object? parameter) => await RunAsync(parameter).ConfigureAwait(true);

    public async Task RunAsync(object? parameter = null)
    {
        if (!CanExecute(parameter))
        {
            return;
        }

        _running = true;
        Refresh();
        try
        {
            await _run(parameter).ConfigureAwait(true);
        }
        finally
        {
            _running = false;
            Refresh();
        }
    }

    public void Refresh() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}
