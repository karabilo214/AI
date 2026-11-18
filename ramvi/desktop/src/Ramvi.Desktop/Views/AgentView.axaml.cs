using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Ramvi.Desktop.ViewModels;

namespace Ramvi.Desktop.Views;

public sealed partial class AgentView : UserControl
{
    public AgentView()
    {
        InitializeComponent();
        AgentInput.AddHandler(KeyDownEvent, OnInputKeyDown, RoutingStrategies.Tunnel);
    }
    private void OnInputKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Enter || e.KeyModifiers.HasFlag(KeyModifiers.Shift) || DataContext is not AgentViewModel model) return;
        e.Handled = true;
        if (model.StartCommand.CanExecute(null)) model.StartCommand.Execute(null);
    }
}
