using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Ramvi.Desktop.ViewModels;
using Avalonia.Threading;

namespace Ramvi.Desktop.Views;

public sealed partial class CharacterDialogueView : UserControl
{
    public CharacterDialogueView()
    {
        InitializeComponent();
        CharacterInput.AddHandler(KeyDownEvent, OnInputKeyDown, RoutingStrategies.Tunnel);
        SizeChanged += (_, _) => FitControls();
        CharacterComposer.SizeChanged += (_, _) => FitControls();
        DialogueText.PropertyChanged += (_, e) =>
        {
            if (e.Property == TextBlock.TextProperty) Dispatcher.UIThread.Post(DialogueScroll.ScrollToEnd);
        };
    }
    private void OnInputKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Enter || e.KeyModifiers.HasFlag(KeyModifiers.Shift) || DataContext is not CharacterViewModel model) return;
        e.Handled = true;
        if (model.Send.CanExecute(null)) model.Send.Execute(null);
    }
    private void FitControls()
    {
        // Preserve the text composer and a readable transcript even with a device
        // failure banner and a multiline draft in a short window.
        var available=Bounds.Height-CharacterComposer.Bounds.Height-48-36-64;
        CharacterControls.MaxHeight=Math.Max(48,Math.Min(Bounds.Height*.45,available));
    }
}
