using Avalonia.Controls;
using Avalonia.Threading;

namespace Ramvi.Desktop.Views;

public sealed partial class CharacterDialogueView : UserControl
{
    public CharacterDialogueView()
    {
        InitializeComponent();
        SizeChanged += (_, _) => FitControls();
        CharacterComposer.SizeChanged += (_, _) => FitControls();
        CharacterSelectors.SizeChanged += (_, _) => FitControls();
        DialogueText.PropertyChanged += (_, e) =>
        {
            if (e.Property == TextBlock.TextProperty) Dispatcher.UIThread.Post(DialogueScroll.ScrollToEnd);
        };
    }
    private void FitControls()
    {
        // Preserve the text composer and a readable transcript even with a device
        // failure banner and a multiline draft in a short window.
        var available=Bounds.Height-CharacterComposer.Bounds.Height-CharacterSelectors.Bounds.Height-48-36-64;
        CharacterControls.MaxHeight=Math.Max(48,Math.Min(Bounds.Height*.45,available));
    }
}
