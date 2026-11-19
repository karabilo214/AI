using Avalonia.Controls;
using Avalonia.Threading;

namespace Ramvi.Desktop.Views;

public sealed partial class CharacterDialogueView : UserControl
{
    public CharacterDialogueView()
    {
        InitializeComponent();
        DialogueText.PropertyChanged += (_, e) =>
        {
            if (e.Property == TextBlock.TextProperty) Dispatcher.UIThread.Post(DialogueScroll.ScrollToEnd);
        };
    }
}
