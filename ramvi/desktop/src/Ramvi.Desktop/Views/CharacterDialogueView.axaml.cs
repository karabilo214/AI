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
        SizeChanged += (_, _) => { FitWidth(); FitControls(); };
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
    /// <summary>Ширина, ниже которой кнопки режима уходят под поле ввода (узкое окно персонажа).</summary>
    internal const double NarrowWidth = 520;

    /// <summary>
    /// 9.26: в узком отдельном окне колонка «Только текст»/«Остановить всё» сжимала поле ввода
    /// до половины ширины — там кнопки встают строкой под полем.
    /// </summary>
    private void FitWidth()
    {
        var narrow = Bounds.Width > 0 && Bounds.Width < NarrowWidth;
        Grid.SetRow(CharacterModeButtons, narrow ? 1 : 0);
        Grid.SetColumn(CharacterModeButtons, narrow ? 0 : 1);
        Grid.SetColumnSpan(CharacterComposer, narrow ? 2 : 1);
        CharacterModeButtons.Orientation = narrow ? Avalonia.Layout.Orientation.Horizontal : Avalonia.Layout.Orientation.Vertical;
        CharacterModeButtons.HorizontalAlignment = narrow ? Avalonia.Layout.HorizontalAlignment.Right : Avalonia.Layout.HorizontalAlignment.Stretch;
    }

    private void FitControls()
    {
        // Preserve the text composer and a readable transcript even with a device
        // failure banner and a multiline draft in a short window.
        var available=Bounds.Height-CharacterComposer.Bounds.Height-48-36-64;
        CharacterControls.MaxHeight=Math.Max(48,Math.Min(Bounds.Height*.45,available));
    }
}
