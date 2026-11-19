using System.ComponentModel;
using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Ramvi.Desktop.Character;
using Ramvi.Desktop.Localization;
using Ramvi.Desktop.Services;
using Ramvi.Desktop.ViewModels;

namespace Ramvi.Desktop.Views;

public sealed partial class CharacterView : UserControl, IDisposable
{
    public static readonly StyledProperty<CharacterDirective?> ResponseProperty = AvaloniaProperty.Register<CharacterView,CharacterDirective?>(nameof(Response));
    public CharacterDirective? Response { get => GetValue(ResponseProperty); set => SetValue(ResponseProperty,value); }
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if(change.Property==ResponseProperty && Response is {} directive) {
            _player?.Apply(directive); CharacterSurface?.InvalidateVisual();
        }
    }
    private readonly DispatcherTimer _timer = new() { Interval = TimeSpan.FromMilliseconds(33) };
    private readonly Stopwatch _clock = new();
    private CharacterPack? _pack;
    private CharacterPlayer? _player;
    private ModuleSectionViewModel? _model;
    private bool _attached;
    private string _emotion = "neutral";
    private readonly List<Visual> _visibilityOwners = [];
    internal bool IsAnimationRunning => _timer.IsEnabled;
    public CharacterView()
    {
        InitializeComponent();
        CharacterSurface.ResourceFailed += (_, _) => Dispatcher.UIThread.Post(() =>
        {
            Release(); CharacterError.Text = Localizer.Current["character.pack_error"];
        });
        _timer.Tick += (_, _) =>
        {
            if (_player is null) return;
            _player.Tick(Math.Min(_clock.Elapsed.TotalMilliseconds, 100));
            _clock.Restart();
            CharacterSurface.InvalidateVisual();
        };
        DataContextChanged += (_, _) =>
        {
            if (_model is not null) _model.PropertyChanged -= ModelChanged;
            _model = DataContext as ModuleSectionViewModel;
            if (_model is not null) _model.PropertyChanged += ModelChanged;
            Refresh();
        };
        CharacterEmotion.SelectionChanged += (_, _) =>
        {
            if (CharacterEmotion.SelectedItem is EmotionItem item && _player is not null)
            {
                _emotion = item.Id;
                _player.Present(item.Id, _player.Animation);
            }
        };
        CharacterIdle.Click += (_, _) => Motion("idle");
        CharacterBlink.Click += (_, _) => Motion("blink");
        CharacterTalk.Click += (_, _) => Motion("talk");
        CharacterNod.Click += (_, _) => Motion("nod");
        CharacterShake.Click += (_, _) => Motion("shake");
    }
    private void ModelChanged(object? sender, PropertyChangedEventArgs e) => Refresh();
    private void Motion(string name)
    {
        if (_player is not null) _player.Present(_player.Emotion, name);
    }
    protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
    {
        base.OnAttachedToVisualTree(e);
        _attached = true;
        foreach (var owner in this.GetVisualAncestors().Prepend(this))
        {
            owner.PropertyChanged += VisibilityChanged;
            _visibilityOwners.Add(owner);
        }
        if (_model is not null) { _model.PropertyChanged -= ModelChanged; _model.PropertyChanged += ModelChanged; }
        Refresh();
    }
    protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
    {
        _attached = false;
        foreach (var owner in _visibilityOwners) owner.PropertyChanged -= VisibilityChanged;
        _visibilityOwners.Clear();
        if (_model is not null) _model.PropertyChanged -= ModelChanged;
        Release(); base.OnDetachedFromVisualTree(e);
    }
    private void VisibilityChanged(object? sender, AvaloniaPropertyChangedEventArgs e)
    {
        if (e.Property == IsVisibleProperty) Dispatcher.UIThread.Post(Refresh);
    }
    public void Dispose()
    {
        foreach (var owner in _visibilityOwners) owner.PropertyChanged -= VisibilityChanged;
        _visibilityOwners.Clear();
        if (_model is not null) _model.PropertyChanged -= ModelChanged;
        Release();
        GC.SuppressFinalize(this);
    }
    private void Refresh()
    {
        if (!_attached || !IsEffectivelyVisible || _model?.CanPresent != true) { Release(); return; }
        if (_pack is not null) return;
        try
        {
            var root = ServiceOptions.FromEnvironment().RamviRoot;
            _pack = new CharacterPack(Path.Combine(root, "character", "ramvi_character", "assets", "pack.json"));
            _player = new CharacterPlayer(_pack);
            if(Response is {} directive) _player.Apply(directive);
            CharacterSurface.Pack = _pack; CharacterSurface.Player = _player;
            CharacterEmotion.ItemsSource = _pack.Emotions.Select(id => new EmotionItem(id)).ToArray();
            CharacterEmotion.SelectedIndex = _pack.Emotions.ToList().IndexOf(_emotion);
            CharacterError.Text = "";
            _clock.Restart(); _timer.Start();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or System.Text.Json.JsonException or InvalidOperationException or KeyNotFoundException)
        {
            Release(); CharacterError.Text = Localizer.Current["character.pack_error"];
        }
    }
    private void Release()
    {
        _timer.Stop(); _clock.Stop(); CharacterSurface.Release();
        _player = null; _pack?.Dispose(); _pack = null;
    }
    private sealed record EmotionItem(string Id)
    {
        public override string ToString() => Localizer.Current["character.emotion." + Id];
    }
}
