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
    public static readonly StyledProperty<CharacterDirective?> ReplyEndProperty = AvaloniaProperty.Register<CharacterView,CharacterDirective?>(nameof(ReplyEnd));
    public CharacterDirective? ReplyEnd { get => GetValue(ReplyEndProperty); set => SetValue(ReplyEndProperty, value); }
    public static readonly StyledProperty<CharacterDirective?> ResponseProperty = AvaloniaProperty.Register<CharacterView,CharacterDirective?>(nameof(Response));
    public CharacterDirective? Response { get => GetValue(ResponseProperty); set => SetValue(ResponseProperty,value); }
    public static readonly StyledProperty<CharacterPlayback?> PlaybackProperty = AvaloniaProperty.Register<CharacterView,CharacterPlayback?>(nameof(Playback));
    public CharacterPlayback? Playback { get => GetValue(PlaybackProperty); set => SetValue(PlaybackProperty,value); }
    public static readonly StyledProperty<string> PackPathProperty = AvaloniaProperty.Register<CharacterView, string>(nameof(PackPath), "");
    public string PackPath { get => GetValue(PackPathProperty); set => SetValue(PackPathProperty, value); }
    public static readonly StyledProperty<bool> AvatarEnabledProperty = AvaloniaProperty.Register<CharacterView, bool>(nameof(AvatarEnabled), true);
    public bool AvatarEnabled { get => GetValue(AvatarEnabledProperty); set => SetValue(AvatarEnabledProperty, value); }
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == AvatarEnabledProperty) Refresh();
        if (change.Property == PackPathProperty && _packPath != (PackPath.Length == 0 ? null : PackPath))
        {
            if (PackPath.Length == 0) { _packPath = null; Release(); Refresh(); }
            else if (!_attached || !IsEffectivelyVisible || !AvatarEnabled) { _packPath = PackPath; Release(); }
            else
            {
                try { LoadPack(PackPath); }
                catch (Exception error) when (error is IOException or UnauthorizedAccessException or System.Text.Json.JsonException or InvalidOperationException or KeyNotFoundException or ArgumentException)
                { CharacterError.Text = Localizer.Current["character.pack_error"]; }
            }
        }
        if (change.Property == PlaybackProperty && Playback is {} playback) _player?.SetPlayback(playback);
        if(change.Property==ResponseProperty && Response is {} directive) {
            _player?.Apply(directive);
            CharacterSurface?.InvalidateVisual();
        }
        if(change.Property==ReplyEndProperty && ReplyEnd is {} end) _player?.Apply(end);
    }
    private readonly DispatcherTimer _timer = new() { Interval = TimeSpan.FromMilliseconds(33) };
    private readonly Stopwatch _clock = new();
    private CharacterPack? _pack;
    private CharacterPlayer? _player;
    private ModuleSectionViewModel? _model;
    private bool _attached;
    private string? _packPath;
    private string _emotion = "neutral";
    public void LoadPack(string path)
    {
        // Validate the replacement before touching the current surface.
        var next = new CharacterPack(path);
        var previous = _pack;
        if (_player is not null)
        {
            if (_clock.IsRunning) _player.Tick(_clock.Elapsed.TotalMilliseconds);
            _player.ReplacePack(next);
        }
        else
        {
            _player = new CharacterPlayer(next);
            if (Response is {} directive) _player.Apply(directive);
            if (Playback is {} playback) _player.SetPlayback(playback);
        }
        _packPath = Path.GetFullPath(path);
        SetCurrentValue(PackPathProperty, _packPath);
        _pack = next;
        CharacterSurface.Pack = next; CharacterSurface.Player = _player;
        previous?.Dispose();
        if (!_attached || !IsEffectivelyVisible || !AvatarEnabled || _model?.CanPresent != true) Release();
        else { CharacterError.Text = ""; _clock.Restart(); _timer.Start(); }
    }
    private readonly List<Visual> _visibilityOwners = [];
    internal bool IsAnimationRunning => _timer.IsEnabled;
    public CharacterView()
    {
        InitializeComponent();
        CharacterSurface.ResourceFailed += (_, _) =>
        {
            var failed = _pack;
            Dispatcher.UIThread.Post(() =>
            {
                if (!ReferenceEquals(failed, _pack)) return;
                Release(); CharacterError.Text = Localizer.Current["character.pack_error"];
            });
        };
        _timer.Tick += (_, _) =>
        {
            if (_player is null) return;
            _player.Tick(_clock.Elapsed.TotalMilliseconds);
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
    }
    private void ModelChanged(object? sender, PropertyChangedEventArgs e) => Refresh();
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
        if (!_attached || !IsEffectivelyVisible || !AvatarEnabled || _model?.CanPresent != true) { Release(); return; }
        if (_pack is not null) return;
        try
        {
            var root = ServiceOptions.FromEnvironment().RamviRoot;
            _pack = new CharacterPack(_packPath ?? Path.Combine(root, "character", "ramvi_character", "assets", "pack.json"));
            _player = new CharacterPlayer(_pack);
            if (_pack.Emotions.Contains(_emotion, StringComparer.Ordinal)) _player.Present(_emotion, "idle");
            if(Response is {} directive) _player.Apply(directive);
            if (Playback is {} playback) _player.SetPlayback(playback);
            CharacterSurface.Pack = _pack; CharacterSurface.Player = _player;
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
}
