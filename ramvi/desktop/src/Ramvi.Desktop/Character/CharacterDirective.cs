namespace Ramvi.Desktop.Character;

/// <summary>Validated segment or cancellation fence. Epoch is owned by this view session.</summary>
public sealed record CharacterDirective(long Epoch, string? ResponseId, string? SegmentId, string Emotion, string Animation, string? Eyes = null, string? Mouth = null, IReadOnlyList<CharacterMove>? Moves = null);
