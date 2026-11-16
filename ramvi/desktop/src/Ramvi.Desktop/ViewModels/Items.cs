using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

public sealed class ChatItemViewModel(ChatInfo chat) : ObservableObject
{
    private ChatInfo _chat = chat;

    public string Id => _chat.Id;

    public string? ModelId => _chat.ModelId;

    public string Title => string.IsNullOrWhiteSpace(_chat.Title) ? Localizer.Current["chat.untitled"] : _chat.Title;

    public void Update(ChatInfo chat)
    {
        _chat = chat;
        Raise(nameof(Title));
        Raise(nameof(ModelId));
    }
}

public sealed class AttachmentViewModel(Attachment attachment) : ObservableObject
{
    public Attachment Attachment { get; } = attachment;

    public string Label => Localizer.Current.Format(
        "chat.attachment_size",
        ("name", Attachment.Name),
        ("kb", Math.Max(1, (int)Math.Ceiling(Attachment.Bytes / 1024.0))));
}

public sealed class MessageViewModel : ObservableObject
{
    private string _content;
    private string _reasoning;
    private string? _finish;
    private string? _error;
    private bool _isThinkingOpen;
    private bool _isStreaming;
    private bool _isLastAnswer;

    public MessageViewModel(MessageInfo message)
    {
        Id = message.Id;
        Role = message.Role;
        _content = message.Content;
        _reasoning = message.Reasoning;
        _finish = message.Finish;
        Attachments = [.. (message.Attachments ?? []).Select(a => new AttachmentViewModel(a))];
    }

    public MessageViewModel(string role)
        : this(new MessageInfo("", "", role, "", 0))
    {
        _isStreaming = true;
    }

    public string Id { get; set; }

    public string Role { get; }

    public bool IsUser => Role == "user";

    public bool IsAssistant => Role == "assistant";

    public string Author => Localizer.Current[IsUser ? "chat.role_user" : "chat.role_assistant"];

    public IReadOnlyList<AttachmentViewModel> Attachments { get; }

    public bool HasAttachments => Attachments.Count > 0;

    public string Content
    {
        get => _content;
        set => Set(ref _content, value);
    }

    public string Reasoning
    {
        get => _reasoning;
        set
        {
            if (Set(ref _reasoning, value))
            {
                Raise(nameof(HasReasoning));
            }
        }
    }

    /// <summary>Thinking показывается отдельной сворачиваемой областью; по умолчанию свёрнута.</summary>
    public bool HasReasoning => !string.IsNullOrEmpty(_reasoning);

    public bool IsThinkingOpen
    {
        get => _isThinkingOpen;
        set => Set(ref _isThinkingOpen, value);
    }

    public bool IsStreaming
    {
        get => _isStreaming;
        set
        {
            if (Set(ref _isStreaming, value))
            {
                Raise(nameof(ShowActions));
                Raise(nameof(ShowRegenerate));
            }
        }
    }

    /// <summary>Последний ответ модели в диалоге: у него есть «Ответить заново».</summary>
    public bool IsLastAnswer
    {
        get => _isLastAnswer;
        set
        {
            if (Set(ref _isLastAnswer, value))
            {
                Raise(nameof(ShowRegenerate));
            }
        }
    }

    /// <summary>Кнопки под ответом (копировать) — когда ответ допечатан.</summary>
    public bool ShowActions => IsAssistant && !_isStreaming;

    public bool ShowRegenerate => ShowActions && _isLastAnswer;

    public string? Finish
    {
        get => _finish;
        set
        {
            if (Set(ref _finish, value))
            {
                Raise(nameof(Note));
                Raise(nameof(HasNote));
            }
        }
    }

    public string? Error
    {
        get => _error;
        set
        {
            if (Set(ref _error, value))
            {
                Raise(nameof(Note));
                Raise(nameof(HasNote));
            }
        }
    }

    /// <summary>Пометка под ответом: остановлен, обрезан или ошибка.</summary>
    public string Note => _error ?? _finish switch
    {
        "cancelled" => Localizer.Current["chat.stopped"],
        "length" => Localizer.Current["chat.truncated"],
        _ => "",
    };

    public bool HasNote => !string.IsNullOrEmpty(Note);
}
