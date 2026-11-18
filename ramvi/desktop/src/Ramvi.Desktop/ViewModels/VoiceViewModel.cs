using System.Globalization;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Localization;

namespace Ramvi.Desktop.ViewModels;

/// <summary>
/// 5.2 (VOI-01): кнопка микрофона у поля ввода. Первое нажатие начинает запись, второе —
/// останавливает и распознаёт; текст добавляется в поле ввода, в чат сам не уходит.
/// Пока идёт запись, окно спрашивает сервис о длительности и громкости; на пределе длины
/// запись останавливается сама.
/// </summary>
public sealed class VoiceViewModel : ObservableObject
{
    private readonly Func<IChatApi> _api;
    private readonly Action<string> _insert;
    private readonly Action<Exception> _showError;
    private readonly Func<bool> _canUse;
    private readonly TimeSpan _poll;
    private string _state = "idle";
    private double _seconds;
    private double _level;
    private int _watch;

    /// <param name="api">Клиент сервиса.</param>
    /// <param name="insert">Добавить распознанный текст в поле ввода.</param>
    /// <param name="showError">Показать ошибку в строке состояния.</param>
    /// <param name="canUse">Можно ли сейчас начать запись (сервис подключён).</param>
    /// <param name="poll">Как часто спрашивать сервис во время записи.</param>
    public VoiceViewModel(
        Func<IChatApi> api, Action<string> insert, Action<Exception> showError, Func<bool> canUse, TimeSpan? poll = null)
    {
        _api = api;
        _insert = insert;
        _showError = showError;
        _canUse = canUse;
        _poll = poll ?? TimeSpan.FromMilliseconds(200);
        ToggleCommand = new Command(ToggleAsync, () => _canUse() && !IsTranscribing);
        CancelCommand = new Command(CancelAsync, () => IsRecording);
    }

    private static Localizer L => Localizer.Current;

    /// <summary>Начать запись или остановить её и распознать.</summary>
    public Command ToggleCommand { get; }

    /// <summary>Прекратить запись без распознавания; звук забывается.</summary>
    public Command CancelCommand { get; }

    public bool IsRecording => _state == "recording";

    public bool IsTranscribing => _state == "transcribing";

    public bool IsIdle => _state == "idle";

    /// <summary>Длительность записи «0:07».</summary>
    public string Elapsed
    {
        get
        {
            var total = (int)Math.Floor(_seconds);
            return string.Create(CultureInfo.InvariantCulture, $"{total / 60}:{total % 60:00}");
        }
    }

    /// <summary>Громкость 0…1 для индикатора записи.</summary>
    public double Level
    {
        get => _level;
        private set => Set(ref _level, value);
    }

    public string Tip => L[IsRecording ? "voice.stop_hint" : IsTranscribing ? "voice.transcribing" : "voice.start_hint"];

    private void SetState(string state)
    {
        if (_state == state)
        {
            return;
        }

        _state = state;
        Raise(nameof(IsRecording));
        Raise(nameof(IsTranscribing));
        Raise(nameof(IsIdle));
        Raise(nameof(Tip));
        Refresh();
    }

    /// <summary>Обновить доступность кнопки (сервис подключился или отключился).</summary>
    public void Refresh()
    {
        ToggleCommand.Refresh();
        CancelCommand.Refresh();
    }

    private void SetProgress(double seconds, double level)
    {
        if (Math.Abs(_seconds - seconds) > 0.001)
        {
            _seconds = seconds;
            Raise(nameof(Elapsed));
        }

        Level = level;
    }

    private Task ToggleAsync() => IsRecording ? StopAsync() : StartAsync();

    private async Task StartAsync()
    {
        try
        {
            var status = await _api().StartVoiceAsync().ConfigureAwait(true);
            SetProgress(status.Seconds, status.Level);
            SetState("recording");
            _ = WatchAsync(++_watch); // кнопка свободна для второго нажатия
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            _showError(error);
        }
    }

    private async Task WatchAsync(int watch)
    {
        try
        {
            while (true)
            {
                await Task.Delay(_poll).ConfigureAwait(true);
                if (watch != _watch || !IsRecording)
                {
                    return;
                }

                var status = await _api().VoiceStatusAsync().ConfigureAwait(true);
                if (watch != _watch || !IsRecording)
                {
                    return;
                }

                SetProgress(status.Seconds, status.Level);
                if (status.State != "recording")
                {
                    SetState("idle"); // запись прервана в сервисе
                    return;
                }

                if (status.LimitReached)
                {
                    await StopAsync().ConfigureAwait(true); // предел длины — распознать записанное
                    return;
                }
            }
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            if (watch == _watch)
            {
                SetState("idle");
                _showError(error);
            }
        }
    }

    private async Task StopAsync()
    {
        if (!IsRecording)
        {
            return;
        }

        _watch++;
        SetState("transcribing");
        try
        {
            var result = await _api().StopVoiceAsync().ConfigureAwait(true);
            _insert(result.Text);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            _showError(error);
        }
        finally
        {
            SetProgress(0, 0);
            SetState("idle");
        }
    }

    private async Task CancelAsync()
    {
        _watch++;
        try
        {
            await _api().CancelVoiceAsync().ConfigureAwait(true);
        }
        catch (Exception error) when (error is ApiException or HttpRequestException)
        {
            _showError(error);
        }
        finally
        {
            SetProgress(0, 0);
            SetState("idle");
        }
    }
}
