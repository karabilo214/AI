using System.Text;
using Ramvi.Desktop.Api;

namespace Ramvi.Desktop.Services;

/// <summary>Файл, перетащенный в окно: имя, размер и способ открыть.</summary>
public sealed record DroppedFile(string Name, long Size, Func<Task<Stream>> Open);

/// <summary>Результат: принятые вложения или ключ первой ошибки с параметрами.</summary>
public sealed record AttachmentResult(
    IReadOnlyList<Attachment> Accepted,
    string? ErrorKey,
    IReadOnlyDictionary<string, object?> ErrorParams);

/// <summary>
/// Проверка до отправки по тем же лимитам, что и в сервисе (`ramvi_service.chat.Limits`):
/// число файлов, размер файла и суммарный размер, только текст UTF-8 без нулевых байтов.
/// Большой файл не читается целиком: размер проверяется до чтения.
/// </summary>
public static class AttachmentReader
{
    private static readonly UTF8Encoding Strict = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    public static async Task<AttachmentResult> ReadAsync(
        IReadOnlyList<Attachment> current, IReadOnlyList<DroppedFile> files, Limits limits)
    {
        var accepted = new List<Attachment>();
        var total = current.Sum(a => a.Bytes);
        foreach (var file in files)
        {
            if (current.Count + accepted.Count >= limits.MaxFiles)
            {
                return Fail(accepted, "chat.too_many_files", ("limit", limits.MaxFiles));
            }

            if (file.Size > limits.MaxFileBytes)
            {
                return Fail(accepted, "chat.file_too_large", ("name", file.Name), ("limit", limits.MaxFileBytes));
            }

            byte[] bytes;
            try
            {
                using var stream = await file.Open().ConfigureAwait(true);
                bytes = await ReadLimitedAsync(stream, limits.MaxFileBytes).ConfigureAwait(true);
            }
            catch (IOException)
            {
                return Fail(accepted, "ui.file_unreadable", ("name", file.Name));
            }
            catch (UnauthorizedAccessException)
            {
                return Fail(accepted, "ui.file_unreadable", ("name", file.Name));
            }

            if (bytes.Length > limits.MaxFileBytes)
            {
                return Fail(accepted, "chat.file_too_large", ("name", file.Name), ("limit", limits.MaxFileBytes));
            }

            string text;
            try
            {
                var offset = bytes.AsSpan().StartsWith((byte[])[0xEF, 0xBB, 0xBF]) ? 3 : 0;
                text = Strict.GetString(bytes, offset, bytes.Length - offset);
            }
            catch (DecoderFallbackException)
            {
                return Fail(accepted, "chat.attachment_not_text", ("name", file.Name));
            }

            if (text.Contains('\0', StringComparison.Ordinal))
            {
                return Fail(accepted, "chat.attachment_not_text", ("name", file.Name));
            }

            var size = Encoding.UTF8.GetByteCount(text);
            if (total + size > limits.MaxTotalBytes)
            {
                return Fail(accepted, "chat.files_too_large", ("limit", limits.MaxTotalBytes));
            }

            total += size;
            accepted.Add(new Attachment(file.Name, text, size));
        }

        return new AttachmentResult(accepted, null, new Dictionary<string, object?>());
    }

    private static async Task<byte[]> ReadLimitedAsync(Stream stream, int limit)
    {
        using var buffer = new MemoryStream();
        var chunk = new byte[81920];
        int read;
        while ((read = await stream.ReadAsync(chunk).ConfigureAwait(true)) > 0)
        {
            buffer.Write(chunk, 0, read);
            if (buffer.Length > limit)
            {
                break; // размер мог быть неизвестен заранее
            }
        }

        return buffer.ToArray();
    }

    private static AttachmentResult Fail(
        List<Attachment> accepted, string key, params (string Name, object? Value)[] parameters) =>
        new(accepted, key, parameters.ToDictionary(p => p.Name, p => p.Value, StringComparer.Ordinal));
}
