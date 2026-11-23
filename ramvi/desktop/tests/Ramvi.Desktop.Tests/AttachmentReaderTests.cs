using System.Text;
using Ramvi.Desktop.Api;
using Ramvi.Desktop.Services;
using Xunit;

namespace Ramvi.Desktop.Tests;

public sealed class AttachmentReaderTests
{
    private static readonly Limits Small = new(1000, 2, 100, 150, 255);

    internal static DroppedFile File(string name, string text) => Bytes(name, Encoding.UTF8.GetBytes(text));

    private static DroppedFile Bytes(string name, byte[] bytes, long? size = null) =>
        new(name, size ?? bytes.Length, () => Task.FromResult<Stream>(new MemoryStream(bytes)));

    private static Task<AttachmentResult> Read(params DroppedFile[] files) =>
        AttachmentReader.ReadAsync([], files, Small);

    [Fact]
    public async Task Utf8_text_is_accepted_and_bom_removed()
    {
        var result = await Read(File("a.txt", "привет"), Bytes("b.txt", [0xEF, 0xBB, 0xBF, (byte)'o', (byte)'k']));

        Assert.Null(result.ErrorKey);
        Assert.Equal(["привет", "ok"], result.Accepted.Select(a => a.Text));
        Assert.Equal([12, 2], result.Accepted.Select(a => a.Bytes));
    }

    [Fact]
    public async Task Large_file_is_rejected_before_reading()
    {
        var opened = false;
        var big = new DroppedFile("big.log", 101, () =>
        {
            opened = true;
            return Task.FromResult<Stream>(new MemoryStream());
        });

        var result = await Read(big);

        Assert.Equal("chat.file_too_large", result.ErrorKey);
        Assert.Equal(100, result.ErrorParams["limit"]);
        Assert.False(opened);
    }

    [Fact]
    public async Task File_larger_than_reported_size_is_rejected()
    {
        var result = await Read(Bytes("lies.txt", new byte[500], size: 0));

        Assert.Equal("chat.file_too_large", result.ErrorKey);
    }

    [Theory]
    [InlineData(new byte[] { 0xFF, 0xFE, 0x41 })]
    [InlineData(new byte[] { 0x41, 0x00, 0x42 })]
    public async Task Binary_is_rejected(byte[] bytes)
    {
        var result = await Read(Bytes("x.bin", bytes));

        Assert.Equal("chat.attachment_not_text", result.ErrorKey);
        Assert.Empty(result.Accepted);
    }

    [Fact]
    public async Task Count_and_total_limits_count_already_attached()
    {
        var current = new[] { new Attachment("old.txt", new string('a', 90), 90) };

        var total = await AttachmentReader.ReadAsync(current, [File("b.txt", new string('b', 70))], Small);
        Assert.Equal("chat.files_too_large", total.ErrorKey);
        Assert.Equal(150, total.ErrorParams["limit"]);

        var count = await AttachmentReader.ReadAsync(current, [File("b.txt", "b"), File("c.txt", "c")], Small);
        Assert.Equal("chat.too_many_files", count.ErrorKey);
        Assert.Equal(["b.txt"], count.Accepted.Select(a => a.Name));
    }

    [Fact]
    public async Task Unreadable_file_is_reported_by_name()
    {
        var locked = new DroppedFile("locked.txt", 3, () => throw new IOException("занят"));

        var result = await Read(locked);

        Assert.Equal("ui.file_unreadable", result.ErrorKey);
        Assert.Equal("locked.txt", result.ErrorParams["name"]);
    }

    [Fact]
    public async Task Zero_limits_mean_no_limit()
    {
        var unlimited = new Limits(1000, 0, 0, 0, 255);
        var files = Enumerable.Range(0, 7).Select(i => File($"{i}.txt", new string('x', 500))).ToArray();
        var result = await AttachmentReader.ReadAsync([], files, unlimited);
        Assert.Null(result.ErrorKey);
        Assert.Equal(7, result.Accepted.Count); // больше Small: 2 файла, 100 и 150 байт
    }
}
