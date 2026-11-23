using System.Runtime.InteropServices;
using Avalonia.Controls;

namespace Ramvi.Desktop.Services;

/// <summary>
/// 3.13: возможности ОС для окна персонажа вне RAMVI. Пропуск кликов (click-through) есть на
/// Windows (WS_EX_TRANSPARENT, ADR-0003) и macOS (NSWindow.ignoresMouseEvents); на других
/// системах это ограничение адаптера, о котором честно сообщает интерфейс.
/// </summary>
public static partial class WindowPlatform
{
    public static bool SupportsClickThrough => OperatingSystem.IsWindows() || OperatingSystem.IsMacOS();

    /// <summary>Включить или выключить пропуск кликов; false — ОС не дала это сделать.</summary>
    public static bool SetClickThrough(Window window, bool enabled)
    {
        var handle = window.TryGetPlatformHandle();
        if (handle is null || handle.Handle == IntPtr.Zero) return false;
        try
        {
            if (OperatingSystem.IsWindows()) return SetWindows(handle.Handle, enabled);
            if (OperatingSystem.IsMacOS()) return SetMac(handle.Handle, handle.HandleDescriptor, enabled);
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException)
        {
            return false;
        }
        return false;
    }

    private const int GwlExStyle = -20;
    private const long WsExLayered = 0x80000, WsExTransparent = 0x20;

    private static bool SetWindows(IntPtr hwnd, bool enabled)
    {
        var style = GetWindowLongPtrW(hwnd, GwlExStyle).ToInt64();
        style = enabled ? style | WsExLayered | WsExTransparent : style & ~WsExTransparent;
        SetWindowLongPtrW(hwnd, GwlExStyle, new IntPtr(style));
        return true;
    }

    private static bool SetMac(IntPtr handle, string? descriptor, bool enabled)
    {
        var window = handle;
        if (descriptor == "NSView") window = Send(handle, Selector("window"));
        if (window == IntPtr.Zero) return false;
        SendBool(window, Selector("setIgnoresMouseEvents:"), enabled);
        return true;
    }

    private static IntPtr Selector(string name) => sel_registerName(name);

    [LibraryImport("user32.dll", EntryPoint = "GetWindowLongPtrW")]
    private static partial IntPtr GetWindowLongPtrW(IntPtr hwnd, int index);

    [LibraryImport("user32.dll", EntryPoint = "SetWindowLongPtrW")]
    private static partial IntPtr SetWindowLongPtrW(IntPtr hwnd, int index, IntPtr value);

    [LibraryImport("/usr/lib/libobjc.A.dylib", StringMarshalling = StringMarshalling.Utf8)]
    private static partial IntPtr sel_registerName(string name);

    [LibraryImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    private static partial IntPtr Send(IntPtr receiver, IntPtr selector);

    [LibraryImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    private static partial void SendBool(IntPtr receiver, IntPtr selector, [MarshalAs(UnmanagedType.I1)] bool value);
}
