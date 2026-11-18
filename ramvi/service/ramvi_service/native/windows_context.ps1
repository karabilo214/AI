param([Parameter(Mandatory=$true)][string]$RequestFile)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class RamviWindow {
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out Rect rect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
    public struct Rect { public int Left, Top, Right, Bottom; }
}
'@
function Target([IntPtr]$Handle) {
    if ($Handle -eq [IntPtr]::Zero -or -not [RamviWindow]::IsWindow($Handle)) { throw 'window unavailable' }
    [uint32]$targetPid = 0
    [void][RamviWindow]::GetWindowThreadProcessId($Handle, [ref]$targetPid)
    $process = Get-Process -Id $targetPid
    return @{ application = "$($targetPid):$($process.StartTime.ToUniversalTime().Ticks)"; window_id = "$($Handle.ToInt64())" }
}
function CheckTarget($Request) {
    $handle = [IntPtr]([long]$Request.window_id)
    $target = Target $handle
    if ($target.application -cne $Request.application) { throw 'application identity changed' }
    if ([RamviWindow]::GetForegroundWindow() -ne $handle) { throw 'window is not foreground' }
    return $handle
}
function ElementId($Element) { return (($Element.GetRuntimeId() | ForEach-Object { "$_" }) -join '.') }
function Nodes($Root) {
    $queue = New-Object System.Collections.Queue
    $queue.Enqueue(@{ element=$Root; depth=0 })
    $result = New-Object System.Collections.ArrayList
    $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
    while ($queue.Count -gt 0 -and $result.Count -lt 128) {
        $entry = $queue.Dequeue()
        $element = $entry.element
        $current = $element.Current
        # Password fields and descendants are omitted entirely.
        if ($current.IsPassword) { continue }
        [void]$result.Add(@{ element=$element; id=(ElementId $element); depth=$entry.depth })
        if ($entry.depth -ge 5) { continue }
        $child = $walker.GetFirstChild($element)
        $count = 0
        while ($null -ne $child -and $count -lt 128 -and $queue.Count -lt 128) {
            $queue.Enqueue(@{ element=$child; depth=($entry.depth+1) })
            $child = $walker.GetNextSibling($child)
            $count++
        }
    }
    return ,$result
}
try {
    $request = Get-Content -LiteralPath $RequestFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($request.operation -in @('target', 'target_title')) {
        $result = Target ([RamviWindow]::GetForegroundWindow())
        if ($request.operation -eq 'target_title') {
            $result.title = [System.Windows.Automation.AutomationElement]::FromHandle([RamviWindow]::GetForegroundWindow()).Current.Name
            $result.process_name = (Get-Process -Id ([int]($result.application.Split(':')[0]))).ProcessName
        }
    } else {
        $handle = CheckTarget $request
        $root = [System.Windows.Automation.AutomationElement]::FromHandle($handle)
        if ($request.kind -eq 'screen') {
            $rect = New-Object RamviWindow+Rect
            if (-not [RamviWindow]::GetWindowRect($handle, [ref]$rect)) { throw 'window bounds unavailable' }
            $width = $rect.Right-$rect.Left; $height = $rect.Bottom-$rect.Top
            if ($width -lt 1 -or $height -lt 1 -or $width*$height -gt 16000000) { throw 'window size exceeds budget' }
            $bitmap = New-Object System.Drawing.Bitmap($width, $height)
            try {
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                try {
                    $dc = $graphics.GetHdc()
                    try { $ok = [RamviWindow]::PrintWindow($handle, $dc, 2) }
                    finally { $graphics.ReleaseHdc($dc) }
                } finally { $graphics.Dispose() }
                if (-not $ok) { throw 'window capture unavailable' }
                # Window-only capture; never fall back to a whole-desktop screenshot.
                $scale = [Math]::Min(1.0, 1024.0/[Math]::Max($width,$height))
                $thumb = New-Object System.Drawing.Bitmap($bitmap, [int]($width*$scale), [int]($height*$scale))
                try {
                    $stream = New-Object System.IO.MemoryStream
                    try {
                        $thumb.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
                        $result = @{ image=('data:image/png;base64,'+[Convert]::ToBase64String($stream.ToArray())) }
                    } finally { $stream.Dispose() }
                } finally { $thumb.Dispose() }
            } finally { $bitmap.Dispose() }
        } else {
            $items = New-Object System.Collections.ArrayList
            $selected = ''
            $nodes = Nodes $root
            $signature = New-Object System.Text.StringBuilder
            foreach ($node in $nodes) {
                $element = $node.element; $current = $element.Current
                $name = [string]$current.Name
                if ($name.Length -gt 512) { $name = $name.Substring(0,512) }
                [void]$items.Add(@{ id=$node.id; name=$name; role=$current.ControlType.ProgrammaticName; enabled=$current.IsEnabled; class_name=$current.ClassName; patterns=@($element.GetSupportedPatterns() | ForEach-Object { $_.ProgrammaticName }) })
                foreach ($part in @($node.id, $current.ControlType.ProgrammaticName, $name, "$($current.IsEnabled)")) {
                    [void]$signature.Append("$($part.Length):$part")
                }
                $valuePattern = $null
                if ($element.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$valuePattern)) {
                    $valueText = [string]$valuePattern.Current.Value
                    if ($valueText.Length -gt 8192) { throw 'UI value exceeds budget' }
                    [void]$signature.Append("$($valueText.Length):$valueText")
                }
                if ($current.HasKeyboardFocus) {
                    $pattern = $null
                    if ($element.TryGetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern, [ref]$pattern)) {
                        $selected = (($pattern.GetSelection() | ForEach-Object { $_.GetText(8192) }) -join "`n")
                        if ($selected.Length -gt 8192) { $selected = $selected.Substring(0,8192) }
                    }
                }
            }
            $sha = [System.Security.Cryptography.SHA256]::Create()
            try { $fingerprint = [Convert]::ToBase64String($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($signature.ToString()))) }
            finally { $sha.Dispose() }
            if ($request.operation -eq 'act') {
                if ($fingerprint -cne $request.fingerprint) { throw 'UI state changed' }
                $matches = @($nodes | Where-Object { $_.id -ceq $request.element_id })
                if ($matches.Count -ne 1) { throw 'element is missing or ambiguous' }
                $element = $matches[0].element
                if ($element.Current.IsPassword -or -not $element.Current.IsEnabled) { throw 'element is unavailable' }
                [void](CheckTarget $request)
                $pattern = $null
                if ($request.action -eq 'invoke') {
                    if (-not $element.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$pattern)) { throw 'invoke unavailable' }
                    [void](CheckTarget $request)
                    $pattern.Invoke()
                } elseif ($request.action -eq 'set_value') {
                    if (-not $element.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$pattern)) { throw 'value unavailable' }
                    if ($pattern.Current.IsReadOnly) { throw 'element is read only' }
                    [void](CheckTarget $request)
                    $pattern.SetValue([string]$request.value)
                } else { throw 'unknown action' }
                $result = @{ applied=$true }
            } else {
                $result = @{ text=$selected; structure=@{ elements=@($items); bounded=$true; fingerprint=$fingerprint } }
            }
        }
        if ($request.operation -ne 'act') { [void](CheckTarget $request) }
    }
    @{ result=$result } | ConvertTo-Json -Depth 12 -Compress
} catch {
    # No native error text: it can contain selected text, app titles or private paths.
    @{ error='system_context_unavailable' } | ConvertTo-Json -Compress
    exit 1
}
