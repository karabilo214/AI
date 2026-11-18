param([string]$Root)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName WindowsBase
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class FixtureFocus {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
}
'@
$form=New-Object Windows.Window
$form.Title='RAMVI isolated agent test'
$form.Width=480; $form.Height=240; $form.WindowStartupLocation='CenterScreen'
$panel=New-Object Windows.Controls.StackPanel
$panel.Margin=New-Object Windows.Thickness(20)
$field=New-Object Windows.Controls.TextBox
$field.Text='reference initial'
$field.Height=28
[Windows.Automation.AutomationProperties]::SetName($field,'Agent input field')
$panel.Children.Add($field) | Out-Null
$button=New-Object Windows.Controls.Button
$button.Content='Apply test'; $button.Height=30; $button.Margin=New-Object Windows.Thickness(0,20,0,0)
[Windows.Automation.AutomationProperties]::SetName($button,'Agent apply button')
$panel.Children.Add($button) | Out-Null
$form.Content=$panel
$script:clicks=0
$button.Add_Click({ $script:clicks++ })
$other=New-Object Windows.Window
$other.Title='RAMVI other test window'; $other.Width=240; $other.Height=150
$timer=New-Object Windows.Threading.DispatcherTimer
$timer.Interval=[timespan]::FromMilliseconds(100)
$timer.Add_Tick({
    @{text=$field.Text; clicks=$script:clicks} | ConvertTo-Json -Compress | Set-Content "$Root\fixture_state.json" -Encoding UTF8
    if (Test-Path "$Root\fixture_mutate.txt") {
        $field.Text=Get-Content "$Root\fixture_mutate.txt" -Raw
        Remove-Item "$Root\fixture_mutate.txt" -Force
    }
    if (Test-Path "$Root\focus.txt") {
        $target=Get-Content "$Root\focus.txt" -Raw
        Remove-Item "$Root\focus.txt" -Force
        if ($target.Trim() -eq 'other') { $other.Show(); $other.Activate(); $selected=$other }
        else { $form.Activate(); $selected=$form }
        $helper=New-Object Windows.Interop.WindowInteropHelper($selected)
        [void][FixtureFocus]::SetForegroundWindow($helper.Handle)
    }
    if (Test-Path "$Root\close.txt") { $other.Close(); $form.Close() }
})
$form.Add_ContentRendered({
    $process=Get-Process -Id $PID
    $helper=New-Object Windows.Interop.WindowInteropHelper($form)
    @{application="$($PID):$($process.StartTime.ToUniversalTime().Ticks)"; window_id="$($helper.Handle.ToInt64())"} | ConvertTo-Json -Compress | Set-Content "$Root\fixture_ready.json" -Encoding UTF8
    $field.Focus(); $field.SelectAll(); $form.Activate(); [void][FixtureFocus]::SetForegroundWindow($helper.Handle)
    $timer.Start()
})
try { $form.ShowDialog() | Out-Null }
finally { $timer.Stop(); $other.Close(); $form.Close() }
