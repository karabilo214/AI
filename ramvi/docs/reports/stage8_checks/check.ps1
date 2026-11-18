# Проверка переносимого комплекта RAMVI на чистой Windows (Windows Sandbox).
$ErrorActionPreference = 'Continue'
$out = 'C:\out'
Start-Transcript -Path "$out\transcript.txt" -Force | Out-Null
function Save($name, $value) { $value | Out-File -Encoding utf8 "$out\$name" }

# 1. Чистая ли система: нет VC++ и Python.
$clean = [ordered]@{
  os = (Get-CimInstance Win32_OperatingSystem).Caption + ' ' + [Environment]::OSVersion.Version
  vcruntime140_system32 = Test-Path 'C:\Windows\System32\vcruntime140.dll'
  msvcp140_system32 = Test-Path 'C:\Windows\System32\msvcp140.dll'
  vcruntime140_1_system32 = Test-Path 'C:\Windows\System32\vcruntime140_1.dll'
  python_in_path = [bool](Get-Command python -ErrorAction SilentlyContinue | Where-Object { $_.Source -notlike '*WindowsApps*' })
  nvidia_smi = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
  gpus = (Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' / ' + $_.DriverVersion }) -join '; '
}
Save 'clean.json' ($clean | ConvertTo-Json)

# 2. Распаковать комплект (копия на диск песочницы).
$t = Measure-Command { robocopy 'C:\src\RAMVI' 'C:\RAMVI' /E /NFL /NDL /NJH /NJS /NP | Out-Null }
Save 'copy.txt' ("{0:N1} s" -f $t.TotalSeconds)
$py = 'C:\RAMVI\runtime\python\python.exe'

# 3. Встроенный Python и torch без VC++ системы.
& $py -c "import sys, platform; print(sys.version); import numpy; print('numpy', numpy.__version__); import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" *> "$out\python.txt"
Save 'python_exit.txt' $LASTEXITCODE

# 4. Диагностика свежим кодом (проверка драйвера 8.1).
& $py 'C:\in\driver_check.py' *> "$out\diagnostics.txt"
Save 'diagnostics_exit.txt' $LASTEXITCODE

# 5. Сервис комплекта: старт и ответ интерфейсному API.
$env:RAMVI_UI_TOKEN = 'sandbox-ui-token-0123456789abcdef'
$info = New-Object System.Diagnostics.ProcessStartInfo
$info.FileName = $py
$info.Arguments = '-m ramvi_service.app --data-dir C:\RAMVI\data_sandbox --app-dir C:\RAMVI'
$info.RedirectStandardOutput = $true; $info.RedirectStandardInput = $true; $info.UseShellExecute = $false
$service = [System.Diagnostics.Process]::Start($info)
$ready = $service.StandardOutput.ReadLine()
Save 'service_ready.txt' $ready
try {
  $port = ($ready | ConvertFrom-Json).port
  $headers = @{ Authorization = "Bearer $env:RAMVI_UI_TOKEN" }
  $status = Invoke-RestMethod "http://127.0.0.1:$port/ui/v1/status" -Headers $headers
  Save 'service_status.json' ($status | ConvertTo-Json -Depth 6)
  $system = Invoke-RestMethod "http://127.0.0.1:$port/ui/v1/system" -Headers $headers -TimeoutSec 300
  Save 'service_system.json' ($system | ConvertTo-Json -Depth 8)
} catch { Save 'service_error.txt' $_.ToString() }
$service.StandardInput.Close()
Start-Sleep 5
if (-not $service.HasExited) { $service.Kill() }

# 6. Само приложение: запуск, снимок окна через 45 с.
$app = Start-Process 'C:\RAMVI\RAMVI.exe' -PassThru
Start-Sleep 45
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$bitmap.Save("$out\screen.png", [System.Drawing.Imaging.ImageFormat]::Png)
$appState = [ordered]@{ alive = -not $app.HasExited; exit = $(if ($app.HasExited) { $app.ExitCode } else { $null }) }
$appState.processes = (Get-Process | Where-Object { $_.ProcessName -match 'RAMVI|python' } | ForEach-Object { $_.ProcessName }) -join ', '
Save 'app.json' ($appState | ConvertTo-Json)
Get-Process | Where-Object { $_.ProcessName -match 'RAMVI|python' } | Stop-Process -Force -ErrorAction SilentlyContinue
Stop-Transcript | Out-Null
Save 'done.txt' (Get-Date -Format s)
