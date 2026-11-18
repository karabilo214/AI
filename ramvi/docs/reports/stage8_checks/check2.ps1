# Повторная проверка: библиотеки Visual C++ рядом с python.exe, исправленная диагностика.
$out = 'C:\out\run2'
New-Item -ItemType Directory -Force $out | Out-Null
$py = 'C:\src\RAMVI\runtime\python\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); import numba, triton; print('numba', numba.__version__, 'triton', triton.__version__)" *> "$out\python.txt"
$LASTEXITCODE | Out-File "$out\python_exit.txt"
(Get-Content 'C:\in\driver_check.py') -replace 'C:\\RAMVI', 'C:\src\RAMVI' | Set-Content 'C:\driver_check2.py'
& $py 'C:\driver_check2.py' *> "$out\diagnostics.txt"
$env:RAMVI_UI_TOKEN = 'sandbox-ui-token-0123456789abcdef'
$info = New-Object System.Diagnostics.ProcessStartInfo
$info.FileName = $py
$info.Arguments = '-m ramvi_service.app --data-dir C:\data_sandbox'
$info.RedirectStandardOutput = $true; $info.RedirectStandardInput = $true; $info.UseShellExecute = $false
$service = [System.Diagnostics.Process]::Start($info)
$ready = $service.StandardOutput.ReadLine()
$ready | Out-File "$out\service_ready.txt"
try {
  $port = ($ready | ConvertFrom-Json).port
  $system = Invoke-RestMethod "http://127.0.0.1:$port/ui/v1/system" -Headers @{ Authorization = "Bearer $env:RAMVI_UI_TOKEN" } -TimeoutSec 300
  ($system | ConvertTo-Json -Depth 8) | Out-File "$out\service_system.json"
} catch { $_.ToString() | Out-File "$out\service_error.txt" }
$service.StandardInput.Close(); Start-Sleep 5; if (-not $service.HasExited) { $service.Kill() }
Get-Date -Format s | Out-File "$out\done.txt"
