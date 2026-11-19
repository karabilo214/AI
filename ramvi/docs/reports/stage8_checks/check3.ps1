# 8.1: релиз 0.8.0 в чистой Windows Sandbox (сети нет): оболочка, проверка компьютера, модель, ответ.
$ErrorActionPreference = 'Continue'
$out = 'C:\out\run3'
New-Item -ItemType Directory -Force $out | Out-Null
function Log($t) { "$(Get-Date -Format s) $t" | Out-File -Append "$out\log.txt" }
Log 'copy bundle'
robocopy C:\src\RAMVI C:\RAMVI /E /NFL /NDL /NJH /NJS /NP | Out-Null
Log "copied: $((Get-ChildItem C:\RAMVI -Recurse -File | Measure-Object Length -Sum).Sum)"
# Библиотеки Visual C++ в системе песочницы есть? (комплект должен работать и без них)
Test-Path C:\Windows\System32\msvcp140.dll | Out-File "$out\system_vcruntime.txt"
$py = 'C:\RAMVI\runtime\python\python.exe'
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" *> "$out\python.txt"
# 1. Оболочка: запуск RAMVI.exe и снимок экрана
Log 'start shell'
$shell = Start-Process C:\RAMVI\RAMVI.exe -PassThru
Start-Sleep 45
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
[System.Drawing.Graphics]::FromImage($bmp).CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save("$out\shell.png")
"shell alive: $(-not $shell.HasExited)" | Out-File "$out\shell.txt"
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, Path | Out-File -Append "$out\shell.txt"
Stop-Process -Id $shell.Id -Force -ErrorAction SilentlyContinue
Start-Sleep 5
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
# 2. Сервис комплекта: проверка компьютера, подключение модели, ответ
Log 'service'
$env:RAMVI_UI_TOKEN = 'sandbox-ui-token-0123456789abcdef'
$env:RAMVI_TOKENIZER = 'C:\RAMVI\runtime\tokenizers\Qwen3-8B'
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'; $env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Force C:\app | Out-Null
'{"data_dir": "C:\\data"}' | Out-File -Encoding ascii C:\app\settings.json
$info = New-Object System.Diagnostics.ProcessStartInfo
$info.FileName = $py
$info.Arguments = '-P -m ramvi_service.app --data-dir C:\data --app-dir C:\app'
$info.WorkingDirectory = 'C:\RAMVI\ramvi'
$info.RedirectStandardOutput = $true; $info.RedirectStandardInput = $true; $info.RedirectStandardError = $true; $info.UseShellExecute = $false
$service = [System.Diagnostics.Process]::Start($info)
$errTask = $service.StandardError.ReadToEndAsync()
$ready = $service.StandardOutput.ReadLine()
$ready | Out-File "$out\service_ready.txt"
$h = @{ Authorization = "Bearer $env:RAMVI_UI_TOKEN" }
try {
  $port = ($ready | ConvertFrom-Json).port
  $base = "http://127.0.0.1:$port/ui/v1"
  (Invoke-RestMethod "$base/system" -Headers $h -TimeoutSec 300 | ConvertTo-Json -Depth 8) | Out-File "$out\system.json"
  $blob = (Get-ChildItem C:\models -File | Sort-Object Length -Descending | Select-Object -First 1).FullName
  Log "model $blob"
  $model = Invoke-RestMethod "$base/models/import" -Method Post -Headers $h -ContentType 'application/json' -Body (@{ path = $blob } | ConvertTo-Json) -TimeoutSec 300
  ($model | ConvertTo-Json -Depth 6) | Out-File "$out\model.json"
  $chat = Invoke-RestMethod "$base/chats" -Method Post -Headers $h -ContentType 'application/json' -Body (@{ model_id = $model.model.id } | ConvertTo-Json) -TimeoutSec 60
  Log 'send'
  $t0 = Get-Date
  $raw = Invoke-WebRequest "$base/chats/$($chat.chat.id)/send" -Method Post -Headers $h -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes('{"text": "What is 17*23? Answer with the number only."}')) -TimeoutSec 900 -UseBasicParsing
  $text = [Text.Encoding]::UTF8.GetString($raw.RawContentStream.ToArray())
  $text | Out-File -Encoding utf8 "$out\answer_events.txt"
  "seconds: $([int]((Get-Date) - $t0).TotalSeconds)" | Out-File "$out\answer_time.txt"
} catch { $_.ToString() | Out-File "$out\service_error.txt" }
$service.StandardInput.Close(); Start-Sleep 8; if (-not $service.HasExited) { $service.Kill() }
$errTask.Result | Out-File "$out\service_stderr.txt"
Log 'done'
Get-Date -Format s | Out-File "$out\done.txt"
