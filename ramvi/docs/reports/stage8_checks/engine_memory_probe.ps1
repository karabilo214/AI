$ErrorActionPreference = 'Stop'
$b = "$env:USERPROFILE\Desktop\RAMVI"
$m = Get-Content "$env:USERPROFILE\.ollama\models\manifests\registry.ollama.ai\library\qwen3\8b" -Raw | ConvertFrom-Json
$blob = "$env:USERPROFILE\.ollama\models\blobs\" + (($m.layers | Where-Object { $_.mediaType -like '*.model' }).digest -replace ':','-')
$env:RAMVI_ENGINE_PORT = '11499'; $env:RAMVI_API_KEY = 'k' * 32; $env:RAMVI_INSTANCE_ID = 'memprobe'
$env:PYTHONUTF8 = '1'; $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
$log = 'C:\anm\offline_exam\engine_stderr.log'
$p = Start-Process -FilePath "$b\runtime\python\python.exe" -ArgumentList @('-P','-m','ramvi_engine','serve','--model',$blob,'--port','11499','--maxt','2048','--tokenizer',"$b\runtime\tokenizers\Qwen3-8B") -RedirectStandardError $log -RedirectStandardOutput 'C:\anm\offline_exam\engine_stdout.log' -PassThru -WindowStyle Hidden
$deadline = (Get-Date).AddMinutes(8)
while ((Get-Date) -lt $deadline) {
  Start-Sleep 3
  try { Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:11499/health" -TimeoutSec 2 | Out-Null; break } catch {}
  if ($p.HasExited) { break }
}
Start-Sleep 5
$g = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
function Probe($label) {
  $g = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
  $w = Get-CimInstance Win32_PerfRawData_PerfProc_Process | Where-Object { $_.IDProcess -eq $p.Id }
  $mm = Get-CimInstance Win32_PerfRawData_PerfOS_Memory
  '{0}: ws={1} ws_private={2} ws_shared={3} commit_private={4} | available={5}' -f $label, [int]($g.WorkingSet64/1MB), [int]($w.WorkingSetPrivate/1MB), [int](($g.WorkingSet64-$w.WorkingSetPrivate)/1MB), [int]($g.PrivateMemorySize64/1MB), $mm.AvailableMBytes
}
Probe 'loaded'
# trim working set
Add-Type -Namespace W -Name K -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(System.IntPtr h);'
[W.K]::EmptyWorkingSet((Get-Process -Id $p.Id).Handle) | Out-Null
Start-Sleep 2
Probe 'after_trim'
$body = '{"model":"x","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":40,"temperature":0}'
try { Invoke-WebRequest -UseBasicParsing -Method Post -Uri http://127.0.0.1:11499/v1/chat/completions -Headers @{Authorization = "Bearer $env:RAMVI_API_KEY"} -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 120 | Select-Object -ExpandProperty StatusCode } catch { "answer failed: $_" }
Probe 'after_answer'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$b*" } | Stop-Process -Force
Get-Content $log -Encoding UTF8 | Select-Object -Last 25
