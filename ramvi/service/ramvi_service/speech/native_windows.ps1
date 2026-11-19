$ErrorActionPreference='Stop'
[Console]::InputEncoding=New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Speech
$request=[Console]::In.ReadToEnd()|ConvertFrom-Json
$speaker=New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
 if($request.operation -eq 'voices') {
  $voices=@($speaker.GetInstalledVoices()|Where-Object Enabled|ForEach-Object {
   @{id=$_.VoiceInfo.Name;name=$_.VoiceInfo.Name;languages=@($_.VoiceInfo.Culture.TwoLetterISOLanguageName)}
  })
  @{voices=$voices}|ConvertTo-Json -Depth 5 -Compress
 } elseif($request.operation -eq 'synthesize') {
  if($request.voice){$speaker.SelectVoice([string]$request.voice)}
  $format=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono)
  $speaker.SetOutputToWaveFile([string]$request.path,$format)
  $speaker.Speak([string]$request.text)
 } else {throw 'unknown native speech operation'}
} finally {$speaker.Dispose()}
