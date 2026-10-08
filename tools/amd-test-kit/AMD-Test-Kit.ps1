# AMD device-lost test kit: runs Astro Bot several times in a row, each time with one emulator
# switch changed, and records how far each run gets. One session answers which switch avoids the
# "device lost" after the intro video, and the GPU breadcrumbs say what the GPU was running.
#
# Put this file (and AMD-Test-Kit.cmd) next to kyty_emulator.exe and start AMD-Test-Kit.cmd.
# Nothing in the emulator folder or the game folder is changed: every switch is set only in the
# environment of the emulator process the kit starts, the same way the launcher applies
# u59-preset.json. The kit writes only its results folder and zip.
#
# Optional parameters (AMD-Test-Kit.cmd passes them through):
#   -Only fix,barrier-old      run only these configurations (names from the lists below)
#   -SecondsAfterVideo 45      how long a run continues after the intro video closes
#   -MaxSeconds 300            hard limit per run
#   -GamePath <folder>         the game folder, if the kit cannot find it in Kyty.ini
#   -TitleId PPSA21564         the title id, if the kit cannot tell it from the game folder
#   -CaptureFromLauncher       instead of reading Kyty.ini, start the game once from the launcher;
#                              the kit takes the launcher's exact command line from that process
#   -ListOnly                  print the configurations and the command line, run nothing
#   -ExtraEnv 'KYTY_X=1,KYTY_Y=0'  extra switches for every run (if we ask for them)
param(
    [string]$EmulatorDir = $PSScriptRoot,
    [string[]]$Only = @(),
    [int]$SecondsAfterVideo = 45,
    [int]$MaxSeconds = 300,
    [string]$GamePath = '',
    [string]$TitleId = '',
    [switch]$CaptureFromLauncher,
    [switch]$ListOnly,
    [string]$OutDir = '',
    [switch]$NoZip,
    [string]$ExtraEnv = '',
    # Appended to the emulator command line (later options win), e.g. '--vulkan-validation true'.
    [string]$ExtraArgs = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

# ------------------------------------------------------------------------------------------------
# Configurations. Every run has the GPU breadcrumbs on. "fix" is this build as released (the
# bundled u59-preset.json and the new defaults); every other entry changes one thing from it. Order:
# the most informative first. The AMD test 1 runs named the hanging draw: Astro Bot's title-screen
# particle mesh shader (VS 0x20c94d46ce55a14b, merged program 0xc739f9614016bed4, PS
# 0xef31694ed8d87754). It has no loops; its LDS waitcnt barriers sit where only some waves go.
$Common = [ordered]@{
    KYTY_DEVICE_FAULT_DIAGNOSTICS = '1'
    KYTY_GPU_BREADCRUMBS          = '1'
}
$HangShaders = '0xc739f9614016bed4,0x20c94d46ce55a14b,0xef31694ed8d87754'
$Configs = @(
    @{ Name = 'fix';              Text = 'this build: LDS waitcnt barrier at subgroup scope, wave32 clusters'; Env = [ordered]@{} }
    @{ Name = 'barrier-old';      Text = 'LDS waitcnt barrier at workgroup scope again (as int16.1 and AMD test 1)'; Env = [ordered]@{ KYTY_LDS_WAITCNT_BARRIER = 'workgroup' } }
    @{ Name = 'barrier-off';      Text = 'no LDS waitcnt barrier (the community workaround; pink clouds expected)'; Env = [ordered]@{ KYTY_LDS_WAITCNT_BARRIER = '0' } }
    @{ Name = 'old-loop-guard';   Text = 'barrier-old with a loop guard on the hanging draw''s shaders'; Env = [ordered]@{
        KYTY_LDS_WAITCNT_BARRIER = 'workgroup'; KYTY_LOOP_GUARD = '200000'; KYTY_LOOP_GUARD_SHADERS = $HangShaders } }
    @{ Name = 'clusters-off';     Text = 'fix without wave32 clusters (wave32 programs share 64-wide subgroups)'; Env = [ordered]@{ KYTY_WAVE32_CLUSTERS = '0' } }
    @{ Name = 'fix-long';         Text = 'fix again for 2 minutes at the title, with the emulator log file'; Env = [ordered]@{}; LogFile = $true; Seconds = 120 }
)
# Earlier suspects, not run by default; -Only <name> runs them.
$MoreConfigs = @(
    @{ Name = 'fastfirst-off';    Text = 'pipelines built optimized at once (no unoptimized first build)'; Env = [ordered]@{ KYTY_PIPELINE_FAST_FIRST = '0' } }
    @{ Name = 'wave64-split';     Text = 'wave64 compute shaders on 32-wide subgroups, as on NVIDIA';  Env = [ordered]@{ KYTY_COMPUTE_WAVE64 = '0' } }
    @{ Name = 'loop-guard';       Text = 'every shader loop ends after 200000 iterations and is named'; Env = [ordered]@{
        KYTY_LOOP_GUARD = '200000'; KYTY_LOOP_GUARD_SHADERS = 'all' } }
    @{ Name = 'sparse-off';       Text = 'no sparse residency (textures and the BDA page table)';      Env = [ordered]@{
        KYTY_TEXTURE_SPARSE_RESIDENCY = '0'; KYTY_BDA_PAGETABLE_SPARSE = '0' } }
    @{ Name = 'queues-off';       Text = 'neither extra queue (side copies and upload DMA on queue 0)'; Env = [ordered]@{ KYTY_SIDE_QUEUE = '0'; KYTY_UPLOAD_DMA = '0' } }
    @{ Name = 'drawrun-off';      Text = 'no draw-run reuse (it starts at the title scene)';           Env = [ordered]@{ KYTY_DRAW_RUN = '0' } }
    @{ Name = 'helper-old';       Text = 'helper lanes run pixel-shader compare-exchange loops (int16.1)'; Env = [ordered]@{ KYTY_PS_HELPER_ATOMICS_SKIP = '0' } }
    @{ Name = 'no-breadcrumbs';   Text = 'fix without breadcrumbs';                                     Env = [ordered]@{ KYTY_GPU_BREADCRUMBS = '0' } }
)
# ------------------------------------------------------------------------------------------------
function Write-Step([string]$text) { Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $text) }

function Read-QtIni([string]$path) {
    # QSettings IniFormat: [Section] and key=value; arrays as "N\key"; strings may be quoted.
    $sections = @{}
    $section = 'General'
    $sections[$section] = @{}
    foreach ($line in [IO.File]::ReadAllLines($path)) {
        if ($line -match '^\s*[;#]') { continue }
        if ($line -match '^\s*\[(.+)\]\s*$') {
            $section = $matches[1]
            if (-not $sections.ContainsKey($section)) { $sections[$section] = @{} }
            continue
        }
        $eq = $line.IndexOf('=')
        if ($eq -le 0) { continue }
        $key = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim()
        if ($value.Length -ge 2 -and $value.StartsWith('"') -and $value.EndsWith('"')) {
            $value = $value.Substring(1, $value.Length - 2) -replace '\\"', '"' -replace '\\\\', '\'
        }
        $sections[$section][$key] = $value
    }
    return $sections
}

function Get-Value($table, [string]$key, [string]$default) {
    if ($table.ContainsKey($key) -and $table[$key] -ne '' -and $table[$key] -notlike '@Invalid*') { return [string]$table[$key] }
    return $default
}

function Format-Arguments([string[]]$list) {
    ($list | ForEach-Object {
        if ($_ -eq '' -or $_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' '
}

function Get-LauncherArguments([string]$dir) {
    # The launcher's CreateEmulatorArgs (src/launcher/src/mainDialog.cpp), from its Kyty.ini: next to
    # the launcher when that file exists, otherwise the shared settings file in ProgramData.
    $ini = Join-Path $dir 'Kyty.ini'
    if (-not [IO.File]::Exists($ini)) { $ini = Join-Path $env:ProgramData 'Kyty\Kyty.ini' }
    if (-not [IO.File]::Exists($ini)) { throw "No launcher settings found (Kyty.ini next to the emulator or in $env:ProgramData\Kyty). Start the game from the launcher once, or use -CaptureFromLauncher." }
    $s = Read-QtIni $ini
    $global = @{}
    if ($s.ContainsKey('GlobalConfiguration')) { $global = $s['GlobalConfiguration'] }
    $games = @{}
    if ($s.ContainsKey('GameConfigurations')) { $games = $s['GameConfigurations'] }
    $count = [int](Get-Value $games 'size' '0')
    $index = 0
    $candidates = [System.Collections.Generic.List[string]]::new()
    for ($i = 1; $i -le $count; $i++) {
        $base = Get-Value $games "$i\basedir" ''
        if ($GamePath -and $base -and ([IO.Path]::GetFullPath($base.Replace('/', '\')).TrimEnd('\') -ieq [IO.Path]::GetFullPath($GamePath).TrimEnd('\'))) { $index = $i }
        elseif (-not $GamePath -and $base -match 'PPSA2156[47]|ASTRO' -and
                ([IO.Directory]::Exists($base.Replace('/', '\')) -or [IO.File]::Exists($base.Replace('/', '\')))) {
            $candidates.Add($base)
            if ($index -eq 0) { $index = $i }
        }
    }
    if ($candidates.Count -gt 1) {
        Write-Host "The launcher has $($candidates.Count) Astro Bot entries; the kit uses the first one:"
        foreach ($c in $candidates) { Write-Host "    $c" }
        Write-Host '  To test another one, start the kit with -GamePath "<that folder>".'
    }
    if ($index -eq 0 -and $count -eq 1 -and -not $GamePath) { $index = 1 }
    $config = @{}
    foreach ($k in $global.Keys) { $config[$k] = $global[$k] }
    $basedir = $GamePath
    if ($index -gt 0) {
        $basedir = (Get-Value $games "$index\basedir" '').Replace('/', '\')
        if ((Get-Value $games "$index\custom_settings" 'false') -eq 'true') {
            foreach ($k in @($games.Keys)) {
                if ($k.StartsWith("$index\")) { $config[$k.Substring("$index\".Length)] = $games[$k] }
            }
            # Controller, audio mix and occlusion mode always come from the global settings.
            foreach ($k in 'controller_color', 'controller_speaker_volume', 'controller_vibration_intensity',
                     'audio_master_volume', 'audio_main_volume', 'audio_music_volume', 'audio_pad_speaker_main_volume',
                     'gpu_occlusion_accurate') {
                if ($global.ContainsKey($k)) { $config[$k] = $global[$k] } else { $config.Remove($k) }
            }
        }
    }
    if (-not $basedir) { throw "Kyty.ini ($ini) has no Astro Bot game configuration. Pass -GamePath <game folder>." }
    $resolution = (Get-Value $config 'screen_resolution' 'R1280X720') -replace '^R', ''
    $wh = $resolution -split '[xX]'
    $a = [System.Collections.Generic.List[string]]::new()
    $a.AddRange([string[]]@('--screen-width', $wh[0], '--screen-height', $wh[1]))
    $a.AddRange([string[]]@('--user-name', (Get-Value $config 'user_name' 'Kyty'), '--user-id', (Get-Value $config 'user_id' '1')))
    $mic = Get-Value $config 'audio_input_device' ''
    if ($mic) { $a.AddRange([string[]]@('--mic', $mic)) }
    $color = Get-Value $config 'controller_color' ''
    if ($color) { $a.AddRange([string[]]@('--controller-color', $color)) }
    $a.AddRange([string[]]@('--controller-volume', (Get-Value $config 'controller_speaker_volume' '50'),
        '--controller-vibration', (Get-Value $config 'controller_vibration_intensity' '100'),
        '--audio-master-volume', (Get-Value $config 'audio_master_volume' '100'),
        '--audio-main-volume', (Get-Value $config 'audio_main_volume' '100'),
        '--audio-music-volume', (Get-Value $config 'audio_music_volume' '100'),
        '--audio-pad-speaker-volume', (Get-Value $config 'audio_pad_speaker_main_volume' '30')))
    if ((Get-Value $config 'gpu_occlusion_accurate' 'false') -eq 'true') { $a.AddRange([string[]]@('--gpu-occlusion', 'on')) }
    $a.AddRange([string[]]@('--present-mode', (Get-Value $config 'present_mode' 'Mailbox')))
    $gpu = [int](Get-Value $config 'gpu_index' '-1')
    if ($gpu -ge 0) { $a.AddRange([string[]]@('--gpu', "$gpu")) }
    if ((Get-Value $config 'fullscreen_enabled' 'false') -eq 'true') { $a.Add('--fullscreen') }
    if ((Get-Value $config 'hide_cursor_enabled' 'false') -eq 'true') { $a.Add('--hide-cursor') }
    $a.AddRange([string[]]@('--readback-linear-images', (Get-Value $config 'readback_linear_images' 'false'),
        '--trophy-notifications', (Get-Value $config 'trophy_enabled' 'true')))
    if ((Get-Value $config 'tessellation_enabled' 'false') -eq 'true') { $a.Add('--tessellation') }
    $a.AddRange([string[]]@('--vblank-frequency', (Get-Value $config 'vblank_frequency' '60'),
        '--console-language', (Get-Value $config 'console_language' '1'),
        '--vulkan-validation', 'false',
        '--shader-validation', (Get-Value $config 'shader_validation_enabled' 'true'),
        '--shader-optimization-type', (Get-Value $config 'shader_optimization_type' 'Performance'),
        '--shader-log-direction', 'Silent', '--shader-log-folder', (Get-Value $config 'shader_log_folder' '_Shaders'),
        '--command-buffer-dump', 'false', '--command-buffer-dump-folder', (Get-Value $config 'command_buffer_dump_folder' '_Buffers'),
        '--printf-direction', (Get-Value $config 'printf_direction' 'Silent'),
        '--printf-output-file', (Get-Value $config 'printf_output_file' '_kyty.txt'),
        '--spirv-debug-printf', 'false'))
    if ((Get-Value $config 'amd_cpu_enabled' 'false') -eq 'true') { $a.Add('--amd-cpu') }
    if ((Get-Value $config 'red_zone_protection_enabled' 'false') -eq 'true') { $a.Add('--redzone') }
    $game = $basedir
    if ([IO.Directory]::Exists($basedir)) { $game = Join-Path $basedir (Get-Value $config 'elf' 'eboot.bin') }
    $a.AddRange([string[]]@('--game', $game))
    # The patch plan the launcher passes: _Patches\<title id>.json next to the launcher.
    $title = $TitleId
    if (-not $title -and $basedir -match '(PPSA\d{5})') { $title = $matches[1].ToUpper() }
    if (-not $title) {
        $plans = @(Get-ChildItem -LiteralPath (Join-Path $dir '_Patches') -Filter 'PPSA*.json' -ErrorAction SilentlyContinue)
        foreach ($p in $plans) { if ($basedir -match [regex]::Escape($p.BaseName)) { $title = $p.BaseName } }
        if (-not $title) {
            $astro = @($plans | Where-Object { $_.BaseName -match '^PPSA2156[47]$' })
            if ($astro.Count -eq 1) { $title = $astro[0].BaseName }
        }
    }
    $patch = ''
    if ($title) { $patch = Join-Path $dir "_Patches\$title.json" }
    if ($patch -and [IO.File]::Exists($patch)) { $a.AddRange([string[]]@('--game-patch', $patch)) } else { $patch = '' }
    return [pscustomobject]@{ Arguments = (Format-Arguments $a.ToArray()); Source = $ini; Title = $title; Patch = $patch; Game = $game }
}

function Get-CapturedArguments([string]$exe) {
    Write-Step 'Start Astro Bot from the launcher now (as you normally do). The kit waits for it, reads its command line and closes it again.'
    $deadline = (Get-Date).AddMinutes(10)
    while ((Get-Date) -lt $deadline) {
        $p = @(Get-CimInstance Win32_Process -Filter "Name='kyty_emulator.exe'" -ErrorAction SilentlyContinue)
        foreach ($proc in $p) {
            if ($proc.CommandLine -and $proc.CommandLine -match '--game') {
                $line = $proc.CommandLine.Trim()
                # Drop the program token (quoted or not).
                if ($line.StartsWith('"')) { $rest = $line.Substring($line.IndexOf('"', 1) + 1) } else { $rest = $line.Substring($line.IndexOf(' ') + 1) }
                Start-Sleep -Seconds 2
                Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
                Write-Step 'Got the launcher command line; the kit closed that game window. You can close the launcher and its console window now.'
                Start-Sleep -Seconds 5
                return [pscustomobject]@{ Arguments = $rest.Trim(); Source = 'launcher process'; Title = ''; Patch = ''; Game = '' }
            }
        }
        Start-Sleep -Milliseconds 500
    }
    throw 'The game was not started from the launcher within 10 minutes.'
}

function Read-Tail([string]$path) {
    try {
        $fs = [IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
        try { $sr = New-Object IO.StreamReader($fs); return $sr.ReadToEnd() } finally { $fs.Dispose() }
    } catch { return '' }
}

# ------------------------------------------------------------------------------------------------
$exe = Join-Path $EmulatorDir 'kyty_emulator.exe'
if (-not [IO.File]::Exists($exe)) { throw "kyty_emulator.exe not found in $EmulatorDir. Put the kit next to kyty_emulator.exe." }
$presetPath = Join-Path $EmulatorDir 'u59-preset.json'
$preset = [ordered]@{}
if ([IO.File]::Exists($presetPath)) {
    $json = [IO.File]::ReadAllText($presetPath) | ConvertFrom-Json
    foreach ($p in $json.PSObject.Properties) { $preset[$p.Name] = [string]$p.Value }
}
# "-Only a,b" arrives as one string through AMD-Test-Kit.cmd (powershell -File).
$Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$selected = @(if ($Only.Count -eq 0) { $Configs } else { @($Configs) + @($MoreConfigs) | Where-Object { $Only -contains $_.Name } })
if ($selected.Count -eq 0) { throw "No configuration matches -Only $($Only -join ','). Names: $((@($Configs) + @($MoreConfigs) | ForEach-Object Name) -join ', ')" }

if ($CaptureFromLauncher) {
    if (Get-Process kyty_emulator -ErrorAction SilentlyContinue) { throw 'Close the running game first.' }
    $launch = Get-CapturedArguments $exe
} else {
    $launch = Get-LauncherArguments $EmulatorDir
}

if ($ExtraArgs) { $launch.Arguments = $launch.Arguments + ' ' + $ExtraArgs }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if (-not $OutDir) { $OutDir = Join-Path $EmulatorDir "AMD-Test-Results-$stamp" }
Write-Step "Emulator:     $exe"
Write-Step "Settings:     $($launch.Source)"
Write-Step "Command line: $($launch.Arguments)"
if (-not $launch.Patch -and $launch.Source -ne 'launcher process') {
    Write-Host "WARNING: no patch file _Patches\$($launch.Title).json next to the emulator: the game would run WITHOUT the launcher patches." -ForegroundColor Yellow
    Write-Host "         Copy your _Patches folder from your int16.1 folder, or turn the patches on once in this launcher, then start the kit again." -ForegroundColor Yellow
}
Write-Step ("Runs:         {0} ({1}), about {2} minutes" -f $selected.Count, (($selected | ForEach-Object Name) -join ', '), [math]::Ceiling($selected.Count * 1.7 + 1.5))
if ($ListOnly) { foreach ($c in $selected) { Write-Host ("  {0,-14} {1}" -f $c.Name, $c.Text) }; return }
if (Get-Process kyty_emulator -ErrorAction SilentlyContinue) { throw 'A kyty_emulator.exe is already running. Close the game (and the launcher) first.' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# System facts for the report (GPU, driver, OS); no personal files.
$sys = [System.Collections.Generic.List[string]]::new()
$sys.Add("Kit run $stamp; emulator $exe")
$sys.Add("Command line: $($launch.Arguments)")
try { $os = Get-CimInstance Win32_OperatingSystem; $sys.Add("OS: $($os.Caption) $($os.Version) build $($os.BuildNumber)") } catch {}
try { foreach ($v in Get-CimInstance Win32_VideoController) { $sys.Add("GPU: $($v.Name); driver $($v.DriverVersion) ($($v.DriverDate)); status $($v.Status)") } } catch {}
try { $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1; $sys.Add("CPU: $($cpu.Name)") } catch {}
try { $mem = Get-CimInstance Win32_ComputerSystem; $sys.Add(("RAM: {0:N1} GB" -f ($mem.TotalPhysicalMemory / 1GB))) } catch {}
try {
    $tdr = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -ErrorAction SilentlyContinue
    if ($tdr) { foreach ($n in 'TdrDelay', 'TdrDdiDelay', 'TdrLevel', 'HwSchMode') { if ($tdr.PSObject.Properties[$n]) { $sys.Add("GraphicsDrivers\$n = $($tdr.$n)") } } }
} catch {}
[IO.File]::WriteAllLines((Join-Path $OutDir 'system.txt'), $sys)

$kitStart = Get-Date
$results = [System.Collections.Generic.List[object]]::new()
$savedEnv = @{}
foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -match '^(KYTY_|TRACY_)' })) { $savedEnv[$v.Name] = $v.Value }
$n = 0
try {
    $queue = [System.Collections.Generic.List[object]]::new()
    foreach ($c in $selected) { $queue.Add($c) }
    for ($qi = 0; $qi -lt $queue.Count; $qi++) {
        $c = $queue[$qi]
        $n++
        $name = '{0:D2}-{1}' -f $n, $c.Name
        $log = Join-Path $OutDir "$name.txt"
        $err = Join-Path $OutDir "$name-stderr.txt"
        # The launcher's environment: no inherited KYTY_/TRACY_ variables, then the preset; the kit's
        # switches go on top.
        foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -match '^(KYTY_|TRACY_)' })) { [Environment]::SetEnvironmentVariable($v.Name, $null, 'Process') }
        $envs = [ordered]@{}
        foreach ($k in $preset.Keys) { $envs[$k] = $preset[$k] }
        foreach ($k in $Common.Keys) { $envs[$k] = $Common[$k] }
        foreach ($k in $c.Env.Keys) { $envs[$k] = $c.Env[$k] }
        foreach ($pair in ($ExtraEnv -split ',' | Where-Object { $_ -match '=' })) { $kv = $pair -split '=', 2; $envs[$kv[0].Trim()] = $kv[1].Trim() }
        foreach ($k in $envs.Keys) { [Environment]::SetEnvironmentVariable($k, $envs[$k], 'Process') }
        $changed = ($c.Env.Keys | ForEach-Object { "$_=$($c.Env[$_])" }) -join ' '
        Write-Step ("Run {0}/{1}: {2} - {3} {4}" -f $n, $queue.Count, $c.Name, $c.Text, $(if ($changed) { "($changed)" } else { '' }))
        $header = "=== AMD test kit run ${name}: $($c.Text)`r`n=== switches: $changed`r`n=== started $(Get-Date -Format o)`r`n"
        [IO.File]::WriteAllText((Join-Path $OutDir "$name-env.txt"), $header + (($envs.Keys | ForEach-Object { "$_=$($envs[$_])" }) -join "`r`n"))
        $arguments = $launch.Arguments
        $kytyLog = ''
        if ($c.ContainsKey('LogFile') -and $c.LogFile) {
            # The emulator's own log (LOGF lines: device, subgroup and memory details) for this run.
            $kytyLog = Join-Path $OutDir "$name-kyty-log.txt"
            $arguments = $arguments -replace '--printf-direction\s+\S+', '--printf-direction File'
            $arguments = $arguments -replace '--printf-output-file\s+("[^"]*"|\S+)', ('--printf-output-file "' + $kytyLog + '"')
        }
        $start = Get-Date
        $after = if ($c.ContainsKey('Seconds')) { [int]$c.Seconds } else { $SecondsAfterVideo }
        $proc = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $EmulatorDir -PassThru `
            -RedirectStandardOutput $log -RedirectStandardError $err
        $null = $proc.Handle # keeps the exit code readable after the process ends
        $videoClosed = $null
        $lostAt = $null
        $verdict = ''
        while ($true) {
            Start-Sleep -Milliseconds 500
            $text = Read-Tail $log
            $now = Get-Date
            if (-not $videoClosed -and $text -cmatch 'sceAvPlayerClose|AvPlayer video stopped') { $videoClosed = $now; Write-Step '  intro video closed' }
            if (-not $lostAt -and $text -cmatch '--- Device loss:|ErrorDeviceLost|VK_ERROR_DEVICE_LOST') { $lostAt = $now; Write-Step '  device lost' }
            if ($proc.HasExited) { break }
            if ($lostAt -and ($now - $lostAt).TotalSeconds -ge 15) { break }
            if ($videoClosed -and ($now - $videoClosed).TotalSeconds -ge $after) { break }
            if (($now - $start).TotalSeconds -ge $MaxSeconds) { break }
        }
        $exited = $proc.HasExited
        if (-not $exited) {
            try { $null = $proc.CloseMainWindow() } catch {}
            if (-not $proc.WaitForExit(10000)) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue; $null = $proc.WaitForExit(10000) }
        }
        $text = Read-Tail $log
        $since = ''
        if ($videoClosed -and $lostAt) { $since = ('{0:N1} s after the video closed' -f ($lostAt - $videoClosed).TotalSeconds) }
        if ($lostAt) { $verdict = "DEVICE LOST $since".Trim() }
        elseif ($exited -and $text -match 'Fatal Error|Unhandled host exception') { $verdict = 'CRASHED (other error, see log)' }
        elseif ($exited) { $verdict = "EXITED (code $($proc.ExitCode))" }
        elseif ($videoClosed) { $verdict = "OK: still running $after s after the video" }
        else { $verdict = "NO VIDEO END within $MaxSeconds s (stopped by the kit)" }
        $title = ''
        if ($text -match 'Title ID: (\S+)') { $title = $matches[1] }
        $results.Add([pscustomobject]@{ Run = $name; Verdict = $verdict; Switches = $changed; Title = $title; Seconds = [int]((Get-Date) - $start).TotalSeconds })
        Write-Step "  -> $verdict"
        if ($text -match 'Loop guard: (\d+) invocations[^\r\n]*shader (0x[0-9a-f]+)') { Write-Step "  loop guard fired: $($matches[1]) invocations, shader $($matches[2])" }
        if ($kytyLog -and [IO.File]::Exists($kytyLog)) {
            # Keep the start (device setup) and the end (the crash) of a large log.
            $info = New-Object IO.FileInfo($kytyLog)
            if ($info.Length -gt 6MB) {
                $bytes = [IO.File]::ReadAllBytes($kytyLog)
                $mark = [Text.Encoding]::ASCII.GetBytes("`r`n... ($($bytes.Length - 6MB) bytes left out by the AMD test kit) ...`r`n")
                $out = New-Object IO.MemoryStream
                $out.Write($bytes, 0, 2MB); $out.Write($mark, 0, $mark.Length); $out.Write($bytes, $bytes.Length - 4MB, 4MB)
                [IO.File]::WriteAllBytes($kytyLog, $out.ToArray())
            }
        }
        [IO.File]::AppendAllText($log, "`r`n=== AMD test kit verdict: $verdict`r`n")
        # A GPU reset takes a few seconds; let the driver settle before the next run.
        Start-Sleep -Seconds $(if ($lostAt) { 15 } else { 5 })
    }
} finally {
    foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -match '^(KYTY_|TRACY_)' })) { [Environment]::SetEnvironmentVariable($v.Name, $null, 'Process') }
    foreach ($k in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $savedEnv[$k], 'Process') }
    # Windows' record of GPU resets (TDR: Display 4101, LiveKernelEvent 141/117) during the session.
    $events = [System.Collections.Generic.List[string]]::new()
    try {
        $sysEvents = @(Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $kitStart } -ErrorAction SilentlyContinue |
            Where-Object { $_.ProviderName -match 'Display|amdkmdag|amdkmdap|amdfendr|dxgkrnl|nvlddmkm|Graphics' -or $_.Id -eq 4101 })
        foreach ($e in $sysEvents) {
            $text = if ($e.Message) { $e.Message -replace '\s+', ' ' } else { '(no message text) data: ' + (($e.Properties | ForEach-Object { [string]$_.Value }) -join '; ') }
            if ($text.Length -gt 600) { $text = $text.Substring(0, 600) }
            $events.Add(("System {0:o} {1} id {2}: {3}" -f $e.TimeCreated, $e.ProviderName, $e.Id, $text))
        }
        $appEvents = @(Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $kitStart } -ErrorAction SilentlyContinue |
            Where-Object { $_.ProviderName -match 'Application Error|Windows Error Reporting' -and $_.Message -match 'kyty|LiveKernelEvent|141|117' })
        foreach ($e in $appEvents) { $events.Add(("Application {0:o} {1} id {2}: {3}" -f $e.TimeCreated, $e.ProviderName, $e.Id, ($e.Message -replace '\s+', ' ' | ForEach-Object { if ($_.Length -gt 600) { $_.Substring(0, 600) } else { $_ } }))) }
    } catch { $events.Add("event log read failed: $_") }
    try {
        foreach ($f in @(Get-ChildItem 'C:\Windows\LiveKernelReports' -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -ge $kitStart })) {
            $events.Add(("LiveKernelReport {0} {1:o} {2:N0} bytes (not copied)" -f $f.FullName, $f.LastWriteTime, $f.Length))
        }
    } catch {}
    if ($events.Count -eq 0) { $events.Add('No display-driver events in the System log during the session.') }
    if ($OutDir -and (Test-Path -LiteralPath $OutDir)) {
        [IO.File]::WriteAllLines((Join-Path $OutDir 'windows-events.txt'), $events)
        $summary = [System.Collections.Generic.List[string]]::new()
        $summary.Add("AMD test kit $stamp - $($results.Count) run(s)")
        foreach ($r in $results) { $summary.Add(("{0,-18} {1,-52} {2}" -f $r.Run, $r.Verdict, $r.Switches)) }
        $titles = @($results | Where-Object { $_.Title } | ForEach-Object Title | Select-Object -Unique)
        if ($launch.Title -and $titles.Count -and ($titles -notcontains $launch.Title)) {
            $summary.Add("WARNING: the kit passed the patch plan of $($launch.Title), but the game reports $($titles -join ','). Rerun with -TitleId $($titles[0]).")
        }
        if (-not $launch.Patch -and $launch.Source -ne 'launcher process') { $summary.Add('WARNING: no patch plan (_Patches\<title>.json) was found; the game ran without the launcher patches.') }
        [IO.File]::WriteAllLines((Join-Path $OutDir 'summary.txt'), $summary)
        Write-Host ''
        $summary | ForEach-Object { Write-Host $_ }
        if (-not $NoZip -and $results.Count -gt 0) {
            $zip = "$OutDir.zip"
            Compress-Archive -Path (Join-Path $OutDir '*') -DestinationPath $zip -Force
            Write-Host ''
            Write-Step "Send this file: $zip"
        }
    }
}
