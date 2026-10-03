<#
.SYNOPSIS
  Boot benchmark for kyty_emulator with the hang trace on (Windows PowerShell 5.1+).

.DESCRIPTION
  Runs a game for -Seconds per variant and repeat, then prints and saves fps per 10 s window,
  worst second, shader/pipeline compile cost, compile threads and APR read counts, all taken from
  the hang trace CSV files (summary.csv, compiles.csv, apr.csv). fps is flips per second from the
  trace, not the window title (the title is a single instant and hid the dips that matter).

  The emulator is run from a frozen copy of -InstallDir (executables, DLLs and non-underscore
  folders only), with a private copy of the game's pipeline cache that is restored before every
  run, so the order of the variants does not matter and the real caches/saves are untouched.

  Use -AnalyzeOnly <hang-trace dir> to summarize an existing trace without launching anything.

  A run needs the machine to itself: it refuses to start while kyty_emulator.exe is running.

.PARAMETER Variant
  "name:KEY=VALUE;KEY=VALUE". The name "preset" (or "preset-<anything>") also loads -PresetFile (a JSON object of
  environment variables, e.g. tools/u59-preset.json); pairs after the colon override it.
  Every variant also gets KYTY_HANG_TRACE=1. Any KYTY_* variable set in this shell is cleared first.

.PARAMETER ScreenshotSeconds
  Opt-in elapsed seconds from process start at which to capture each variant's SDL client.
  Example: -ScreenshotSeconds 30,60,90,120. This adds overhead. Matching elapsed time does
  not guarantee the same game state. Hidden/minimized windows or unsupported Vulkan capture
  can fail; tools/compare_captures.py rejects failed, missing, late and black captures.
  Compare two captures-<run>/captures.json manifests with that Python script (no packages needed).

.PARAMETER KeyTaps
  Opt-in KEY@SECONDS taps to the exact emulator process's SDL window. Example:
  -KeyTaps 'J@25,J@32,J@40' sends Cross (J). Focus must be verified before key-down.
  Each tap holds for -KeyTapHoldMs (default 120), and the emulator gets
  KYTY_HOST_INPUT_MIN_PRESS_MS and KYTY_HOST_INPUT_ONLY=1 only with this option.
  key-taps-<run>.json records requested/actual timing, PID, HWND, focus and insertion status.
  Taps add overhead; inspect screenshots/input trace to confirm entry into gameplay.

.EXAMPLE
  .\tools\bench_boot.ps1 -InstallDir _Build\windows\install-video-fix -GameDir C:/Games/PPSA01325 `
      -TitleId PPSA01325 -Seconds 60 -Repeat 3 -PresetFile tools\u59-preset.json `
      -Variant 'base:KYTY_DCC_GPU=1;KYTY_PROGRAM_CACHE=0;KYTY_PIPELINE_LIBRARY=1','preset'

.EXAMPLE
  .\tools\bench_boot.ps1 -AnalyzeOnly C:\path\to\_HangTrace\20260930-pid1234
#>
[CmdletBinding()]
param(
    [string]$InstallDir,
    [string]$GameDir,
    [int]$IniGame = 0,  # take the game directory from C:\ProgramData\Kyty\Kyty.ini entry N (avoids typing odd paths)
    [string]$TitleId,
    [ValidateSet('Silent', 'File')][string]$Printf = 'Silent',  # guest printf log; Silent for speed, File to keep a crash report
    [ValidateSet('Fifo', 'Mailbox', 'Immediate')][string]$PresentMode = 'Mailbox',  # Mailbox falls back to Fifo on this GPU (vsync)
    [int]$Seconds = 60,
    [int]$Repeat = 1,
    [string[]]$Variant = @('base:KYTY_DCC_GPU=1;KYTY_PROGRAM_CACHE=0;KYTY_PIPELINE_LIBRARY=0'),
    [string]$PresetFile,
    [string]$OutDir,
    [string[]]$ExtraArgs = @(),
    [switch]$NoFreeze,
    [switch]$KeepCache,  # restore the pipeline/program cache only before the first run: later runs see what earlier ones compiled (a returning player)
    [switch]$Profile,   # pass --profile (Tracy). Off by default: docs/EXPERIMENTAL.md says to disable it for clean timing
    [int]$SlowAfter = 50,      # do not start the slow-scene sampler before this many seconds into the run
    [int]$SlowSeconds = 8,     # consecutive seconds below 30 fps before the slow-scene sampler starts
    [int]$SampleSeconds = 20,  # how long thread_sampler.exe samples the busiest threads
    [int[]]$ScreenshotSeconds = @(),
    [string[]]$KeyTaps = @(),
    [ValidateRange(120, 2000)][int]$KeyTapHoldMs = 120,
    [string]$AnalyzeOnly
)

$ErrorActionPreference = 'Stop'

function Get-Num($value) {
    if ($null -eq $value -or "$value" -eq '') { return 0.0 }
    return [double]$value
}

function Get-ColumnSum($rows, $column) {
    if (@($rows).Count -eq 0 -or -not ($rows[0].PSObject.Properties.Name -contains $column)) { return 0.0 }
    $sum = 0.0
    foreach ($row in $rows) { $sum += Get-Num $row.$column }
    return $sum
}

# One row per second. The closing rows written at shutdown are shorter than a second: skipped.
function Get-SecondRows($rows) {
    $result = @()
    $previous = $null
    foreach ($row in $rows) {
        $t = Get-Num $row.t_ms
        if ($null -ne $previous -and ($t - $previous) -lt 500) { continue }
        $result += $row
        $previous = $t
    }
    return $result
}

function Get-TraceMetrics($dir) {
    $summaryFile = Join-Path $dir 'summary.csv'
    if (-not (Test-Path $summaryFile)) { throw "no summary.csv in $dir" }
    # Totals use every row (the closing partial rows hold real counts); fps uses full seconds only.
    $all = @(Import-Csv $summaryFile)
    $seconds = $all
    $flips = @(Get-SecondRows $all | ForEach-Object { [int](Get-Num $_.flips) })
    $windows = @()
    for ($i = 0; $i -lt $flips.Count; $i += 10) {
        $end = [Math]::Min($i + 9, $flips.Count - 1)
        $windows += [int][Math]::Round((($flips[$i..$end]) | Measure-Object -Average).Average)
    }
    # Ignore the first 5 s (startup) and the last second (shutdown) for worst/mean.
    $steady = @($flips | Select-Object -Skip 5 | Select-Object -SkipLast 1)
    $metrics = [ordered]@{
        seconds       = $flips.Count
        fps_windows   = ($windows -join ' ')
        fps_mean      = if ($steady.Count) { [Math]::Round(($steady | Measure-Object -Average).Average, 1) } else { 0 }
        worst_second  = if ($steady.Count) { ($steady | Measure-Object -Minimum).Minimum } else { 0 }
        seconds_lt30  = @($steady | Where-Object { $_ -lt 30 }).Count
        programs      = [int](Get-ColumnSum $seconds 'compile_programs')
        gfx_pipelines = [int](Get-ColumnSum $seconds 'compile_gfx_pipelines')
        cs_pipelines  = [int](Get-ColumnSum $seconds 'compile_cs_pipelines')
        translate_ms  = [int]((Get-ColumnSum $seconds 'compile_translate_us') / 1000)
        emit_ms       = [int]((Get-ColumnSum $seconds 'compile_emit_us') / 1000)
        module_ms     = [int]((Get-ColumnSum $seconds 'compile_module_us') / 1000)
        pipeline_ms   = [int](((Get-ColumnSum $seconds 'compile_gfx_pipeline_us') + (Get-ColumnSum $seconds 'compile_cs_pipeline_us')) / 1000)
        gpu_busy_ms_s = [Math]::Round((Get-ColumnSum $seconds 'gpu_busy_us') / 1000 / [Math]::Max(1, $flips.Count), 1)
        apr_reads     = [int](Get-ColumnSum $seconds 'apr_reads')
        apr_mb        = [int]((Get-ColumnSum $seconds 'apr_bytes') / 1MB)
        # Memory predication packets that drained the GPU: waits per second and ms blocked per second.
        pred_waits_s  = [int][Math]::Round((Get-ColumnSum $seconds 'pred_flush_waits') / [Math]::Max(1, $flips.Count))
        pred_wait_ms_s = [Math]::Round((Get-ColumnSum $seconds 'pred_flush_wait_us') / 1000 / [Math]::Max(1, $flips.Count), 1)
    }

    $compilesFile = Join-Path $dir 'compiles.csv'
    $metrics['compile_threads'] = 0
    $metrics['compile_worst_s_ms'] = 0
    if (Test-Path $compilesFile) {
        $compiles = @(Import-Csv $compilesFile)
        if ($compiles.Count) {
            $metrics['compile_threads'] = @($compiles | Select-Object -ExpandProperty host_tid -Unique).Count
            $perSecond = $compiles | Group-Object { [int][Math]::Floor((Get-Num $_.t_ms) / 1000) } | ForEach-Object {
                [int]((($_.Group | ForEach-Object { Get-Num $_.total_us }) | Measure-Object -Sum).Sum / 1000)
            }
            $metrics['compile_worst_s_ms'] = ($perSecond | Measure-Object -Maximum).Maximum
        }
    }

    $aprFile = Join-Path $dir 'apr.csv'
    $metrics['apr_main_small_reads'] = 0
    if (Test-Path $aprFile) {
        $apr = @(Import-Csv $aprFile)
        $metrics['apr_main_small_reads'] = @($apr | Where-Object { $_.thread -eq 'MainThread' -and (Get-Num $_.size) -le 4096 }).Count
    }
    return [pscustomobject]$metrics
}

if ($AnalyzeOnly) {
    Get-TraceMetrics $AnalyzeOnly | Format-List
    return
}

# ---- run mode -----------------------------------------------------------------------------

if ($KeyTaps.Count) {
    . (Join-Path $PSScriptRoot 'bench_key_taps.ps1')
    $keyTapPlan = @(ConvertTo-BenchKeyTapPlan $KeyTaps $Seconds)
    # Compile the optional Win32 adapter before timing the launched process.
    Initialize-BenchKeyTapNative
}
if ($ScreenshotSeconds.Count) {
    if (@($ScreenshotSeconds | Where-Object { $_ -lt 0 -or $_ -ge $Seconds }).Count) {
        throw '-ScreenshotSeconds must be nonnegative and less than -Seconds'
    }
    $ScreenshotSeconds = @($ScreenshotSeconds | Sort-Object -Unique)
    $captureNames = @($Variant | ForEach-Object { ($_ -split ':', 2)[0] })
    if (@($captureNames | Where-Object { $_ -notmatch '^[A-Za-z0-9_-]+$' }).Count -or
        @($captureNames | Sort-Object -Unique).Count -ne $captureNames.Count) {
        throw 'Screenshots require unique variant names containing only letters, digits, underscore or hyphen'
    }
    . (Join-Path $PSScriptRoot 'bench_captures.ps1')
}

if (-not $GameDir -and $IniGame -gt 0) {
    $iniPath = 'C:\ProgramData\Kyty\Kyty.ini'
    $line = [IO.File]::ReadAllLines($iniPath, [Text.Encoding]::UTF8) | Where-Object { $_.StartsWith("$IniGame\basedir=") } | Select-Object -First 1
    if (-not $line) { throw "no '$IniGame\basedir=' in $iniPath" }
    $GameDir = $line.Substring("$IniGame\basedir=".Length).TrimEnd("`r")
    Write-Host "game dir from ini: $GameDir"
}
foreach ($required in 'InstallDir', 'GameDir', 'TitleId') {
    if (-not (Get-Variable $required -ValueOnly)) { throw "-$required is required (or use -AnalyzeOnly)" }
}
if (Get-Process kyty_emulator -ErrorAction SilentlyContinue) {
    throw 'kyty_emulator.exe is already running: a benchmark needs the machine to itself.'
}
$InstallDir = (Resolve-Path $InstallDir).Path
if (-not $OutDir) { $OutDir = Join-Path $env:TEMP ('kyty-bench-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$runtime = $InstallDir
$runCacheDir  = $null
$cacheOrigDir = $null
if (-not $NoFreeze) {
    $runtime = Join-Path $OutDir 'rt'
    New-Item -ItemType Directory -Force $runtime | Out-Null
    Get-ChildItem $InstallDir -File | Copy-Item -Destination $runtime -Force
    Get-ChildItem $InstallDir -Directory | Where-Object { $_.Name -notmatch '^_' } |
        ForEach-Object { Copy-Item $_.FullName (Join-Path $runtime $_.Name) -Recurse -Force }
    # Every cache file of this title (driver cache <id>.bin and the program cache <id>.programs.bin)
    # is kept as an original and put back before each run, so repeats start from the same state.
    $cacheFiles = @(Get-ChildItem (Join-Path $InstallDir '_PipelineCache') -File -Filter "$TitleId*" -ErrorAction SilentlyContinue)
    if ($cacheFiles.Count) {
        $runCacheDir  = Join-Path $runtime '_PipelineCache'
        $cacheOrigDir = Join-Path $OutDir 'cache-orig'
        New-Item -ItemType Directory -Force $runCacheDir, $cacheOrigDir | Out-Null
        $cacheFiles | Copy-Item -Destination $cacheOrigDir -Force
    }
}
# The game's save is copied into the frozen runtime (and restored before every run), so a run
# starts where the real save is, repeatably, and never writes to the real save.
$srcSave = Join-Path $InstallDir "_SaveData\$TitleId"
$runSave = $null
if (-not $NoFreeze -and (Test-Path $srcSave)) {
    New-Item -ItemType Directory -Force (Join-Path $runtime '_SaveData') | Out-Null
    $runSave = Join-Path $runtime "_SaveData\$TitleId"
}
$exe = Join-Path $runtime 'kyty_emulator.exe'

$presetEnv = @{}
if ($PresetFile) {
    $json = Get-Content $PresetFile -Raw | ConvertFrom-Json
    foreach ($property in $json.PSObject.Properties) {
        if ("$($property.Value)" -ne '') { $presetEnv[$property.Name] = "$($property.Value)" }
    }
}

function Quote-Arg($text) { if ($text -match '\s') { return '"' + $text + '"' } else { return $text } }

$table = @()
$cacheRestored = $false
foreach ($spec in $Variant) {
    $name, $pairs = $spec -split ':', 2
    $env_vars = @{}
    if ($name -eq 'preset' -or $name -like 'preset-*') {
        if (-not $PresetFile) { throw "variant '$name' needs -PresetFile" }
        foreach ($key in $presetEnv.Keys) { $env_vars[$key] = $presetEnv[$key] }
    }
    if ($pairs) {
        foreach ($pair in ($pairs -split ';')) {
            if ($pair -match '^([^=]+)=(.*)$') { $env_vars[$Matches[1]] = $Matches[2] }
        }
    }
    for ($n = 1; $n -le $Repeat; $n++) {
        $tag = "$name-$n"
        $trace = Join-Path $OutDir "trace-$tag"
        $log = Join-Path $OutDir "$tag.log"
        if (Test-Path $trace) { Remove-Item $trace -Recurse -Force }
        if ($runCacheDir -and -not ($KeepCache -and $cacheRestored)) {
            $cacheRestored = $true
            Get-ChildItem $runCacheDir -File -Filter "$TitleId*" | Remove-Item -Force
            Copy-Item (Join-Path $cacheOrigDir '*') $runCacheDir -Force
        }

        if ($runSave) {
            if (Test-Path $runSave) { Remove-Item $runSave -Recurse -Force }
            Copy-Item $srcSave $runSave -Recurse -Force
        }
        Get-ChildItem Env: | Where-Object { $_.Name -like 'KYTY_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
        foreach ($key in $env_vars.Keys) { Set-Item "Env:$key" $env_vars[$key] }
        $env:KYTY_HANG_TRACE = '1'
        $env:KYTY_HANG_TRACE_DIR = $trace
        if ($KeyTaps.Count) {
            $env:KYTY_HOST_INPUT_MIN_PRESS_MS = [string]$KeyTapHoldMs
            $env:KYTY_HOST_INPUT_ONLY = '1'
        }

        $emuArgs = @('--screen-width', '1280', '--screen-height', '720', '--user-name', 'Kyty', '--user-id', '1000',
            '--present-mode', $PresentMode, '--gpu', '0', '--readback-linear-images', 'false',
            '--vblank-frequency', '60', '--console-language', '7',
            '--vulkan-validation', 'false', '--shader-validation', 'false',
            '--shader-optimization-type', 'Performance', '--shader-log-direction', 'Silent',
            '--shader-log-folder', '_Shaders', '--command-buffer-dump', 'false',
            '--command-buffer-dump-folder', '_Buffers', '--printf-direction', $Printf,
            '--printf-output-file', $log, '--spirv-debug-printf', 'false', '--amd-cpu') +
            $(if ($Profile) { @('--profile') } else { @() }) + $ExtraArgs +
            @('--game', "$GameDir/eboot.bin")
        $argLine = ($emuArgs | ForEach-Object { Quote-Arg $_ }) -join ' '

        Write-Host ("[{0}] running {1} s ..." -f $tag, $Seconds)
        if ($ScreenshotSeconds.Count -and (Test-Path -LiteralPath (Join-Path $OutDir "captures-$tag"))) {
            throw "Capture folder for $tag already exists (choose a new -OutDir)"
        }
        if ($KeyTaps.Count -and (Test-Path -LiteralPath (Join-Path $OutDir "key-taps-$tag.json"))) {
            throw "Key-tap manifest for $tag already exists (choose a new -OutDir)"
        }
        $process = Start-Process -FilePath $exe -ArgumentList $argLine -WorkingDirectory $runtime -PassThru
        $keyTapSession = $null
        if ($KeyTaps.Count) {
            $keyTapSession = New-BenchKeyTaps $OutDir $tag $process $keyTapPlan $KeyTapHoldMs
            Update-BenchKeyTaps $keyTapSession
        }
        $captureSession = $null
        if ($ScreenshotSeconds.Count) {
            $captureSession = New-BenchCaptures $OutDir $tag $process $ScreenshotSeconds
            Update-BenchCaptures $captureSession
        }
        # Once a second: CPU time per host thread. One thread near 100% while the GPU idles means a
        # serialized CPU limit; low CPU everywhere with a slow game means waiting (I/O, GPU, locks).
        $perThread = @{}
        $previous  = @{}
        $series    = New-Object System.Collections.ArrayList   # (second, tid, cpu%) for threads above 1%
        $samples   = 0
        $cpuSampleClock = [Diagnostics.Stopwatch]::StartNew()
        $previousSampleSeconds = $null
        $cpuObservedSeconds = 0.0
        $deadline  = [DateTime]::UtcNow.AddSeconds($Seconds)
        # Slow-scene profiling: when the game has been below 30 fps for $SlowSeconds in a row (after
        # the first 50 s), thread_sampler.exe samples the busiest threads for $SampleSeconds.
        $samplerExe   = @("$PSScriptRoot\thread_sampler.exe", (Join-Path $runtime 'thread_sampler.exe')) | Where-Object { Test-Path $_ } | Select-Object -First 1
        $slowRun      = 0
        $samplerProc  = $null
        $samplesCsv   = Join-Path $OutDir "samples-$tag.csv"
        try {
          while (-not $process.HasExited -and [DateTime]::UtcNow -lt $deadline) {
            if ($keyTapSession) { Wait-BenchKeyTaps $keyTapSession 1000 }
            else { Start-Sleep -Milliseconds 1000 }
            if ($captureSession) { Update-BenchCaptures $captureSession }
            $profileElapsed = if ($keyTapSession) { $cpuSampleClock.Elapsed.TotalSeconds } else { $samples }
            if ($samplerExe -and -not $samplerProc -and $profileElapsed -ge $SlowAfter) {
                try {
                    $lastRows = @(Import-Csv (Join-Path $trace 'summary.csv') | Select-Object -Last 2)
                    $flipsNow  = if ($lastRows.Count -ge 2) { [int](Get-Num $lastRows[0].flips) } else { 0 }
                    # Seconds spent compiling pipelines are not the steady slow scene: do not trigger on them.
                    $compiling = $lastRows.Count -ge 2 -and ((Get-Num $lastRows[0].compile_gfx_pipelines) -gt 0 -or
                        (Get-Num $lastRows[0].compile_cs_pipelines) -gt 0 -or (Get-Num $lastRows[0].compile_programs) -gt 0)
                    if ($flipsNow -ge 1 -and $flipsNow -le 30 -and -not $compiling) { $slowRun++ } else { $slowRun = 0 }
                } catch { }
                if ($slowRun -ge $SlowSeconds) {
                    $recent = @($series | Where-Object { $_.t -gt $profileElapsed - 5 } | Group-Object tid |
                        ForEach-Object { [pscustomobject]@{ tid = $_.Name; sum = ($_.Group | Measure-Object cpu_pct -Sum).Sum } } |
                        Sort-Object sum -Descending | Select-Object -First 5)
                    $tidArgs = ($recent | ForEach-Object { $_.tid }) -join ' '
                    Write-Host ("[{0}] slow scene detected ({1} s below 30 fps): sampling threads {2} for {3} s" -f $tag, $slowRun, $tidArgs, $SampleSeconds)
                    $samplerProc = Start-Process -FilePath $samplerExe -PassThru -WindowStyle Hidden -ArgumentList ("{0} {1} 250 `"{2}`" {3}" -f $process.Id, $SampleSeconds, $samplesCsv, $tidArgs)
                }
            }
            $current = @{}
            try {
                $process.Refresh()
                foreach ($thread in $process.Threads) {
                    try {
                        $current[$thread.Id] = $thread.TotalProcessorTime.TotalSeconds
                        if (-not $perThread.ContainsKey($thread.Id)) {
                            $perThread[$thread.Id] = @{ sum = 0.0; peak = 0.0; start = ('0x{0:x}' -f [int64]$thread.StartAddress) }
                        }
                    } catch { }
                }
            } catch { break }
            $sampleSeconds = $cpuSampleClock.Elapsed.TotalSeconds
            $sampleInterval = if ($null -eq $previousSampleSeconds) { 0.0 } else { $sampleSeconds - $previousSampleSeconds }
            if ($sampleInterval -gt 0) { $cpuObservedSeconds += $sampleInterval }
            foreach ($id in $current.Keys) {
                if ($previous.ContainsKey($id)) {
                    $delta = $current[$id] - $previous[$id]
                    $perThread[$id].sum += $delta
                    # A hold at a tap deadline can extend this interval. Keep its
                    # cost in the run, but report CPU rate per actual wall second.
                    $cpuPercent = if ($keyTapSession) { Get-BenchKeyTapCpuPercent $delta $sampleInterval } else { $delta * 100 }
                    if ($cpuPercent / 100 -gt $perThread[$id].peak) { $perThread[$id].peak = $cpuPercent / 100 }
                    if ($cpuPercent -ge 1) {
                        $sampleTime = if ($keyTapSession) { $sampleSeconds } else { $samples + 1 }
                        [void]$series.Add([pscustomobject]@{ t = $sampleTime; tid = $id; cpu_pct = [Math]::Round($cpuPercent) })
                    }
                }
            }
            $previous = $current
            $previousSampleSeconds = $sampleSeconds
            $samples++
          }
        } finally {
            if ($keyTapSession) { Complete-BenchKeyTaps $keyTapSession }
        }
        $exited = $process.HasExited
        if ($captureSession) { Complete-BenchCaptures $captureSession }
        if ($samplerProc -and -not $samplerProc.HasExited) { $null = $samplerProc.WaitForExit(($SampleSeconds + 5) * 1000) }
        if ($series.Count) { $series | Export-Csv (Join-Path $OutDir "threads-$tag.csv") -NoTypeInformation }
        $exitCode = $null
        if ($exited) { $exitCode = $process.ExitCode }
        else {
            $null = $process.CloseMainWindow()
            if (-not $process.WaitForExit(10000)) { $process.Kill() }
        }
        $fatal = $false
        if (Test-Path $log) {
            $fatal = [bool](Select-String -Path $log -Pattern 'ErrorDeviceLost|Fatal Error|--- Error ---' -Quiet)
        }
        $row = [ordered]@{ run = $tag; early_exit = $exited; exit_code = $exitCode; fatal = $fatal }
        $totalCpu = 0.0
        foreach ($entry in $perThread.Values) { $totalCpu += $entry.sum }
        $cpuDenominator = if ($keyTapSession) { $cpuObservedSeconds } else { $samples }
        $row['cpu_total_pct'] = if ($cpuDenominator -gt 0) { [Math]::Round($totalCpu / $cpuDenominator * 100) } else { 0 }
        $topThreads = @($perThread.GetEnumerator() | Sort-Object { $_.Value.sum } -Descending | Select-Object -First 8)
        $threadText = ($topThreads | ForEach-Object {
            '{0}@{1}:{2:N0}%/{3:N0}%' -f $_.Key, $_.Value.start, ($_.Value.sum / [Math]::Max(0.001, $cpuDenominator) * 100), ($_.Value.peak * 100)
        }) -join '  '
        $row['top_threads'] = $threadText
        Write-Host ("[{0}] cpu total {1}% of one core; top threads (tid@start avg%/peak%): {2}" -f $tag, $row['cpu_total_pct'], $threadText)
        try { (Get-TraceMetrics $trace).PSObject.Properties | ForEach-Object { $row[$_.Name] = $_.Value } }
        catch { $row['error'] = $_.Exception.Message }
        $table += [pscustomobject]$row
        # Per-thread profile of the slow scene (needs Python 3; names and guest modules need -Printf File).
        if (Test-Path $samplesCsv) {
            $python = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)
            if (-not $python) { $python = 'C:\Users\blade\AppData\Local\Programs\Python\Python312\python.exe' }
            $analysis = Join-Path $OutDir "profile-$tag.txt"
            $logArgs = if (Test-Path $log) { @('--log', $log) } else { @() }
            & $python (Join-Path $PSScriptRoot 'analyze_samples.py') $samplesCsv --exe $exe @logArgs 2>&1 | Tee-Object -FilePath $analysis | Write-Host
            Write-Host "profile: $analysis"
        }
    }
}

$table | Format-Table -AutoSize | Out-String -Width 300 | Write-Host
$table | Export-Csv (Join-Path $OutDir 'results.csv') -NoTypeInformation
Write-Host "results: $OutDir\results.csv"
