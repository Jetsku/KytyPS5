# Dot-sourced by bench_boot. Workers are isolated because PrintWindow can hang.
function New-BenchCaptures($OutDir, $Tag, $Process, $Schedule) {
    $folder = Join-Path $OutDir "captures-$Tag"
    if (Test-Path -LiteralPath $folder) { throw "Capture folder already exists: $folder (choose a new -OutDir)" }
    New-Item -ItemType Directory -Path $folder | Out-Null
    $records = @($Schedule | ForEach-Object {
        [pscustomobject]@{ requested_seconds = $_; actual_seconds = $null; completed_seconds = $null
            status = 'not-reached'; image = "second-$_.bmp"; error = 'Capture time was not reached' }
    })
    $session = [pscustomobject]@{
        Folder = $folder; Manifest = Join-Path $folder 'captures.json'; Records = $records
        Process = $Process; Started = $Process.StartTime.ToUniversalTime(); Next = 0
        Jobs = [Collections.ArrayList]::new(); Tag = $Tag
    }
    Save-BenchCaptureManifest $session
    return $session
}

function Save-BenchCaptureManifest($Session) {
    [ordered]@{ alignment = 'elapsed-time'; run = $Session.Tag
        started_utc = $Session.Started.ToString('o'); captures = @($Session.Records)
        note = 'Elapsed process time, not deterministic game-state/frame alignment; capture adds overhead.'
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Session.Manifest -Encoding UTF8
}

function Update-BenchCaptures($Session, [switch]$Finish) {
    foreach ($job in @($Session.Jobs)) {
        $timedOut = $job.Clock.Elapsed.TotalSeconds -ge 5
        if (-not $job.Process.HasExited -and -not $timedOut) { continue }
        $row = $Session.Records[$job.Index]
        if (-not $job.Process.HasExited) {
            try { $job.Process.Kill() } catch { }
            $row.status = 'timeout'
            $row.error = 'Capture worker exceeded five seconds'
        } else {
            try {
                $result = Get-Content -LiteralPath $job.Result -Raw | ConvertFrom-Json
                if ($result.status -notin @('ok', 'error')) { throw 'Invalid worker status' }
                $row.status = $result.status
                $row.actual_seconds = $result.actual_seconds
                $row.completed_seconds = $result.completed_seconds
                $row.error = $result.error
            } catch {
                $row.status = 'error'
                $row.error = "Capture worker failed: $($_.Exception.Message)"
            }
        }
        $Session.Jobs.Remove($job)
        $job.Process.Dispose()
        Save-BenchCaptureManifest $Session
    }
    if ($Finish -or $Session.Process.HasExited) { return }
    $elapsed = ([DateTime]::UtcNow - $Session.Started).TotalSeconds
    while ($Session.Next -lt $Session.Records.Count -and $Session.Records[$Session.Next].requested_seconds -le $elapsed) {
        $index = $Session.Next
        $Session.Next++
        $row = $Session.Records[$index]
        $requestFile = Join-Path $Session.Folder "request-$index.json"
        $resultFile = Join-Path $Session.Folder "result-$index.json"
        [ordered]@{ target_pid = $Session.Process.Id; started_utc = $Session.Started.ToString('o')
            requested_seconds = $row.requested_seconds; image = $row.image; result_file = $resultFile
        } | ConvertTo-Json | Set-Content -LiteralPath $requestFile -Encoding UTF8
        $row.status = 'pending'
        $row.error = 'Worker has not completed'
        try {
            $workerScript = Join-Path $PSScriptRoot 'capture_window.ps1'
            $arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}" -RequestFile "{1}"' -f $workerScript, $requestFile
            $worker = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru
            [void]$Session.Jobs.Add([pscustomobject]@{
                Process = $worker; Clock = [Diagnostics.Stopwatch]::StartNew(); Index = $index; Result = $resultFile
            })
        } catch {
            $row.status = 'error'
            $row.error = $_.Exception.Message
        }
        Save-BenchCaptureManifest $Session
    }
}

function Complete-BenchCaptures($Session) {
    while ($Session.Jobs.Count) {
        Update-BenchCaptures $Session -Finish
        if ($Session.Jobs.Count) { Start-Sleep -Milliseconds 100 }
    }
    Save-BenchCaptureManifest $Session
    Write-Host "captures: $($Session.Manifest) (elapsed-time pairs; inspect with tools/compare_captures.py)"
}
