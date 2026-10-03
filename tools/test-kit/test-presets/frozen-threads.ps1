# Called by "While frozen - GPU load.cmd" while the game is frozen. Read-only: it measures how much
# processor time each kyty_emulator thread uses over 5 seconds and appends the busiest threads to
# the file given as the first argument. One thread near 100 % means something is spinning or
# compiling; every thread near 0 % means the emulator is waiting.
param([Parameter(Mandatory = $true)][string]$OutFile)

$lines = New-Object System.Collections.Generic.List[string]
$process = Get-Process kyty_emulator -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $process) {
    $lines.Add('kyty_emulator is not running')
} else {
    $before = @{}
    foreach ($thread in $process.Threads) {
        try { $before[$thread.Id] = $thread.TotalProcessorTime.TotalMilliseconds } catch {}
    }
    $processBefore = $process.TotalProcessorTime.TotalMilliseconds
    Start-Sleep -Seconds 5
    $process.Refresh()
    $rows = foreach ($thread in $process.Threads) {
        try {
            if ($before.ContainsKey($thread.Id)) {
                $busy = ($thread.TotalProcessorTime.TotalMilliseconds - $before[$thread.Id]) / 50.0
                $wait = if ($thread.ThreadState -eq 'Wait') { [string]$thread.WaitReason } else { '' }
                [pscustomobject]@{ Thread = $thread.Id; CpuPercentOfOneCore = [math]::Round($busy, 1)
                    State = [string]$thread.ThreadState; WaitReason = $wait }
            }
        } catch {}
    }
    $total = ($process.TotalProcessorTime.TotalMilliseconds - $processBefore) / 50.0
    $lines.Add(('kyty_emulator: {0} threads, {1:N1} % of one core over 5 s in total' -f $process.Threads.Count, $total))
    $lines.Add('Busiest threads:')
    $table = $rows | Sort-Object CpuPercentOfOneCore -Descending | Select-Object -First 10 |
        Format-Table -AutoSize | Out-String
    foreach ($line in ($table -split "`r?`n")) { if ($line.Trim()) { $lines.Add($line) } }
}
Add-Content -LiteralPath $OutFile -Value $lines
