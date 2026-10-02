# Headless checks: no game, window creation or desktop capture.
$ErrorActionPreference = 'Stop'
foreach ($name in 'bench_boot.ps1', 'bench_captures.ps1', 'capture_window.ps1') {
    $tokens = $null
    $errors = $null
    [void][Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $name), [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw ($errors | Out-String) }
}
. (Join-Path $PSScriptRoot 'bench_captures.ps1')
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('kyty-capture-test-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    $session = New-BenchCaptures $testRoot 'headless' (Get-Process -Id $PID) @(0, 999999)
    Update-BenchCaptures $session
    Complete-BenchCaptures $session
    $manifest = Get-Content -LiteralPath $session.Manifest -Raw | ConvertFrom-Json
    if ($manifest.captures[0].status -ne 'error' -or
        $manifest.captures[0].error -notlike '*No visible non-minimized SDL client*') {
        throw "Expected SDL window rejection, got: $($manifest.captures[0] | ConvertTo-Json -Compress)"
    }
    if ($manifest.captures[1].status -ne 'not-reached') { throw 'Future capture silently lost' }
    if (Get-ChildItem -LiteralPath $session.Folder -Filter '*.bmp') { throw 'Unexpected screenshot of a headless process' }
    $duplicateRejected = $false
    try { New-BenchCaptures $testRoot 'headless' (Get-Process -Id $PID) @(0) } catch { $duplicateRejected = $true }
    if (-not $duplicateRejected) { throw 'Existing capture directory could contaminate a new run' }
    # A real sleeping child stands in for a blocked native capture. Advance only
    # its deadline clock; assert the child is terminated and evidence is invalid.
    $sleeper = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList '-NoProfile -NonInteractive -Command "Start-Sleep -Seconds 30"' -WindowStyle Hidden -PassThru
    $sleeperId = $sleeper.Id
    try {
        [void]$session.Jobs.Add([pscustomobject]@{ Process = $sleeper; Index = 0; Result = 'absent'
            Clock = [pscustomobject]@{ Elapsed = [TimeSpan]::FromSeconds(6) } })
        Update-BenchCaptures $session -Finish
        $manifest = Get-Content -LiteralPath $session.Manifest -Raw | ConvertFrom-Json
        if ($manifest.captures[0].status -ne 'timeout' -or $session.Jobs.Count) { throw 'Blocked capture was not invalidated' }
    } finally {
        Get-Process -Id $sleeperId -ErrorAction SilentlyContinue | Stop-Process -Force
    }
    Write-Host 'PASS: syntax, native capture compilation, target-window rejection, missing schedule, stale-directory rejection, bounded timeout'
} finally {
    $resolved = [IO.Path]::GetFullPath($testRoot)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if (-not $resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path -Leaf $resolved) -notlike 'kyty-capture-test-*') { throw 'Unexpected cleanup path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
