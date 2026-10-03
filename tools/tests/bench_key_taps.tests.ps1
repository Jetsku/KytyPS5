# Standalone headless checks; never creates a window or injects desktop input.
$ErrorActionPreference = 'Stop'
$toolsRoot = Split-Path -Parent $PSScriptRoot
function Assert-KeyTap($Condition, $Message) {
    if (-not $Condition) { throw $Message }
}
function Assert-KeyTapThrows([scriptblock]$Action, $Message) {
    $threw = $false
    try { & $Action | Out-Null } catch { $threw = $true }
    Assert-KeyTap $threw $Message
}
$tokens = $null
$errors = $null
$bootAst = [Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $toolsRoot 'bench_boot.ps1'), [ref]$tokens, [ref]$errors)
Assert-KeyTap ($errors.Count -eq 0) 'bench_boot syntax is invalid'
$parameterNames = @($bootAst.ParamBlock.Parameters | ForEach-Object { $_.Name.VariablePath.UserPath })
Assert-KeyTap ($parameterNames -contains 'KeyTaps') 'bench_boot is missing the opt-in -KeyTaps parameter'
Assert-KeyTap ($parameterNames -contains 'KeyTapHoldMs') 'bench_boot is missing the minimum hold parameter'
. (Join-Path $toolsRoot 'bench_key_taps.ps1')

$plan = @(ConvertTo-BenchKeyTapPlan @('j@25,J@32', 'ENTER@1.5', 'Left@3', 'J@32') 60)
Assert-KeyTap ($plan.Count -eq 5) 'Comma-separated entries or explicit repeated taps were dropped'
Assert-KeyTap ($plan[0].key -eq 'ENTER' -and $plan[0].requested_seconds -eq 1.5) 'Schedule was not sorted numerically'
Assert-KeyTap ($plan[1].key -eq 'LEFT' -and $plan[2].key -eq 'J') 'Key names were not normalized'
Assert-KeyTap ($plan[2].virtual_key -eq 0x4a) 'J did not map to a Win32 virtual J key'
Assert-KeyTap ($plan[3].sequence -lt $plan[4].sequence) 'Equal-time taps lost their explicit input order'
Assert-KeyTap (@(ConvertTo-BenchKeyTapPlan @() 60).Count -eq 0) 'An empty opt-in schedule created a tap'
foreach ($bad in @('J@-1', 'J@NaN', 'J@Infinity', 'J@60', 'J@', '@25', 'J@1@2', 'J@1,,J@2', 'WIN@1', 'J@1e3')) {
    Assert-KeyTapThrows { ConvertTo-BenchKeyTapPlan @($bad) 60 } "Accepted invalid schedule: $bad"
}
Assert-KeyTapThrows { ConvertTo-BenchKeyTapPlan @('J@1') 0 } 'Accepted zero benchmark duration'
Assert-KeyTap ((Get-BenchKeyTapCpuPercent 3.0 3.0) -eq 100.0) 'A full core over a hold-extended three-second sample must report 100%, not 300%'
Assert-KeyTap ((Get-BenchKeyTapCpuPercent 0.3 3.0) -eq 10.0) 'CPU idle time during a hold was lost from the sample denominator'
Assert-KeyTap ((Get-BenchKeyTapCpuPercent 0.0 0.0) -eq 0.0) 'Initial sample divided by zero elapsed time'

$script:keyTapTestState = @{ elapsed = 2.0; window = 123; focus = $true; foreground = $true;
    alive = $true; failDown = $false; failUp = $false; failHold = $false; loseFocus = $false; translateLosesFocus = $false;
    downs = 0; ups = 0; waits = @(); focusCalls = 0; keySpecs = 0 }
$adapter = [pscustomobject]@{
    Clock = { param($Started) $script:keyTapTestState.elapsed }
    Alive = { param($Process, $Started) $script:keyTapTestState.alive }
    FindWindow = { param($TargetId) $script:keyTapTestState.window }
    Focus = { param($TargetId, $Window) $script:keyTapTestState.focusCalls++; $script:keyTapTestState.focus }
    Foreground = { param($TargetId, $Window) $script:keyTapTestState.foreground }
    KeySpec = { param($VirtualKey, $Window) $script:keyTapTestState.keySpecs++;
        if ($script:keyTapTestState.translateLosesFocus) { $script:keyTapTestState.foreground = $false }
        [pscustomobject]@{ ScanCode = 0x24; Extended = $false } }
    Send = { param($TargetId, $Window, $Spec, $Down)
        if ($Down) { $script:keyTapTestState.downs++; if ($script:keyTapTestState.failDown) { return 0 } }
        else { $script:keyTapTestState.ups++; if ($script:keyTapTestState.failUp) { return 0 } }
        return 1 }
    Wait = { param($Milliseconds)
        $script:keyTapTestState.waits += $Milliseconds
        $script:keyTapTestState.elapsed += $Milliseconds / 1000.0
        if ($script:keyTapTestState.loseFocus) { $script:keyTapTestState.foreground = $false }
        if ($script:keyTapTestState.failHold) { throw 'Interrupted hold' } }
}
$fakeProcess = [pscustomobject]@{ Id = 456; StartTime = [DateTime]::UtcNow; HasExited = $false }
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('kyty-key-taps-test-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    $session = New-BenchKeyTaps $testRoot 'sent' $fakeProcess @(ConvertTo-BenchKeyTapPlan @('J@1,J@10') 60) 120 $adapter
    Update-BenchKeyTaps $session
    Complete-BenchKeyTaps $session
    $manifest = Get-Content -LiteralPath $session.Manifest -Raw | ConvertFrom-Json
    Assert-KeyTap ($manifest.taps[0].status -eq 'sent' -and $manifest.taps[1].status -eq 'not-reached') 'Due/future taps reported wrong status'
    Assert-KeyTap ($manifest.taps[0].target_pid -eq 456 -and $manifest.taps[0].hwnd -eq '0x7b') 'Exact target identity was not logged'
    Assert-KeyTap ($manifest.taps[0].actual_seconds -ge 2 -and $manifest.taps[0].completed_seconds -gt $manifest.taps[0].actual_seconds) 'Actual timing was not logged'
    Assert-KeyTap ($manifest.taps[0].hold_ms_actual -ge 120 -and $manifest.taps[0].overhead_ms -ge 0) 'Hold/overhead were not logged'
    Assert-KeyTap ($script:keyTapTestState.downs -eq 1 -and $script:keyTapTestState.ups -eq 1 -and $script:keyTapTestState.waits[0] -eq 120) 'Tap did not hold and release exactly once'
    Update-BenchKeyTaps $session
    Assert-KeyTap ($script:keyTapTestState.downs -eq 1) 'A completed tap was replayed'
    Assert-KeyTapThrows { New-BenchKeyTaps $testRoot 'sent' $fakeProcess $plan 120 $adapter } 'Existing tap manifest was overwritten'
    Assert-KeyTapThrows { New-BenchKeyTaps $testRoot 'short' $fakeProcess $plan 119 $adapter } 'Hold below 120ms was accepted'

    foreach ($scenario in @('missing-window', 'focus-denied', 'foreground-mismatch', 'process-exited', 'translate-focus-lost', 'keydown-failed', 'keyup-failed', 'hold-error', 'focus-lost')) {
        $script:keyTapTestState.elapsed = 2.0
        $script:keyTapTestState.window = 123
        $script:keyTapTestState.focus = $true
        $script:keyTapTestState.foreground = $true
        $script:keyTapTestState.alive = $true
        $script:keyTapTestState.failDown = $false
        $script:keyTapTestState.failUp = $false
        $script:keyTapTestState.failHold = $false
        $script:keyTapTestState.loseFocus = $false
        $script:keyTapTestState.translateLosesFocus = $false
        $script:keyTapTestState.downs = 0
        $script:keyTapTestState.ups = 0
        switch ($scenario) {
            'missing-window' { $script:keyTapTestState.window = 0 }
            'focus-denied' { $script:keyTapTestState.focus = $false }
            'foreground-mismatch' { $script:keyTapTestState.foreground = $false }
            'process-exited' { $script:keyTapTestState.alive = $false }
            'translate-focus-lost' { $script:keyTapTestState.translateLosesFocus = $true }
            'keydown-failed' { $script:keyTapTestState.failDown = $true }
            'keyup-failed' { $script:keyTapTestState.failUp = $true }
            'hold-error' { $script:keyTapTestState.failHold = $true }
            'focus-lost' { $script:keyTapTestState.loseFocus = $true }
        }
        $failure = New-BenchKeyTaps $testRoot $scenario $fakeProcess @(ConvertTo-BenchKeyTapPlan @('J@1') 60) 120 $adapter
        Update-BenchKeyTaps $failure
        Complete-BenchKeyTaps $failure
        $tap = (Get-Content -LiteralPath $failure.Manifest -Raw | ConvertFrom-Json).taps[0]
        Assert-KeyTap ($tap.status -ne 'sent') "$scenario was incorrectly reported as sent"
        Assert-KeyTap ($null -ne $tap.actual_seconds -and $tap.completed_seconds -ge $tap.actual_seconds) "$scenario lost the real attempt time"
        if ($scenario -in @('missing-window', 'focus-denied', 'foreground-mismatch', 'process-exited', 'translate-focus-lost')) {
            Assert-KeyTap ($script:keyTapTestState.downs -eq 0 -and $script:keyTapTestState.ups -eq 0) "$scenario sent input to an unverified window"
        } else {
            Assert-KeyTap ($script:keyTapTestState.ups -eq 1) "$scenario failed to release its attempted key in finally"
        }
    }

    # Exercise the real between-sample scheduler with only fake focus/input.
    # Its wall clock is real, while no native event is inserted.
    $script:keyTapTestState.focus = $true
    $script:keyTapTestState.foreground = $true
    $script:keyTapTestState.alive = $true
    $script:keyTapTestState.failDown = $false
    $script:keyTapTestState.failUp = $false
    $script:keyTapTestState.failHold = $false
    $script:keyTapTestState.loseFocus = $false
    $script:keyTapTestState.translateLosesFocus = $false
    $script:keyTapTestState.downs = 0
    $script:keyTapTestState.ups = 0
    $script:keyTapSchedulerClock = [Diagnostics.Stopwatch]::StartNew()
    $schedulerAdapter = $adapter.PSObject.Copy()
    $schedulerAdapter.Clock = { param($Started) $script:keyTapSchedulerClock.Elapsed.TotalSeconds }
    $scheduled = New-BenchKeyTaps $testRoot 'scheduler' $fakeProcess @(ConvertTo-BenchKeyTapPlan @('J@0.03,J@10') 60) 120 $schedulerAdapter
    Wait-BenchKeyTaps $scheduled 60
    Assert-KeyTap ($scheduled.Next -eq 1 -and $scheduled.Records[0].actual_seconds -ge 0.03 -and
        $scheduled.Records[0].status -eq 'sent' -and $scheduled.Records[1].status -eq 'not-reached') 'Between-sample scheduler fired before the deadline or missed the due tap'
    Assert-KeyTap ($script:keyTapTestState.downs -eq 1 -and $script:keyTapTestState.ups -eq 1) 'Between-sample scheduler duplicated an input edge'

    # Native helper compilation and a real headless-process lookup. There is no
    # SDL window for this PowerShell PID, so no focus or SendInput is attempted.
    $native = New-BenchKeyTapNativeAdapter
    Assert-KeyTap ([KytyBenchKeyInput]::InputSize() -eq $(if ([IntPtr]::Size -eq 8) { 40 } else { 28 })) 'Win32 INPUT ABI size is wrong'
    Assert-KeyTap ((& ($native.FindWindow) $PID) -eq 0) 'Headless PowerShell unexpectedly matched an SDL window'
    Assert-KeyTap (-not (& ($native.Foreground) $PID 0)) 'Zero HWND was accepted as foreground'
    $nativeSpec = New-Object 'KytyBenchKeyInput+KeySpec'
    $nativeSpec.ScanCode = 0x24
    Assert-KeyTap ((& ($native.Send) $PID 0 $nativeSpec $true) -eq 0) 'Native zero-HWND key-down guard did not return before SendInput'
    $headless = New-BenchKeyTaps $testRoot 'native-headless' (Get-Process -Id $PID) @(ConvertTo-BenchKeyTapPlan @('J@0') 60) 120 $native
    Update-BenchKeyTaps $headless
    Complete-BenchKeyTaps $headless
    Assert-KeyTap ($headless.Records[0].status -eq 'window-missing' -and $headless.Records[0].keydown_events -eq 0) 'Native missing-window route did not fail closed'
    Write-Host 'PASS: schedule parser, exact PID/foreground gates, timing manifest, finally release, native INPUT ABI and headless-window rejection'
} finally {
    $resolved = [IO.Path]::GetFullPath($testRoot)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if (-not $resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path -Leaf $resolved) -notlike 'kyty-key-taps-test-*') { throw 'Unexpected key-tap cleanup path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
