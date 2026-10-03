# Opt-in bench_boot keyboard scheduler. SendInput inserts events into the OS
# input stream; the manifest records insertion/focus, not receipt by the game.
function ConvertTo-BenchKeyTapPlan([string[]]$KeyTaps, [double]$Seconds) {
    if ($Seconds -le 0 -or [double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds)) {
        throw '-Seconds must be positive for a key-tap schedule'
    }
    $named = @{ ENTER = 0x0d; RETURN = 0x0d; SPACE = 0x20; ESC = 0x1b; ESCAPE = 0x1b;
        LEFT = 0x25; UP = 0x26; RIGHT = 0x27; DOWN = 0x28; LSHIFT = 0xa0; LCTRL = 0xa2 }
    $sequence = 0
    $plan = @()
    foreach ($group in $KeyTaps) {
        foreach ($entry in $group.Split(',')) {
            if ($entry.Trim() -notmatch '^([A-Za-z0-9]+)@([0-9]+(?:\.[0-9]+)?|\.[0-9]+)$') {
                throw "Invalid -KeyTaps entry '$entry'; use KEY@SECONDS, for example J@25"
            }
            $key = $Matches[1].ToUpperInvariant()
            $time = [double]::Parse($Matches[2], [Globalization.CultureInfo]::InvariantCulture)
            if ($time -ge $Seconds -or [double]::IsInfinity($time)) {
                throw "Key tap '$entry' must be nonnegative and earlier than -Seconds"
            }
            if ($key -match '^[A-Z0-9]$') { $virtualKey = [int][char]$key }
            elseif ($named.ContainsKey($key)) { $virtualKey = $named[$key] }
            else { throw "Unsupported key '$key'; use letters, digits, ENTER, SPACE, ESC, arrows, LSHIFT or LCTRL" }
            if ($key -eq 'RETURN') { $key = 'ENTER' }
            if ($key -eq 'ESCAPE') { $key = 'ESC' }
            $plan += [pscustomobject]@{ key = $key; virtual_key = [int]$virtualKey;
                requested_seconds = $time; sequence = $sequence }
            $sequence++
        }
    }
    # Preserve explicit repeated taps and their input order at the same second.
    $plan | Sort-Object requested_seconds, sequence
}

function Initialize-BenchKeyTapNative {
    if ('KytyBenchKeyInput' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class KytyBenchKeyInput {
    delegate bool EnumProc(IntPtr hwnd, IntPtr arg);
    [StructLayout(LayoutKind.Sequential)] struct Rect { public int left, top, right, bottom; }
    [StructLayout(LayoutKind.Sequential)] struct MouseInput {
        public int dx, dy; public uint mouseData, flags, time; public UIntPtr extraInfo;
    }
    [StructLayout(LayoutKind.Sequential)] struct KeyboardInput {
        public ushort virtualKey, scan; public uint flags, time; public UIntPtr extraInfo;
    }
    [StructLayout(LayoutKind.Sequential)] struct HardwareInput {
        public uint message; public ushort paramLow, paramHigh;
    }
    [StructLayout(LayoutKind.Explicit)] struct InputUnion {
        [FieldOffset(0)] public MouseInput mouse;
        [FieldOffset(0)] public KeyboardInput keyboard;
        [FieldOffset(0)] public HardwareInput hardware;
    }
    [StructLayout(LayoutKind.Sequential)] struct Input { public uint type; public InputUnion data; }
    public struct KeySpec { public ushort ScanCode; public bool Extended; }
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc callback, IntPtr arg);
    [DllImport("user32.dll")] static extern bool IsWindow(IntPtr hwnd);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll")] static extern bool IsIconic(IntPtr hwnd);
    [DllImport("user32.dll")] static extern bool GetClientRect(IntPtr hwnd, out Rect rect);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr hwnd, StringBuilder name, int length);
    [DllImport("user32.dll")] static extern bool ShowWindowAsync(IntPtr hwnd, int command);
    [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr hwnd);
    [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] static extern IntPtr GetKeyboardLayout(uint thread);
    [DllImport("user32.dll")] static extern uint MapVirtualKeyEx(uint key, uint mode, IntPtr layout);
    [DllImport("user32.dll", SetLastError=true)] static extern uint SendInput(uint count, [In] Input[] inputs, int size);
    public static int InputSize() { return Marshal.SizeOf(typeof(Input)); }
    static bool ExactWindow(int processId, IntPtr hwnd) {
        if (hwnd == IntPtr.Zero || !IsWindow(hwnd) || !IsWindowVisible(hwnd)) return false;
        uint owner; GetWindowThreadProcessId(hwnd, out owner);
        if (owner != (uint)processId) return false;
        var name = new StringBuilder(256);
        GetClassName(hwnd, name, name.Capacity);
        return name.ToString().StartsWith("SDL", StringComparison.Ordinal);
    }
    public static long FindWindow(int processId) {
        IntPtr target = IntPtr.Zero;
        long area = 0;
        EnumWindows(delegate(IntPtr hwnd, IntPtr arg) {
            if (!ExactWindow(processId, hwnd)) return true;
            Rect rect;
            if (!GetClientRect(hwnd, out rect)) return true;
            long size = (long)(rect.right - rect.left) * (rect.bottom - rect.top);
            if (size > area) { target = hwnd; area = size; }
            return true;
        }, IntPtr.Zero);
        return target.ToInt64();
    }
    public static bool Foreground(int processId, long handle) {
        var hwnd = new IntPtr(handle);
        return ExactWindow(processId, hwnd) && !IsIconic(hwnd) && GetForegroundWindow() == hwnd;
    }
    public static bool Focus(int processId, long handle) {
        var hwnd = new IntPtr(handle);
        if (!ExactWindow(processId, hwnd)) return false;
        if (IsIconic(hwnd)) ShowWindowAsync(hwnd, 9); // SW_RESTORE
        SetForegroundWindow(hwnd);
        // Windows may reject a foreground request. Never infer focus from the
        // request alone, and do not send Alt or keys to unrelated applications.
        return Foreground(processId, handle);
    }
    public static KeySpec Translate(int virtualKey, long handle) {
        uint owner;
        uint thread = GetWindowThreadProcessId(new IntPtr(handle), out owner);
        if (thread == 0) throw new InvalidOperationException("Target window disappeared before key translation");
        uint scan = MapVirtualKeyEx((uint)virtualKey, 4, GetKeyboardLayout(thread)); // MAPVK_VK_TO_VSC_EX
        if ((scan & 0xff) == 0) throw new InvalidOperationException("Key has no target-layout scan code");
        return new KeySpec { ScanCode = (ushort)(scan & 0xff), Extended = (scan & 0xff00) != 0 };
    }
    public static int Send(int processId, long handle, KeySpec spec, bool down) {
        // Recheck immediately before DOWN; SendInput routes to the foreground
        // window. UP always releases the exact scan code attempted by this tap,
        // including focus loss during the hold, so no synthetic key stays held.
        if (down && !Foreground(processId, handle)) return 0;
        var input = new Input();
        input.type = 1; // INPUT_KEYBOARD
        input.data.keyboard.scan = spec.ScanCode;
        input.data.keyboard.flags = 0x0008u | (spec.Extended ? 0x0001u : 0u) | (down ? 0u : 0x0002u);
        return (int)SendInput(1, new Input[] { input }, InputSize());
    }
}
'@
}

function New-BenchKeyTapNativeAdapter {
    Initialize-BenchKeyTapNative
    return [pscustomobject]@{
        Clock = { param($Started) ([DateTime]::UtcNow - $Started).TotalSeconds }
        Alive = { param($Process, $Started)
            try {
                $current = Get-Process -Id $Process.Id -ErrorAction Stop
                return -not $current.HasExited -and
                    [Math]::Abs(($current.StartTime.ToUniversalTime() - $Started).TotalMilliseconds) -le 1
            } catch { return $false } }
        FindWindow = { param($TargetId) [KytyBenchKeyInput]::FindWindow($TargetId) }
        Focus = { param($TargetId, $Window) [KytyBenchKeyInput]::Focus($TargetId, $Window) }
        Foreground = { param($TargetId, $Window) [KytyBenchKeyInput]::Foreground($TargetId, $Window) }
        KeySpec = { param($VirtualKey, $Window) [KytyBenchKeyInput]::Translate($VirtualKey, $Window) }
        Send = { param($TargetId, $Window, $Spec, $Down) [KytyBenchKeyInput]::Send($TargetId, $Window, $Spec, $Down) }
        Wait = { param($Milliseconds) Start-Sleep -Milliseconds $Milliseconds }
    }
}

function New-BenchKeyTaps($OutDir, $Tag, $Process, $Plan, [int]$HoldMs = 120, $Adapter = $null) {
    if ($HoldMs -lt 120 -or $HoldMs -gt 2000) { throw '-KeyTapHoldMs must be in 120..2000' }
    $manifest = Join-Path $OutDir "key-taps-$Tag.json"
    if (Test-Path -LiteralPath $manifest) { throw "Key-tap manifest already exists: $manifest (choose a new -OutDir)" }
    if ($null -eq $Adapter) { $Adapter = New-BenchKeyTapNativeAdapter }
    $records = @($Plan | ForEach-Object {
        [pscustomobject]@{ key = $_.key; virtual_key = $_.virtual_key; sequence = $_.sequence
            requested_seconds = $_.requested_seconds; actual_seconds = $null; keydown_seconds = $null; completed_seconds = $null
            hold_ms_requested = $HoldMs; hold_ms_actual = $null; overhead_ms = $null
            target_pid = $Process.Id; hwnd = $null; foreground_verified = $false
            keydown_events = 0; keyup_events = 0; status = 'not-reached'; error = 'Tap time was not reached' }
    })
    $session = [pscustomobject]@{ Manifest = $manifest; Records = $records; Process = $Process
        Started = $Process.StartTime.ToUniversalTime(); Next = 0; Tag = $Tag; Adapter = $Adapter; HoldMs = $HoldMs }
    Save-BenchKeyTapManifest $session
    return $session
}

function Save-BenchKeyTapManifest($Session) {
    [ordered]@{ schema_version = 1; alignment = 'elapsed-process-time'; run = $Session.Tag
        target_pid = $Session.Process.Id; started_utc = $Session.Started.ToString('o')
        host_input_min_press_ms = $Session.HoldMs; host_input_only = $true; taps = @($Session.Records)
        note = 'sent means OS events were inserted with target foreground verified; inspect captures/input trace to confirm gameplay. Focus and holds add overhead.'
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Session.Manifest -Encoding UTF8
}

function Invoke-BenchKeyTap($Session, $Row) {
    $adapter = $Session.Adapter
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $attemptedDown = $false
    $spec = $null
    $window = 0L
    try {
        $Row.status = 'pending'
        $Row.error = 'Tap has not completed'
        $Row.actual_seconds = [double](& ($adapter.Clock) $Session.Started)
        Save-BenchKeyTapManifest $Session
        if (-not (& ($adapter.Alive) $Session.Process $Session.Started)) {
            $Row.status = 'process-exited'; $Row.error = 'Target process exited or its PID was reused'; return
        }
        $window = [long](& ($adapter.FindWindow) $Session.Process.Id)
        $Row.hwnd = '0x{0:x}' -f $window
        if ($window -eq 0) { $Row.status = 'window-missing'; $Row.error = 'No visible SDL client for the target PID'; return }
        if (-not (& ($adapter.Focus) $Session.Process.Id $window) -or
            -not (& ($adapter.Foreground) $Session.Process.Id $window)) {
            $Row.status = 'focus-denied'; $Row.error = 'The exact target window could not be verified as foreground'; return
        }
        # Translation uses the target window's keyboard layout and physical scan
        # events, rather than Unicode WM_CHAR packets that SDL may not treat as keys.
        $spec = & ($adapter.KeySpec) $Row.virtual_key $window
        if (-not (& ($adapter.Alive) $Session.Process $Session.Started) -or
            -not (& ($adapter.Foreground) $Session.Process.Id $window)) {
            $Row.status = 'focus-denied'; $Row.error = 'Target identity or foreground changed before key-down'; return
        }
        $Row.foreground_verified = $true
        $Row.keydown_seconds = [double](& ($adapter.Clock) $Session.Started)
        $attemptedDown = $true
        $Row.keydown_events = [int](& ($adapter.Send) $Session.Process.Id $window $spec $true)
        if ($Row.keydown_events -ne 1) { $Row.status = 'input-error'; $Row.error = 'SendInput did not insert key-down'; return }
        & ($adapter.Wait) $Session.HoldMs
        $Row.hold_ms_actual = [Math]::Round(([double](& ($adapter.Clock) $Session.Started) - $Row.keydown_seconds) * 1000, 3)
        if (-not (& ($adapter.Alive) $Session.Process $Session.Started) -or
            -not (& ($adapter.Foreground) $Session.Process.Id $window)) {
            $Row.status = 'focus-lost'; $Row.error = 'Target foreground or process identity was lost during the hold'; return
        }
        $Row.status = 'sent'; $Row.error = $null
    } catch {
        $Row.status = 'error'; $Row.error = $_.Exception.Message
    } finally {
        if ($attemptedDown) {
            try {
                $Row.keyup_events = [int](& ($adapter.Send) $Session.Process.Id $window $spec $false)
                if ($Row.keyup_events -ne 1) { $Row.status = 'release-error'; $Row.error = 'SendInput did not insert key-up' }
            } catch { $Row.status = 'release-error'; $Row.error = $_.Exception.Message }
        }
        $Row.completed_seconds = [double](& ($adapter.Clock) $Session.Started)
        if ($attemptedDown -and $null -eq $Row.hold_ms_actual) {
            $Row.hold_ms_actual = [Math]::Round(($Row.completed_seconds - $Row.keydown_seconds) * 1000, 3)
        }
        $Row.overhead_ms = [Math]::Round($clock.Elapsed.TotalMilliseconds, 3)
        Save-BenchKeyTapManifest $Session
        Write-Host ("[{0}] key {1}@{2}s: {3} (actual {4}s, PID {5}, HWND {6})" -f
            $Session.Tag, $Row.key, $Row.requested_seconds, $Row.status,
            $Row.actual_seconds, $Row.target_pid, $Row.hwnd)
    }
}

function Update-BenchKeyTaps($Session) {
    while ($Session.Next -lt $Session.Records.Count) {
        $row = $Session.Records[$Session.Next]
        $elapsed = [double](& ($Session.Adapter.Clock) $Session.Started)
        if ($row.requested_seconds -gt $elapsed) { break }
        $Session.Next++
        Invoke-BenchKeyTap $Session $row
    }
}

function Wait-BenchKeyTaps($Session, [int]$Milliseconds = 1000) {
    # Preserve the benchmark's roughly once-per-second CPU sampling while waking
    # for tap deadlines between samples. Holds count toward this wait budget.
    $clock = [Diagnostics.Stopwatch]::StartNew()
    while ($clock.Elapsed.TotalMilliseconds -lt $Milliseconds) {
        Update-BenchKeyTaps $Session
        $remaining = $Milliseconds - $clock.Elapsed.TotalMilliseconds
        if ($remaining -le 0) { break }
        $sleep = $remaining
        if ($Session.Next -lt $Session.Records.Count) {
            $elapsed = [double](& ($Session.Adapter.Clock) $Session.Started)
            $untilTap = ($Session.Records[$Session.Next].requested_seconds - $elapsed) * 1000
            $sleep = [Math]::Min($sleep, [Math]::Max(1, $untilTap))
        }
        Start-Sleep -Milliseconds ([int][Math]::Ceiling($sleep))
    }
    Update-BenchKeyTaps $Session
}

function Complete-BenchKeyTaps($Session) {
    Save-BenchKeyTapManifest $Session
    Write-Host "key taps: $($Session.Manifest) (inspect captures to verify gameplay)"
}

function Get-BenchKeyTapCpuPercent([double]$CpuSeconds, [double]$WallSeconds) {
    if ($WallSeconds -le 0) { return 0.0 }
    return $CpuSeconds / $WallSeconds * 100.0
}
