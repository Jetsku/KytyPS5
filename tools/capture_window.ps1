<#
Private bench_boot worker. Captures only an SDL client owned by TargetPid using
PrintWindow; never copies the desktop. Parent enforces a five-second timeout.
Some Vulkan/driver combinations return black: compare_captures.py rejects those.
#>
[CmdletBinding()]
param([Parameter(Mandatory)][string]$RequestFile)
$ErrorActionPreference = 'Stop'
$request = Get-Content -LiteralPath $RequestFile -Raw | ConvertFrom-Json
$result = [ordered]@{
    requested_seconds = $request.requested_seconds
    actual_seconds = $null
    completed_seconds = $null
    status = 'error'
    image = $request.image
    method = 'PrintWindow-client-only'
    error = $null
}
try {
    Add-Type -AssemblyName System.Drawing
    Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;
public static class KytyClientCapture {
    delegate bool EnumProc(IntPtr hwnd, IntPtr arg);
    [StructLayout(LayoutKind.Sequential)] struct Rect { public int left, top, right, bottom; }
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc callback, IntPtr arg);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll")] static extern bool IsIconic(IntPtr hwnd);
    [DllImport("user32.dll")] static extern bool GetClientRect(IntPtr hwnd, out Rect rect);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr hwnd, StringBuilder text, int count);
    [DllImport("user32.dll")] static extern bool PrintWindow(IntPtr hwnd, IntPtr dc, uint flags);
    [DllImport("user32.dll")] static extern bool SetProcessDPIAware();
    public static void Save(int processId, string path) {
        SetProcessDPIAware();
        IntPtr target = IntPtr.Zero;
        int width = 0, height = 0;
        EnumWindows(delegate(IntPtr hwnd, IntPtr arg) {
            uint pid; GetWindowThreadProcessId(hwnd, out pid);
            if (pid != processId || !IsWindowVisible(hwnd) || IsIconic(hwnd)) return true;
            var name = new StringBuilder(256);
            GetClassName(hwnd, name, name.Capacity);
            if (!name.ToString().StartsWith("SDL", StringComparison.Ordinal)) return true;
            Rect rect;
            if (!GetClientRect(hwnd, out rect)) return true;
            int w = rect.right - rect.left, h = rect.bottom - rect.top;
            if ((long)w * h > (long)width * height) { target = hwnd; width = w; height = h; }
            return true;
        }, IntPtr.Zero);
        if (target == IntPtr.Zero || width <= 0 || height <= 0)
            throw new InvalidOperationException("No visible non-minimized SDL client for target PID");
        using (var bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb)) {
            using (var graphics = Graphics.FromImage(bitmap)) {
                graphics.Clear(Color.Black);
                IntPtr dc = graphics.GetHdc();
                try {
                    // PW_CLIENTONLY | PW_RENDERFULLCONTENT. No CopyFromScreen fallback.
                    if (!PrintWindow(target, dc, 3)) throw new InvalidOperationException("PrintWindow failed");
                } finally { graphics.ReleaseHdc(dc); }
            }
            bitmap.Save(path, ImageFormat.Bmp);
        }
    }
}
'@
    $target = Get-Process -Id $request.target_pid -ErrorAction Stop
    $start = [DateTime]::Parse($request.started_utc, [Globalization.CultureInfo]::InvariantCulture).ToUniversalTime()
    if ([Math]::Abs(($target.StartTime.ToUniversalTime() - $start).TotalMilliseconds) -gt 1) {
        throw 'Target PID was reused'
    }
    $result.actual_seconds = ([DateTime]::UtcNow - $start).TotalSeconds
    [KytyClientCapture]::Save($target.Id, (Join-Path (Split-Path -Parent $RequestFile) $request.image))
    $result.completed_seconds = ([DateTime]::UtcNow - $start).TotalSeconds
    $result.status = 'ok'
} catch {
    $result.error = $_.Exception.Message
} finally {
    $result | ConvertTo-Json | Set-Content -LiteralPath $request.result_file -Encoding UTF8
}
if ($result.status -ne 'ok') { exit 1 }
