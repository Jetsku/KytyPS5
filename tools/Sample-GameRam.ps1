# Read-only Windows process RAM sampler. Private bytes and working set are byte counts;
# VirtualQueryEx committed regions are virtual extents, not an additive physical RAM total.
# Paging-file-backed guest views can alias one section. File-backed mapped/image pages and
# driver mappings also need separate attribution. Global system commit is context, not a
# process-only charge. CSV state/type/protection values and addresses are hexadecimal.
# Region totals update only with a map snapshot; region_age_ms records their age. Private
# bytes and working set remain fresh each sample, avoiding a full address-space walk each second.
# Peak counters come from the same GetProcessMemoryInfo call. Peak commit is PeakPagefileUsage;
# it excludes the separate guest shared-section commitment just as current private commit does.
# Large-region resident_estimate_bytes uses QueryWorkingSetEx at a 64 KiB stride; -1 means
# not sampled. Estimates include aliases and are not a count of unique physical pages.
# No guest bytes are read, no target threads are paused, no target memory is changed.
# Run alongside an existing harness-owned PID; this script exits when that PID exits.
# The sampler moves only its own process to BelowNormal priority and logical CPUs 16-31.
# Example: ./tools/Sample-GameRam.ps1 -GameProcessId 1234 -OutputDir ./ram-samples
param([Parameter(Mandatory=$true)][int]$GameProcessId,[Parameter(Mandatory=$true)][string]$OutputDir,[ValidateRange(250,60000)][int]$IntervalMs=1000,[ValidateRange(1,600)][int]$MapSeconds=10)
$ErrorActionPreference='Stop'
[IO.Directory]::CreateDirectory($OutputDir) | Out-Null
$me=Get-Process -Id $PID
try { $me.PriorityClass='BelowNormal'; $me.ProcessorAffinity=[IntPtr][int64]4294901760 } catch {}
Add-Type -TypeDefinition @"
using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics;
public static class RamSampler {
 static ulong[] cachedTotals=new ulong[6]; static int cachedCount;
 static DateTime lastMap=DateTime.MinValue;
 [StructLayout(LayoutKind.Sequential)] public struct MBI { public UIntPtr Base, Allocation; public uint AllocationProtect; public ushort Partition; public UIntPtr Size; public uint State, Protect, Type; }
 [StructLayout(LayoutKind.Sequential)] public struct PMC { public uint cb, Faults; public UIntPtr PeakWS, WS, PeakPaged, Paged, PeakNonPaged, NonPaged, Pagefile, PeakPagefile, Private; }
 [StructLayout(LayoutKind.Sequential)] public struct PERF { public uint cb; public UIntPtr Commit, CommitLimit, CommitPeak, PhysicalTotal, PhysicalAvailable, SystemCache, KernelTotal, KernelPaged, KernelNonPaged, PageSize; public uint Handles, Processes, Threads; }
 [DllImport("psapi.dll")] static extern bool GetPerformanceInfo(out PERF p,uint size);
 [DllImport("kernel32.dll")] static extern IntPtr OpenProcess(uint access,bool inherit,int pid);
 [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
 [DllImport("kernel32.dll")] static extern UIntPtr VirtualQueryEx(IntPtr h,UIntPtr p,out MBI m,UIntPtr size);
 [DllImport("psapi.dll")] static extern bool GetProcessMemoryInfo(IntPtr h,out PMC m,uint size);
 [DllImport("psapi.dll",CharSet=CharSet.Unicode)] static extern uint GetMappedFileName(IntPtr h,UIntPtr p,StringBuilder s,uint n);
 [StructLayout(LayoutKind.Sequential)] public struct WSENTRY {public UIntPtr Address, Flags;}
 [DllImport("psapi.dll")] static extern bool QueryWorkingSetEx(IntPtr h,[In,Out] WSENTRY[] entries,uint bytes);
 static long ResidentEstimate(IntPtr h,ulong b,ulong size) {
  if(size<1048576) return -1;
  const ulong stride=65536; ulong valid=0, total=0;
  for(ulong start=0;start<size;) {
   int count=(int)Math.Min(16384ul,(size-start+stride-1)/stride); var entries=new WSENTRY[count];
   for(int i=0;i<count;i++) entries[i].Address=new UIntPtr(b+start+(ulong)i*stride);
   if(!QueryWorkingSetEx(h,entries,(uint)(count*16))) return -1;
   for(int i=0;i<count;i++){ulong weight=Math.Min(stride,size-start-(ulong)i*stride); total+=weight;if((entries[i].Flags.ToUInt64()&1)!=0)valid+=weight;}
   start+=(ulong)count*stride;
  } return (long)valid;
 }
 public static void Sample(int pid,string dir,bool map) {
  IntPtr h=OpenProcess(0x410,false,pid); if(h==IntPtr.Zero) throw new Exception("OpenProcess failed");
  try {
   string stamp=DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fff"); PMC p; if(!GetProcessMemoryInfo(h,out p,(uint)Marshal.SizeOf(typeof(PMC)))) return;
   ulong[] totals=map?new ulong[6]:cachedTotals; int count=map?0:cachedCount; var watch=Stopwatch.StartNew();
   StreamWriter w=map?new StreamWriter(Path.Combine(dir,"regions-"+DateTime.Now.ToString("yyyyMMdd-HHmmss-fff")+".csv")):null;
   if(w!=null) w.WriteLine("base,allocation,size,state,type,protect,allocationProtect,resident_estimate_bytes,path");
   try {
    ulong a=0; MBI m; while(map && VirtualQueryEx(h,new UIntPtr(a),out m,new UIntPtr((uint)Marshal.SizeOf(typeof(MBI)))).ToUInt64()!=0) {
     ulong b=m.Base.ToUInt64(), size=m.Size.ToUInt64(); if(size==0 || b+size<=a) break;
     if(m.State!=0x10000) {
      int t=m.Type==0x20000?0:m.Type==0x40000?1:m.Type==0x1000000?2:-1;
      if(t>=0) totals[t+(m.State==0x1000?0:3)]+=size;
      if(w!=null) { string path=""; if(m.State==0x1000 && (m.Type==0x40000 || m.Type==0x1000000)) {var sb=new StringBuilder(1024); if(GetMappedFileName(h,m.Base,sb,1024)!=0) path=sb.ToString();}
       w.WriteLine("{0:X},{1:X},{2},{3:X},{4:X},{5:X},{6:X},{7},\"{8}\"",b,m.Allocation.ToUInt64(),size,m.State,m.Type,m.Protect,m.AllocationProtect,m.State==0x1000?ResidentEstimate(h,b,size):-1,path.Replace("\"","\"\"")); }
      count++;
     } a=b+size;
    }
   } finally {if(w!=null) w.Dispose();}
   if(map){cachedTotals=totals;cachedCount=count;lastMap=DateTime.UtcNow;}
   PERF global; GetPerformanceInfo(out global,(uint)Marshal.SizeOf(typeof(PERF)));
   string csv=Path.Combine(dir,"timeline.csv"); if(!File.Exists(csv)) File.WriteAllText(csv,"time,pid,private_bytes,working_set,pagefile,private_commit,mapped_commit,image_commit,private_reserve,mapped_reserve,image_reserve,regions,scan_ms,system_commit,physical_available,region_age_ms,peak_working_set,peak_commit\n");
   File.AppendAllText(csv,string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17}\n",stamp,pid,p.Private.ToUInt64(),p.WS.ToUInt64(),p.Pagefile.ToUInt64(),totals[0],totals[1],totals[2],totals[3],totals[4],totals[5],count,watch.ElapsedMilliseconds,global.Commit.ToUInt64()*global.PageSize.ToUInt64(),global.PhysicalAvailable.ToUInt64()*global.PageSize.ToUInt64(),(long)(DateTime.UtcNow-lastMap).TotalMilliseconds,p.PeakWS.ToUInt64(),p.PeakPagefile.ToUInt64()));
  } finally {CloseHandle(h);}
 }
}
"@
$me=Get-Process -Id $PID
try { $me.PriorityClass='BelowNormal'; $me.ProcessorAffinity=[IntPtr][int64]4294901760 } catch {}
$nextMap=Get-Date
$lastCheckpoint=''
while(Get-Process -Id $GameProcessId -ErrorAction SilentlyContinue) {
 $request=Join-Path $OutputDir 'checkpoint-request.txt'
 $checkpoint=if(Test-Path -LiteralPath $request){[IO.File]::ReadAllText($request)}else{''}
 $newCheckpoint=$checkpoint -and $checkpoint -ne $lastCheckpoint
 $map=(Get-Date) -ge $nextMap -or $newCheckpoint
 try { [RamSampler]::Sample($GameProcessId,$OutputDir,$map) } catch { $_ | Out-File (Join-Path $OutputDir 'errors.txt') -Append }
 if($map) {
  $nextMap=(Get-Date).AddSeconds($MapSeconds)
  if($newCheckpoint) {
   $snapshot=Get-ChildItem -LiteralPath $OutputDir -Filter 'regions-*.csv' | Sort-Object Name | Select-Object -Last 1
   Add-Content -LiteralPath (Join-Path $OutputDir 'checkpoint-snapshots.txt') -Value ("$checkpoint|$($snapshot.Name)")
   $lastCheckpoint=$checkpoint
  }
 }
 Start-Sleep -Milliseconds $IntervalMs
}
