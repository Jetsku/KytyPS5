#include "graphics/host_gpu/renderer/hazardTracker.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Libs::Graphics::Hazard;

static void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "FAILED: %s\n", message);
		std::exit(1);
	}
}

static Range G(uint64_t lo, uint64_t hi, bool write) {
	return {0, lo, hi, write};
}

static Range H(uint64_t handle, uint64_t lo, uint64_t hi, bool write) {
	return {handle, lo, hi, write};
}

static Verdict CheckOne(const Tracker& tracker, Range range) {
	return tracker.Check(std::span<const Range>(&range, 1));
}

static void AddOne(Tracker& tracker, Range range) {
	tracker.Add(std::span<const Range>(&range, 1));
}

// The masks the renderer queues (shaderResourceBarrier.cpp, graphicsRun.cpp, renderDraw.cpp).
constexpr uint64_t AllCommands = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
constexpr uint64_t Compute     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
constexpr uint64_t AllGraphics = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
constexpr uint64_t Transfer    = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
constexpr uint64_t DrawIndirect = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
constexpr uint64_t MemRead     = VK_ACCESS_2_MEMORY_READ_BIT;
constexpr uint64_t MemWrite    = VK_ACCESS_2_MEMORY_WRITE_BIT;
constexpr uint64_t ShRead      = VK_ACCESS_2_SHADER_READ_BIT;
constexpr uint64_t ShWrite     = VK_ACCESS_2_SHADER_WRITE_BIT;
constexpr uint64_t XferWrite   = VK_ACCESS_2_TRANSFER_WRITE_BIT;
constexpr uint64_t IndRead     = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

static void TestOverlaps() {
	Tracker t;
	Check(t.Empty(), "a new tracker is empty");
	AddOne(t, G(0x1000, 0x2000, true));  // chain wrote [0x1000, 0x2000)
	AddOne(t, G(0x8000, 0x9000, false)); // chain read [0x8000, 0x9000)
	Check(CheckOne(t, G(0x1800, 0x1900, false)) == Verdict::Raw, "read inside a written range is RAW");
	Check(CheckOne(t, G(0x0, 0x1001, false)) == Verdict::Raw, "read overlapping the first written byte is RAW");
	Check(CheckOne(t, G(0x1fff, 0x3000, true)) == Verdict::Waw, "write overlapping the last written byte is WAW");
	Check(CheckOne(t, G(0x8800, 0x8801, true)) == Verdict::War, "write inside a read range is WAR");
	Check(CheckOne(t, G(0x8000, 0x9000, false)) == Verdict::Free, "read after read is free");
	Check(CheckOne(t, G(0x2000, 0x8000, true)) == Verdict::Free, "ranges that only touch are free");
	Check(CheckOne(t, G(0x0, 0x1000, true)) == Verdict::Free, "a range ending at the written start is free");
	// One access with several ranges: the RAW wins over a WAR found first.
	const std::vector<Range> both {G(0x8000, 0x8100, true), G(0x1000, 0x1100, false)};
	Check(t.Check(both) == Verdict::Raw, "RAW reported before WAR");
	const std::vector<Range> ww {G(0x8000, 0x8100, true), G(0x1000, 0x1100, true)};
	Check(t.Check(ww) == Verdict::Waw, "WAW reported before WAR");
	t.Reset();
	Check(t.Empty() && CheckOne(t, G(0x1000, 0x2000, true)) == Verdict::Free, "reset forgets the chain");
}

static void TestAliasing() {
	// A dispatch writes a storage buffer over guest [0x10000, 0x20000) (host buffer 7); the next one
	// samples an image whose guest data lies inside it (host image 9). The host handles differ, so
	// only the guest space sees the read-after-write.
	Tracker guest;
	Tracker host;
	AddOne(guest, G(0x10000, 0x20000, true));
	AddOne(host, H(7, 0x400, 0x10400, true));
	Check(CheckOne(guest, G(0x18000, 0x19000, false)) == Verdict::Raw, "image over written buffer memory: RAW");
	Check(CheckOne(host, H(9, 0, UINT64_MAX, false)) == Verdict::Free, "host check alone misses the alias");
	// Two images over one memory range (different handles): an image write against an image read.
	Tracker images;
	AddOne(images, G(0x50000, 0x60000, false));
	Check(CheckOne(images, G(0x5f000, 0x70000, true)) == Verdict::War, "second image writing read memory: WAR");
	// One host buffer holding two disjoint guest ranges (a joined buffer): the guest ranges decide,
	// the host check of the same bytes agrees, and different host offsets in one buffer are free.
	Tracker joined_guest;
	Tracker joined_host;
	AddOne(joined_guest, G(0x1000, 0x2000, true));
	AddOne(joined_host, H(5, 0x0, 0x1000, true));
	Check(CheckOne(joined_guest, G(0x3000, 0x4000, true)) == Verdict::Free, "disjoint guest ranges of one buffer");
	Check(CheckOne(joined_host, H(5, 0x2000, 0x3000, true)) == Verdict::Free, "disjoint host ranges of one buffer");
	Check(CheckOne(joined_host, H(5, 0x800, 0x900, false)) == Verdict::Raw, "same host bytes: RAW");
	// Spaces never compare across: a host range at guest-looking offsets is not a guest range.
	Check(!Overlap(G(0x1000, 0x2000, true), H(0x1000, 0x1000, 0x2000, true)), "spaces are separate");
	// A whole image (host range 0..max) against a later binding of the same image.
	Tracker whole;
	AddOne(whole, H(11, 0, UINT64_MAX, true));
	Check(CheckOne(whole, H(11, 0, UINT64_MAX, false)) == Verdict::Raw, "same host image written then read");
	Check(CheckOne(whole, H(12, 0, UINT64_MAX, true)) == Verdict::Free, "another host image");
}

static void TestIndirectArguments() {
	// Indirect argument records are reads of their guest range: a chain dispatch that wrote the
	// records conflicts, one that wrote next to them does not.
	Tracker t;
	AddOne(t, G(0x40000, 0x40100, true)); // culling output: draw records
	Check(CheckOne(t, G(0x40020, 0x40034, false)) == Verdict::Raw, "args inside written records: RAW");
	Check(CheckOne(t, G(0x40100, 0x40114, false)) == Verdict::Free, "args after the written records: free");
	Check(CheckOne(t, G(0x3fff0, 0x40000, false)) == Verdict::Free, "args before the written records: free");
}

static void TestCapacity() {
	Tracker t;
	std::vector<Range> many(Tracker::Capacity, G(0x0, 0x1, false));
	for (size_t i = 0; i < many.size(); i++) {
		many[i] = G(i * 16, i * 16 + 8, false);
	}
	t.Add(many);
	Check(!t.Poisoned() && t.Size() == Tracker::Capacity, "exactly the capacity fits");
	AddOne(t, G(0x100000, 0x100008, false));
	Check(t.Poisoned(), "past the capacity the tracker is poisoned");
	Check(CheckOne(t, G(0x900000, 0x900008, false)) != Verdict::Free, "a poisoned tracker reports a hazard");
	t.Reset();
	Check(!t.Poisoned() && t.Empty(), "reset clears the poison");
}

static void TestChainReset() {
	// ShaderAccess (after a dispatch): compute -> all commands, shader read|write -> memory read|write.
	Check(ChainReset(Compute, ShRead | ShWrite, AllCommands, MemRead | MemWrite) == Reset::Compute,
	      "ShaderAccess restarts an intact chain");
	// ShaderWriteHazard (before a writing dispatch): all -> compute, memory -> shader.
	Check(ChainReset(AllCommands, MemRead | MemWrite, Compute, ShRead | ShWrite) == Reset::Full,
	      "ShaderWriteHazard is a full reset for compute");
	// Their union (the batch.mixed barrier at a dispatch) and a guest global barrier.
	Check(ChainReset(AllCommands | Compute, MemRead | MemWrite | ShRead | ShWrite, AllCommands | Compute,
	                 MemRead | MemWrite | ShRead | ShWrite) == Reset::Full,
	      "the merged dispatch barrier is a full reset");
	Check(ChainReset(AllCommands, MemWrite, AllCommands, MemRead | MemWrite) == Reset::Full, "guest barrier");
	// Indirect argument barrier: never reaches compute shader accesses.
	Check(ChainReset(AllGraphics | Compute | Transfer, ShWrite | XferWrite, DrawIndirect, IndRead) == Reset::None,
	      "the indirect argument barrier does not reset");
	// Upload post barrier (transfer -> all): does not cover compute writes made available.
	Check(ChainReset(Transfer, XferWrite, AllCommands, MemRead | MemWrite) == Reset::None,
	      "an upload post barrier does not reset");
	// A read-only second scope does not order a later write after the chain's writes being visible
	// to it... it is still an execution dependency, but the reset demands read and write access.
	Check(ChainReset(AllCommands, MemWrite, AllCommands, MemRead) == Reset::None, "second scope must allow writes");
	Check(ChainReset(AllGraphics, ShWrite, AllCommands, MemRead | MemWrite) == Reset::None,
	      "a graphics-only first scope does not reset");
}

static void TestChain() {
	Chain c;
	uint64_t foreign = 0;
	const std::vector<Range> a_guest {G(0x1000, 0x2000, true)};
	const std::vector<Range> a_host {H(3, 0, 0x1000, true)};
	Check(!c.Intact(foreign), "no chain before a reset");
	// A dispatch before any reset: stays invalid.
	c.NoteDispatch(foreign, foreign + 1, true, a_guest, a_host);
	foreign++;
	Check(!c.Intact(foreign), "a dispatch alone does not start a chain");
	// A compute-only reset without an intact chain does nothing.
	c.OnFlush(foreign, true, Compute, ShRead | ShWrite, AllCommands, MemRead | MemWrite, false);
	Check(!c.Intact(foreign), "ShaderAccess cannot start a chain");
	// A full reset starts it.
	c.OnFlush(foreign, true, AllCommands, MemRead | MemWrite, Compute, ShRead | ShWrite, false);
	Check(c.Intact(foreign) && c.guest.Empty(), "a full reset starts an empty chain");
	// The dispatch command (Sink() bumps the foreign count once).
	c.NoteDispatch(foreign, foreign + 1, true, a_guest, a_host);
	foreign++;
	Check(c.Intact(foreign) && c.guest.Size() == 1, "the dispatch joins the chain");
	Check(c.guest.Check(std::span<const Range>(a_guest)) == Verdict::Waw, "the next dispatch sees the chain");
	// An imprecise dispatch poisons the chain: still intact, but every check is a hazard, until a
	// ShaderAccess flush restarts it clean.
	Chain d0 = c;
	d0.NoteDispatch(foreign, foreign + 1, false, {}, {}, false);
	Check(!d0.Intact(foreign + 1), "modes 1-3: an imprecise dispatch invalidates");
	Chain d = c;
	d.NoteDispatch(foreign, foreign + 1, false, {}, {});
	Check(d.Intact(foreign + 1), "an imprecise dispatch keeps the chain intact");
	Check(d.guest.Check(std::span<const Range>(a_host)) != Verdict::Free &&
	          d.guest.Check(std::span<const Range>()) != Verdict::Free,
	      "a poisoned chain reports a hazard for any access");
	d.OnFlush(foreign + 1, true, Compute, ShRead | ShWrite, AllCommands, MemRead | MemWrite, false);
	Check(d.Intact(foreign + 1) && d.guest.Empty() && !d.guest.Poisoned(), "ShaderAccess after it restarts a clean chain");
	// After a foreign command a poisoned chain is just invalid.
	Chain d2 = c;
	d2.NoteDispatch(foreign, foreign + 1, false, {}, {});
	d2.OnFlush(foreign + 2, true, Compute, ShRead | ShWrite, AllCommands, MemRead | MemWrite, false);
	Check(!d2.Intact(foreign + 2), "a foreign command after the imprecise dispatch invalidates");
	// Two commands recorded (a label, then the dispatch): not the chain's only command.
	Chain e = c;
	e.NoteDispatch(foreign + 1, foreign + 2, true, a_guest, a_host);
	Check(!e.Intact(foreign + 2), "another command before the dispatch invalidates");
	// Any foreign command (a draw, a copy) breaks intactness without a flush.
	Check(!c.Intact(foreign + 1), "a foreign command breaks the chain");
	// Image barriers recorded alone that are not the dispatch's own: invalidate.
	Chain f = c;
	f.OnFlush(foreign, false, 0, 0, 0, 0, true);
	Check(!f.Intact(foreign), "foreign side barriers invalidate");
	// The dispatch's own side barriers: the chain stays.
	Chain g = c;
	g.OnFlush(foreign, false, 0, 0, 0, 0, false);
	Check(g.Intact(foreign) && g.guest.Size() == 1, "own side barriers keep the chain");
	// A ShaderAccess flush of an intact chain restarts it empty.
	c.OnFlush(foreign, true, Compute, ShRead | ShWrite, AllCommands, MemRead | MemWrite, false);
	Check(c.Intact(foreign) && c.guest.Empty(), "ShaderAccess restarts an intact chain empty");
	// A flush of a barrier that does not reach compute keeps an intact chain as it is.
	c.NoteDispatch(foreign, foreign + 1, true, a_guest, a_host);
	foreign++;
	c.OnFlush(foreign, true, AllGraphics | Compute | Transfer, ShWrite | XferWrite, DrawIndirect, IndRead, false);
	Check(c.Intact(foreign) && c.guest.Size() == 1, "an unrelated memory barrier keeps the chain");
	// ... but not after a foreign command.
	c.OnFlush(foreign + 1, true, AllGraphics | Compute | Transfer, ShWrite | XferWrite, DrawIndirect, IndRead, false);
	Check(!c.Intact(foreign + 1), "a flush after a foreign command invalidates");
}

int main() {
	TestOverlaps();
	TestAliasing();
	TestIndirectArguments();
	TestCapacity();
	TestChainReset();
	TestChain();
	std::printf("gpu hazard tests passed\n");
	return 0;
}
