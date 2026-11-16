#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HAZARDTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HAZARDTRACKER_H_

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <span>
#include <vector>

// KYTY_GPU_HAZARD (render.h GpuHazardMode): resource-precise hazards between guest dispatches.
//
// A "chain" is the dispatches recorded in one command buffer since the last barrier that ordered
// everything earlier before later compute shader accesses (ChainReset below), with nothing else
// recorded in between. The tracker holds the byte ranges those dispatches read and wrote. A later
// dispatch whose ranges have no read-after-write, write-after-read or write-after-write overlap with
// the chain needs no barrier against it: everything recorded before the chain is ordered against it
// by the barrier that started the chain, and the chain's dispatches touched none of its memory.
//
// Ranges live in "spaces": space 0 is guest addresses (every guest buffer and image binding, so two
// resources over one guest memory range - a buffer and an image, two images, a joined buffer - are
// compared by their memory, not by their handles); any other space is a host resource handle (the
// VkBuffer / VkImage) with host byte offsets, the conservative check the verify counters use.
namespace Libs::Graphics::Hazard {

struct Range {
	uint64_t space = 0; // 0: guest address space; else a host resource handle
	uint64_t lo    = 0; // first byte
	uint64_t hi    = 0; // one past the last byte (hi > lo)
	bool     write = false;
};

enum class Verdict : uint8_t { Free = 0, Raw = 1, War = 2, Waw = 3 };

[[nodiscard]] constexpr bool Overlap(const Range& a, const Range& b) noexcept {
	return a.space == b.space && a.lo < b.hi && b.lo < a.hi;
}

class Tracker {
public:
	// The chain may hold at most this many ranges; Add() beyond it poisons the tracker (every later
	// Check() is a hazard until Reset()).
	static constexpr size_t Capacity = 1024;

	void Reset() noexcept {
		m_ranges.clear();
		m_poisoned = false;
	}
	[[nodiscard]] bool   Empty() const noexcept { return m_ranges.empty() && !m_poisoned; }
	[[nodiscard]] size_t Size() const noexcept { return m_ranges.size(); }
	[[nodiscard]] bool   Poisoned() const noexcept { return m_poisoned; }
	// Every later Check() is a hazard until Reset() (accesses not known).
	void Poison() noexcept {
		m_ranges.clear();
		m_poisoned = true;
	}

	// The first hazard of `access` against the chain, in the order RAW, WAW, WAR; Free when none.
	[[nodiscard]] Verdict Check(std::span<const Range> access) const noexcept {
		if (m_poisoned) {
			return Verdict::Raw;
		}
		Verdict result = Verdict::Free;
		for (const auto& a: access) {
			for (const auto& c: m_ranges) {
				if (!Overlap(a, c)) {
					continue;
				}
				if (c.write && !a.write) {
					return Verdict::Raw;
				}
				if (c.write && a.write) {
					result = Verdict::Waw;
				} else if (a.write && result == Verdict::Free) {
					result = Verdict::War;
				}
			}
		}
		return result;
	}

	void Add(std::span<const Range> access) {
		if (m_poisoned) {
			return;
		}
		if (m_ranges.size() + access.size() > Capacity) {
			m_ranges.clear();
			m_poisoned = true;
			return;
		}
		m_ranges.insert(m_ranges.end(), access.begin(), access.end());
	}

private:
	std::vector<Range> m_ranges;
	bool               m_poisoned = false;
};

// Barrier scopes (synchronization2 bit values) that reset a chain when a batch with this memory
// dependency is recorded.
//  - Full: the first scope is every stage with every write made available, the second scope covers
//    compute shader reads and writes: everything recorded before is ordered against every later
//    dispatch. Starts a chain whatever came before.
//  - Compute: the first scope covers compute shader stages and shader writes, the second compute
//    shader reads and writes. Orders the chain's dispatches before every later dispatch, so it
//    restarts a chain that is valid (nothing but its dispatches since the last Full reset).
namespace Scope {
constexpr uint64_t AllCommands   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
constexpr uint64_t ComputeShader = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
constexpr uint64_t MemoryRead    = VK_ACCESS_2_MEMORY_READ_BIT;
constexpr uint64_t MemoryWrite   = VK_ACCESS_2_MEMORY_WRITE_BIT;
constexpr uint64_t ShaderRead    = VK_ACCESS_2_SHADER_READ_BIT;
constexpr uint64_t ShaderWrite   = VK_ACCESS_2_SHADER_WRITE_BIT;
constexpr uint64_t StorageRead   = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
constexpr uint64_t StorageWrite  = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
constexpr uint64_t SampledRead   = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
} // namespace Scope

// Second scope reaches compute shader reads and writes (every kind).
[[nodiscard]] constexpr bool ReachesComputeAccess(uint64_t dst_stages, uint64_t dst_access) noexcept {
	using namespace Scope;
	const bool stage = (dst_stages & (AllCommands | ComputeShader)) != 0;
	const bool reads = (dst_access & (MemoryRead | ShaderRead)) != 0 ||
	                   ((dst_access & StorageRead) != 0 && (dst_access & SampledRead) != 0);
	const bool writes = (dst_access & (MemoryWrite | ShaderWrite)) != 0;
	return stage && reads && writes;
}

enum class Reset : uint8_t { None = 0, Compute = 1, Full = 2 };

[[nodiscard]] constexpr Reset ChainReset(uint64_t src_stages, uint64_t src_access, uint64_t dst_stages,
                                         uint64_t dst_access) noexcept {
	using namespace Scope;
	if (!ReachesComputeAccess(dst_stages, dst_access)) {
		return Reset::None;
	}
	if ((src_stages & AllCommands) != 0 && (src_access & MemoryWrite) != 0) {
		return Reset::Full;
	}
	if ((src_stages & (AllCommands | ComputeShader)) != 0 && (src_access & (MemoryWrite | ShaderWrite)) != 0) {
		return Reset::Compute;
	}
	return Reset::None;
}

// The chain state of one command buffer.
//  - valid: a Full reset happened since the buffer began and everything recorded since the last
//    reset is the chain's dispatches (Note()).
//  - The caller bumps `foreign` for every recorded command that is not a chain dispatch.
struct Chain {
	Tracker  guest;
	Tracker  host;
	bool     valid  = false;
	uint64_t serial = 0; // `foreign` value at the last reset or chain dispatch

	void Invalidate() noexcept {
		valid = false;
		guest.Reset();
		host.Reset();
	}
	// A batch was recorded: its memory dependency (`has_memory`), and whether it also recorded
	// image/buffer barriers or upload copies (`side`) that are not known to concern only the next
	// dispatch's own bindings (those invalidate the chain unless the memory dependency resets it).
	void OnFlush(uint64_t foreign, bool has_memory, uint64_t src_stages, uint64_t src_access,
	             uint64_t dst_stages, uint64_t dst_access, bool side) noexcept {
		const auto reset = has_memory ? ChainReset(src_stages, src_access, dst_stages, dst_access) : Reset::None;
		const bool intact = valid && foreign == serial;
		if (reset == Reset::Full || (reset == Reset::Compute && intact)) {
			guest.Reset();
			host.Reset();
			valid  = true;
			serial = foreign;
			return;
		}
		if (!intact || side) {
			Invalidate();
		}
	}
	// Whether the chain can be compared against: valid and nothing else recorded since.
	[[nodiscard]] bool Intact(uint64_t foreign) const noexcept { return valid && foreign == serial; }
	// A dispatch was recorded as the only command since the chain state at `before` (the `foreign`
	// value just before the dispatch command, after its batch flush).
	// An imprecise dispatch (accesses not all known) keeps the chain valid but poisoned: no later
	// dispatch can pass it, yet a ShaderAccess flush after it (Compute reset) orders it like any chain
	// dispatch and starts a clean chain.
	// keep_poisoned false: an imprecise dispatch invalidates the chain instead (KYTY_GPU_HAZARD 1-3).
	void NoteDispatch(uint64_t before, uint64_t after, bool precise, std::span<const Range> guest_access,
	                  std::span<const Range> host_access, bool keep_poisoned = true) {
		if (!valid || before != serial || after != before + 1 || (!precise && !keep_poisoned)) {
			Invalidate();
			return;
		}
		if (!precise) {
			guest.Poison();
			host.Poison();
			serial = after;
			return;
		}
		guest.Add(guest_access);
		host.Add(host_access);
		serial = after;
	}
};

} // namespace Libs::Graphics::Hazard

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HAZARDTRACKER_H_
