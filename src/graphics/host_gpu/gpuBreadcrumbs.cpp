#include "graphics/host_gpu/gpuBreadcrumbs.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/breadcrumbLogic.h"
#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <chrono>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace Libs::Graphics::GpuBreadcrumbs {

namespace {

constexpr uint64_t QueueStride  = 64;   // bytes per queue in the marker buffer
constexpr uint64_t BufferBytes  = 4096;
constexpr size_t   ShaderSlots  = 8192; // direct-mapped address -> hash table
constexpr uint64_t DumpMaxLines = 32;

std::atomic<bool> g_active {false};
vk::Device        g_device        = nullptr;
vk::Buffer        g_buffer        = nullptr;
vk::DeviceMemory  g_memory        = nullptr;
volatile uint32_t* g_mapped       = nullptr;
std::array<std::atomic<uint64_t>, static_cast<size_t>(Queue::Count)> g_newest {};
std::atomic_flag  g_dumped = ATOMIC_FLAG_INIT;
std::mutex        g_state_mutex; // the marker buffer's lifetime against the probe thread

struct ShaderSlot {
	std::atomic<uint64_t> address {0};
	std::atomic<uint64_t> hash {0};
};
std::array<ShaderSlot, ShaderSlots> g_shaders;

const char* QueueName(uint32_t queue) {
	switch (queue) {
		case 0: return "graphics queue";
		case 1: return "side-copy queue";
		case 2: return "upload DMA queue";
		default: return "queue ?";
	}
}

size_t ShaderSlotIndex(uint64_t address) {
	return static_cast<size_t>(((address >> 4u) * 0x9e3779b97f4a7c15ull) >> 51u) % ShaderSlots;
}

uint64_t ShaderHash(uint64_t address) {
	if (address == 0) {
		return 0;
	}
	const auto& slot = g_shaders[ShaderSlotIndex(address)];
	return slot.address.load(std::memory_order_acquire) == address
	           ? slot.hash.load(std::memory_order_relaxed)
	           : 0;
}

void PrintCheckpoint(const char* mark, const DiagnosticCheckpoint& c) {
	char shaders[192] = {};
	int  used         = 0;
	const auto add    = [&](const char* stage, uint64_t address) {
		if (address == 0 || used < 0 || used >= static_cast<int>(sizeof(shaders))) {
			return;
		}
		used += std::snprintf(shaders + used, sizeof(shaders) - static_cast<size_t>(used),
		                      " %s=0x%016" PRIx64 "@0x%010" PRIx64, stage, ShaderHash(address), address);
	};
	add("VS", c.vs);
	add("PS", c.ps);
	add("CS", c.cs);
	std::printf("  %s #%" PRIu64 " %s submit=%" PRIu64 " tick=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64
	            "%s\n",
	            mark, c.sequence, DiagnosticOpName(c.op), c.submit_id, c.tick, c.arg0, c.arg1, c.arg2,
	            c.arg3, c.arg4, shaders);
	LOGF("  %s #%" PRIu64 " %s submit=%" PRIu64 " tick=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64 "%s\n",
	     mark, c.sequence, DiagnosticOpName(c.op), c.submit_id, c.tick, c.arg0, c.arg1, c.arg2, c.arg3,
	     c.arg4, shaders);
}

uint32_t ChooseMemoryType(const vk::PhysicalDeviceMemoryProperties& memory, uint32_t type_bits,
                          bool device_coherent) {
	using F = vk::MemoryPropertyFlagBits;
	const vk::MemoryPropertyFlags host = F::eHostVisible | F::eHostCoherent;
	const vk::MemoryPropertyFlags amd  = F::eDeviceCoherentAMD | F::eDeviceUncachedAMD;
	// Device-uncached first (AMD's recommendation for breadcrumbs: the writes reach memory even
	// when the GPU hangs), then host memory outside VRAM, then any host-visible coherent type.
	for (int pass = 0; pass < 3; pass++) {
		for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
			if ((type_bits & (1u << i)) == 0) {
				continue;
			}
			const auto flags = memory.memoryTypes[i].propertyFlags;
			if ((flags & host) != host) {
				continue;
			}
			const bool has_amd = (flags & amd) == amd;
			if (has_amd && !device_coherent) {
				continue;
			}
			if (pass == 0 && has_amd) return i;
			if (pass == 1 && !has_amd && !(flags & F::eDeviceLocal)) return i;
			if (pass == 2 && !has_amd) return i;
		}
	}
	return UINT32_MAX;
}

} // namespace

// The markers of every queue and the operations each may still run (Dump, the probe).
void PrintState(const char* when, bool loss);

bool Requested(bool has_amd_buffer_marker, bool has_nv_checkpoints) {
	if (!has_amd_buffer_marker) {
		return false;
	}
	const auto* value = std::getenv("KYTY_GPU_BREADCRUMBS");
	if (value != nullptr && std::strcmp(value, "0") == 0) {
		return false;
	}
	if (value != nullptr && std::strcmp(value, "1") == 0) {
		return true;
	}
	return DeviceFaultDiagnosticsEnabled() && !has_nv_checkpoints;
}

void Initialize(GraphicContext& graphics, bool device_coherent) {
	vk::BufferCreateInfo info {};
	info.size  = BufferBytes;
	info.usage = vk::BufferUsageFlagBits::eTransferDst;
	const std::array<uint32_t, 2> families {graphics.queue_family, graphics.transfer_queue_family};
	if (graphics.transfer_queue != nullptr && graphics.transfer_queue_family != graphics.queue_family) {
		info.sharingMode           = vk::SharingMode::eConcurrent;
		info.queueFamilyIndexCount = 2;
		info.pQueueFamilyIndices   = families.data();
	}
	vk::Buffer buffer = nullptr;
	if (graphics.device.createBuffer(&info, nullptr, &buffer) != vk::Result::eSuccess) {
		std::printf("Kyty GPU breadcrumbs (KYTY_GPU_BREADCRUMBS): marker buffer not created; off\n");
		std::fflush(stdout);
		return;
	}
	vk::MemoryRequirements requirements {};
	graphics.device.getBufferMemoryRequirements(buffer, &requirements);
	const auto type = ChooseMemoryType(graphics.GetPhysicalDeviceMemoryProperties(),
	                                   requirements.memoryTypeBits, device_coherent);
	vk::DeviceMemory memory = nullptr;
	void*            mapped = nullptr;
	if (type != UINT32_MAX) {
		vk::MemoryAllocateInfo allocate {};
		allocate.allocationSize  = requirements.size;
		allocate.memoryTypeIndex = type;
		if (graphics.device.allocateMemory(&allocate, nullptr, &memory) != vk::Result::eSuccess ||
		    graphics.device.bindBufferMemory(buffer, memory, 0) != vk::Result::eSuccess ||
		    graphics.device.mapMemory(memory, 0, VK_WHOLE_SIZE, {}, &mapped) != vk::Result::eSuccess) {
			mapped = nullptr;
		}
	}
	if (mapped == nullptr) {
		if (memory != nullptr) {
			graphics.device.freeMemory(memory, nullptr);
		}
		graphics.device.destroyBuffer(buffer, nullptr);
		std::printf("Kyty GPU breadcrumbs (KYTY_GPU_BREADCRUMBS): no host-visible memory for markers; off\n");
		std::fflush(stdout);
		return;
	}
	std::memset(mapped, 0, BufferBytes);
	g_device = graphics.device;
	g_buffer = buffer;
	g_memory = memory;
	g_mapped = static_cast<volatile uint32_t*>(mapped);
	const auto flags = graphics.GetPhysicalDeviceMemoryProperties().memoryTypes[type].propertyFlags;
	std::printf("Kyty GPU breadcrumbs (KYTY_GPU_BREADCRUMBS): VK_AMD_buffer_marker at the top and bottom "
	            "of the pipe around every guest operation, memory type %u (%s); printed if the GPU "
	            "is lost\n",
	            type,
	            (flags & vk::MemoryPropertyFlagBits::eDeviceUncachedAMD) ? "device-uncached"
	            : (flags & vk::MemoryPropertyFlagBits::eDeviceLocal)     ? "device-local, host-visible"
	                                                                     : "host memory");
	std::fflush(stdout);
	g_active.store(true, std::memory_order_release);
	// KYTY_GPU_BREADCRUMBS_PROBE_S=<n> (test): print the markers every n seconds, three times, to
	// check on a healthy device that every queue's markers advance.
	if (const auto* probe = std::getenv("KYTY_GPU_BREADCRUMBS_PROBE_S"); probe != nullptr) {
		const auto seconds = std::strtoul(probe, nullptr, 10);
		if (seconds != 0) {
			std::thread([seconds] {
				for (int i = 0; i < 3 && Active(); i++) {
					std::this_thread::sleep_for(std::chrono::seconds(seconds));
					PrintState("probe", false);
				}
			}).detach();
		}
	}
}

void Shutdown(GraphicContext& graphics) {
	const std::lock_guard lock(g_state_mutex);
	if (!g_active.exchange(false)) {
		return;
	}
	graphics.device.destroyBuffer(g_buffer, nullptr);
	graphics.device.freeMemory(g_memory, nullptr); // unmaps
	g_buffer = nullptr;
	g_memory = nullptr;
	g_mapped = nullptr;
}

bool Active() noexcept {
	return g_active.load(std::memory_order_relaxed);
}

uint32_t Note(Queue queue, const DiagnosticCheckpoint& checkpoint) {
	if (!Active()) {
		return 0;
	}
	DiagnosticCheckpoint copy = checkpoint;
	copy.queue                = static_cast<uint32_t>(queue);
	const auto sequence       = RecordDiagnosticSequence(copy);
	auto&      newest         = g_newest[static_cast<size_t>(queue)];
	auto       known          = newest.load(std::memory_order_relaxed);
	while (known < sequence &&
	       !newest.compare_exchange_weak(known, sequence, std::memory_order_relaxed)) {
	}
	// Zero means "no marker"; the 2^32nd sequence skips it.
	const auto marker = static_cast<uint32_t>(sequence);
	return marker != 0 ? marker : 1u;
}

vk::Buffer Buffer() noexcept {
	return g_buffer;
}

uint64_t TopOffset(Queue queue) noexcept {
	return static_cast<uint64_t>(queue) * QueueStride;
}

uint64_t BottomOffset(Queue queue) noexcept {
	return static_cast<uint64_t>(queue) * QueueStride + 4u;
}

void Write(vk::CommandBuffer command, Queue queue, uint32_t marker) {
	if (marker == 0 || g_buffer == nullptr) {
		return;
	}
	command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_buffer, TopOffset(queue), marker);
	command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, g_buffer, BottomOffset(queue),
	                             marker);
}

void NoteShader(uint64_t address, uint64_t hash) noexcept {
	if (address == 0 || !Active()) {
		return;
	}
	auto& slot = g_shaders[ShaderSlotIndex(address)];
	if (slot.address.load(std::memory_order_relaxed) == address &&
	    slot.hash.load(std::memory_order_relaxed) == hash) {
		return;
	}
	slot.address.store(0, std::memory_order_relaxed);
	slot.hash.store(hash, std::memory_order_relaxed);
	slot.address.store(address, std::memory_order_release);
}

void PrintState(const char* when, bool loss) {
	const std::lock_guard lock(g_state_mutex);
	if (!Active() || g_mapped == nullptr) {
		return;
	}
	const auto newest = NewestDiagnosticSequence();
	std::printf("--- GPU breadcrumbs (VK_AMD_buffer_marker, KYTY_GPU_BREADCRUMBS) %s: newest breadcrumb #%" PRIu64
	            " ---\n",
	            when, newest);
	LOGF("--- GPU breadcrumbs: newest breadcrumb #%" PRIu64 " ---\n", newest);
	for (uint32_t q = 0; q < static_cast<uint32_t>(Queue::Count); q++) {
		const auto queue  = static_cast<Queue>(q);
		const auto top    = g_mapped[TopOffset(queue) / 4u];
		const auto bottom = g_mapped[BottomOffset(queue) / 4u];
		const auto begun  = Breadcrumbs::SequenceFromMarker(top, newest);
		const auto done   = Breadcrumbs::SequenceFromMarker(bottom, newest);
		const auto recorded = g_newest[q].load(std::memory_order_relaxed);
		if (begun == 0 && recorded == 0) {
			continue; // the queue never ran breadcrumbed work
		}
		std::printf("  %s: began #%" PRIu64 ", everything before #%" PRIu64
		            " completed, newest recorded #%" PRIu64 "\n",
		            QueueName(q), begun, done, recorded);
		LOGF("  %s: began #%" PRIu64 ", everything before #%" PRIu64 " completed, newest recorded #%" PRIu64
		     "\n",
		     QueueName(q), begun, done, recorded);
		const auto flight = Breadcrumbs::QueueInFlight(begun, done);
		if (flight.Empty()) {
			continue;
		}
		DiagnosticCheckpoint last {};
		const bool           known_last = LookupDiagnosticCheckpoint(flight.last, &last);
		if (flight.first == flight.last && known_last &&
		    (last.op == OpCommandBufferEnd || last.op == OpSideReadbackEnd || last.op == OpUploadDmaEnd)) {
			std::printf("    idle: its last command buffer completed\n");
			continue;
		}
		// The queue's operations in [first, last]; others' sequences interleave and are skipped.
		uint64_t printed = 0;
		uint64_t skipped = 0;
		const auto first = flight.Count() > 4096 ? flight.last - 4096 : flight.first;
		for (uint64_t s = flight.last + 1; s-- > first;) {
			DiagnosticCheckpoint c {};
			if (!LookupDiagnosticCheckpoint(s, &c) || c.queue != q) {
				continue;
			}
			if (printed == DumpMaxLines) {
				skipped++;
				continue;
			}
			const char* mark = s == flight.first ? "oldest unfinished" : s == flight.last ? "newest begun" : "in flight";
			PrintCheckpoint(mark, c);
			printed++;
		}
		if (skipped != 0) {
			std::printf("    ... and %" PRIu64 " older operation(s) of this queue in flight\n", skipped);
		}
		if (DiagnosticCheckpoint oldest {};
		    loss && LookupDiagnosticCheckpoint(flight.first, &oldest) && oldest.queue == q) {
			std::printf("    -> the oldest unfinished operation of this queue is #%" PRIu64
			            " (%s): the likely hang\n",
			            flight.first, DiagnosticOpName(oldest.op));
			LOGF("    -> likely hang: #%" PRIu64 " (%s)\n", flight.first, DiagnosticOpName(oldest.op));
		}
	}
	std::fflush(stdout);
}

void Dump() {
	if (!Active() || g_dumped.test_and_set(std::memory_order_acq_rel)) {
		return;
	}
	PrintState("after the device loss", true);
}

} // namespace Libs::Graphics::GpuBreadcrumbs
