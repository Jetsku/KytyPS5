#include "common/hangWatchdog.h"
#include "common/staticOnce.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/aftermath.h"
#include "graphics/host_gpu/gpuBreadcrumbs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr size_t CHECKPOINT_RING_SIZE = 1u << 16u;

std::array<DiagnosticCheckpoint, CHECKPOINT_RING_SIZE> g_checkpoints {};
uint64_t g_checkpoint_sequence = 0;
std::mutex g_checkpoint_mutex;
std::atomic_flag g_dumping = ATOMIC_FLAG_INIT;
std::atomic<bool> g_dump_done {false};

const char* OpName(uint32_t op) {
	switch (op) {
		case 0: return "DispatchDirect";
		case 1: return "DrawIndex";
		case 2: return "DrawIndexAuto";
		case 3: return "EopWrite";
		case 4: return "EopInterrupt";
		case 5: return "EopWriteBack";
		case 6: return "EopFlip";
		case 7: return "EopWriteBackFlip";
		case 8: return "EopOnlyFlip";
		case 9: return "DispatchIndirect";
		case GpuBreadcrumbs::OpCommandBufferEnd: return "CommandBufferEnd";
		case GpuBreadcrumbs::OpSideReadback: return "SideReadback";
		case GpuBreadcrumbs::OpSideReadbackEnd: return "SideReadbackEnd";
		case GpuBreadcrumbs::OpUploadDma: return "UploadDma";
		case GpuBreadcrumbs::OpUploadDmaEnd: return "UploadDmaEnd";
		default: return "Unknown";
	}
}

void Print(const char* stage, const DiagnosticCheckpoint& checkpoint) {
	std::printf("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64
	            "\n",
	            stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id,
	            checkpoint.arg0, checkpoint.arg1, checkpoint.arg2, checkpoint.arg3,
	            checkpoint.arg4);
	std::printf("    tick=%" PRIu64 " shader VS=0x%016" PRIx64 " PS=0x%016" PRIx64 " CS=0x%016" PRIx64 "\n",
	            checkpoint.tick, checkpoint.vs, checkpoint.ps, checkpoint.cs);
	LOGF("    tick=%" PRIu64 " shader VS=0x%016" PRIx64 " PS=0x%016" PRIx64 " CS=0x%016" PRIx64 "\n",
	     checkpoint.tick, checkpoint.vs, checkpoint.ps, checkpoint.cs);
	LOGF("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id, checkpoint.arg0,
	     checkpoint.arg1, checkpoint.arg2, checkpoint.arg3, checkpoint.arg4);
}

}

const char* DiagnosticOpName(uint32_t op) {
	return OpName(op);
}

uint64_t RecordDiagnosticSequence(const DiagnosticCheckpoint& checkpoint) {
	const std::lock_guard lock(g_checkpoint_mutex);
	const auto sequence = ++g_checkpoint_sequence;
	auto& slot = g_checkpoints[sequence % CHECKPOINT_RING_SIZE];
	slot = checkpoint;
	slot.sequence = sequence;
	return sequence;
}

bool LookupDiagnosticCheckpoint(uint64_t sequence, DiagnosticCheckpoint* checkpoint) {
	const std::lock_guard lock(g_checkpoint_mutex);
	const auto& slot = g_checkpoints[sequence % CHECKPOINT_RING_SIZE];
	if (sequence == 0 || slot.sequence != sequence) {
		return false;
	}
	*checkpoint = slot;
	return true;
}

uint64_t NewestDiagnosticSequence() {
	const std::lock_guard lock(g_checkpoint_mutex);
	return g_checkpoint_sequence;
}

bool DeviceFaultDiagnosticsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DEVICE_FAULT_DIAGNOSTICS");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

const void* RecordDiagnosticCheckpoint(const DiagnosticCheckpoint& checkpoint) {
	const std::lock_guard lock(g_checkpoint_mutex);
	const auto sequence = ++g_checkpoint_sequence;
	auto& slot = g_checkpoints[sequence % CHECKPOINT_RING_SIZE];
	slot = checkpoint;
	slot.sequence = sequence;
	// Vulkan treats this as an opaque token. Never hand it a pointer to a reusable ring slot.
	return reinterpret_cast<const void*>(static_cast<uintptr_t>(sequence));
}

namespace {

// KYTY_ADDRESS_BINDING_REPORT state. Never destroyed (the driver may report unbinds during static
// destruction).
struct AddressBindings {
	struct Event {
		uint64_t ns     = 0;
		uint64_t base   = 0;
		uint64_t size   = 0;
		uint64_t handle = 0;
		uint32_t type   = 0; // VkObjectType
		uint32_t flags  = 0; // VkDeviceAddressBindingFlagsEXT (internal object)
		bool     bind   = false;
		char     name[120] {};
	};
	static constexpr size_t RingSize = 1u << 17u;
	std::mutex                                 mutex;
	std::map<std::pair<uint64_t, uint64_t>, Event> live; // by base address, handle
	std::vector<Event>                         ring = std::vector<Event>(RingSize);
	uint64_t                                   count = 0;
	std::unordered_map<uint64_t, std::string>  names;
	std::shared_mutex                          names_mutex;
	VkDebugUtilsMessengerEXT                   messenger = VK_NULL_HANDLE;
	std::chrono::steady_clock::time_point      start = std::chrono::steady_clock::now();
};
AddressBindings& GetAddressBindings() {
	static auto* bindings = new AddressBindings;
	return *bindings;
}
std::atomic<bool> g_address_bindings_active {false};

uint64_t AddressBindingNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now() - GetAddressBindings().start)
	                                 .count());
}

VKAPI_ATTR VkBool32 VKAPI_CALL AddressBindingCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /*severity*/, VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*user*/) {
	if ((type & VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT) == 0 || data == nullptr) {
		return VK_FALSE;
	}
	const VkDeviceAddressBindingCallbackDataEXT* binding = nullptr;
	for (auto* next = static_cast<const VkBaseInStructure*>(data->pNext); next != nullptr; next = next->pNext) {
		if (next->sType == VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT) {
			binding = reinterpret_cast<const VkDeviceAddressBindingCallbackDataEXT*>(next);
			break;
		}
	}
	if (binding == nullptr) {
		return VK_FALSE;
	}
	AddressBindings::Event event {};
	event.ns    = AddressBindingNs();
	event.base  = binding->baseAddress;
	event.size  = binding->size;
	event.flags = binding->flags;
	event.bind  = binding->bindingType == VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT;
	if (data->objectCount != 0 && data->pObjects != nullptr) {
		event.handle = data->pObjects[0].objectHandle;
		event.type   = static_cast<uint32_t>(data->pObjects[0].objectType);
		if (data->pObjects[0].pObjectName != nullptr) {
			std::snprintf(event.name, sizeof(event.name), "%s", data->pObjects[0].pObjectName);
		}
	}
	auto& bindings = GetAddressBindings();
	if (!event.bind && event.name[0] == '\0' && event.handle != 0) {
		std::unique_lock lock(bindings.names_mutex);
		if (const auto found = bindings.names.find(event.handle); found != bindings.names.end()) {
			std::snprintf(event.name, sizeof(event.name), "%s", found->second.c_str());
			bindings.names.erase(found);
		}
	}
	const std::lock_guard lock(bindings.mutex);
	bindings.ring[bindings.count++ % AddressBindings::RingSize] = event;
	if (event.bind) {
		bindings.live[{event.base, event.handle}] = event;
	} else {
		bindings.live.erase({event.base, event.handle});
	}
	return VK_FALSE;
}

void PrintBinding(const char* what, const AddressBindings::Event& event, uint64_t address, uint64_t now) {
	std::string name = event.name;
	if (name.empty()) {
		auto&            bindings = GetAddressBindings();
		std::shared_lock lock(bindings.names_mutex);
		if (const auto found = bindings.names.find(event.handle); found != bindings.names.end()) {
			name = found->second;
		}
	}
	const auto end = event.base + event.size;
	const auto rel = address >= end ? static_cast<int64_t>(address - end) : -static_cast<int64_t>(end - address);
	std::printf("  binding %s: %s [0x%010" PRIx64 ", 0x%010" PRIx64 ") size 0x%" PRIx64
	            " type %u handle 0x%016" PRIx64 "%s, %.3f s before the fault, fault-end %+" PRId64
	            " %s\n",
	            what, event.bind ? "bind" : "UNBIND", event.base, end, event.size, event.type,
	            event.handle, (event.flags & VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT) != 0 ? " internal" : "",
	            static_cast<double>(now - std::min(now, event.ns)) / 1e9, rel, name.c_str());
	LOGF("  binding %s: %s [0x%010" PRIx64 ", 0x%010" PRIx64 ") type %u handle 0x%016" PRIx64
	     " %.3f s before, %s\n",
	     what, event.bind ? "bind" : "UNBIND", event.base, end, event.type, event.handle,
	     static_cast<double>(now - std::min(now, event.ns)) / 1e9, name.c_str());
}

} // namespace

bool AddressBindingReportRequested() {
	static const bool requested = [] {
		const auto* value = std::getenv("KYTY_ADDRESS_BINDING_REPORT");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return requested;
}

bool AddressBindingReportActive() {
	return g_address_bindings_active.load(std::memory_order_relaxed);
}

bool AddressBindingReportWantsName(uint64_t handle) {
	auto&            bindings = GetAddressBindings();
	std::shared_lock lock(bindings.names_mutex);
	return !bindings.names.contains(handle);
}

void AddressBindingReportNoteName(uint64_t handle, std::string name) {
	auto&            bindings = GetAddressBindings();
	std::unique_lock lock(bindings.names_mutex);
	bindings.names.try_emplace(handle, std::move(name));
}

void AddressBindingReportCreateMessenger(vk::Instance instance) {
	if (!AddressBindingReportRequested() || instance == nullptr) {
		return;
	}
	auto create = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDebugUtilsMessengerEXT;
	if (create == nullptr) {
		std::printf("KYTY_ADDRESS_BINDING_REPORT: VK_EXT_debug_utils missing; off\n");
		return;
	}
	VkDebugUtilsMessengerCreateInfoEXT info {};
	info.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
	                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
	info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT;
	info.pfnUserCallback = AddressBindingCallback;
	auto& bindings = GetAddressBindings();
	if (create(static_cast<VkInstance>(instance), &info, nullptr, &bindings.messenger) != VK_SUCCESS) {
		std::printf("KYTY_ADDRESS_BINDING_REPORT: messenger creation failed; off\n");
		return;
	}
	g_address_bindings_active.store(true);
	std::printf("KYTY_ADDRESS_BINDING_REPORT: recording GPU address bindings\n");
}

void AddressBindingReportDump(uint64_t address) {
	if (!AddressBindingReportActive()) {
		return;
	}
	auto&      bindings = GetAddressBindings();
	const auto now      = AddressBindingNs();
	const std::lock_guard lock(bindings.mutex);
	std::printf("--- Address bindings near 0x%010" PRIx64 " (%" PRIu64 " events, %zu live) ---\n", address,
	            bindings.count, bindings.live.size());
	// Live ranges covering the address or ending within 64 MiB below it, and the next one above.
	constexpr uint64_t Near = 64ull << 20u;
	auto               above = bindings.live.upper_bound({address, UINT64_MAX});
	if (above != bindings.live.end()) {
		PrintBinding("live-above", above->second, address, now);
	}
	int shown = 0;
	for (auto it = std::make_reverse_iterator(above); it != bindings.live.rend() && shown < 8; ++it) {
		const auto& event = it->second;
		if (event.base + Near * 8 < address) {
			break;
		}
		if (event.base + event.size + Near < address) {
			continue;
		}
		PrintBinding(address < event.base + event.size ? "live-COVERS" : "live-below", event, address, now);
		++shown;
	}
	// Objects placed inside a reported block (images: "block_off=0x.. alloc=0x.." in the name) whose
	// own range ends within 1 MiB of the address or covers it, live or unbound.
	const auto placed = [&](const AddressBindings::Event& event, const char* what) {
		if (address < event.base || address > event.base + event.size + Near) {
			return false;
		}
		std::string name = event.name;
		if (name.empty()) {
			std::shared_lock names_lock(bindings.names_mutex);
			if (const auto found = bindings.names.find(event.handle); found != bindings.names.end()) {
				name = found->second;
			}
		}
		const auto off_at   = name.find("block_off=0x");
		const auto alloc_at = name.find("alloc=0x");
		if (off_at == std::string::npos || alloc_at == std::string::npos) {
			return false;
		}
		const auto offset = std::strtoull(name.c_str() + off_at + 12, nullptr, 16);
		const auto bytes  = std::strtoull(name.c_str() + alloc_at + 8, nullptr, 16);
		const auto begin  = event.base + offset;
		const auto end    = begin + bytes;
		if ((address >= begin && address < end) || (address >= end && address - end < (1ull << 20u)) ||
		    (end > address && end - address < (1ull << 20u))) {
			std::printf("  placed %s: [0x%010" PRIx64 ", 0x%010" PRIx64 ") fault-end %+" PRId64 " %s %.3f s ago: %s\n",
			            what, begin, end, static_cast<int64_t>(address - end), event.bind ? "bind" : "UNBIND",
			            static_cast<double>(now - std::min(now, event.ns)) / 1e9, name.c_str());
			return true;
		}
		return false;
	};
	int placed_shown = 0;
	for (const auto& [key, event]: bindings.live) {
		if (placed_shown < 16 && placed(event, "live")) {
			++placed_shown;
		}
	}
	// Ring events (newest first) whose range covers the address or ends within 64 MiB below it.
	const auto first = bindings.count > AddressBindings::RingSize ? bindings.count - AddressBindings::RingSize : 0;
	shown            = 0;
	for (auto index = bindings.count; index > first; --index) {
		const auto& event = bindings.ring[(index - 1) % AddressBindings::RingSize];
		if (placed_shown < 32 && !event.bind && placed(event, "ring")) {
			++placed_shown;
		}
		if (shown < 12 && event.base <= address + Near && address <= event.base + event.size + Near) {
			PrintBinding(address >= event.base && address < event.base + event.size ? "ring-COVERS" : "ring-near",
			             event, address, now);
			++shown;
		}
	}
	std::fflush(stdout);
}

static void DumpDeviceFault(GraphicContext& graphics) {
	if (!graphics.device_fault_enabled) {
		return;
	}
	vk::DeviceFaultCountsEXT counts {};
	auto result = graphics.device.getFaultInfoEXT(&counts, nullptr);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		std::printf("--- Device fault: query failed: %s ---\n", vk::to_string(result).c_str());
		return;
	}
	if (counts.addressInfoCount > 4096 || counts.vendorInfoCount > 4096 || counts.vendorBinarySize > 64u * 1024u * 1024u) {
		std::printf("--- Device fault: driver returned oversized counts; skipping allocation ---\n");
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	std::vector<uint8_t>                       binary(static_cast<size_t>(counts.vendorBinarySize));
	vk::DeviceFaultInfoEXT                     info {};
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	info.pVendorBinaryData = binary.empty() ? nullptr : binary.data();
	result = graphics.device.getFaultInfoEXT(&counts, &info);
	std::printf("--- Device fault (%s): \"%s\" addresses=%u vendor=%u binary=%" PRIu64 " ---\n",
	            vk::to_string(result).c_str(), info.description.data(), counts.addressInfoCount,
	            counts.vendorInfoCount, static_cast<uint64_t>(counts.vendorBinarySize));
	LOGF("--- Device fault (%s): \"%s\" addresses=%u vendor=%u binary=%" PRIu64 " ---\n",
	     vk::to_string(result).c_str(), info.description.data(), counts.addressInfoCount,
	     counts.vendorInfoCount, static_cast<uint64_t>(counts.vendorBinarySize));
	if (!binary.empty()) {
		if (auto* file = std::fopen("_device_fault.nv-gpudmp", "wb"); file != nullptr) {
			std::fwrite(binary.data(), 1, std::min<size_t>(binary.size(), counts.vendorBinarySize), file);
			std::fflush(file);
			std::fclose(file);
			std::printf("  vendor binary written to _device_fault.nv-gpudmp\n");
		}
	}
	for (uint32_t i = 0; i < std::min<size_t>(counts.addressInfoCount, addresses.size()); i++) {
		const auto& address = addresses[i];
		std::printf("  address[%u]: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n", i,
		            vk::to_string(address.addressType).c_str(),
		            static_cast<uint64_t>(address.reportedAddress),
		            static_cast<uint64_t>(address.addressPrecision));
		LOGF("  address[%u]: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n", i,
		     vk::to_string(address.addressType).c_str(),
		     static_cast<uint64_t>(address.reportedAddress),
		     static_cast<uint64_t>(address.addressPrecision));
		AddressBindingReportDump(static_cast<uint64_t>(address.reportedAddress));
	}
	for (uint32_t i = 0; i < std::min<size_t>(counts.vendorInfoCount, vendors.size()); i++) {
		const auto& vendor = vendors[i];
		std::printf("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i,
		            vendor.description.data(), static_cast<uint64_t>(vendor.vendorFaultCode),
		            static_cast<uint64_t>(vendor.vendorFaultData));
		LOGF("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i,
		     vendor.description.data(), static_cast<uint64_t>(vendor.vendorFaultCode),
		     static_cast<uint64_t>(vendor.vendorFaultData));
	}
	std::fflush(stdout);
}

void WaitForDeviceLossDiagnostics(uint32_t start_ms) {
	using Clock = std::chrono::steady_clock;
	const auto start = Clock::now();
	while (!g_dumping.test(std::memory_order_acquire)) {
		if (Clock::now() - start > std::chrono::milliseconds(start_ms)) {
			return; // nobody is reporting
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	const auto begun = Clock::now();
	while (!g_dump_done.load(std::memory_order_acquire) &&
	       Clock::now() - begun < std::chrono::seconds(10)) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
}

namespace {
std::array<std::atomic<void (*)()>, 4> g_device_loss_notes {};
} // namespace

void SetDeviceLossNote(void (*note)()) {
	for (auto& slot: g_device_loss_notes) {
		void (*expected)() = nullptr;
		if (slot.load(std::memory_order_acquire) == note ||
		    slot.compare_exchange_strong(expected, note, std::memory_order_acq_rel)) {
			return;
		}
	}
}

void DumpDeviceLossDiagnostics(GraphicContext& graphics, uint64_t tick, bool queue_locked) {
	HangWatchdog::Scope diagnostics(
	    "device-loss-diagnostics",
	    reinterpret_cast<uint64_t>(static_cast<VkDevice>(graphics.device)), tick);
	if (g_dumping.test_and_set(std::memory_order_acquire)) return;
	// Exiting threads wait for this (WaitForDeviceLossDiagnostics) so the report is complete.
	struct Done {
		~Done() {
			std::fflush(stdout);
			g_dump_done.store(true, std::memory_order_release);
		}
	} done;
	std::printf("--- Device loss: submission/wait tick=%" PRIu64 ", KYTY_DEVICE_FAULT_DIAGNOSTICS=%d ---\n",
	            tick, DeviceFaultDiagnosticsEnabled());
	LOGF("--- Device loss: submission/wait tick=%" PRIu64 " ---\n", tick);
	DumpDeviceFault(graphics);
	for (auto& slot: g_device_loss_notes) {
		if (auto* note = slot.load(std::memory_order_acquire); note != nullptr) {
			note();
			std::fflush(stdout);
		}
	}
	if (const auto violations = CommandRecorder::GuardViolations(); violations != 0) {
		std::printf("  Kyty command guard (KYTY_CMD_GUARD): %" PRIu64 " violation(s) before the loss\n", violations);
	}
	GpuBreadcrumbs::Dump();
	Aftermath::WaitForDump();
	if (!graphics.diagnostic_checkpoints_enabled || graphics.queue == nullptr) {
		if (!GpuBreadcrumbs::Active()) {
			std::printf("  No GPU breadcrumbs: NV checkpoints and VK_AMD_buffer_marker breadcrumbs are off (KYTY_DEVICE_FAULT_DIAGNOSTICS=1, KYTY_GPU_BREADCRUMBS).\n");
		}
		std::fflush(stdout);
		return;
	}
	std::vector<vk::CheckpointDataNV> data;
	if (queue_locked) {
		data = graphics.queue.getCheckpointDataNV();
	} else if (graphics.queue_mutex.TryLock()) {
		data = graphics.queue.getCheckpointDataNV();
		graphics.queue_mutex.Unlock();
	} else {
		// Fatal reporting must not wait behind another stalled driver submission.
		std::printf("  Queue busy: NV checkpoint query skipped; latest CPU breadcrumb follows (not GPU completion).\n");
		DiagnosticCheckpoint latest;
		{
			const std::lock_guard lock(g_checkpoint_mutex);
			latest = g_checkpoints[g_checkpoint_sequence % CHECKPOINT_RING_SIZE];
		}
		if (latest.sequence != 0) Print("latest CPU record", latest);
		std::fflush(stdout);
		return;
	}
	std::printf("--- Diagnostic checkpoints (%zu) ---\n", data.size());
	for (const auto& entry: data) {
		const auto sequence = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(entry.pCheckpointMarker));
		DiagnosticCheckpoint checkpoint;
		{
			const std::lock_guard lock(g_checkpoint_mutex);
			checkpoint = g_checkpoints[sequence % CHECKPOINT_RING_SIZE];
		}
		const auto stage = vk::to_string(entry.stage);
		if (sequence == 0 || checkpoint.sequence != sequence) {
			std::printf("  [%s] breadcrumb seq=%" PRIu64 " retired from bounded history\n", stage.c_str(), sequence);
			continue;
		}
		Print(stage.c_str(), checkpoint);
	}
	std::fflush(stdout);
}

namespace {

// Recently destroyed image views (NoteImageViewDestroyed), oldest first; a view created again with
// the same handle leaves the set. Never destroyed: images may still be released during static
// destruction.
struct DeadView {
	uint64_t   guest_address = 0;
	vk::Format format        = vk::Format::eUndefined;
};
struct DeadViews {
	static constexpr size_t                   Limit = 1u << 16u;
	std::shared_mutex                         mutex;
	std::unordered_map<VkImageView, DeadView> views;
	std::deque<VkImageView>                   order;
};
DeadViews& GetDeadViews() {
	static auto* registry = new DeadViews;
	return *registry;
}
std::atomic<bool> g_any_dead_views {false};

void ReportDeadView(const char* where, const char* what, uint32_t index, VkImageView view) {
	static std::atomic<uint32_t> reported {0};
	const auto                   count = reported.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (count > 32 && (count & (count - 1u)) != 0) {
		return;
	}
	DeadView dead {};
	{
		auto&            registry = GetDeadViews();
		std::shared_lock lock(registry.mutex);
		if (const auto found = registry.views.find(view); found != registry.views.end()) {
			dead = found->second;
		}
	}
	LOGF("DEAD image view in %s: %s %u, view %p of guest 0x%016" PRIx64
	     " format %d; replaced with a null view (%u so far)\n",
	     where, what, index, static_cast<void*>(view), dead.guest_address,
	     static_cast<int>(dead.format), count);
}

} // namespace

bool DeadViewScrubEnabled() {
	static constexpr auto enabled_init = [] {
		const char* value = std::getenv("KYTY_DEAD_VIEW_SCRUB");
		return value != nullptr && std::strcmp(value, "0") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

void NoteImageViewCreated(vk::ImageView view) {
	if (view == nullptr || !g_any_dead_views.load(std::memory_order_relaxed)) {
		return;
	}
	auto&            registry = GetDeadViews();
	std::unique_lock lock(registry.mutex);
	registry.views.erase(static_cast<VkImageView>(view));
}

void NoteImageViewDestroyed(vk::ImageView view, uint64_t guest_address, vk::Format format) {
	if (view == nullptr) {
		return;
	}
	const auto       key      = static_cast<VkImageView>(view);
	auto&            registry = GetDeadViews();
	std::unique_lock lock(registry.mutex);
	const auto [entry, inserted] =
	    registry.views.insert_or_assign(key, DeadView {guest_address, format});
	if (inserted) {
		registry.order.push_back(key);
	}
	while (registry.order.size() > DeadViews::Limit) {
		registry.views.erase(registry.order.front());
		registry.order.pop_front();
	}
	g_any_dead_views.store(true, std::memory_order_relaxed);
}

bool DiagnosticViewIsLive(vk::ImageView view) {
	if (view == nullptr || !g_any_dead_views.load(std::memory_order_relaxed)) {
		return true;
	}
	auto&            registry = GetDeadViews();
	std::shared_lock lock(registry.mutex);
	return !registry.views.contains(static_cast<VkImageView>(view));
}

void ScrubDeadRenderingViews(const vk::RenderingInfo& info, const char* where) {
	if (!g_any_dead_views.load(std::memory_order_relaxed)) {
		return;
	}
	const auto clear = [](const vk::RenderingAttachmentInfo* attachment) {
		auto* writable             = const_cast<vk::RenderingAttachmentInfo*>(attachment);
		writable->imageView        = nullptr;
		writable->resolveImageView = nullptr;
		writable->resolveMode      = vk::ResolveModeFlagBits::eNone;
	};
	const auto scrub = [&](const vk::RenderingAttachmentInfo* attachment, const char* what,
	                       uint32_t index) {
		if (attachment == nullptr || DiagnosticViewIsLive(attachment->imageView)) {
			return;
		}
		ReportDeadView(where, what, index, attachment->imageView);
		clear(attachment);
	};
	for (uint32_t i = 0; i < info.colorAttachmentCount && info.pColorAttachments != nullptr; i++) {
		scrub(info.pColorAttachments + i, "color attachment", i);
	}
	const auto depth_view =
	    info.pDepthAttachment != nullptr ? info.pDepthAttachment->imageView : vk::ImageView {};
	scrub(info.pDepthAttachment, "depth attachment", 0);
	if (info.pStencilAttachment != nullptr && depth_view &&
	    info.pStencilAttachment->imageView == depth_view && !DiagnosticViewIsLive(depth_view)) {
		clear(info.pStencilAttachment); // the depth attachment's view, reported with it
	} else {
		scrub(info.pStencilAttachment, "stencil attachment", 0);
	}
}

void ScrubDeadDescriptorViews(const vk::WriteDescriptorSet* writes, uint32_t count,
                              const char* where) {
	if (!g_any_dead_views.load(std::memory_order_relaxed) || writes == nullptr) {
		return;
	}
	for (uint32_t i = 0; i < count; i++) {
		const auto& write = writes[i];
		// Storage images only: a write through a stale view faults, a read through one samples
		// garbage (and checking every sampled image costs a lookup per texture per draw).
		if (write.pImageInfo == nullptr ||
		    write.descriptorType != vk::DescriptorType::eStorageImage) {
			continue;
		}
		for (uint32_t k = 0; k < write.descriptorCount; k++) {
			const auto& info = write.pImageInfo[k];
			if (DiagnosticViewIsLive(info.imageView)) {
				continue;
			}
			ReportDeadView(where, "descriptor binding", write.dstBinding, info.imageView);
			const_cast<vk::DescriptorImageInfo&>(info).imageView = nullptr;
		}
	}
}

} // namespace Libs::Graphics
