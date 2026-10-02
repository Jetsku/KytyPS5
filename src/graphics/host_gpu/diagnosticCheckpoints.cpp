#include "graphics/host_gpu/vulkanCommon.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr size_t CHECKPOINT_RING_SIZE = 1u << 16u;

std::array<DiagnosticCheckpoint, CHECKPOINT_RING_SIZE> g_checkpoints {};
uint64_t g_checkpoint_sequence = 0;
std::mutex g_checkpoint_mutex;
std::atomic_flag g_dumping = ATOMIC_FLAG_INIT;

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

void DumpDeviceLossDiagnostics(GraphicContext& graphics, uint64_t tick, bool queue_locked) {
	if (g_dumping.test_and_set(std::memory_order_acquire)) return;
	std::printf("--- Device loss: submission/wait tick=%" PRIu64 ", KYTY_DEVICE_FAULT_DIAGNOSTICS=%d ---\n",
	            tick, DeviceFaultDiagnosticsEnabled());
	LOGF("--- Device loss: submission/wait tick=%" PRIu64 " ---\n", tick);
	DumpDeviceFault(graphics);
	if (!graphics.diagnostic_checkpoints_enabled || graphics.queue == nullptr) {
		std::printf("  NV checkpoints unavailable; enable KYTY_DEVICE_FAULT_DIAGNOSTICS=1 before launch for supported driver diagnostics.\n");
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

} // namespace Libs::Graphics
