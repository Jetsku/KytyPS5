#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUBREADCRUMBS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUBREADCRUMBS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

// GPU breadcrumbs for devices without NV checkpoints (AMD), KYTY_GPU_BREADCRUMBS:
//   unset/auto: on with KYTY_DEVICE_FAULT_DIAGNOSTICS=1 when the device has VK_AMD_buffer_marker
//               but not VK_NV_device_diagnostic_checkpoints; 1: on whenever VK_AMD_buffer_marker
//               exists (NVIDIA drivers have it too); 0: off.
// Before each guest operation (the CommandBuffer::SetDebugInfo sites, the same ones NV checkpoints
// use), at the end of each guest command buffer, and around side-queue readbacks and upload DMA
// batches, the queue writes the breadcrumb's marker at the top and at the bottom of the pipe into a
// host-visible (on AMD device-uncached) buffer. After a device loss Dump() prints, per queue, the
// operations from the last one whose predecessors all completed to the last one begun, with their
// guest shader addresses and hashes, submit ids and ticks (breadcrumbLogic.h). Costs nothing off.
namespace Libs::Graphics {
struct GraphicContext;
struct DiagnosticCheckpoint;
} // namespace Libs::Graphics

namespace Libs::Graphics::GpuBreadcrumbs {

enum class Queue : uint32_t { Graphics = 0, Side = 1, Transfer = 2, Count = 3 };

// Breadcrumb operations beyond CommandBufferDebugOp (diagnosticCheckpoints.cpp OpName).
inline constexpr uint32_t OpCommandBufferEnd = 100;
inline constexpr uint32_t OpSideReadback     = 101;
inline constexpr uint32_t OpSideReadbackEnd  = 102;
inline constexpr uint32_t OpUploadDma        = 103;
inline constexpr uint32_t OpUploadDmaEnd     = 104;

// Whether the switch asks for breadcrumbs on a device with these extensions (device creation).
[[nodiscard]] bool Requested(bool has_amd_buffer_marker, bool has_nv_checkpoints);
// After device creation, with VK_AMD_buffer_marker enabled (and VK_AMD_device_coherent_memory's
// deviceCoherentMemory when `device_coherent`): creates the marker buffer. Prints one line.
void Initialize(GraphicContext& graphics, bool device_coherent);
void Shutdown(GraphicContext& graphics);
// Cheap check for the recording sites.
[[nodiscard]] bool Active() noexcept;

// Records the CPU half of a breadcrumb and returns its marker (0 when inactive).
[[nodiscard]] uint32_t Note(Queue queue, const DiagnosticCheckpoint& checkpoint);
// The marker writes into a native command buffer (top and bottom of the pipe).
void Write(vk::CommandBuffer command, Queue queue, uint32_t marker);
// The marker buffer and the byte offsets of a queue's two words, for recorder-encoded writes.
[[nodiscard]] vk::Buffer Buffer() noexcept;
[[nodiscard]] uint64_t   TopOffset(Queue queue) noexcept;
[[nodiscard]] uint64_t   BottomOffset(Queue queue) noexcept;

// Guest shader address -> hash, noted where programs are prepared, for the dump.
void NoteShader(uint64_t address, uint64_t hash) noexcept;

// After a device loss: what each queue was running. Safe to call more than once.
void Dump();

} // namespace Libs::Graphics::GpuBreadcrumbs

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUBREADCRUMBS_H_ */
