#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_PACKETCLASS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_PACKETCLASS_H_

#include "graphics/guest_gpu/pm4.h"

#include <cstdint>

namespace Libs::Graphics::DrawPrep {

// PM4 packet classes for the draw-preparation window (S0 histogram, S6 fences).
enum class PacketClass : uint8_t {
	WindowSafe, // only CP register/state writes, markers, control flow: no drain needed
	Draw,       // direct draw
	Fence,      // anything else: commit the window first
};

// `header` without the predication bit; `body` points after the header; `remaining_dw` counts
// the dwords from the header to the end of the command buffer. Conservative: every packet that
// is not known to only touch command-processor state is a fence.
//
// Window-safe packets and why (pm4Handlers.cpp):
// - SET_CONTEXT_REG / SET_SH_REG / SET_UCONFIG_REG(_INDEX): register parsers only decode into
//   the register files (and the index type / user-data marker CP state). The *_REG_INDIRECT
//   forms read guest memory and are fences.
// - INDEX_TYPE, INDEX_BASE, INDEX_BUFFER_SIZE, NUM_INSTANCES, SET_BASE: CP draw state, captured
//   into a draw's arguments when its packet is parsed.
// - CLEAR_STATE and NOP/R_CONTEXT_STATE: register-file copies (the latter also counts a DE
//   event, which the constant engine only observes after this command stream yields, and the
//   window is committed whenever a stream yields).
// - PFP_SYNC_ME: no-op. Plain NOPs, push/pop markers, user-data markers (0x4, 0xd).
// - INDIRECT_BUFFER call/chain (4 dwords): moves the fetcher. The 14-dword form is a
//   conditional branch on guest memory: a fence.
// Draws: DRAW_INDEX_2, DRAW_INDEX_OFFSET_2, DRAW_INDEX_AUTO, DISPATCH_DRAW_PREAMBLE (handled as
// DRAW_INDEX_2). Indirect draws read GPU-written arguments and are fences.
[[nodiscard]] inline PacketClass ClassifyPacket(uint32_t header, const uint32_t* body,
                                                uint32_t remaining_dw) {
	const auto opcode = (header >> 8u) & 0xffu;
	switch (opcode) {
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_SET_BASE:
		case Pm4::IT_CLEAR_STATE:
		case Pm4::IT_PFP_SYNC_ME: return PacketClass::WindowSafe;
		case Pm4::IT_INDIRECT_BUFFER:
			return KYTY_PM4_LEN(header) == 4u ? PacketClass::WindowSafe : PacketClass::Fence;
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE: return PacketClass::Draw;
		case Pm4::IT_NOP: {
			const auto r = KYTY_PM4_R(header);
			if (r == Pm4::R_ZERO) {
				if (remaining_dw >= 2 && (body[0] & 0xffff0000u) == 0x68750000u) {
					// Markers: 0 (none), 0x4 and 0xd (user-data markers) only set CP state;
					// the others are flips.
					const auto id = body[0] & 0xfffu;
					return id == 0x0u || id == 0x4u || id == 0xdu ? PacketClass::WindowSafe
					                                               : PacketClass::Fence;
				}
				return PacketClass::WindowSafe;
			}
			return r == Pm4::R_CONTEXT_STATE || r == Pm4::R_PUSH_MARKER || r == Pm4::R_POP_MARKER
			           ? PacketClass::WindowSafe
			           : PacketClass::Fence;
		}
		default: break;
	}
	return PacketClass::Fence;
}

// What a fence packet is (FrameEvent DrawPrepFence<kind>): which packets end the preparation
// windows in practice, to plan which of them a window could keep open.
enum class FenceKind : uint8_t {
	RegIndirect,    // SET_SH/CONTEXT/UCONFIG_REG_INDIRECT: registers loaded from guest memory
	EventWrite,     // EVENT_WRITE, EVENT_WRITE_EOS (flushes, occlusion dumps, ...)
	EndOfPipe,      // EVENT_WRITE_EOP, RELEASE_MEM and its custom NOP form
	AcquireMem,     // ACQUIRE_MEM, SURFACE_SYNC and the custom NOP form
	Wait,           // WAIT_REG_MEM(_64), MEM_SEMAPHORE, COND_EXEC, SET_PREDICATION, CE/DE waits, flip waits
	DataWrite,      // WRITE_DATA, COPY_DATA, DMA_DATA, CP_DMA and the custom NOP forms
	ConstantEngine, // WRITE_CONST_RAM, DUMP_CONST_RAM, CE/DE counter increments, INDIRECT_BUFFER_CNST
	Marker,         // other NOPs: markers other than 0/4/0xd, flips, remaining custom codes
	Dispatch,       // DISPATCH_DIRECT/INDIRECT and the custom dispatch reset
	IndirectDraw,   // indirect draws and DISPATCH_DRAW
	ContextControl, // CONTEXT_CONTROL
	Other,          // everything else (conditional INDIRECT_BUFFER, GET_LOD_STATS, REWIND, ...)
	Count,
};

// SET_SH/CONTEXT/UCONFIG_REG_INDIRECT: the guest range of the (offset, value) register pairs the
// command processor loads (pm4Handlers.cpp CpOpIndirect*Regs; size 0 when there are none). False
// for any other packet or a truncated one.
struct RegisterIndirectRange {
	uint64_t address = 0;
	uint64_t size    = 0;
};

[[nodiscard]] inline bool RegisterIndirectPairs(uint32_t header, const uint32_t* body,
                                                uint32_t remaining_dw, RegisterIndirectRange& range) {
	const auto opcode = (header >> 8u) & 0xffu;
	if ((opcode != Pm4::IT_SET_SH_REG_INDIRECT && opcode != Pm4::IT_SET_UCONFIG_REG_INDIRECT &&
	     opcode != Pm4::IT_SET_CONTEXT_REG_INDIRECT) ||
	    KYTY_PM4_LEN(header) != 5u || remaining_dw < 5u) {
		return false;
	}
	range.address = (static_cast<uint64_t>(body[0]) & 0xfffffffcu) |
	                (static_cast<uint64_t>(body[1]) << 32u);
	range.size    = static_cast<uint64_t>(body[3] & 0x3fffu) * 2u * sizeof(uint32_t);
	return true;
}

[[nodiscard]] inline FenceKind ClassifyFence(uint32_t header) {
	const auto opcode = (header >> 8u) & 0xffu;
	switch (opcode) {
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT: return FenceKind::RegIndirect;
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOS: return FenceKind::EventWrite;
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_RELEASE_MEM: return FenceKind::EndOfPipe;
		case Pm4::IT_ACQUIRE_MEM:
		case Pm4::IT_SURFACE_SYNC: return FenceKind::AcquireMem;
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_MEM_SEMAPHORE:
		case Pm4::IT_COND_EXEC:
		case Pm4::IT_SET_PREDICATION:
		case Pm4::IT_WAIT_ON_CE_COUNTER:
		case Pm4::IT_WAIT_ON_DE_COUNTER_DIFF: return FenceKind::Wait;
		case Pm4::IT_WRITE_DATA:
		case Pm4::IT_COPY_DATA:
		case Pm4::IT_DMA_DATA:
		case Pm4::IT_CP_DMA: return FenceKind::DataWrite;
		case Pm4::IT_WRITE_CONST_RAM:
		case Pm4::IT_DUMP_CONST_RAM:
		case Pm4::IT_INCREMENT_CE_COUNTER:
		case Pm4::IT_INCREMENT_DE_COUNTER:
		case Pm4::IT_INDIRECT_BUFFER_CNST: return FenceKind::ConstantEngine;
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return FenceKind::Dispatch;
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW: return FenceKind::IndirectDraw;
		case Pm4::IT_CONTEXT_CONTROL: return FenceKind::ContextControl;
		case Pm4::IT_NOP:
			switch (KYTY_PM4_R(header)) {
				case Pm4::R_RELEASE_MEM: return FenceKind::EndOfPipe;
				case Pm4::R_ACQUIRE_MEM: return FenceKind::AcquireMem;
				case Pm4::R_WRITE_DATA:
				case Pm4::R_DMA_DATA: return FenceKind::DataWrite;
				case Pm4::R_WAIT_FLIP_DONE: return FenceKind::Wait;
				case Pm4::R_DISPATCH_RESET:
				case Pm4::R_PSSR: return FenceKind::Dispatch;
				default: return FenceKind::Marker;
			}
		default: break;
	}
	return FenceKind::Other;
}

// KYTY_SYNC_EPOCH (syncEpoch.h): whether guest CPU writes must be visible to the GPU work after
// this packet, so the synchronization epoch advances before it. Every fence does, except a
// register load from memory (SET_*_REG_INDIRECT), which only reads guest memory.
[[nodiscard]] inline bool AdvancesSyncEpoch(uint32_t header, const uint32_t* body,
                                            uint32_t remaining_dw) {
	return ClassifyPacket(header, body, remaining_dw) == PacketClass::Fence &&
	       ClassifyFence(header) != FenceKind::RegIndirect;
}

} // namespace Libs::Graphics::DrawPrep

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_PACKETCLASS_H_
