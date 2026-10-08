#include "graphics/shader/shader.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/gpuBreadcrumbs.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderVertexMetadata.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <xxhash.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace Libs::Graphics {

struct ShaderBinaryInfo {
	uint8_t  signature[7];
	uint8_t  version;
	uint32_t pssl_or_cg  : 1;
	uint32_t cached      : 1;
	uint32_t type        : 4;
	uint32_t source_type : 2;
	uint32_t length      : 24;
	uint8_t  chunk_usage_base_offset_dw;
	uint8_t  num_input_usage_slots;
	uint8_t  is_srt                 : 1;
	uint8_t  is_srt_used_info_valid : 1;
	uint8_t  is_extended_usage_info : 1;
	uint8_t  reserved2              : 5;
	uint8_t  reserved3;
	uint32_t hash0;
	uint32_t hash1;
	uint32_t crc32;
};

// Startup-only: registration captures identities and workers retain code certificates.
static bool RegisteredShaderCodeEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_REGISTERED_SHADER_CODE");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

struct ShaderMapEntry : ShaderMappedData {
	uint64_t code_hash = 0;
	uint64_t registered_hash = 0;
	bool has_declared_hash = false;
};

static uint64_t GetDeclaredShaderHash(uint64_t shader_addr);

static uint64_t RegisteredOrDeclaredShaderHash(uint64_t addr, const ShaderMapEntry& entry) {
	if (RegisteredShaderCodeEnabled()) {
		// Headerless serial preparations keep exact code hashing, including guest writes
		// without re-registration. Declared IDs continue to select title-specific options.
		return entry.has_declared_hash ? entry.registered_hash : 0;
	}
	return GetDeclaredShaderHash(addr);
}

static std::unique_ptr<std::unordered_map<uint64_t, ShaderMapEntry>> g_shader_map;
static std::mutex                                                      g_shader_map_mutex;
// Bumped under g_shader_map_mutex after every map update. Starts at 1 so that zero-initialized
// memo entries never match.
static std::atomic<uint64_t> g_shader_map_generation {1};

void ShaderInit() {
	EXIT_IF(g_shader_map != nullptr);

	g_shader_map = std::make_unique<std::unordered_map<uint64_t, ShaderMapEntry>>();
}

void ShaderMapUserData(uint64_t addr, const ShaderMappedData& data) {
	EXIT_IF(g_shader_map == nullptr);

	ShaderMapEntry entry;
	static_cast<ShaderMappedData&>(entry) = data;
	if (RegisteredShaderCodeEnabled()) {
		if (addr == 0 || data.code_size_bytes == 0 || data.code_size_bytes % sizeof(uint32_t) != 0) {
			EXIT("ShaderMapUserData: invalid code address/size 0x%016" PRIx64 " / %u\n", addr, data.code_size_bytes);
		}
		// Hash before taking the map lock: a guest read can fault and ask the GPU to publish bytes.
		entry.code_hash = XXH3_64bits(reinterpret_cast<const void*>(addr), data.code_size_bytes);
		const auto declared = GetDeclaredShaderHash(addr);
		// Keep declared IDs used by our title-specific codegen options; headerless code uses XXH3.
		entry.registered_hash = declared != 0 ? declared : entry.code_hash;
		entry.has_declared_hash = declared != 0;
	}
	std::scoped_lock lock(g_shader_map_mutex);

	(*g_shader_map)[addr] = entry;
	g_shader_map_generation.fetch_add(1, std::memory_order_release);
}

void ShaderUnmapCode(uint64_t addr, uint64_t size) {
	if (!RegisteredShaderCodeEnabled() || size == 0 || g_shader_map == nullptr) return;
	std::scoped_lock lock(g_shader_map_mutex);
	const auto removed = std::erase_if(*g_shader_map, [=](const auto& pair) {
		const auto start = pair.first;
		const auto bytes = pair.second.code_size_bytes;
		return start >= addr ? start - addr < size : addr - start < bytes;
	});
	if (removed != 0) g_shader_map_generation.fetch_add(1, std::memory_order_release);
}

uint64_t ShaderMapGeneration() {
	return g_shader_map_generation.load(std::memory_order_acquire);
}

static bool ShaderMapMemoEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SHADER_MAP_MEMO");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

static ShaderMapEntry ShaderGetMappedDataLocked(uint64_t addr, const char* label,
                                                  bool* found = nullptr) {
	std::scoped_lock lock(g_shader_map_mutex);

	if (auto iter = g_shader_map->find(addr); iter != g_shader_map->end()) {
		return iter->second;
	}
	if (DrawPrep::Speculative()) {
		// A draw prepared ahead may see a shader the guest has not registered yet; the serial
		// path decides at commit.
		DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
		if (found != nullptr) {
			*found = false;
		}
		return {};
	}

	EXIT("%s shader=0x%016" PRIx64 " is missing from ShaderMap\n", label, addr);
}

static ShaderMapEntry ShaderGetMappedData(uint64_t addr, const char* label) {
	EXIT_IF(g_shader_map == nullptr);

	if (!ShaderMapMemoEnabled()) {
		return ShaderGetMappedDataLocked(addr, label);
	}
	// Per-thread memo of recent lookups (KYTY_SHADER_MAP_MEMO), tagged with the map generation
	// loaded before the locked lookup: an update racing that lookup leaves the entry stale, and
	// any later update changes the generation, so a hit always returns the current mapping.
	struct Entry {
		uint64_t         addr       = 0;
		uint64_t         generation = 0;
		ShaderMapEntry data;
	};
	static thread_local std::array<Entry, 16> memo {};
	auto&      entry      = memo[(addr >> 8u) % memo.size()];
	const auto generation = g_shader_map_generation.load(std::memory_order_acquire);
	if (entry.generation == generation && entry.addr == addr) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderMapMemoHits);
		return entry.data;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderMapMemoMisses);
	bool       found = true;
	const auto data  = ShaderGetMappedDataLocked(addr, label, &found);
	if (found) {
		entry = {addr, generation, data};
	}
	return data;
}

// KYTY_SHADER_METADATA_BATCH=0 restores one clean-backing probe per header word, vertex
// attribute and vertex-buffer descriptor.
static bool ShaderMetadataBatchEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SHADER_METADATA_BATCH");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

static const ShaderBinaryInfo* GetBinaryInfo(const uint32_t* code) {
	EXIT_IF(code == nullptr);

	if (code[0] == 0xBEEB03FF) {
		return reinterpret_cast<const ShaderBinaryInfo*>(code +
		                                                 static_cast<size_t>(code[1] + 1) * 2);
	}

	return nullptr;
}

// Draw-prep: the guest AGC metadata a stage preparation parses (the user-data header, its
// direct-resource offsets and the input semantics) is copied through the recorder, and the
// parsers consume the copies. A preparing worker therefore never dereferences guest memory, and
// the certificate covers every parsed byte (and a little more: the whole header and all
// semantics are copied even when fewer are used, which only costs occasional fallbacks).
struct PreparationMetadata {
	static constexpr uint32_t MaxEntries = 64;

	ShaderUserData                           header {};
	std::array<uint16_t, MaxEntries>        direct_offsets {};
	std::array<ShaderSemantic, MaxEntries>  semantics {};
};

// Outside a preparation returns `data` unchanged. Inside one, returns a view whose pointers refer
// to `copy`; false (the preparation failed) when a copy could not be read.
static bool CopyMetadataForPreparation(const ShaderMappedData& data, PreparationMetadata& copy,
                                       ShaderMappedData& view) {
	view = data;
	if (!DrawPrep::Speculative()) {
		return true;
	}
	const auto read = [](const void* address, void* destination, uint64_t size) {
		return size == 0 ||
		       LibKernel::Memory::TryReadGpuCleanBacking(reinterpret_cast<uint64_t>(address),
		                                                destination, size);
	};
	if (data.user_data != nullptr) {
		if (!read(data.user_data, &copy.header, sizeof(copy.header))) {
			return false;
		}
		const uint32_t direct = copy.header.direct_resource_count;
		if (direct > PreparationMetadata::MaxEntries) {
			// Beyond every known domain: the parser would reject it; let the serial path.
			DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
			return false;
		}
		if (direct != 0 && copy.header.direct_resource_offset != nullptr) {
			if (!read(copy.header.direct_resource_offset, copy.direct_offsets.data(),
			          uint64_t {direct} * sizeof(uint16_t))) {
				return false;
			}
			copy.header.direct_resource_offset = copy.direct_offsets.data();
		}
		view.user_data = &copy.header;
	}
	if (data.input_semantics != nullptr) {
		// The count lives in the host-side shader map entry.
		const uint32_t count = data.num_input_semantics;
		if (count > PreparationMetadata::MaxEntries) {
			DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
			return false;
		}
		if (!read(data.input_semantics, copy.semantics.data(),
		          uint64_t {count} * sizeof(ShaderSemantic))) {
			return false;
		}
		view.input_semantics = copy.semantics.data();
	}
	return true;
}

static uint64_t GetDeclaredShaderHash(uint64_t shader_addr) {
	static const bool use_clean_backing = [] {
		const auto* value = std::getenv("KYTY_SHADER_METADATA_BACKING");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	// A draw-prep preparation reads only through recorded probes (a clean probe returns the
	// bytes the mapped read would) and never falls back to the guest mapping.
	const bool speculative = DrawPrep::Speculative();
	if (use_clean_backing || speculative) {
		// Probe only the bytes this path consumes, in its original dependent order.
		// A shader header may share a protected page with unrelated GPU-owned bytes.
		// No values persist across calls; failed proofs retain the fault/readback path.
		const auto read_word = [speculative](const uint32_t* address) {
			uint32_t value = 0;
			const bool clean = LibKernel::Memory::TryReadGpuCleanBacking(
			    reinterpret_cast<uint64_t>(address), &value, sizeof(value));
			Profiler::CountFrameEvent(clean ? Profiler::FrameEvent::ShaderHeaderProbeHits
			                               : Profiler::FrameEvent::ShaderHeaderProbeMisses);
			return clean || speculative ? value : *address;
		};
		const auto* code = reinterpret_cast<const uint32_t*>(shader_addr);
		EXIT_IF(code == nullptr);
		// Both leading words in one silent probe. It reads the second word even when the
		// first is not the marker, which a silent probe may do: it cannot fault, and a
		// failure falls back to the exact per-word reads below.
		std::array<uint32_t, 2> head {};
		uint32_t                offset = 0;
		if (ShaderMetadataBatchEnabled() &&
		    LibKernel::Memory::TryReadGpuCleanBacking(shader_addr, head.data(), sizeof(head))) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderHeaderProbeHits);
			if (head[0] != 0xBEEB03FF) return 0;
			offset = head[1];
		} else {
			if (read_word(code) != 0xBEEB03FF) return 0;
			offset = read_word(code + 1);
		}
		const auto* header = reinterpret_cast<const ShaderBinaryInfo*>(
		    code + static_cast<size_t>(offset + 1u) * 2);
		std::array<uint32_t, 2> hashes {};
		static_assert(offsetof(ShaderBinaryInfo, hash1) == offsetof(ShaderBinaryInfo, hash0) + sizeof(uint32_t));
		const bool clean = LibKernel::Memory::TryReadGpuCleanBacking(
		    reinterpret_cast<uint64_t>(&header->hash0), hashes.data(), sizeof(hashes));
		Profiler::CountFrameEvent(clean ? Profiler::FrameEvent::ShaderHeaderProbeHits
		                               : Profiler::FrameEvent::ShaderHeaderProbeMisses);
		if (!clean && speculative) {
			return 0; // the failed probe already failed the preparation
		}
		return clean ? (static_cast<uint64_t>(hashes[1]) << 32u) | hashes[0]
		             : (static_cast<uint64_t>(header->hash1) << 32u) | header->hash0;
	}
	const auto* header = GetBinaryInfo(reinterpret_cast<const uint32_t*>(shader_addr));
	return header != nullptr ? (static_cast<uint64_t>(header->hash1) << 32u) | header->hash0 : 0;
}

// KYTY_SHADER_HASH_BACKING=0 hashes headerless shader code in place. By default the code bytes are
// first copied from the clean backing: code often shares a 4 KiB tracker page with GPU-written
// storage data, and hashing through the protected guest mapping then faults and drains the GPU
// on every draw (u37 Sky Garden: ~7.8k CP-thread faults at XXH3_64bits, ~33 ms per flip). The
// clean-backing read proves the exact code bytes are not GPU-owned; otherwise the in-place hash
// keeps the fault/readback path.
static bool ShaderHashBackingEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SHADER_HASH_BACKING");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_DRAW_PREP_CODE_CERT=0 restores failing every draw-prep preparation of a shader without an
// AGC header hash (Uncertified): the preparation then never reads or hashes the code.
static bool DrawPrepCodeCertEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DRAW_PREP_CODE_CERT");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_DRAW_PREP_CODE_DIGEST=0 certifies the code of headerless shaders by its bytes, as before,
// instead of by the digest the preparation computes from them (readSet.h, digest reads).
static bool DrawPrepCodeDigestEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DRAW_PREP_CODE_DIGEST");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// Draw-prep: the hash of shader code without an AGC header hash (every Astro Bot shader), taken
// over bytes read through the preparation's recorder, so they become part of its certificate
// (readSet.h). At commit every certified range must be clean for a backing read and hold exactly
// the recorded bytes; the serial path would then read those same bytes (HashShaderCode takes its
// clean-backing branch, and a clean range's mapping holds the backing bytes) and compute the same
// hash. A read that cannot be served fails the preparation (the serial path runs at commit).
// KYTY_DRAW_PREP_CODE_DIGEST (default on): the code is certified by this hash itself, the only
// value the preparation derives from those bytes and the key the serial path would look the
// program up by. The commit re-reads the code (same clean read) and re-hashes it instead of
// comparing it with a recorded copy, and the recorder keeps no copy of it.
// KYTY_BACKING_INPLACE: the code is hashed where it is in the backing (same gate, same bytes), not
// copied out first.
static uint64_t HashCleanBackingInPlace(uint64_t address, uint64_t size, bool& ok) {
	uint64_t                        digest = 0;
	LibKernel::Memory::InPlaceStats stats;
	ok = LibKernel::Memory::HashGpuCleanBacking(address, size, digest, &stats);
	Profiler::CountFrameEvent(Profiler::FrameEvent::BackingInPlaceHashes, stats.inspected);
	if (stats.locked != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BackingInPlaceHashesLocked, stats.locked);
	}
	return digest;
}

static uint64_t HashShaderCodeCertified(std::span<const uint32_t> code) {
	if (!DrawPrepCodeCertEnabled()) {
		DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
		return 0;
	}
	const auto address = reinterpret_cast<uint64_t>(code.data());
	if (DrawPrepCodeDigestEnabled() && LibKernel::Memory::BackingInPlaceEnabled()) {
		bool       ok     = false;
		const auto digest = HashCleanBackingInPlace(address, code.size_bytes(), ok);
		if (!ok) {
			return 0; // the recorder has failed the preparation
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderCodeHashCertified);
		return digest;
	}
	static thread_local std::vector<uint32_t> scratch;
	scratch.resize(code.size());
	if (DrawPrepCodeDigestEnabled()) {
		uint64_t digest = 0;
		if (!LibKernel::Memory::TryReadGpuCleanBackingDigest(address, scratch.data(),
		                                                    code.size_bytes(), digest)) {
			return 0; // the recorder has failed the preparation
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderCodeHashCertified);
		return digest;
	}
	if (!LibKernel::Memory::TryReadGpuCleanBacking(address, scratch.data(), code.size_bytes())) {
		return 0; // the recorder has failed the preparation
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderCodeHashCertified);
	return XXH3_64bits(scratch.data(), code.size_bytes());
}

static uint64_t HashShaderCode(std::span<const uint32_t> code) {
	if (DrawPrep::Speculative()) {
		// A draw-prep preparation must not read code outside its certificate, nor fall back to
		// the mapping: it hashes through its recorder instead.
		return HashShaderCodeCertified(code);
	}
	if (ShaderHashBackingEnabled()) {
		if (LibKernel::Memory::BackingInPlaceEnabled()) {
			bool       ok = false;
			const auto digest =
			    HashCleanBackingInPlace(reinterpret_cast<uint64_t>(code.data()), code.size_bytes(), ok);
			if (ok) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderCodeHashBacking);
				return digest;
			}
		} else {
			static thread_local std::vector<uint32_t> scratch;
			scratch.resize(code.size());
			if (LibKernel::Memory::TryReadGpuCleanBacking(reinterpret_cast<uint64_t>(code.data()),
			                                              scratch.data(), code.size_bytes())) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderCodeHashBacking);
				return XXH3_64bits(scratch.data(), code.size_bytes());
			}
		}
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderCodeHashDirect);
	return XXH3_64bits(code.data(), code.size_bytes());
}

static ShaderParams GetShaderParams(uint64_t shader_addr, const char* label, uint64_t declared_hash,
	                                std::span<const uint32_t> user_data,
	                                const ShaderMapEntry& data, uint32_t user_data_base = 0) {
	if (DrawPrep::SpeculativeFailed()) {
		return {};
	}
	if (data.code_size_bytes == 0 || data.code_size_bytes % sizeof(uint32_t) != 0) {
		EXIT("%s hash=0x%016" PRIx64 " shader=0x%016" PRIx64
		     " has invalid AGC shader_size=0x%08" PRIx32 "\n",
		     label, declared_hash, shader_addr, data.code_size_bytes);
	}
	const auto code_words = data.code_size_bytes / sizeof(uint32_t);
	const auto code = std::span {reinterpret_cast<const uint32_t*>(shader_addr), code_words};
	uint64_t effective_hash = declared_hash;
	if (RegisteredShaderCodeEnabled() && DrawPrep::Speculative()) {
		// Keep the worker's exact code certificate, including failed-read fallback at commit.
		const auto certified_hash = HashShaderCodeCertified(code);
		if (certified_hash != data.code_hash || DrawPrep::SpeculativeFailed()) {
			DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
			return {};
		}
		if (effective_hash == 0) effective_hash = certified_hash;
	}
	// A draw-prep preparation hashes headerless code through its recorder (the code becomes part
	// of its certificate; KYTY_DRAW_PREP_CODE_CERT=0 fails it instead).
	ShaderParams params {
	    .code            = code,
	    .user_data_count = static_cast<uint32_t>(user_data.size()) + user_data_base,
	    .hash            = effective_hash != 0 ? effective_hash : HashShaderCode(code),
	};
	EXIT_IF(user_data.size() > HW::UserSgprInfo::SGPRS_MAX ||
	        params.user_data_count > params.user_data.size());
	std::copy(user_data.begin(), user_data.end(), params.user_data.begin() + user_data_base);
	// KYTY_GPU_BREADCRUMBS: names the shaders of the operations a device-loss report lists.
	GpuBreadcrumbs::NoteShader(shader_addr, params.hash);
	return params;
}

#if 0
// Kept as disabled debugging guards for investigating unusual stage register state.
static void vs_check(const HW::VertexShaderInfo& vs, const HW::ShaderRegisters& sh) {
	const auto is_zero_or_wave64_subgroup = [](uint32_t value) {
		return value == 0 || value <= 0x40;
	};
	const auto is_known_gs_out_prim_type = [](uint32_t value) {
		switch (static_cast<Prospero::GsOutputPrimitiveType>(value)) {
			case Prospero::GsOutputPrimitiveType::kPoints:
			case Prospero::GsOutputPrimitiveType::kLines:
			case Prospero::GsOutputPrimitiveType::kTriangles:
			case Prospero::GsOutputPrimitiveType::k2dRectangle:
			case Prospero::GsOutputPrimitiveType::kRectList: return true;
		}

		return false;
	};
	const bool ps5_ngg_passthrough_triangle_path =
	    vs.es_regs.data_addr != 0 && vs.gs_regs.data_addr == vs.es_regs.data_addr &&
	    sh.m_geNggSubgrpCntl == 0x00000001 && sh.m_vgtGsMaxVertOut == 0x00000003 &&
	    sh.m_vgtGsOutPrimType == 0x00000002 && sh.m_geMaxOutputPerSubgroup <= 0x000000c0;

	if (vs.es_regs.data_addr != 0 || vs.gs_regs.data_addr != 0) {
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.priority != 0);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.float_mode != 192);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.dx10_clamp != true);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.debug_mode != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.ieee_mode != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.cu_group_enable != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.require_forward_progress != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.threadgroup_configuration != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.gs_vgpr_component_count != 3);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc1.fp16_overflow != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.scratch_en != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.offchip_lds != false);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.es_vgpr_component_count != 3);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.lds_size != 0);
		EXIT_NOT_IMPLEMENTED(vs.gs_regs.rsrc2.shared_vgprs != 0);
	}

	for (uint32_t value = sh.m_spiShaderPosFormat; value != 0; value >>= 4u) {
		EXIT_NOT_IMPLEMENTED((value & 0xfu) != 0 && (value & 0xfu) != 0x4u);
	}
	if (sh.m_paClVsOutCntl != 0x00000000) {
		static bool logged = false;
		if (!logged) {
			LOGF("\t temporary: accepting PA_CL_VS_OUT_CNTL = 0x%08" PRIx32 "\n",
			     sh.m_paClVsOutCntl);
			logged = true;
		}
	}

	EXIT_NOT_IMPLEMENTED(sh.m_spiShaderIdxFormat != 0x00000000 &&
	                     sh.m_spiShaderIdxFormat != 0x00000001);
	EXIT_NOT_IMPLEMENTED(sh.m_geNggSubgrpCntl != 0x00000000 && sh.m_geNggSubgrpCntl != 0x00000001);
	EXIT_NOT_IMPLEMENTED(sh.m_vgtGsInstanceCnt != 0x00000000);
	EXIT_NOT_IMPLEMENTED(!is_zero_or_wave64_subgroup(sh.GetEsVertsPerSubgrp()));
	EXIT_NOT_IMPLEMENTED(!is_zero_or_wave64_subgroup(sh.GetGsPrimsPerSubgrp()));
	EXIT_NOT_IMPLEMENTED(!is_zero_or_wave64_subgroup(sh.GetGsInstPrimsInSubgrp()));
	EXIT_NOT_IMPLEMENTED(!is_zero_or_wave64_subgroup(sh.m_geMaxOutputPerSubgroup) &&
	                     !ps5_ngg_passthrough_triangle_path);
	EXIT_NOT_IMPLEMENTED(sh.m_vgtEsgsRingItemsize != 0x00000000 &&
	                     sh.m_vgtEsgsRingItemsize != 0x00000004);
	EXIT_NOT_IMPLEMENTED(sh.m_vgtGsMaxVertOut != 0x00000000 && !ps5_ngg_passthrough_triangle_path);
	EXIT_NOT_IMPLEMENTED(!is_known_gs_out_prim_type(sh.m_vgtGsOutPrimType));
}

static void ps_check(const HW::PsStageRegisters& ps, const HW::ShaderRegisters& sh) {
	if (sh.target_output_mode[0] != 0 && sh.target_output_mode[0] != 2 &&
	    sh.target_output_mode[0] != 4 && sh.target_output_mode[0] != 5 &&
	    sh.target_output_mode[0] != 7 && sh.target_output_mode[0] != 9) {
		EXIT("Not implemented (sh.target_output_mode[0] != 0 && sh.target_output_mode[0] != 2 && "
		     "sh.target_output_mode[0] != 4 && sh.target_output_mode[0] != 5 && "
		     "sh.target_output_mode[0] != 7 && sh.target_output_mode[0] != 9)\n");
	}
	EXIT_NOT_IMPLEMENTED(sh.db_shader_control.conservative_z_export_value != 0x00000000);
	EXIT_NOT_IMPLEMENTED(sh.db_shader_control.shader_z_behavior != 0x00000001 &&
	                     sh.db_shader_control.shader_z_behavior != 0x00000000);
	// EXIT_NOT_IMPLEMENTED(ps.shader_kill_enable != false);
	// EXIT_NOT_IMPLEMENTED(ps.shader_execute_on_noop != false);
	// EXIT_NOT_IMPLEMENTED(ps.m_spiShaderPgmRsrc1Ps != 0x002c0000);
	// EXIT_NOT_IMPLEMENTED(ps.m_spiShaderPgmRsrc2Ps != 0x00000000);
	// EXIT_NOT_IMPLEMENTED(ps.vgprs != 0x00 && ps.vgprs != 0x01);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.priority != 0);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.float_mode != 192);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.dx10_clamp != true);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.debug_mode != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.ieee_mode != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.cu_group_disable != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.require_forward_progress != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc1.fp16_overflow != false);
	EXIT_NOT_IMPLEMENTED(ps.rsrc2.scratch_en != false);
	// EXIT_NOT_IMPLEMENTED(ps.user_sgpr != 0 && ps.user_sgpr != 4 && ps.user_sgpr != 12);
	EXIT_NOT_IMPLEMENTED(ps.rsrc2.wave_cnt_en != false);
	if (ps.rsrc2.extra_lds_size != 0) {
		static std::atomic_uint log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("\t PS extra LDS reservation = 0x%02" PRIx8 ", continuing\n",
			     ps.rsrc2.extra_lds_size);
		}
	}
	EXIT_NOT_IMPLEMENTED(ps.rsrc2.raster_ordered_shading != 0);
	EXIT_NOT_IMPLEMENTED(ps.rsrc2.shared_vgprs != 0);

	if (sh.shader_z_format != 0x00000000 && sh.shader_z_format != 0x00000001 &&
	    !sh.db_shader_control.shader_z_export_enable) {
		static std::atomic_uint log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("\t shader_z_format = 0x%08" PRIx32
			     " with z export disabled, ignoring depth export format\n",
			     sh.shader_z_format);
		}
	}
	EXIT_NOT_IMPLEMENTED(sh.db_shader_control.shader_z_export_enable &&
	                     sh.shader_z_format != 0x00000000 && sh.shader_z_format != 0x00000001);
	constexpr uint32_t ps_input_linear_center = 0x00000020u;
	constexpr uint32_t ps_input_pos_w         = 0x00000800u;
	constexpr uint32_t ps_input_front_face    = 0x00001000u;
	constexpr uint32_t supported_ps_input_bits =
	    0x00000702u | ps_input_linear_center | ps_input_pos_w | ps_input_front_face;
	EXIT_NOT_IMPLEMENTED((sh.ps_input_ena & ~supported_ps_input_bits) != 0);
	EXIT_NOT_IMPLEMENTED((sh.ps_input_addr & ~supported_ps_input_bits) != 0);
	EXIT_NOT_IMPLEMENTED(sh.ps_input_ena != sh.ps_input_addr);
	// EXIT_NOT_IMPLEMENTED(ps.m_spiPsInControl != 0x00000000);
	constexpr uint32_t baryc_persp_mask =
	    0x00000003u | 0x00000030u | 0x00000300u | 0x00003000u;
	constexpr uint32_t baryc_linear_mask = 0x00030000u | 0x00300000u | 0x03000000u;
	constexpr uint32_t baryc_known_mask  = baryc_persp_mask | baryc_linear_mask;
	EXIT_NOT_IMPLEMENTED((sh.baryc_cntl & ~baryc_known_mask) != 0);
	EXIT_NOT_IMPLEMENTED((sh.baryc_cntl & baryc_persp_mask) != 0);
	if ((sh.ps_input_ena & ps_input_linear_center) == 0 && (sh.baryc_cntl & baryc_linear_mask) != 0) {
		static std::atomic_uint log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("\t ignoring inactive linear SPI_BARYC_CNTL bits: 0x%08" PRIx32 "\n",
			     sh.baryc_cntl & baryc_linear_mask);
		}
	} else {
		EXIT_NOT_IMPLEMENTED((sh.baryc_cntl & baryc_linear_mask) != 0x00000000 &&
		                     (sh.baryc_cntl & baryc_linear_mask) != 0x01000000);
	}
	if ((sh.m_cbShaderMask & 0x0000000f) != 0x0000000f) {
		static bool logged = false;
		if (!logged) {
			LOGF("\t temporary: accepting partial CB_SHADER_MASK = 0x%08" PRIx32 "\n",
			     sh.m_cbShaderMask);
			logged = true;
		}
	}
	if ((sh.m_cbShaderMask & ~0x0000000fu) != 0) {
		static bool logged = false;
		if (!logged) {
			LOGF("\t temporary: ignoring extra CB_SHADER_MASK MRT bits: 0x%08" PRIx32 "\n",
			     sh.m_cbShaderMask);
			logged = true;
		}
	}

	if (sh.db_shader_control.other_bits != 0x00000000) {
		static std::atomic_uint log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("\t temporary: ignoring unsupported DB_SHADER_CONTROL bits 0x%08" PRIx32 "\n",
			     sh.db_shader_control.other_bits);
		}
	}
	EXIT_NOT_IMPLEMENTED(sh.m_paScShaderControl != 0x00000000);
}

static void cs_check(const HW::CsStageRegisters& cs, const HW::ShaderRegisters& /*sh*/) {
	// EXIT_NOT_IMPLEMENTED(cs.num_thread_x != 0x00000040);
	// EXIT_NOT_IMPLEMENTED(cs.num_thread_y != 0x00000001);
	// EXIT_NOT_IMPLEMENTED(cs.num_thread_z != 0x00000001);
	// EXIT_NOT_IMPLEMENTED(cs.vgprs != 0x00 && cs.vgprs != 0x01);
	EXIT_NOT_IMPLEMENTED(cs.priority != 0x00);
	EXIT_NOT_IMPLEMENTED(cs.debug_mode != false);
	EXIT_NOT_IMPLEMENTED(cs.require_forward_progress != false);
	EXIT_NOT_IMPLEMENTED(cs.shared_vgprs != 0x00);
	EXIT_NOT_IMPLEMENTED(cs.scratch_en != 0x00);
	// EXIT_NOT_IMPLEMENTED(cs.user_sgpr != 0x0c);
	if (cs.tgid_x_en == 0x00) {
		static bool logged = false;
		if (!logged) {
			LOGF("\t temporary: compute shader has TGID X disabled\n");
			logged = true;
		}
	} else {
		EXIT_NOT_IMPLEMENTED(cs.tgid_x_en != 0x01);
	}
	// EXIT_NOT_IMPLEMENTED(cs.tgid_y_en != 0x00);
	// EXIT_NOT_IMPLEMENTED(cs.tgid_z_en != 0x00);
	EXIT_NOT_IMPLEMENTED(cs.tg_size_en != 0x00);
	EXIT_NOT_IMPLEMENTED(cs.tidig_comp_cnt > 2);

	//	EXIT_NOT_IMPLEMENTED(cs.m_computePgmRsrc1 != 0x002c0040);
	//	EXIT_NOT_IMPLEMENTED(cs.m_computePgmRsrc2 != 0x00000098);
	//	EXIT_NOT_IMPLEMENTED(cs.m_computeNumThreadX != 0x00000040);
	//	EXIT_NOT_IMPLEMENTED(cs.m_computeNumThreadY != 0x00000001);
	//	EXIT_NOT_IMPLEMENTED(cs.m_computeNumThreadZ != 0x00000001);
}
#endif

static void ShaderDetectBuffers(ShaderVertexInputInfo& info) {
	KYTY_PROFILER_FUNCTION();

	info.buffers_num = 0;

	// A failed draw-prep preparation is discarded; its partial tables may be stale.
	const bool speculative = DrawPrep::Speculative();
	if (speculative && DrawPrep::ActiveRecorder()->reads->Failed()) {
		return;
	}

	for (int ri = 0; ri < info.resources_num; ri++) {
		const auto& r = info.resources[ri];

		bool merged = false;
		for (int bi = 0; bi < info.buffers_num; bi++) {
			auto& b = info.buffers[bi];

			uint64_t stride = b.stride;

			if (stride == r.Stride() &&
			    b.fetch_index == static_cast<uint32_t>(info.resources_dst[ri].fetch_index)) {
				uint64_t rbase   = r.Base48();
				uint64_t base    = std::min(rbase, b.addr);
				uint64_t offset1 = rbase - base;
				uint64_t offset2 = b.addr - base;

				if (offset1 < stride && offset2 < stride) {
					if (speculative && (b.num_records != r.NumRecords() ||
					                    b.attr_num >= ShaderVertexInputBuffer::ATTR_MAX)) {
						DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
						return;
					}
					EXIT_NOT_IMPLEMENTED(b.num_records != r.NumRecords());
					b.addr = base;
					EXIT_NOT_IMPLEMENTED(b.attr_num >= ShaderVertexInputBuffer::ATTR_MAX);
					b.attr_indices[b.attr_num++] = ri;
					merged                       = true;
					break;
				}
			}
		}

		if (!merged) {
			EXIT_NOT_IMPLEMENTED(info.buffers_num >= ShaderVertexInputInfo::RES_MAX);
			int bi                           = info.buffers_num++;
			info.buffers[bi].addr            = r.Base48();
			info.buffers[bi].stride          = r.Stride();
			info.buffers[bi].num_records     = r.NumRecords();
			info.buffers[bi].fetch_index     = info.resources_dst[ri].fetch_index;
			info.buffers[bi].attr_num        = 1;
			info.buffers[bi].attr_indices[0] = ri;
		}
	}

	for (int bi = 0; bi < info.buffers_num; bi++) {
		auto& b = info.buffers[bi];
		for (int ri = 0; ri < b.attr_num; ri++) {
			b.attr_offsets[ri] = info.resources[b.attr_indices[ri]].Base48() - b.addr;
		}
	}
}

// Opt-in, bounded forensic output. All guest bytes use non-faulting backing reads;
// this captures evidence only and never supplies replacement renderer data.
static void DumpVertexFailure(uint64_t shader_addr, const ShaderMappedData& data,
                              const HW::UserSgprInfo& user_sgpr, uint32_t user_sgpr_num,
                              const ShaderVertexMetadata& metadata, uint64_t attrib,
                              uint64_t buffer, bool speculative) {
	static const char* prefix = std::getenv("KYTY_VERTEX_FAILURE_DUMP");
	if (prefix == nullptr || *prefix == '\0') return;
	static std::atomic<uint32_t> count {0};
	const auto index = count.fetch_add(1, std::memory_order_relaxed);
	if (index >= 8) return;
	const auto path = std::string(prefix) + "-" + std::to_string(index) + ".txt";
	auto* file = std::fopen(path.c_str(), "w");
	if (file == nullptr) return;
	std::fprintf(file, "shader=%016" PRIx64 " code_bytes=%u speculative=%u "
	                  "sgpr_count=%u attrib_reg=%d buffer_reg=%d semantics=%u\n",
	             shader_addr, data.code_size_bytes, static_cast<unsigned>(speculative),
	             user_sgpr_num, metadata.vertex_attrib_reg, metadata.vertex_buffer_reg,
	             metadata.input_semantics_count);
	for (uint32_t i = 0; i < HW::UserSgprInfo::SGPRS_MAX; ++i) {
		std::fprintf(file, "sgpr[%u]=%08x\n", i, user_sgpr.value[i]);
	}
	for (uint32_t i = 0; i < metadata.input_semantics_count; ++i) {
		uint32_t word = 0;
		static_assert(sizeof(ShaderSemantic) == sizeof(word));
		std::memcpy(&word, &metadata.input_semantics[i], sizeof(word));
		std::fprintf(file, "semantic[%u]=%08x\n", i, word);
	}
	const auto dump_words = [file](const char* label, uint64_t address) {
		std::fprintf(file, "%s=%016" PRIx64 " (128 bytes before, 512 bytes from pointer)\n",
		             label, address);
		if (address < 128 || address > UINT64_MAX - 512) return;
		for (int offset = -128; offset < 512; offset += 16) {
			std::array<uint32_t, 4> words {};
			const auto current = offset < 0 ? address - static_cast<uint64_t>(-offset)
			                                : address + static_cast<uint64_t>(offset);
			const bool read = LibKernel::Memory::TryReadBacking(current, words.data(), sizeof(words));
			std::fprintf(file, "%016" PRIx64 " read=%u %08x %08x %08x %08x\n", current,
			             static_cast<unsigned>(read), words[0], words[1], words[2], words[3]);
		}
	};
	dump_words("attribute_table", attrib);
	dump_words("buffer_table", buffer);
	std::fclose(file);
	const auto code_size = std::min<uint32_t>(data.code_size_bytes, 64u * 1024u);
	std::vector<uint8_t> code(code_size);
	if (code_size != 0 && LibKernel::Memory::TryReadBacking(shader_addr, code.data(), code_size)) {
		const auto code_path = path + ".shader.bin";
		if (auto* code_file = std::fopen(code_path.c_str(), "wb")) {
			std::fwrite(code.data(), 1, code.size(), code_file);
			std::fclose(code_file);
		}
	}
	std::fprintf(stderr, "VertexFailureDump: %s\n", path.c_str());
	std::fflush(stderr);
}

static void ShaderApplyAttribSemantics(ShaderVertexInputInfo& info,
                                       const ShaderSemantic*  input_semantics,
                                       uint32_t num_input_semantics, const uint32_t* attrib,
                                       const uint32_t* buffer, uint64_t shader_addr,
                                       const ShaderMappedData& data,
                                       const HW::UserSgprInfo& user_sgpr, uint32_t user_sgpr_num,
                                       const ShaderVertexMetadata& metadata) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(attrib == nullptr || buffer == nullptr);

	const bool debug_dump = Config::GraphicsDebugDumpEnabled();
	static const bool use_clean_backing_env = [] {
		const auto* value = std::getenv("KYTY_SHADER_METADATA_BACKING");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	// A draw-prep preparation reads only through recorded probes and stops at the first failed
	// one (which already failed the preparation) instead of reading the guest mapping.
	const bool speculative       = DrawPrep::Speculative();
	const bool use_clean_backing = use_clean_backing_env || speculative;
	EXIT_NOT_IMPLEMENTED(num_input_semantics > ShaderVertexInputInfo::RES_MAX);
	// Diagnostics of refused reads (HangTrace unclean.csv): the attribute table, then the
	// vertex-buffer descriptors.
	DrawPrep::ScopedReadPurpose read_purpose("vertex-attributes");

	// KYTY_SHADER_METADATA_BATCH: read the used span of each table with one silent probe. A
	// successful probe proves every word of the span clean, so each word equals what its own
	// probe returns; a failed probe falls back to exactly the per-word reads.
	const bool batch = use_clean_backing && ShaderMetadataBatchEnabled() && num_input_semantics > 1;
	constexpr uint32_t          MaxBatchWords = 64;
	std::array<uint32_t, MaxBatchWords> attribute_words {};
	uint32_t                    attribute_first   = 0;
	bool                        attributes_batched = false;
	if (batch) {
		uint32_t first = UINT32_MAX;
		uint32_t last  = 0;
		for (uint32_t i = 0; i < num_input_semantics; i++) {
			first = std::min<uint32_t>(first, input_semantics[i].semantic);
			last  = std::max<uint32_t>(last, input_semantics[i].semantic);
		}
		if (last - first < MaxBatchWords) {
			attribute_first    = first;
			attributes_batched = LibKernel::Memory::TryReadGpuCleanBacking(
			    reinterpret_cast<uint64_t>(attrib + first), attribute_words.data(),
			    (last - first + 1u) * sizeof(uint32_t));
			Profiler::CountFrameEvent(attributes_batched ? Profiler::FrameEvent::VertexTableBatchHits
			                                             : Profiler::FrameEvent::VertexTableBatchMisses);
		}
	}

	// Pass 1: validate each semantic and read its attribute word, in the original order.
	std::array<uint32_t, ShaderVertexInputInfo::RES_MAX> attributes {};
	for (uint32_t i = 0; i < num_input_semantics; i++) {
		const auto& in = input_semantics[i];

		if (speculative && (in.static_vb_index == 1 || in.static_attribute == 1)) {
			DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
			return;
		}
		EXIT_NOT_IMPLEMENTED(in.static_vb_index == 1 || in.static_attribute == 1);

		uint32_t attribute = 0;
		if (attributes_batched) {
			attribute = attribute_words[in.semantic - attribute_first];
		} else {
			const bool attribute_clean = use_clean_backing &&
			    LibKernel::Memory::TryReadGpuCleanBacking(
			        reinterpret_cast<uint64_t>(attrib + in.semantic), &attribute, sizeof(attribute));
			if (use_clean_backing) {
				Profiler::CountFrameEvent(attribute_clean
				                              ? Profiler::FrameEvent::VertexMetadataProbeHits
				                              : Profiler::FrameEvent::VertexMetadataProbeMisses);
			}
			if (!attribute_clean) {
				if (speculative) return;
				attribute = attrib[in.semantic];
			}
		}
		attributes[i] = attribute;

		if (debug_dump) {
			LOGF("reg = %u, size = %u, va[%u] = 0x%08" PRIx32 "\n",
			     static_cast<uint32_t>(in.hardware_mapping),
			     static_cast<uint32_t>(in.size_in_elements), i, attribute);
		}
	}

	// The vertex-buffer descriptors those attributes select, again as one span when possible.
	read_purpose.Set("vertex-buffers");
	std::array<uint32_t, MaxBatchWords * 2> sharp_words {};
	uint32_t                                sharp_first  = 0;
	bool                                    sharps_batched = false;
	if (batch) {
		uint32_t first = UINT32_MAX;
		uint32_t last  = 0;
		bool     valid = true;
		for (uint32_t i = 0; i < num_input_semantics; i++) {
			const uint32_t index = attributes[i] & 0x1fu;
			valid &= index < static_cast<uint32_t>(ShaderVertexInputInfo::RES_MAX);
			first = std::min(first, index);
			last  = std::max(last, index);
		}
		if (valid && (last - first + 1u) * 4u <= sharp_words.size()) {
			sharp_first    = first;
			sharps_batched = LibKernel::Memory::TryReadGpuCleanBacking(
			    reinterpret_cast<uint64_t>(buffer + static_cast<size_t>(first) * 4u),
			    sharp_words.data(), (last - first + 1u) * 4u * sizeof(uint32_t));
			Profiler::CountFrameEvent(sharps_batched ? Profiler::FrameEvent::VertexTableBatchHits
			                                         : Profiler::FrameEvent::VertexTableBatchMisses);
		}
	}

	for (uint32_t i = 0; i < num_input_semantics; i++) {
		const auto& in = input_semantics[i];

		uint32_t reg  = in.hardware_mapping;
		uint32_t size = in.size_in_elements;
		const uint32_t attribute = attributes[i];

		size_t index = attribute & 0x1fu;
		auto   format =
		    static_cast<Prospero::VertexAttribFormat>((attribute >> 5u) & 0x1ffu);
		uint32_t offset      = (attribute >> 14u) & 0xfffu;
		uint32_t fetch_index = (attribute >> 26u) & 0x1u;

		if (speculative && index >= ShaderVertexInputInfo::RES_MAX) {
			// Possibly stale bytes: let the serial path decide.
			DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
			return;
		}
		EXIT_NOT_IMPLEMENTED(index >= ShaderVertexInputInfo::RES_MAX);

		const auto* sharp = &buffer[index * 4];

		EXIT_NOT_IMPLEMENTED(info.resources_num >= ShaderVertexInputInfo::RES_MAX);

		auto& r           = info.resources[info.resources_num];
		auto& rd          = info.resources_dst[info.resources_num];
		rd.register_start = static_cast<int>(reg);
		rd.registers_num  = static_cast<int>(size);
		rd.attr_id        = static_cast<int>(in.semantic);
		rd.fetch_index    = fetch_index;
		bool descriptor_clean = false;
		if (sharps_batched) {
			std::copy_n(sharp_words.data() + (index - sharp_first) * 4u, 4, r.fields);
			descriptor_clean = true;
		} else {
			descriptor_clean = use_clean_backing &&
			    LibKernel::Memory::TryReadGpuCleanBacking(reinterpret_cast<uint64_t>(sharp),
			                                             r.fields, 4 * sizeof(uint32_t));
			if (use_clean_backing) {
				Profiler::CountFrameEvent(descriptor_clean
				                              ? Profiler::FrameEvent::VertexMetadataProbeHits
				                              : Profiler::FrameEvent::VertexMetadataProbeMisses);
			}
		}
		if (!descriptor_clean) {
			if (speculative) return;
			r.fields[0] = sharp[0];
			r.fields[1] = sharp[1];
			r.fields[2] = sharp[2];
			r.fields[3] = sharp[3];
		}
		EXIT_NOT_IMPLEMENTED(r.AddTid());
		EXIT_NOT_IMPLEMENTED(r.SwizzleEnabled());
		if (format != Prospero::VertexAttribFormat::kInvalid) {
			const auto                   format_raw    = static_cast<uint32_t>(format);
			const auto                   buffer_format = format_raw >> 2u;
			const auto                   channels      = (format_raw & 3u) + 1u;
			// AGC vertex formats encode the buffer format above the two channel-count bits.
			// The fetch prolog selects X001, XY01, XYZ1, or XYZW from that count.
			r.fields[3] = (r.fields[3] & ~((0x7fu << 12u) | 0xfffu)) |
			              (buffer_format << 12u) |
			              DstSel(4, channels > 1u ? 5u : 0u, channels > 2u ? 6u : 0u,
			                     channels > 3u ? 7u : 1u);
		}
		// Speculative table reads can observe data before its producing draw has committed.
		// Such a descriptor must fall back to ordered preparation, not reach pipeline creation.
		// Use the same component rules as TranslateEmbeddedFetch: constants need no
		// memory format, and selectors beyond this semantic's width are not consumed.
		const auto format_info = ShaderRecompiler::Format::GetFormatInfo(r.Format());
		bool invalid_descriptor = size > 4;
		for (uint32_t component = 0; component < std::min(size, 4u); component++) {
			invalid_descriptor |= ShaderRecompiler::Format::ResolveFormattedSource(
			    format_info, GetDstSel(r.DstSelXYZW(), component)).kind ==
			    ShaderRecompiler::Format::FormattedSourceKind::Invalid;
		}
		if (invalid_descriptor) {
			DumpVertexFailure(shader_addr, data, user_sgpr, user_sgpr_num, metadata,
			                  reinterpret_cast<uint64_t>(attrib),
			                  reinterpret_cast<uint64_t>(buffer), speculative);
			static std::atomic<uint32_t> invalid_logs {0};
			if (invalid_logs.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::fprintf(stderr, "VertexDescriptorInvalid: speculative=%u clean=%u batched=%u "
				     "attrib_table=0x%016" PRIx64 " buffer_table=0x%016" PRIx64
				     " semantic=%u attribute=0x%08x descriptor_index=%zu "
				     "descriptor=%08x,%08x,%08x,%08x format=%u selectors=%u,%u,%u,%u\n",
				     static_cast<unsigned>(speculative), static_cast<unsigned>(descriptor_clean),
				     static_cast<unsigned>(sharps_batched), reinterpret_cast<uint64_t>(attrib),
				     reinterpret_cast<uint64_t>(buffer), static_cast<unsigned>(in.semantic),
				     attribute, index, r.fields[0], r.fields[1], r.fields[2], r.fields[3],
				     static_cast<unsigned>(r.RawFormat()), static_cast<unsigned>(r.DstSelX()),
				     static_cast<unsigned>(r.DstSelY()), static_cast<unsigned>(r.DstSelZ()),
				     static_cast<unsigned>(r.DstSelW()));
				uint32_t live_attribute = 0;
				std::array<uint32_t, 4> live_descriptor {};
				const bool live_attribute_read = LibKernel::Memory::TryReadBacking(
				    reinterpret_cast<uint64_t>(attrib + in.semantic), &live_attribute,
				    sizeof(live_attribute));
				const bool live_descriptor_read = LibKernel::Memory::TryReadBacking(
				    reinterpret_cast<uint64_t>(sharp), live_descriptor.data(),
				    sizeof(live_descriptor));
				std::fprintf(stderr, "VertexDescriptorBacking: attribute_read=%u attribute=%08x "
				    "descriptor_read=%u descriptor=%08x,%08x,%08x,%08x\n",
				    static_cast<unsigned>(live_attribute_read), live_attribute,
				    static_cast<unsigned>(live_descriptor_read), live_descriptor[0],
				    live_descriptor[1], live_descriptor[2], live_descriptor[3]);
				std::fflush(stderr);
			}
			if (speculative) {
				DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
				return;
			}
		}
		if (offset != 0) {
			r.UpdateAddress48(r.Base48() + offset);
		}

		info.resources_num++;
	}
}

static uint32_t ShaderCalcPsSystemInputBase(const HW::ShaderRegisters& regs) {
	constexpr uint32_t ps_input_persp_sample    = 0x00000001u;
	constexpr uint32_t ps_input_persp_center    = 0x00000002u;
	constexpr uint32_t ps_input_persp_centroid  = 0x00000004u;
	constexpr uint32_t ps_input_persp_pull      = 0x00000008u;
	constexpr uint32_t ps_input_linear_sample   = 0x00000010u;
	constexpr uint32_t ps_input_linear_center   = 0x00000020u;
	constexpr uint32_t ps_input_linear_centroid = 0x00000040u;
	constexpr uint32_t ps_input_line_stipple    = 0x00000080u;
	constexpr uint32_t ps_input_pos_xy          = 0x00000300u;
	constexpr uint32_t ps_input_pos_z           = 0x00000400u;
	constexpr uint32_t ps_input_pos_w           = 0x00000800u;
	constexpr uint32_t ps_input_front_face      = 0x00001000u;
	constexpr uint32_t ps_input_ancillary       = 0x00002000u;
	constexpr uint32_t ps_input_sample_coverage = 0x00004000u;
	constexpr uint32_t ps_input_pos_fixed_pt    = 0x00008000u;
	constexpr uint32_t supported_ps_input_bits =
	    ps_input_persp_sample | ps_input_persp_center | ps_input_persp_centroid |
	    ps_input_persp_pull | ps_input_linear_sample | ps_input_linear_center |
	    ps_input_linear_centroid | ps_input_line_stipple | ps_input_pos_xy | ps_input_pos_z |
	    ps_input_pos_w | ps_input_front_face | ps_input_ancillary | ps_input_sample_coverage |
	    ps_input_pos_fixed_pt;

	EXIT_NOT_IMPLEMENTED((regs.ps_input_ena & ~supported_ps_input_bits) != 0);
	EXIT_NOT_IMPLEMENTED((regs.ps_input_addr & ~supported_ps_input_bits) != 0);
	EXIT_NOT_IMPLEMENTED(regs.ps_input_ena != regs.ps_input_addr);

	const uint32_t inputs = regs.ps_input_addr;
	uint32_t       reg    = 0;
	if ((inputs & ps_input_persp_sample) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_persp_center) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_persp_centroid) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_persp_pull) != 0) {
		reg += 3;
	}
	if ((inputs & ps_input_linear_sample) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_linear_center) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_linear_centroid) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_line_stipple) != 0) {
		reg += 1;
	}
	return reg;
}

static bool ShaderGetStaticVertexInputInfo(uint64_t shader_addr, const HW::UserSgprInfo& user_sgpr,
                                           uint32_t user_sgpr_num, const HW::ShaderRegisters& sh,
                                           const ShaderMappedData& data,
                                           ShaderVertexInputInfo&  info) {
	KYTY_PROFILER_FUNCTION();

	info = {};

	info.pa_cl_vs_out_cntl = sh.m_paClVsOutCntl;

	EXIT_NOT_IMPLEMENTED(shader_addr == 0);
	info.scratch_size_dwords = data.scratch_size_dwords;

	if (data.user_data == nullptr) {
		LOGF("ShaderGetVertexInputInfo(): no AGC user data for shader=0x%016" PRIx64
		     " user_sgpr_num=%u\n",
		     shader_addr, user_sgpr_num);
	}
	ShaderVertexMetadata metadata;
	std::string          metadata_error;
	if (!ShaderReadVertexMetadata(data, HW::UserSgprInfo::SGPRS_MAX, metadata, &metadata_error)) {
		LOGF("ShaderGetVertexInputInfo(): invalid AGC metadata shader=0x%016" PRIx64 ": %s\n",
		     shader_addr, metadata_error.c_str());
		return false;
	}

	if (metadata.vertex_attrib_reg >= 0) {
		info.fetch_external   = false;
		info.fetch_embedded   = true;
		info.fetch_attrib_reg = metadata.vertex_attrib_reg;
		info.fetch_buffer_reg = metadata.vertex_buffer_reg;

		const auto* attrib = reinterpret_cast<const uint32_t*>(
		    static_cast<uint64_t>(user_sgpr.value[metadata.vertex_attrib_reg]) |
		    (static_cast<uint64_t>(user_sgpr.value[metadata.vertex_attrib_reg + 1]) << 32u));
		const auto* buffer = reinterpret_cast<const uint32_t*>(
		    static_cast<uint64_t>(user_sgpr.value[metadata.vertex_buffer_reg]) |
		    (static_cast<uint64_t>(user_sgpr.value[metadata.vertex_buffer_reg + 1]) << 32u));

		if (attrib == nullptr || buffer == nullptr) {
			LOGF("ShaderGetVertexInputInfo(): null vertex table pointer shader=0x%016" PRIx64 "\n",
			     shader_addr);
			return false;
		}
		ShaderApplyAttribSemantics(info, metadata.input_semantics.data(),
		                           metadata.input_semantics_count, attrib, buffer, shader_addr,
		                           data, user_sgpr, user_sgpr_num, metadata);
		if (DrawPrep::SpeculativeFailed()) return false;
		ShaderDetectBuffers(info);
	}
	return true;
}

static void ShaderGetStaticInputInfoPS(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	const ShaderMappedData& data, ShaderPixelInputInfo& ps_info) {
	KYTY_PROFILER_FUNCTION();

	ps_info = {};
	ps_info.scratch_size_dwords = data.scratch_size_dwords;

	// SPI_PS_IN_CONTROL: NUM_INTERP occupies bits 5:0 and PS_W32_EN is bit 15.
	ps_info.input_num            = sh.ps_in_control & 0x3fu;
	if ((sh.ps_in_control & 0x8000u) != 0) {
		ps_info.wave_size = 32;
	}
	EXIT_NOT_IMPLEMENTED(ps_info.input_num > std::size(ps_info.interpolator_settings));
	ps_info.ps_system_input_base = ShaderCalcPsSystemInputBase(sh);
	const uint32_t active_inputs = sh.ps_input_ena & sh.ps_input_addr;
	if ((active_inputs & 0x00000002u) != 0) {
		ps_info.ps_perspective_center_vgpr = (active_inputs & 0x00000001u) != 0 ? 2u : 0u;
	}
	if ((active_inputs & 0x00000004u) != 0) {
		ps_info.ps_perspective_centroid_vgpr = 2u * std::popcount(active_inputs & 0x3u);
	}
	// SPI_PS_INPUT_ADDR order: persp sample, center, centroid (2 VGPRs each), persp pull (3),
	// linear sample, center, centroid (2 each).
	if ((active_inputs & 0x00000001u) != 0) {
		ps_info.ps_perspective_sample_vgpr = 0u;
	}
	const uint32_t linear_base = 2u * std::popcount(active_inputs & 0x7u) +
	                             ((active_inputs & 0x00000008u) != 0 ? 3u : 0u);
	if ((active_inputs & 0x00000010u) != 0) {
		ps_info.ps_linear_sample_vgpr = linear_base;
	}
	if ((active_inputs & 0x00000020u) != 0) {
		ps_info.ps_linear_center_vgpr = linear_base + 2u * std::popcount(active_inputs & 0x10u);
	}
	if ((active_inputs & 0x00000040u) != 0) {
		ps_info.ps_linear_centroid_vgpr = linear_base + 2u * std::popcount(active_inputs & 0x30u);
	}
	for (uint32_t i = 0; i < data.num_input_semantics && i < ps_info.input_num && i < 32u; i++) {
		const auto& semantic = data.input_semantics[i];
		if (semantic.is_custom != 0 && semantic.is_f16 == 0) {
			ps_info.custom_interpolation_mask |= 1u << i;
		}
	}
	ps_info.ps_pos_x                     = (active_inputs & 0x00000100u) != 0;
	ps_info.ps_pos_y                     = (active_inputs & 0x00000200u) != 0;
	ps_info.ps_pos_z                     = (active_inputs & 0x00000400u) != 0;
	ps_info.ps_pos_w                     = (active_inputs & 0x00000800u) != 0;
	ps_info.ps_front_face                = (active_inputs & 0x00001000u) != 0;
	ps_info.ps_ancillary                 = (active_inputs & 0x00002000u) != 0;
	ps_info.ps_sample_shading            = (active_inputs & 0x00000011u) != 0;
	ps_info.ps_no_perspective            = (sh.ps_input_ena & sh.ps_input_addr & 0x00000020u) != 0;
	ps_info.ps_pixel_kill_enable         = sh.db_shader_control.shader_kill_enable;
	ps_info.ps_depth_export_enable       = sh.db_shader_control.shader_z_export_enable;
	ps_info.ps_sample_mask_export_enable = sh.db_shader_control.shader_mask_export_enable;
	ps_info.ps_early_z =
	    (sh.db_shader_control.shader_z_behavior == 1 && !sh.db_shader_control.shader_kill_enable &&
	     !sh.db_shader_control.shader_z_export_enable &&
	     !sh.db_shader_control.shader_mask_export_enable);
	ps_info.ps_execute_on_noop = sh.db_shader_control.shader_execute_on_noop;

	for (uint32_t i = 0; i < ps_info.input_num; i++) {
		ps_info.interpolator_settings[i] = sh.ps_interpolator_settings[i];
	}

	ps_info.target_shader_mask = sh.m_cbShaderMask;
	for (int i = 0; i < 8; i++) {
		const auto slot = ShaderPixelExportTarget(ps_info.target_shader_mask, i);
		ps_info.target_output_mode[i]    = sh.target_output_mode[i];
		ps_info.target_export_mapping[i] = sh.target_output_mode[i] != 0 && slot < 8
		                                       ? target_export_mapping[slot]
		                                       : Prospero::ColorComponentMapping {};
	}
}

static void ShaderGetStaticInputInfoCS(const HW::ComputeShaderInfo& regs,
                                       const HW::ShaderRegisters& /*sh*/,
                                       const ShaderMappedData& data, ShaderComputeInputInfo& info) {
	const bool dispatch_thread_dimensions = info.dispatch_thread_dimensions;
	const auto host_subgroup_size         = info.host_subgroup_size;
	const auto workgroup_counts           = std::to_array(info.workgroup_counts);
	info                                  = {};
	info.dispatch_thread_dimensions       = dispatch_thread_dimensions;
	info.host_subgroup_size               = host_subgroup_size;
	std::ranges::copy(workgroup_counts, info.workgroup_counts);
	info.threads_num[0]                   = regs.cs_regs.num_thread_x;
	info.threads_num[1]                   = regs.cs_regs.num_thread_y;
	info.threads_num[2]                   = regs.cs_regs.num_thread_z;
	info.lds_size_dwords                  = static_cast<uint32_t>(regs.cs_regs.lds_size) * 128u;
	info.scratch_size_dwords              = data.scratch_size_dwords;
	info.float_mode                       = regs.cs_regs.float_mode;
	info.group_id[0]                      = regs.cs_regs.tgid_x_en != 0;
	info.group_id[1]                      = regs.cs_regs.tgid_y_en != 0;
	info.group_id[2]                      = regs.cs_regs.tgid_z_en != 0;
	info.wave_size                        = regs.cs_regs.wave_size;
	info.thread_ids_num      = regs.cs_regs.tidig_comp_cnt + 1;
	info.tg_size_en          = regs.cs_regs.tg_size_en != 0;

	info.workgroup_register = regs.cs_regs.user_sgpr;
}

ShaderParams PrepareProgram(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            const HW::UserConfig& user_config, ShaderVertexInputInfo& info) {
	const auto& sh     = context.GetShaderRegisters();
	const auto data = ShaderGetMappedData(regs.es_regs.data_addr, "ShaderGetInputInfoVS():");
	const bool merged = (context.GetShaderStages() & 0x20u) != 0;
	// GS_EN controls amplification, not whether an NGG shader uses a workgroup.
	const bool native_ngg = !merged && data.type == Prospero::ShaderBinaryType::kGs &&
	                        regs.gs_regs.rsrc2.lds_size != 0;
	auto        params = GetShaderParams(
	    regs.es_regs.data_addr, "ShaderRecompiler VS",
	    RegisteredOrDeclaredShaderHash(regs.es_regs.data_addr, data),
	    std::span<const uint32_t>(regs.gs_user_sgpr.value, regs.gs_regs.rsrc2.user_sgpr), data,
	    merged || native_ngg ? 8u : 0u);
	if (DrawPrep::SpeculativeFailed()) {
		return params; // discarded; the serial path decides at commit
	}
	// The GS user SGPRs start at s8: one the shader receives may hold the draw's start instance.
	const auto start_instance = regs.start_instance_user_sgpr;
	const int32_t start_instance_sgpr =
	    start_instance >= 0 && static_cast<uint32_t>(start_instance) < regs.gs_regs.rsrc2.user_sgpr
	        ? 8 + start_instance
	        : -1;
	if (!merged && !native_ngg) {
		const auto static_info = [&](const ShaderMappedData& metadata) {
			return ShaderGetStaticVertexInputInfo(regs.es_regs.data_addr, regs.gs_user_sgpr,
			                                      regs.gs_regs.rsrc2.user_sgpr, sh, metadata, info);
		};
		bool prepared = false;
		if (DrawPrep::Speculative()) {
			PreparationMetadata metadata_copy;
			ShaderMappedData    metadata = data;
			if (!CopyMetadataForPreparation(data, metadata_copy, metadata)) {
				return params;
			}
			prepared = static_info(metadata);
		} else {
			prepared = static_info(data);
		}
		if (!prepared) {
			if (DrawPrep::Speculative()) {
				DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
				return params;
			}
			EXIT("failed to prepare vertex shader program\n");
		}
		info.wave_size           = (context.GetShaderStages() & 0x00400000u) != 0 ? 32u : 64u;
		info.start_instance_sgpr = start_instance_sgpr;
		return params;
	}
	// NGG user SGPRs start at s8; a separately compiled GS back half also receives
	// its user-data pointer in s0:s1.
	info                     = {};
	info.logical_stage       = ShaderType::Mesh;
	info.start_instance_sgpr = start_instance_sgpr;
	info.pa_cl_vs_out_cntl   = sh.m_paClVsOutCntl;
	auto& mesh               = info.mesh;
	mesh.input_primitive     = static_cast<uint32_t>(user_config.GetPrimType());
	mesh.wave_size           = (context.GetShaderStages() & 0x00400000u) != 0 ? 32u : 64u;
	mesh.max_vertices        = sh.m_geMaxOutputPerSubgroup;
	mesh.provoking_vertex    = context.GetModeControl().provoking_vtx_last ? 2u : 0u;
	mesh.lds_size_dwords     = static_cast<uint32_t>(regs.gs_regs.rsrc2.lds_size) * 128u;
	mesh.scratch_size_dwords = data.scratch_size_dwords;
	if (data.type == Prospero::ShaderBinaryType::kGsFront) {
		EXIT_IF(regs.gs_regs.data_addr == 0);
		const auto back = ShaderGetMappedData(regs.gs_regs.data_addr, "ShaderGetInputInfoGS():");
		const auto back_params =
		    GetShaderParams(regs.gs_regs.data_addr, "ShaderRecompiler GS",
		                    RegisteredOrDeclaredShaderHash(regs.gs_regs.data_addr, back), {}, back);
		if (DrawPrep::SpeculativeFailed()) {
			return params;
		}
		params.back_code = back_params.code;
		params.user_data[0] = static_cast<uint32_t>(regs.gs_regs.user_data_addr);
		params.user_data[1] = static_cast<uint32_t>(regs.gs_regs.user_data_addr >> 32u);
		const uint64_t hashes[] = {params.hash, back_params.hash};
		params.hash = XXH3_64bits(hashes, sizeof(hashes));
		mesh.scratch_size_dwords = std::max(mesh.scratch_size_dwords, back.scratch_size_dwords);
	}
	if (merged) {
		EXIT_NOT_IMPLEMENTED(regs.gs_regs.rsrc1.gs_vgpr_component_count != 3u ||
		                     regs.gs_regs.rsrc2.es_vgpr_component_count != 3u);
	}
	const auto& group = user_config.GetGeControl();
	if ((user_config.GetPrimType() != Prospero::PrimitiveType::kPointList &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kLineList &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriFan &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriStrip &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList) ||
	    sh.m_vgtGsOutPrimType != 2u || (!native_ngg && sh.m_vgtGsMaxVertOut < 3u) ||
	    group.vertex_group_size < mesh.InputPrimitiveSize() ||
	    mesh.max_vertices == 0u) {
		EXIT("unsupported GS assembly: input=%u output=%u vertices=%u GE=%u/%u max_output=%u\n",
		     mesh.input_primitive, sh.m_vgtGsOutPrimType, sh.m_vgtGsMaxVertOut,
		     group.primitive_group_size, group.vertex_group_size, mesh.max_vertices);
	}
	mesh.max_primitives       = native_ngg ? mesh.InputPrimitiveCount(mesh.max_vertices)
	                                       : group.primitive_group_size * (sh.m_vgtGsMaxVertOut - 2u);
	mesh.primitives_per_group = std::min({static_cast<uint32_t>(group.primitive_group_size),
	                                      mesh.InputPrimitiveCount(group.vertex_group_size),
	                                      native_ngg ? mesh.max_primitives :
	                                          mesh.max_vertices / sh.m_vgtGsMaxVertOut});
	EXIT_IF(mesh.primitives_per_group == 0u);
	mesh.vertices_per_group = mesh.InputVertexCount(mesh.primitives_per_group);
	const auto threads  = native_ngg ? mesh.vertices_per_group : mesh.max_vertices;
	mesh.threads_num[0] = ((threads + mesh.wave_size - 1u) / mesh.wave_size) * mesh.wave_size;
	mesh.threads_num[1] = mesh.threads_num[2] = 1u;
	return params;
}

std::array<ShaderParams, 3>
PrepareTessellationPrograms(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            std::array<ShaderVertexInputInfo, 3>& input_info) {
	const auto& sh        = context.GetShaderRegisters();
	const auto  local     = ShaderGetMappedData(regs.ls_regs.data_addr, "ShaderGetInputInfoLS():");
	const auto  control   = ShaderGetMappedData(regs.hs_regs.data_addr, "ShaderGetInputInfoHS():");
	const auto evaluation = ShaderGetMappedData(regs.es_regs.data_addr, "ShaderGetInputInfoTES():");
	EXIT_NOT_IMPLEMENTED(local.type != Prospero::ShaderBinaryType::kHsFront ||
	                     control.type != Prospero::ShaderBinaryType::kHsBack ||
	                     evaluation.type != Prospero::ShaderBinaryType::kGs);
	EXIT_NOT_IMPLEMENTED((context.GetShaderStages() & 0x4u) == 0 ||
	                     (context.GetShaderStages() & 0x00600020u) != 0);
	EXIT_IF(regs.hs_regs.user_data_addr == 0);

	const auto local_users      = std::span(regs.hs_user_sgpr.value, regs.hs_regs.rsrc2.user_sgpr);
	const auto evaluation_users = std::span(regs.gs_user_sgpr.value, regs.gs_regs.rsrc2.user_sgpr);
	std::array<ShaderParams, 3> params {
	    GetShaderParams(regs.ls_regs.data_addr, "ShaderRecompiler LS",
	                    RegisteredOrDeclaredShaderHash(regs.ls_regs.data_addr, local), local_users, local),
	    GetShaderParams(regs.hs_regs.data_addr, "ShaderRecompiler HS",
	                    RegisteredOrDeclaredShaderHash(regs.hs_regs.data_addr, control), local_users, control, 8u),
	    GetShaderParams(regs.es_regs.data_addr, "ShaderRecompiler TES",
	                    RegisteredOrDeclaredShaderHash(regs.es_regs.data_addr, evaluation), evaluation_users,
	                    evaluation),
	};
	// The fused HS back half receives its separate user-data address in s0:s1.
	// RDNA2 reserves s0:s7 before the native HS user SGPRs.
	params[1].user_data[0] = static_cast<uint32_t>(regs.hs_regs.user_data_addr);
	params[1].user_data[1] = static_cast<uint32_t>(regs.hs_regs.user_data_addr >> 32u);

	input_info = {};
	if (!ShaderGetStaticVertexInputInfo(regs.ls_regs.data_addr, regs.hs_user_sgpr,
	                                    regs.hs_regs.rsrc2.user_sgpr, sh, local, input_info[0])) {
		EXIT("failed to prepare local shader program\n");
	}
	input_info[0].logical_stage       = ShaderType::Local;
	input_info[1].logical_stage       = ShaderType::TessellationControl;
	input_info[1].scratch_size_dwords = control.scratch_size_dwords;
	input_info[2].logical_stage       = ShaderType::TessellationEvaluation;
	input_info[2].scratch_size_dwords = evaluation.scratch_size_dwords;
	input_info[2].pa_cl_vs_out_cntl   = sh.m_paClVsOutCntl;

	ShaderTessellationInputInfo tess {
	    .input_control_points  = (sh.m_vgtLsHsConfig >> 8u) & 0x3fu,
	    .output_control_points = (sh.m_vgtLsHsConfig >> 14u) & 0x3fu,
	    .domain                = sh.m_vgtTfParam & 0x3u,
	    .partitioning          = (sh.m_vgtTfParam >> 2u) & 0x3u,
	    .output_topology       = (sh.m_vgtTfParam >> 5u) & 0x3u,
	};
	EXIT_IF(tess.input_control_points == 0 || tess.input_control_points > 32 ||
	        tess.output_control_points == 0 || tess.output_control_points > 32);
	ShaderRecompiler::AnalyzeTessellationPrograms(params[0].code, params[1].code, tess);
	for (auto& stage: input_info) {
		stage.tess = tess;
	}
	return params;
}

ShaderParams PrepareProgram(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    ShaderPixelInputInfo&                               ps_info) {
	const auto data = ShaderGetMappedData(regs.ps_regs.data_addr, "ShaderGetInputInfoPS():");
	if (DrawPrep::Speculative()) {
		PreparationMetadata metadata_copy;
		ShaderMappedData    metadata = data;
		if (DrawPrep::SpeculativeFailed() ||
		    !CopyMetadataForPreparation(data, metadata_copy, metadata)) {
			return {};
		}
		ShaderGetStaticInputInfoPS(regs, sh, target_export_mapping, metadata, ps_info);
	} else {
		ShaderGetStaticInputInfoPS(regs, sh, target_export_mapping, data, ps_info);
	}
	return GetShaderParams(
	    regs.ps_regs.data_addr, "ShaderRecompiler PS", RegisteredOrDeclaredShaderHash(regs.ps_regs.data_addr, data),
	    std::span<const uint32_t>(regs.ps_user_sgpr.value, regs.ps_regs.rsrc2.user_sgpr), data);
}

ShaderParams PrepareProgram(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
                            ShaderComputeInputInfo& info) {
	const auto data = ShaderGetMappedData(regs.cs_regs.data_addr, "ShaderGetInputInfoCS():");
	ShaderGetStaticInputInfoCS(regs, sh, data, info);
	return GetShaderParams(
	    regs.cs_regs.data_addr, "ShaderRecompiler CS", RegisteredOrDeclaredShaderHash(regs.cs_regs.data_addr, data),
	    std::span<const uint32_t>(regs.cs_user_sgpr.value, regs.cs_regs.user_sgpr), data);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ShaderDbgDumpInputInfo(const ShaderVertexInputInfo& info) {
	KYTY_PROFILER_BLOCK("ShaderDbgDumpInputInfo(Vs)");

	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t fetch_external = %s\n"
	     "\t fetch_embedded = %s\n",
	     info.fetch_external ? "true" : "false", info.fetch_embedded ? "true" : "false");

	for (int i = 0; i < info.resources_num; i++) {
		LOGF("\t input %d\n", i);

		const auto& r  = info.resources[i];
		const auto& rd = info.resources_dst[i];

		LOGF("\t\t register_start   = %d\n"
		     "\t\t registers_num    = %d\n"
		     "\t\t fetch_index      = %" PRIu32 "\n",
		     rd.register_start, rd.registers_num, rd.fetch_index);
		LOGF("\t\t fields           = %08" PRIx32 "%08" PRIx32 "%08" PRIx32 "%08" PRIx32 "\n",
		     r.fields[3], r.fields[2], r.fields[1], r.fields[0]);
		LOGF("\t\t Base()           = %" PRIx64 "\n"
		     "\t\t Stride()         = %" PRIu16 "\n"
		     "\t\t SwizzleEnabled() = %s\n"
		     "\t\t NumRecords()     = %" PRIu32 "\n"
		     "\t\t DstSelX()        = %" PRIu8 "\n"
		     "\t\t DstSelY()        = %" PRIu8 "\n"
		     "\t\t DstSelZ()        = %" PRIu8 "\n"
		     "\t\t DstSelW()        = %" PRIu8 "\n",
		     r.Base48(), r.Stride(), r.SwizzleEnabled() ? "true" : "false", r.NumRecords(),
		     r.DstSelX(), r.DstSelY(), r.DstSelZ(), r.DstSelW());
		LOGF("\t\t Format()         = %" PRIu8 "\n"
		     "\t\t OutOfBounds()    = %" PRIu8 "\n",
		     r.RawFormat(), r.OutOfBounds());
		LOGF("\t\t AddTid()         = %s\n", r.AddTid() ? "true" : "false");
	}

	for (int i = 0; i < info.buffers_num; i++) {
		LOGF("\t buffer %d\n", i);

		const auto& r = info.buffers[i];
		LOGF("\t\t addr        = %" PRIx64 "\n"
		     "\t\t stride      = %" PRIu32 "\n"
		     "\t\t num_records = %" PRIu32 "\n"
		     "\t\t fetch_index = %" PRIu32 "\n"
		     "\t\t attr_num    = %" PRId32 "\n",
		     r.addr, r.stride, r.num_records, r.fetch_index, r.attr_num);
		for (int j = 0; j < r.attr_num; j++) {
			LOGF("\t\t attr_indices[%d]  = %d\n"
			     "\t\t attr_offsets[%d]  = %u\n",
			     j, r.attr_indices[j], j, r.attr_offsets[j]);
		}
	}
}

void ShaderDbgDumpInputInfo(const ShaderPixelInputInfo& info) {
	KYTY_PROFILER_BLOCK("ShaderDbgDumpInputInfo(Ps)");

	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t input_num            = %u\n"
	     "\t ps_system_input_base = %u\n"
	     "\t custom_interpolation_mask = 0x%08" PRIx32 "\n"
	     "\t ps_perspective_center_vgpr = %" PRIu32 "\n"
	     "\t ps_perspective_sample_vgpr = %" PRIu32 "\n"
	     "\t ps_perspective_centroid_vgpr = %" PRIu32 "\n"
	     "\t ps_pos_x             = %s\n"
	     "\t ps_pos_y             = %s\n"
	     "\t ps_pos_z             = %s\n"
	     "\t ps_pos_w             = %s\n"
	     "\t ps_front_face        = %s\n"
	     "\t ps_ancillary         = %s\n"
	     "\t ps_sample_shading    = %s\n"
	     "\t ps_no_perspective    = %s\n"
	     "\t ps_pixel_kill_enable = %s\n"
	     "\t ps_early_z           = %s\n"
	     "\t ps_execute_on_noop   = %s\n",
	     info.input_num, info.ps_system_input_base, info.custom_interpolation_mask,
	     info.ps_perspective_center_vgpr, info.ps_perspective_sample_vgpr,
	     info.ps_perspective_centroid_vgpr,
	     info.ps_pos_x ? "true" : "false",
	     info.ps_pos_y ? "true" : "false", info.ps_pos_z ? "true" : "false",
	     info.ps_pos_w ? "true" : "false", info.ps_front_face ? "true" : "false",
	     info.ps_ancillary ? "true" : "false",
	     info.ps_sample_shading ? "true" : "false", info.ps_no_perspective ? "true" : "false",
	     info.ps_pixel_kill_enable ? "true" : "false", info.ps_early_z ? "true" : "false",
	     info.ps_execute_on_noop ? "true" : "false");

	for (uint32_t i = 0; i < info.input_num; i++) {
		LOGF("\t interpolator_settings[%u] = %u\n", i, info.interpolator_settings[i]);
	}
}

void ShaderDbgDumpInputInfo(const ShaderComputeInputInfo& info) {
	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t workgroup_register = %d\n"
	     "\t thread_ids_num     = %d\n"
	     "\t wave_size          = %u\n"
	     "\t lds_size_dwords    = %u\n"
	     "\t threads_num        = {%u, %u, %u}\n"
	     "\t tg_size_en         = %s\n",
	     info.workgroup_register, info.thread_ids_num, info.wave_size, info.lds_size_dwords,
	     info.threads_num[0], info.threads_num[1], info.threads_num[2],
	     info.tg_size_en ? "true" : "false");
	LOGF("\t threadgroup_id     = {%s, %s, %s}\n", info.group_id[0] ? "true" : "false",
	     info.group_id[1] ? "true" : "false", info.group_id[2] ? "true" : "false");
}

} // namespace Libs::Graphics
