#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/NullTrace.h"
#include "common/prepPhases.h"

#include "common/assert.h"
#include "common/liveSwitch.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

Live::Switch g_buffer_refresh_fusion("KYTY_BUFFER_REFRESH_FUSION", Live::ParseDefaultOff);

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
constexpr uint64_t MaxIndirectImageProbes = 65536u;

// KYTY_WORKGROUP_IMAGES: set when a materialization on this thread failed for want of workgroup
// counts (TakeWorkgroupCountsMissing).
thread_local bool t_workgroup_counts_missing = false;

bool SpecializationFail(std::string_view message) {
	// A draw whose descriptors cannot be specialized fails again on every frame; each distinct
	// reason is reported once (the console write costs more than the failed materialization).
	static std::mutex                      mutex;
	static std::unordered_set<std::string> reported;
	{
		const std::lock_guard lock(mutex);
		if (reported.size() >= 64 || !reported.emplace(message).second) {
			return false;
		}
	}
	std::fprintf(stderr, "shader resource specialization failed: %.*s\n",
	             static_cast<int>(message.size()), message.data());
	return false;
}

// KYTY_SPEC_UNMAPPED_NULL (default on; =0 off): a buffer V# the renderer would bind as empty
// (base 0, no records, or a base outside every guest mapping: FindBuffers' ClampRangeSizeQuiet)
// specializes as the null descriptor, as a V# with a non-zero type does. Its stride, swizzle,
// ADD_TID and format bits cannot change what the shader reads (nothing) or writes (dropped), so
// the garbage words a game leaves in a descriptor slot it only uses on a branch it does not take
// no longer compile a new permutation per value (Astro Bot's [Performance] patches: a vertex
// shader's atomic V# gave ~80 permutations in minutes). The probe answers exactly the binding's
// question (is the base mapped), so a V# with a mapped base is never nulled. The decision is made
// per materialization from the current words: the same slot holding a real V# later specializes
// normally (a different key, so the null permutation is not reused for it).
std::atomic<SrtMappedRange> g_unmapped_buffer_probe {nullptr};
std::atomic<uint64_t>       g_unmapped_buffer_nulls {0};

thread_local std::array<uint64_t, 16> t_mapped_blocks {}; // 64 KiB blocks seen mapped (+1)

bool BufferBaseMapped(SrtMappedRange probe, uint64_t base, uint64_t size) {
	const auto block = (base >> 16u) + 1u;
	auto&      slot  = t_mapped_blocks[block & (t_mapped_blocks.size() - 1u)];
	if (slot == block) {
		return true;
	}
	if (!probe(base, size)) {
		return false;
	}
	slot = block; // only positive verdicts are cached: a stale one keeps the V# (old behaviour)
	return true;
}

// Whether the buffer descriptor must specialize (and bind) as null.
bool NullBufferDescriptor(const ShaderBufferResource& descriptor, bool count = true) {
	if (descriptor.Type() != 0) {
		return true;
	}
	const auto probe = g_unmapped_buffer_probe.load(std::memory_order_relaxed);
	if (probe == nullptr) {
		return false;
	}
	const uint64_t base = descriptor.Base48();
	const uint64_t size = descriptor.GetSize();
	if (base != 0u && size != 0u && BufferBaseMapped(probe, base, size)) {
		return false;
	}
	if ((descriptor.fields[0] | descriptor.fields[1] | descriptor.fields[2] | descriptor.fields[3]) == 0u) {
		return false; // already the null descriptor
	}
	if (!count) {
		return true;
	}
	const auto nulls = g_unmapped_buffer_nulls.fetch_add(1, std::memory_order_relaxed);
	NullTraceNote(NullTraceKind::NullBuffer);
	if (nulls < 8u) {
		std::fprintf(stderr,
		             "Kyty spec: buffer V# with unmapped base 0x%012" PRIx64
		             " specialized as null (%08x %08x %08x %08x)%s\n",
		             base, descriptor.fields[0], descriptor.fields[1], descriptor.fields[2],
		             descriptor.fields[3], nulls == 7u ? "; further ones are counted only" : "");
	}
	return true;
}

// KYTY_MS_DEPTH_SAMPLE0=1 (from usa-ratt/KytyPS5-GT7 round 124): -1 = not read yet.
std::atomic<int> g_ms_depth_sample0 {-1};
// KYTY_EQAA_MIXED=1 (from usa-ratt/KytyPS5-GT7 18979e65b): the EQAA FMASK map. -1 = not read yet.
std::atomic<int> g_eqaa_fmask_map {-1};

// KYTY_INDIRECT_LEADING_COORDS=1 (default off; upstream d3768c00e). -1 = not read yet.
std::atomic<int> g_indirect_leading_coords {-1};

bool IndirectLeadingCoordinates() {
	auto value = g_indirect_leading_coords.load(std::memory_order_relaxed);
	if (value < 0) [[unlikely]] {
		const char* text = std::getenv("KYTY_INDIRECT_LEADING_COORDS");
		value            = text != nullptr && text[0] == '1' ? 1 : 0;
		g_indirect_leading_coords.store(value, std::memory_order_relaxed);
	}
	return value != 0;
}

bool EqaaFmaskMap() {
	auto value = g_eqaa_fmask_map.load(std::memory_order_relaxed);
	if (value < 0) [[unlikely]] {
		const char* text = std::getenv("KYTY_EQAA_MIXED");
		value            = text != nullptr && text[0] == '1' ? 1 : 0;
		g_eqaa_fmask_map.store(value, std::memory_order_relaxed);
	}
	return value != 0;
}

// EQAA FMASK formats (kFmask<bits>_S<samples>_F<fragments> with fewer fragments than samples):
// the number of colour fragments stored per pixel. 0 for formats with one fragment per sample,
// whose native mapping is the identity.
uint8_t FmaskEqaaFragmentCount(Prospero::BufferFormat format) {
	using F = Prospero::BufferFormat;
	switch (format) {
		case F::kFmask8_S2_F1:
		case F::kFmask8_S4_F1:
		case F::kFmask8_S8_F1:
		case F::kFmask16_S16_F1: return 1;
		case F::kFmask8_S4_F2:
		case F::kFmask16_S8_F2:
		case F::kFmask32_S16_F2: return 2;
		case F::kFmask32_S8_F4:
		case F::kFmask64_S16_F4: return 4;
		case F::kFmask64_S16_F8: return 8;
		default: return 0;
	}
}

// One 32-bit word of an expanded FMASK value (four bits per sample) covering samples
// first_sample..first_sample+7. Without EQAA (fragments == 0) sample i maps to host sample i;
// with EQAA it maps to fragment (i mod fragments), so every index names a host sample that exists.
constexpr uint32_t FmaskSampleMapWord(uint32_t fragments, uint32_t first_sample) {
	if (fragments == 0) {
		return first_sample == 0 ? 0x76543210u : 0xfedcba98u;
	}
	uint32_t word = 0;
	for (uint32_t i = 0; i < 8; i++) {
		word |= ((first_sample + i) % fragments) << (i * 4u);
	}
	return word;
}
static_assert(FmaskSampleMapWord(0, 0) == 0x76543210u && FmaskSampleMapWord(0, 8) == 0xfedcba98u);
static_assert(FmaskSampleMapWord(8, 8) == 0x76543210u);
static_assert(FmaskSampleMapWord(4, 0) == 0x32103210u);
static_assert(FmaskSampleMapWord(2, 8) == 0x10101010u);
static_assert(FmaskSampleMapWord(1, 0) == 0u);

bool MsDepthSample0Dimension() {
	auto value = g_ms_depth_sample0.load(std::memory_order_relaxed);
	if (value < 0) [[unlikely]] {
		const char* text = std::getenv("KYTY_MS_DEPTH_SAMPLE0");
		value            = text != nullptr && text[0] == '1' ? 1 : 0;
		g_ms_depth_sample0.store(value, std::memory_order_relaxed);
	}
	return value != 0;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	// KYTY_MS_DEPTH_SAMPLE0=1: a multisample depth or stencil T# (depth tiling) read by an
	// instruction of a single-sample dimension (image_sample, image_load without a sample index)
	// reads its first sample on the console. The shader gets a single-sample image and the host
	// binds a single-sample copy of that sample (TextureCache::SingleSampleDepth/Stencil).
	// Otherwise the shader samples a multisample image (not valid SPIR-V): on NVIDIA the samples
	// came back interleaved (GT7 replays: light sprites tested the wrong depth, the 16x16 depth
	// reduction for the CPU read two pictures at once). Multisample colour T#s keep their
	// multisample dimension.
	constexpr uint32_t depth_tiling = static_cast<uint32_t>(Prospero::TileMode::kDepth);
	const bool         single_sample_read =
	    (requested == Decoder::ImageDimension::Dim2D ||
	     requested == Decoder::ImageDimension::Dim2DArray) &&
	    ((descriptor.dwords[3] >> 20u) & 0x1fu) == depth_tiling && MsDepthSample0Dimension();
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (single_sample_read) {
				return is_array ? Decoder::ImageDimension::Dim2DArray
				                : Decoder::ImageDimension::Dim2D;
			}
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa:
			return single_sample_read ? Decoder::ImageDimension::Dim2D
			                          : Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto& words = descriptor.dwords;
	// Reject texture descriptors with nonzero reserved bits.
	if ((words[1] & 0x20000000u) != 0u || (words[2] & 0x70003000u) != 0u ||
	    (!r128 && ((words[4] & 0xe000e000u) != 0u || (words[5] & 0xf9000000u) != 0u ||
	               (words[6] & 0x00007b00u) != 0u))) {
		return false;
	}
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	// The range above leaves the encoding's gaps open, and a value such as 139, which lies between
	// 136 and 156 and names nothing, used to pass here and abort the emulator further down instead of
	// being treated as what it is: eight dwords that are not a descriptor.
	if (!Prospero::IsDefinedBufferFormat(format)) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	if (!IsValidImageSwizzle(words[3] & 0xfffu)) return false;
	const bool array = type == Prospero::ImageType::kColor1DArray ||
	                   type == Prospero::ImageType::kColor2DArray ||
	                   type == Prospero::ImageType::kColor2DMsaaArray ||
	                   type == Prospero::ImageType::kCube;
	if (array && ((words[4] >> 16u) & 0x1fffu) > (words[4] & 0x1fffu)) {
		return false;
	}
	const auto base_level = (words[3] >> 12u) & 0xfu;
	const auto last_level = (words[3] >> 16u) & 0xfu;
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && last_level >= 1 && last_level <= 3 &&
		       (r128 || max_mip == last_level);
	}
	return base_level <= last_level;
}

// A null image keeps the draw alive, but the walk read something that is not a descriptor (all-zero
// dwords are an unbound slot, not reported), so say so once per shader and slot instead of hiding it.
void ReportInvalidImageDescriptor(uint64_t shader_hash, uint32_t slot,
                                  const DescriptorValue& descriptor) {
	static std::mutex                                   mutex;
	static std::set<std::pair<uint64_t, uint32_t>>    reported;
	{
		const std::lock_guard lock(mutex);
		if (reported.size() >= 64 || !reported.emplace(shader_hash, slot).second) {
			return;
		}
	}
	const auto& words = descriptor.dwords;
	std::fprintf(stderr,
	             "image descriptor %u of shader 0x%016" PRIx64
	             " is not a descriptor, binding null: %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	             slot, shader_hash, words[0], words[1], words[2], words[3], words[4], words[5],
	             words[6], words[7]);
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

// KYTY_STORAGE_SWIZZLE_CANON (TheCruZ 72e95dcd): a storage image in its own format (no conversion)
// applies the swizzle only to stores (StoreTexel), and a store keeps just the channels the format
// has. A swizzle that stores each of them from its own component is the identity, so an RG target
// written as (R, G, 0, 1) shares the program of an RGBA target written as (R, G, B, A).
uint32_t CanonicalStorageSwizzle(uint32_t swizzle, Prospero::BufferFormat format) {
	const auto info = Format::GetFormatInfo(format);
	if (info.type == Format::ComponentType::Unknown || info.component_count == 0u ||
	    info.component_count > 4u) {
		return swizzle;
	}
	for (uint32_t channel = 0; channel < info.component_count; channel++) {
		uint32_t source = 0;
		while (source < 4u && ((swizzle >> (source * 3u)) & 7u) != 4u + channel) source++;
		if (source != channel) return swizzle;
	}
	return ShaderImageIdentitySwizzle;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

// FloatCompare: KYTY_COMPARE_SAMPLER_SPLIT (Erlou93/KytyPS5-Yotei, ported there from peepostare
// f0bed387): the depth-compare samples of an S# that also serves plain samples get their own host
// sampler. KYTY_POINT_INTEGER_IMAGES: every UINT image is point sampled, as SINT ones are.
enum class SamplerClass : uint8_t { Float, Integer, PointInteger, FloatCompare };
constexpr uint32_t SamplerClassCount = 4;

template <typename Image>
SamplerClass ClassifySampler(const Image& image, bool depth_compare) {
	if (image.numeric_class == Prospero::TextureNumericClass::Sint ||
	    (image.numeric_class == Prospero::TextureNumericClass::Uint &&
	     (image.conversion_format != Prospero::BufferFormat::kInvalid ||
	      GetCodegenOptions().point_integer_images))) {
		return SamplerClass::PointInteger;
	}
	if (image.numeric_class == Prospero::TextureNumericClass::Uint) {
		return SamplerClass::Integer;
	}
	return depth_compare && GetCodegenOptions().compare_sampler_split ? SamplerClass::FloatCompare
	                                                                   : SamplerClass::Float;
}

template <typename Image>
SamplerClass ClassifySampler(const Image& image) {
	if constexpr (requires { image.depth_compare; }) {
		return ClassifySampler(image, image.depth_compare);
	} else {
		return ClassifySampler(image, false);
	}
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t ImageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::Dynamic || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	// Validated descriptors (ValidImageDescriptor) never have base > last; descriptors that reach
	// specialization without that check still fail it below instead of wrapping around.
	return base <= last ? last - base + 1u : 0u;
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

struct ReadCapture {
	SrtRuntime                                  source;
	std::vector<std::pair<uint64_t, uint64_t>>& ranges;
};

bool CaptureStrictRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (!capture.source.read_specialization_memory(capture.source.userdata, address, values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool CaptureOrdinaryRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (!capture.source.read_memory(capture.source.userdata, address, values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool ProbeOrdinaryBacking(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	return capture.source.try_read_clean_backing(capture.source.userdata, address, values);
}

bool ProbeStrictBacking(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	const auto probe = capture.source.try_read_specialization_backing != nullptr
	                       ? capture.source.try_read_specialization_backing
	                       : capture.source.try_read_clean_backing;
	if (!probe(capture.source.userdata, address, values)) return false;
	// The exact union has the same written-buffer intersections as its dwords.
	// A failed probe is silent; scalar fallback records its actual successful prefix.
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool WrittenBuffersDisjoint(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                            std::span<const std::pair<uint64_t, uint64_t>> reads) {
	for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
		if (!program.info.buffers[i].written) continue;
		ShaderBufferResource buffer;
		if (!DecodeBufferDescriptor(snapshot.buffers[i], buffer)) return false;
		const auto base = buffer.Base48();
		const auto size = buffer.GetSize();
		if (size == 0u) continue;
		if (size - 1u > AddressMask - base) return false;
		for (const auto [address, bytes]: reads) {
			if (bytes == 0u) continue;
			if (address > AddressMask || bytes - 1u > AddressMask - address ||
			    (address <= base ? base - address < bytes : address - base < size)) {
				return false;
			}
		}
	}
	return true;
}

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanFlatSlots(const ResourcePlan& program, const DescriptorSource* source,
                        std::span<uint8_t> slots, Value extra = {}) {
	if (source == nullptr && extra.IsEmpty()) {
		return;
	}
	std::vector<Value>       pending;
	if (source != nullptr) {
		pending.assign(source->dwords.begin(), source->dwords.begin() + source->dword_count);
	}
	if (!extra.IsEmpty()) pending.push_back(extra);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto value = pending.back().Resolve();
		pending.pop_back();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (inst->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = inst->Arg(1).Resolve();
			if (slot.IsImmediate() && slot.GetType() == Type::U32 && slot.U32() < slots.size()) {
				slots[slot.U32()] = 1u;
				pending.push_back(program.srt_reads[slot.U32()].value);
			}
			continue;
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint32_t dynamic_offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto offset = static_cast<uint64_t>(dynamic_offset) & ~uint64_t {3};
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	base &= AddressMask & ~uint64_t {3};
	if (offset > AddressMask - base) {
		ObserveSrtRead(runtime, base, {}, false);
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	const bool read = prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, prefix);
	ObserveSrtRead(runtime, address, prefix, read);
	return read;
}

// Indirect tables can contain many keys but at most MaxImages distinct descriptors. Keep
// small tables on the linear path; build a local index only once that scan becomes long.
// Slots store child ordinals, never descriptor pointers, so vector growth is harmless.
class IndirectImageIndex {
public:
	struct Lookup {
		uint32_t ordinal;
		uint32_t slot = UINT32_MAX;
	};

	Lookup Find(const std::vector<DescriptorValue>& images, size_t children_begin,
	            const DescriptorValue& candidate) {
		const auto count = static_cast<uint32_t>(images.size() - children_begin);
		if (!m_indexed && count < LinearLimit) {
			const auto found = std::find(images.begin() + children_begin, images.end(), candidate);
			return {static_cast<uint32_t>(found - images.begin() - children_begin + 1u)};
		}
		if (!m_indexed) {
			m_ordinals.fill(0);
			for (uint32_t child = 0; child < count; ++child) {
				auto slot = Hash(images[children_begin + child]);
				while (m_ordinals[slot] != 0u) slot = (slot + 1u) & SlotMask;
				m_ordinals[slot] = static_cast<uint16_t>(child + 1u);
			}
			m_indexed = true;
		}
		auto slot = Hash(candidate);
		while (m_ordinals[slot] != 0u) {
			const auto ordinal = m_ordinals[slot];
			if (images[children_begin + ordinal - 1u] == candidate) return {ordinal, slot};
			slot = (slot + 1u) & SlotMask;
		}
		return {count + 1u, slot};
	}

	void Insert(Lookup lookup) {
		if (lookup.slot != UINT32_MAX) {
			m_ordinals[lookup.slot] = static_cast<uint16_t>(lookup.ordinal);
		}
	}

private:
	static constexpr uint32_t LinearLimit = 8;
	static constexpr uint32_t SlotCount = std::bit_ceil(ShaderInfo::MaxImages * 2u);
	static constexpr uint32_t SlotMask = SlotCount - 1u;
	static_assert(ShaderInfo::MaxImages < UINT16_MAX);

	static uint32_t Hash(const DescriptorValue& descriptor) {
		// Hash explicit fields, excluding any padding. Equality below still compares the
		// complete descriptor, so collisions cannot merge different views or formats.
		uint64_t hash = 14695981039346656037ull ^ descriptor.dword_count;
		for (const auto word: descriptor.dwords) {
			hash = (hash ^ word) * 1099511628211ull;
		}
		hash ^= hash >> 32u;
		return static_cast<uint32_t>(hash) & SlotMask;
	}

	std::array<uint16_t, SlotCount> m_ordinals;
	bool                          m_indexed = false;
};

// The shader looks the key up in the bindless translation table: nothing to enumerate. Record
// the heap for the host and reserve the two words it patches (region base, entry count); zeros
// resolve every key to the placeholder in slot 0.
bool MaterializeBindlessImage(const DescriptorSource::IndirectDescriptor& indirect,
                              const DescriptorValue& table_value, uint32_t image_index,
                              ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	ShaderBufferResource heap;
	if (table_value.dword_count != 4u || !DecodeBufferDescriptor(table_value, heap)) {
		return false;
	}
	const auto mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
	snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
	snapshot.images[image_index] = {.dword_count = 8u};
	snapshot.bindless_heaps.push_back({.base           = heap.Base48(),
	                                   .size           = heap.GetSize(),
	                                   .table_offset   = indirect.table_offset,
	                                   .image          = image_index,
	                                   .mapping_offset = mapping_offset,
	                                   // KYTY_BINDLESS_STRIDE: a plain heap's T#s may sit in larger
	                                   // records (table_stride > 32); otherwise 32 bytes apart.
	                                   .stride         = indirect.record_pointer != UINT32_MAX ||
	                                                             indirect.table_stride > 32u
	                                                         ? indirect.table_stride
	                                                         : 32u,
	                                   .record_pointer = indirect.record_pointer,
	                                   .nested_vsharp  = indirect.nested_vsharp,
	                                   .nested_heap    = indirect.nested_heap,
	                                   .nested_keys    = indirect.nested_keys});
	auto& root                      = specialization.images[image_index];
	root.indirect_root              = image_index;
	root.indirect_mapping_offset    = mapping_offset;
	root.indirect_search_iterations = 0;
	root.bindless                   = true;
	return true;
}

template <bool Optimize>
bool MaterializeIndirectImage(const ResourcePlan& program,
                              const DescriptorSource::IndirectDescriptor& indirect,
                              const DescriptorValue& material_value,
                              const DescriptorValue& table_value, uint32_t image_index,
                              const SrtRuntime& runtime, SrtWalker& clean,
                              EvaluationScratch& scratch, ResourceSnapshot& snapshot,
                              ResourceSpecialization& specialization) {
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX; // Scalar addresses have no buffer descriptor bounds.
	ShaderBufferResource table;
	if (table_value.dword_count == 2u) {
		table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
	} else if (DecodeBufferDescriptor(table_value, table)) {
		table_base = table.Base48();
		table_size = table.GetSize();
	} else {
		return false;
	}
	auto& keys = scratch.material_keys;
	keys.clear();
	// KYTY_WORKGROUP_IMAGES: T#s at table_offset + k * table_stride.
	uint32_t key_stride = 32u;
	if (indirect.workgroup_axis != UINT32_MAX) {
		// KYTY_WORKGROUP_IMAGES (adapted from upstream KytyPS5 0ee880090 and c23de6218): the key is
		// the workgroup id, so the keys are the dispatch's workgroups along that axis.
		uint32_t key_count = 0;
		if (runtime.workgroup_counts.size() == 3u && indirect.workgroup_axis < 3u &&
		    std::ranges::find(runtime.workgroup_counts, 0u) == runtime.workgroup_counts.end()) {
			key_count = runtime.workgroup_counts[indirect.workgroup_axis];
		}
		if (key_count == 0u) {
			// An indirect dispatch whose arguments the GPU writes: the renderer reads them and
			// retries (TakeWorkgroupCountsMissing).
			t_workgroup_counts_missing = true;
			return false;
		}
		// Upstream de3ed905c: a runtime base added to the id, and a constant division after it.
		uint32_t first = 0;
		if (!indirect.selector_first.IsEmpty() && !clean.Evaluate(indirect.selector_first, first)) {
			return false;
		}
		if (indirect.table_stride == 0u || key_count > MaxIndirectImageProbes) {
			return false;
		}
		key_stride = indirect.table_stride;
		keys.resize(key_count);
		std::iota(keys.begin(), keys.end(), first);
		if (indirect.workgroup_shift != 0u) {
			for (auto& key: keys) {
				key = static_cast<uint32_t>(uint64_t {key} * indirect.workgroup_multiplier >>
				                            indirect.workgroup_shift);
			}
		}
		std::ranges::sort(keys);
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
		if (uint64_t {indirect.table_offset} + uint64_t {keys.back()} * indirect.table_stride +
		        32u > UINT32_MAX + 1ull) {
			return false;
		}
	} else if (!indirect.selector) {
		uint32_t key_count = 0;
		const bool evaluated = clean.Evaluate(indirect.key_count, key_count);
		if (std::bit_cast<int32_t>(key_count) <= 0) key_count = 0;
		// KYTY_SELECTOR_BITSET (upstream 49ebbfd62): a reversed range starts at a runtime first key.
		uint32_t first = 0;
		if (!indirect.selector_first.IsEmpty() && !clean.Evaluate(indirect.selector_first, first)) {
			return false;
		}
		// A range base - maximum .. base whose base is below the maximum: the wrapped (negative)
		// keys are never the shader's, so only 0 .. base remain.
		if (uint64_t {first} + key_count > UINT32_MAX + 1ull) {
			key_count = static_cast<uint32_t>(uint64_t {first} + key_count - (UINT32_MAX + 1ull));
			first     = 0;
		}
		if (table_value.dword_count != 2u || !evaluated ||
		    key_count > MaxIndirectImageProbes ||
		    uint64_t {indirect.table_offset} + (uint64_t {first} + key_count) * 32u >
		        UINT32_MAX + 1ull) {
			return false;
		}
		keys.resize(key_count);
		std::iota(keys.begin(), keys.end(), first);
	} else if (indirect.selector->bitset) {
		// KYTY_SELECTOR_BITSET (adapted from upstream 145578ac0 and 914a3bb4b): the keys are
		// first + count - 1 - rank for every rank some bitmap record of the buffer sets.
		ShaderBufferResource      material;
		std::array<uint32_t, 8>   masks {};
		const auto                record_words = indirect.selector->stride / 4u;
		uint32_t                  first = 0, count = 0;
		if (table_value.dword_count != 2u || (record_words != 4u && record_words != 8u) ||
		    !DecodeBufferDescriptor(material_value, material) ||
		    !clean.Evaluate(indirect.selector_first, first) ||
		    !clean.Evaluate(indirect.key_count, count) || count != record_words * 32u ||
		    indirect.selector->offset != 0u || material.Type() != 0u ||
		    material.Stride() != indirect.selector->stride || material.SwizzleEnabled() ||
		    material.AddTid() || material.OutOfBounds() != 0u || (material.Base48() & 3u) != 0u) {
			return false;
		}
		// The last key is the range base; ranks above a base below the maximum are negative keys
		// (never the shader's) and are left out.
		const auto last  = static_cast<uint32_t>(uint64_t {first} + count - 1u);
		const bool wraps = uint64_t {first} + count > UINT32_MAX + 1ull;
		if (uint64_t {indirect.table_offset} + (uint64_t {wraps ? last : first + count - 1u} + 1u) * 32u >
		    UINT32_MAX + 1ull) {
			return false;
		}
		if (material.Format() != Prospero::BufferFormat::kInvalid) {
			// Bitmap storage can grow with resolution while the key domain stays at 32 keys per
			// record word: read it in bounded chunks.
			const auto words = material.GetSize() / sizeof(uint32_t);
			for (uint64_t first_word = 0; first_word < words; first_word += keys.size()) {
				keys.resize(static_cast<size_t>(
				    std::min<uint64_t>(words - first_word, MaxIndirectImageProbes)));
				if (first_word * 4u > UINT32_MAX ||
				    !ReadScalarTable(material.Base48(), material.GetSize(),
				                     static_cast<uint32_t>(first_word * 4u), runtime, keys)) {
					return false;
				}
				for (size_t word = 0; word < keys.size(); ++word) {
					masks[(first_word + word) % record_words] |= keys[word];
				}
			}
		}
		keys.clear();
		for (uint32_t word = 0; word < record_words; ++word) {
			for (auto mask = masks[word]; mask != 0u; mask &= mask - 1u) {
				const auto rank = word * 32u + static_cast<uint32_t>(std::countr_zero(mask));
				if (wraps && rank > last) continue;
				keys.push_back(last - rank);
			}
		}
		std::ranges::sort(keys);
	} else if (!indirect.selector_mask.IsEmpty()) {
		uint32_t mask = 0;
		uint32_t count = 0;
		if (material_value.dword_count != 2u || table_value.dword_count != 2u ||
		    !clean.Evaluate(indirect.selector_mask, mask) ||
		    !clean.Evaluate(indirect.key_count, count) || count == 0u || count > 32u) {
			return false;
		}
		if (count < 32u) mask &= (1u << count) - 1u;
		const auto material_base =
		    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
		keys.reserve(std::popcount(mask));
		while (mask != 0u) {
			const auto index = std::countr_zero(mask);
			const auto offset = static_cast<uint64_t>(indirect.selector->offset) +
			                    static_cast<uint64_t>(index) * indirect.selector->stride;
			if (offset > UINT32_MAX) return false;
			uint32_t key = 0;
			if (!ReadScalarTable(material_base, UINT64_MAX, static_cast<uint32_t>(offset),
			                     runtime, {&key, 1})) return false;
			keys.push_back(key);
			mask &= mask - 1u;
		}
		std::ranges::sort(keys);
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
	} else if (!indirect.key_count.IsEmpty()) {
		// KYTY_SELECTOR_BITSCAN / KYTY_SELECTOR_SHARED_LIST (adapted from upstream f297c0bfe,
		// 0e9c77b0a and ffb6ad6ca): the selector index is below key_count, so only those records
		// are read; out-of-bounds scalar reads return zero.
		const auto&          selector = *indirect.selector;
		ShaderBufferResource material;
		uint32_t             count = 0;
		if (!DecodeBufferDescriptor(material_value, material) || table_value.dword_count != 4u ||
		    !clean.Evaluate(indirect.key_count, count) || count > MaxIndirectImageProbes ||
		    (count != 0u && uint64_t {count - 1u} * selector.stride + selector.offset + 4u >
		                        UINT32_MAX + 1ull)) {
			return false;
		}
		const auto read_record = [&](uint32_t record) {
			uint32_t key = 0;
			if (!ReadScalarTable(material.Base48(), material.GetSize(),
			                     record * selector.stride + selector.offset, runtime, {&key, 1})) {
				return false;
			}
			keys.push_back(key);
			return true;
		};
		if (selector.ranks) {
			// KYTY_SELECTOR_MASK_RANKS (upstream 99f4eba22): a scan selects only the ranks some word
			// of its mask's address class sets.
			const auto&          mask = *selector.ranks;
			DescriptorValue      mask_value;
			ShaderBufferResource masks;
			if (count > 32u || mask.stride < 4u || (mask.stride & 3u) != 0u ||
			    !clean.EvaluateDescriptor(mask.source, mask_value) ||
			    !DecodeBufferDescriptor(mask_value, masks)) {
				return false;
			}
			const auto words = mask.stride / 4u;
			// Each chunk holds whole strides of the class, or one of its words.
			const uint64_t chunk = words <= MaxIndirectImageProbes ? MaxIndirectImageProbes / words * words
			                                                       : 1u;
			uint32_t       ranks = 0;
			for (uint64_t at = mask.offset; at + 4u <= masks.GetSize() && at <= UINT32_MAX;
			     at += std::max<uint64_t>(chunk, words) * 4u) {
				keys.resize(static_cast<size_t>(std::min<uint64_t>((masks.GetSize() - at) / 4u, chunk)));
				if (!ReadScalarTable(masks.Base48(), masks.GetSize(), static_cast<uint32_t>(at),
				                     runtime, keys)) {
					return false;
				}
				for (size_t word = 0; word < keys.size(); word += words) ranks |= keys[word];
			}
			keys.clear();
			if (count < 32u) ranks &= (1u << count) - 1u;
			for (; ranks != 0u; ranks &= ranks - 1u) {
				if (!read_record(static_cast<uint32_t>(std::countr_zero(ranks)))) return false;
			}
		} else {
			keys.reserve(count);
			for (uint32_t record = 0; record < count; ++record) {
				if (!read_record(record)) return false;
			}
		}
		if (selector.excluded) std::erase(keys, *selector.excluded);
		std::ranges::sort(keys);
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
	} else {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) || table_value.dword_count != 4u ||
		    material.Stride() != indirect.selector->stride) {
			return false;
		}
		// Enumerate every wrapped scalar-buffer offset that can pass the descriptor bounds.
		const auto step = std::gcd<uint64_t>(indirect.selector->stride, uint64_t {1} << 32u);
		const auto residue = static_cast<uint64_t>(indirect.selector->offset) % step;
		const auto limit = std::min<uint64_t>(UINT32_MAX, material.GetSize() + 3u);
		const auto probe_count = residue <= limit ? (limit - residue) / step + 1u : 0u;
		if (probe_count > MaxIndirectImageProbes) {
			return false;
		}
		keys.reserve(static_cast<size_t>(probe_count) + 1u);
		keys.push_back(0u);
		for (uint64_t offset = residue; offset <= limit && probe_count != 0u; offset += step) {
			uint32_t key = 0;
			if (!ReadScalarTable(material.Base48(), material.GetSize(),
			                     static_cast<uint32_t>(offset), runtime, {&key, 1})) {
				return false;
			}
			keys.push_back(key);
			if (limit - offset < step) {
				break;
			}
		}
		if (indirect.selector->excluded) std::erase(keys, *indirect.selector->excluded);
		std::ranges::sort(keys);
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
	}

	const auto children_begin = snapshot.images.size();
	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto root_image = specialization.images[image_index];
	IndirectImageIndex children;
	snapshot.flattened_srt.resize(mapping_offset + 1u + keys.size() * 2u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(keys.size());
	for (uint32_t entry = 0; entry < keys.size(); ++entry) {
		const auto key = keys[entry];
		DescriptorValue candidate;
		candidate.dword_count = 8u;
		const auto table_offset = key * key_stride + indirect.table_offset;
		if (!ReadScalarTable(table_base, table_size, table_offset, runtime, candidate.dwords)) {
			return false;
		}
		// KYTY_R128_TABLE_IMAGES: a 128-bit T# has no upper words (the next entry's bytes).
		if (program.info.images[image_index].r128 && GetCodegenOptions().r128_table_images) {
			std::fill(candidate.dwords.begin() + 4, candidate.dwords.begin() + 8, 0u);
		}
		if (NullImageDescriptor(candidate) ||
		    !ValidImageDescriptor(candidate, program.info.images[image_index].r128)) {
			candidate.dwords.fill(0);
		}
		uint32_t ordinal = 0;
		if (entry == 0) {
			snapshot.images[image_index] = candidate;
		} else if (snapshot.images[image_index] != candidate) {
			const auto lookup = [&] {
				if constexpr (Optimize) {
					return children.Find(snapshot.images, children_begin, candidate);
				} else {
					const auto found = std::find(snapshot.images.begin() + children_begin,
					                             snapshot.images.end(), candidate);
					return IndirectImageIndex::Lookup {
					    static_cast<uint32_t>(found - snapshot.images.begin() - children_begin + 1u)};
				}
			}();
			ordinal = lookup.ordinal;
			if (ordinal > snapshot.images.size() - children_begin) {
				if (snapshot.images.size() >= ShaderInfo::MaxImages) {
					return false;
				}
				snapshot.images.push_back(candidate);
				auto child = root_image;
				child.indirect_root = image_index;
				specialization.images.push_back(child);
				if constexpr (Optimize) children.Insert(lookup);
			}
		}
		snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = key;
		snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = ordinal;
	}
	if (snapshot.images.size() == children_begin) {
		snapshot.flattened_srt.resize(mapping_offset);
	} else {
		auto& root = specialization.images[image_index];
		root.indirect_root = image_index;
		root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
		root.indirect_search_iterations = std::bit_width(keys.size());
	}
	return true;
}

} // namespace

struct SamplerPlan {
	struct Binding {
		uint32_t     source;
		SamplerClass type;
	};
	std::array<std::array<uint32_t, SamplerClassCount>, ShaderInfo::MaxSamplers> mapping;
	std::array<Binding, ShaderInfo::MaxSamplers>                bindings;
	uint32_t                                                  sampler_count = 0;
};

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization)
	    : source_count(static_cast<uint32_t>(specialization.images.size())) {
		EXIT_IF(specialization.images.size() > indices.size());
		for (uint32_t index = 0; index < source_count; index++) {
			indices[index] = specialization.images[index].fmask ? UINT32_MAX : count++;
		}
	}

	uint32_t operator[](uint32_t index) const {
		EXIT_IF(index >= source_count);
		return indices[index];
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != source_count);
		if (count == source_count) {
			return;
		}
		for (uint32_t index = 0; index < source_count; index++) {
			if (indices[index] != UINT32_MAX && indices[index] != index) {
				images[indices[index]] = std::move(images[index]);
			}
		}
		images.resize(count);
	}

private:
	std::array<uint32_t, ShaderInfo::MaxImages> indices;
	uint32_t                                    source_count;
	uint32_t                                    count = 0;
};

template <typename Images>
bool BuildSamplerPlan(const ShaderInfo& base, const Images& images, SamplerPlan& plan);

bool PlanReadsWorkgroupCounts(const ResourcePlan& program) {
	return std::ranges::any_of(program.info.images, [&](const ImageResource& image) {
		const auto* source = Source(program, image.source);
		return source != nullptr && source->indirect_descriptor.has_value() &&
		       source->indirect_descriptor->workgroup_axis != UINT32_MAX;
	});
}

bool TakeWorkgroupCountsMissing() {
	return std::exchange(t_workgroup_counts_missing, false);
}

void SetEqaaFmaskMapForTesting(bool enabled) {
	g_eqaa_fmask_map.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

void SetMsDepthSample0ForTesting(bool enabled) {
	g_ms_depth_sample0.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

void SetUnmappedBufferProbe(SrtMappedRange probe) {
	g_unmapped_buffer_probe.store(probe, std::memory_order_relaxed);
}

uint64_t UnmappedBufferNulls() {
	return g_unmapped_buffer_nulls.load(std::memory_order_relaxed);
}

// KYTY_NULL_BUFFER_SHARE (CodegenOptions::null_buffer_share): a buffer slot of the plan that
// specializes the same in every permutation but for null vs live. Bindings selected at run time
// (indirect roots, image aliases) and scalar buffers keep their own null specialization.
static bool NullShareSlot(const BufferResource& buffer) {
	return buffer.indirect_root == BufferResource::NoIndirectBuffer &&
	       buffer.image_alias == BufferResource::NoImageAlias && !buffer.scalar &&
	       buffer.indirect_resources.empty();
}

static uint64_t PackNullShare(const ResourceSpecialization::Buffer& b) {
	return 1u | (uint64_t {b.packed_stride & 0xfffffu} << 1u) |
	       (uint64_t {static_cast<uint32_t>(b.descriptor_format) & 0xffu} << 21u) |
	       (uint64_t {b.descriptor_swizzle & 0xfffu} << 29u) | (uint64_t {b.zero_stride_oob ? 1u : 0u} << 41u);
}

// The last live specialization per (shader, stage, user data, slot): an open-addressed table of
// atomics (the materialization hot path takes no lock). A full table shares nothing more.
struct NullShareEntry {
	std::atomic<uint64_t> key {0};
	std::atomic<uint64_t> value {0};
};
constexpr size_t                         NullShareTableSize = size_t {1} << 15u;
static std::array<NullShareEntry, NullShareTableSize> g_null_share_table;

static uint64_t NullShareKey(uint64_t shader_hash, uint32_t stage, uint32_t user_data_base,
                             uint32_t user_data_count, uint32_t i) {
	uint64_t key = shader_hash ^ (uint64_t {stage} << 56u) ^ (uint64_t {user_data_base} << 40u) ^
	               (uint64_t {user_data_count} << 48u) ^ ((uint64_t {i} + 1u) * 0x9e3779b97f4a7c15ull);
	key ^= key >> 29u;
	key *= 0xbf58476d1ce4e5b9ull;
	key ^= key >> 32u;
	return key == 0u ? 1u : key;
}

static std::atomic<uint64_t>* NullShareValue(uint64_t key, bool insert) {
	for (size_t probe = 0; probe < 16u; ++probe) {
		auto&    entry    = g_null_share_table[(key + probe) & (NullShareTableSize - 1u)];
		uint64_t existing = entry.key.load(std::memory_order_acquire);
		if (existing == key) return &entry.value;
		if (existing == 0u) {
			if (!insert) return nullptr;
			if (entry.key.compare_exchange_strong(existing, key, std::memory_order_acq_rel) ||
			    existing == key) {
				return &entry.value;
			}
		}
	}
	return nullptr;
}

static ResourceSpecialization::Buffer UnpackNullShare(uint64_t v) {
	return {
	    .packed_stride      = static_cast<uint32_t>((v >> 1u) & 0xfffffu),
	    .descriptor_format  = static_cast<Prospero::BufferFormat>((v >> 21u) & 0xffu),
	    .descriptor_swizzle = static_cast<uint32_t>((v >> 29u) & 0xfffu),
	    .zero_stride_oob    = ((v >> 41u) & 1u) != 0u,
	};
}

void SeedNullBufferShare(uint64_t shader_hash, uint32_t stage, uint32_t user_data_base,
                         uint32_t user_data_count, const ResourceSpecialization& specialization) {
	for (uint32_t i = 0; i < specialization.buffers.size(); ++i) {
		const auto& buffer = specialization.buffers[i];
		// Only live slots: a null V# specializes with stride 0 and no format (zero_stride_oob).
		if (buffer.zero_stride_oob ||
		    (buffer.packed_stride == 0u && buffer.descriptor_format == Prospero::BufferFormat::kInvalid)) {
			continue;
		}
		if (auto* slot = NullShareValue(NullShareKey(shader_hash, stage, user_data_base, user_data_count, i), true);
		    slot != nullptr) {
			slot->store(PackNullShare(buffer), std::memory_order_relaxed);
		}
	}
}

static bool SpecializeBuffer(const ResourcePlan& program, const BufferResource& buffer,
                             DescriptorValue& descriptor_value,
                             ResourceSpecialization& specialization, uint32_t i) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
		return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
	}
	if (NullBufferDescriptor(descriptor)) {
		descriptor_value.dwords.fill(0);
		descriptor = {};
	}
	// KYTY_NULL_BUFFER_SHARE: a null V# (all zeros: the renderer binds the null buffer with the
	// null marker offset, so every access is out of range and a formatted read gives zeros) takes
	// the last live specialization of its slot, so null and live share one permutation.
	if (GetCodegenOptions().null_buffer_share && NullShareSlot(buffer)) {
		// A raw slot whose V# has stride 0 and OOB_SELECT 0 is out of range for every access (the
		// structured check offset >= stride always fails): it reads 0 and drops writes exactly as a
		// null V#, so it is bound and specialized as one.
		if (!buffer.formatted && descriptor.OutOfBounds() == 0u &&
		    (descriptor.PackedStride() & 0x3fffu) == 0u) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		const auto key  = NullShareKey(program.shader_hash, static_cast<uint32_t>(program.stage),
		                                program.user_data_base, program.user_data_count, i);
		const bool null = std::ranges::all_of(descriptor_value.dwords, [](uint32_t w) { return w == 0u; });
		if (null) {
			if (const auto* slot = NullShareValue(key, false); slot != nullptr) {
				if (const auto live = slot->load(std::memory_order_relaxed); live != 0u) {
					specialization.buffers.push_back(UnpackNullShare(live));
					return true;
				}
			}
		} else if (auto* slot = NullShareValue(key, true);
		           slot != nullptr &&
		           !(descriptor.OutOfBounds() == 0u && (descriptor.PackedStride() & 0x3fffu) == 0u)) {
			// (A live V# whose reads fold to constants, zero_stride_oob, is not shared: its
			// constants are its DST_SELs, not the null V#'s zeros.)
			auto       packed = descriptor.PackedStride();
			const auto st     = packed & 0x3fffu;
			const bool sw     = st != 0u && ((packed >> 14u) & 1u) != 0u;
			if (st == 0u) {
				packed &= ~((1u << 14u) | (3u << 16u));
			} else if (!sw) {
				packed &= ~(3u << 16u);
			}
			slot->store(PackNullShare({
			               .packed_stride      = packed,
			               .descriptor_format  = buffer.formatted ? descriptor.Format()
			                                                      : Prospero::BufferFormat::kInvalid,
			               .descriptor_swizzle = buffer.formatted ? descriptor.DstSelXYZW()
			                                                      : DstSel(4, 5, 6, 7),
			               .zero_stride_oob    = descriptor.OutOfBounds() == 0u && st == 0u,
			           }),
			           std::memory_order_relaxed);
		}
	}
	auto       packed_stride = descriptor.PackedStride();
	const auto stride        = packed_stride & 0x3fffu;
	const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
	if (stride == 0u) {
		packed_stride &= ~((1u << 14u) | (3u << 16u));
	} else if (!swizzle) {
		packed_stride &= ~(3u << 16u);
	}
	// KYTY_BUFFER_OOB_MODES: OOB_SELECT 0 with a stride also checks offset < stride (Table 35);
	// BufferOobOffsetCheck marks such a V# for the emitter (ByteAddress).
	if (GetCodegenOptions().buffer_oob_modes && descriptor.OutOfBounds() == 0u && stride != 0u &&
	    !swizzle) {
		packed_stride |= BufferOobOffsetCheck;
	}
	specialization.buffers.push_back({
	    .packed_stride     = packed_stride,
	    .descriptor_format = buffer.formatted
	                             ? descriptor.Format()
	                             : Prospero::BufferFormat::kInvalid,
	    .descriptor_swizzle =
	        buffer.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7),
	    .zero_stride_oob = descriptor.OutOfBounds() == 0u && stride == 0u,
	});
	return true;
}

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization, bool buffers_ready) {
	if (!buffers_ready) {
		specialization.buffers.clear();
		specialization.buffers.reserve(program.info.buffers.size());
		for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
			if (!SpecializeBuffer(program, program.info.buffers[i], snapshot.buffers[i], specialization, i)) return false;
		}
	}
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		const auto& descriptor = snapshot.images[i];
		auto&       image      = specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = ImageMipCount(base, descriptor);
		if (image.mip_count == 0u) {
			return SpecializationFail(
			    fmt::format("image descriptor {} has an invalid mip range", i));
		}
		if (NullImageDescriptor(descriptor)) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = Decoder::ImageDimension::Dim2D;
			image.cube          = false;
			if (GetCodegenOptions().null_image_keep_dim &&
			    base.dimension != Decoder::ImageDimension::Unknown) {
				// KYTY_NULL_IMAGE_KEEP_DIM (TheCruZ 4f1b2ebc): a null descriptor reads zero through a
				// view of any type (NullTextureDesc makes one per dimension), so the image keeps the
				// dimension the shader samples with and a game that binds it later (a texture still
				// streaming in) reuses the same program instead of compiling another.
				image.dimension = base.dimension;
				image.cube      = base.cube;
			}
			if (image.bindless) {
				// No one descriptor describes a bindless image: keep the dimension the
				// instruction samples, which picks the set-1 array.
				image.dimension = base.dimension == Decoder::ImageDimension::Unknown
				                      ? Decoder::ImageDimension::Dim2D
				                      : base.dimension;
				image.cube      = base.cube;
			}
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32SInt &&
		                         format != Prospero::BufferFormat::k32Float)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		image.fmask_fragments =
		    image.fmask && EqaaFmaskMap() ? FmaskEqaaFragmentCount(format) : uint8_t {0};
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		// KYTY_DEPTH_BITS_READ (upstream aafe014d0, see ReadsDepthBits): raw loads only, since
		// filtering would blend the codes (ours: no sampled pair and no compare uses the image).
		const bool depth_bits =
		    GetCodegenOptions().depth_bits_read && !storage && !base.depth_compare &&
		    format == Prospero::BufferFormat::k16UInt &&
		    static_cast<Prospero::TileMode>((descriptor.dwords[3] >> 20u) & 0x1fu) ==
		        Prospero::TileMode::kDepth &&
		    std::ranges::none_of(program.info.sampled_pairs,
		                         [&](const auto& pair) { return pair.image == i; });
		// KYTY_SRGB8_DECODE: a sampled 8 / 8_8 sRGB texture is decoded in the shader (Format::IsSrgbFormat).
		// KYTY_SRGB8_VIEW: 8 sRGB is decoded by the sampler (an R8_SRGB view), before filtering.
		const bool srgb_decode  = GetCodegenOptions().srgb8_decode && !storage && !base.depth_compare &&
		                         Format::IsSrgbFormat(format) &&
		                         !(format == Prospero::BufferFormat::k8Srgb && Prospero::Srgb8ViewActive());
		image.conversion_format = depth_bits || srgb_decode ? format : ImageConversionFormat(format);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		if (storage && GetCodegenOptions().storage_swizzle_canon &&
		    image.conversion_format == Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = CanonicalStorageSwizzle(image.shader_swizzle, format);
		}
		// KYTY_RAW_SMALL_SINT_STORAGE: write-only 16/8-bit SINT storage, the same low bits through an
		// unsigned view (Erlou93/KytyPS5-Yotei).
		const bool raw_small_sint_storage =
		    storage && GetCodegenOptions().raw_small_sint_storage &&
		    (format == Prospero::BufferFormat::k16SInt || format == Prospero::BufferFormat::k8SInt) &&
		    base.written && !base.read && !base.atomic;
		const bool raw_sint_storage = raw_small_sint_storage ||
		                              (storage && format == Prospero::BufferFormat::k32SInt &&
		                               base.written && (base.atomic ? !base.atomic64 : !base.read));
		image.numeric_class         = Prospero::SampledTextureNumericClass(
		            depth_bits ? Prospero::BufferFormat::k16UNorm : format);
		if (storage) {
			if ((!raw_sint_storage && image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
			image.clamp_half_float =
			    GetCodegenOptions().float_store_clamp && base.written && !base.atomic &&
			    image.numeric_class == Prospero::TextureNumericClass::Float &&
			    (format == Prospero::BufferFormat::k16Float ||
			     format == Prospero::BufferFormat::k16_16Float ||
			     format == Prospero::BufferFormat::k16_16_16_16Float);
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported &&
		           !base.depth_compare && !image.fmask) {
			// A sampled texture in a format the host cannot represent: the specialization treats
			// it as a null image (the renderer binds a null texture for it, ResolveTextureFull)
			// instead of dropping every draw that samples it.
			(void)SpecializationFail(fmt::format(
			    "sampled image descriptor {} uses unsupported format {} (T# {:08x} {:08x} {:08x} "
			    "{:08x} {:08x} {:08x} {:08x} {:08x}); bound as a null texture",
			    i, static_cast<uint32_t>(format), descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
			image.numeric_class     = Prospero::TextureNumericClass::Float;
			image.dimension         = Decoder::ImageDimension::Dim2D;
			image.cube              = false;
			image.conversion_format = Prospero::BufferFormat::kInvalid;
			continue;
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			return SpecializationFail(
			    fmt::format("sampled image descriptor {} uses unsupported format {} "
			                "(T# {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x})",
			                i, static_cast<uint32_t>(format), descriptor.dwords[0],
			                descriptor.dwords[1], descriptor.dwords[2], descriptor.dwords[3],
			                descriptor.dwords[4], descriptor.dwords[5], descriptor.dwords[6],
			                descriptor.dwords[7]));
		}
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index || root.bindless) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < snapshot.flattened_srt.size()
		                           ? snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		if (root.indirect_search_iterations == 0u || key_count < 2u ||
		    static_cast<size_t>(root.indirect_mapping_offset) + 1u +
		            static_cast<size_t>(key_count) * 2u >
		        snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < specialization.images.size(); resource++) {
			if (specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = specialization.images[exemplar];
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		// KYTY_INDIRECT_LEADING_COORDS (upstream d3768c00e): each candidate samples with the
		// coordinate count of its own type, so a 2D or 3D candidate reads the leading coordinates
		// of an equal or larger dimension, except an array with more than one slice, which reads
		// the next coordinate as its slice. Derivative, LOD, clamp and offset operands keep the
		// root's dimension, so roots that use them (or mips) stay strict.
		const auto& base    = program.info.images[root_index];
		const auto  leading = [&](Decoder::ImageDimension dimension, const DescriptorValue& value) {
			const bool sliced =
			    static_cast<Prospero::ImageType>((value.dwords[3] >> 28u) & 0xfu) ==
			        Prospero::ImageType::kColor2DArray &&
			    (value.dwords[4] & 0x1fffu) != ((value.dwords[4] >> 16u) & 0x1fffu);
			return !sliced &&
			       (dimension == Decoder::ImageDimension::Dim2D || dimension == base.dimension);
		};
		const bool leading_coordinates =
		    IndirectLeadingCoordinates() &&
		    (base.dimension == Decoder::ImageDimension::Dim2D ||
		     base.dimension == Decoder::ImageDimension::Dim3D) &&
		    !base.cube && !image_class.cube && leading(image_class.dimension, snapshot.images[exemplar]) &&
		    std::ranges::none_of(program.memory_info, [&](const MemoryInfo& memory) {
			    return memory.kind == ResourceKind::Image && memory.resource == root_index &&
			           (memory.image_has_mip ||
			            (memory.image_sample_flags &
			             (Decoder::ImageSampleFlagLod | Decoder::ImageSampleFlagDerivative |
			              Decoder::ImageSampleFlagLodClamp | Decoder::ImageSampleFlagOffset)) != 0u);
		    });
		for (uint32_t candidate = 0; candidate < specialization.images.size(); candidate++) {
			auto& image = specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			if (NullImageDescriptor(snapshot.images[candidate])) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			const bool same_coordinates = image.dimension == image_class.dimension &&
			                              image.cube == image_class.cube;
			// KYTY_STORAGE_TABLE_LOOSE (adapted from Erlou93/KytyPS5-Yotei M120): a storage-table
			// candidate that differs from the exemplar only in a plain dimension (not cube or MSAA:
			// each write takes its coordinates as the candidate's own type needs them) or in the
			// swizzle of a format without conversion (each write applies its own, StoreTexel) keeps
			// its own type, as the GPU addresses it by its own T#, instead of failing the stage.
			const auto plain_dimension = [](Decoder::ImageDimension dimension) {
				return dimension == Decoder::ImageDimension::Dim1D ||
				       dimension == Decoder::ImageDimension::Dim1DArray ||
				       dimension == Decoder::ImageDimension::Dim2D ||
				       dimension == Decoder::ImageDimension::Dim2DArray ||
				       dimension == Decoder::ImageDimension::Dim3D;
			};
			if (GetCodegenOptions().storage_table_loose &&
			    program.info.images[root_index].resource_class == ImageResourceClass::Storage &&
			    !image.cube && !image_class.cube && plain_dimension(image.dimension) &&
			    plain_dimension(image_class.dimension) &&
			    image.numeric_class == image_class.numeric_class &&
			    image.mip_count == image_class.mip_count &&
			    image.conversion_format == image_class.conversion_format &&
			    image.conversion_format == Prospero::BufferFormat::kInvalid &&
			    (!same_coordinates || image.shader_swizzle != image_class.shader_swizzle)) {
				continue;
			}
			// KYTY_SAMPLED_TABLE_READS (adapted from Erlou93/KytyPS5-Yotei EmitImageAccess): a
			// sampled-table candidate of another plain dimension keeps its own type when every access
			// to the table takes the candidate's own coordinates: loads (one case per candidate) and
			// samples without gradients or offsets; a sample with an LOD or clamp after its
			// coordinates only when the candidate has no more coordinates than the instruction.
			const auto coordinates = [](Decoder::ImageDimension dimension) {
				return dimension == Decoder::ImageDimension::Dim1D ? 1u
				       : dimension == Decoder::ImageDimension::Dim1DArray ||
				               dimension == Decoder::ImageDimension::Dim2D
				           ? 2u
				           : 3u;
			};
			const auto& root_base = program.info.images[root_index];
			if (GetCodegenOptions().sampled_table_reads && !same_coordinates &&
			    root_base.resource_class == ImageResourceClass::Sampled &&
			    (root_base.table_access & (ImageTableAccessWrite | ImageTableAccessOther)) == 0u &&
			    (root_base.table_access & (ImageTableAccessRead | ImageTableAccessSample |
			                               ImageTableAccessSampleExtra)) != 0u &&
			    !image.cube && !image_class.cube && plain_dimension(image.dimension) &&
			    plain_dimension(image_class.dimension) &&
			    plain_dimension(root_base.dimension) &&
			    ((root_base.table_access & ImageTableAccessSampleExtra) == 0u ||
			     (coordinates(image.dimension) <= coordinates(root_base.dimension) &&
			      coordinates(image_class.dimension) <= coordinates(root_base.dimension))) &&
			    image.numeric_class == image_class.numeric_class &&
			    image.mip_count == image_class.mip_count &&
			    image.conversion_format == image_class.conversion_format &&
			    image.shader_swizzle == image_class.shader_swizzle) {
				continue;
			}
			const bool prefix_coordinates = leading_coordinates && !image.cube &&
			                                leading(image.dimension, snapshot.images[candidate]);
			if (image.numeric_class != image_class.numeric_class ||
			    (!same_coordinates && !(is_2d(image.dimension) && is_2d(image_class.dimension)) &&
			     !prefix_coordinates) ||
			    image.mip_count != image_class.mip_count ||
			    image.conversion_format != image_class.conversion_format ||
			    image.shader_swizzle != image_class.shader_swizzle) {
				// The fields that differ (exemplar/candidate): class, dimension, cube, mips,
				// conversion format, swizzle, T# format; and the table's access kinds.
				const auto format_of = [&](uint32_t index) {
					return (snapshot.images[index].dwords[1] >> 20u) & 0x1ffu;
				};
				return SpecializationFail(fmt::format(
				    "indirect image table at pc 0x{:08x} has incompatible candidates (class {}/{} "
				    "dim {}/{} cube {}/{} mips {}/{} conv {}/{} swizzle {:03x}/{:03x} format {}/{}; "
				    "{} dim {} access 0x{:x})",
				    root_base.first_use_pc, static_cast<uint32_t>(image_class.numeric_class),
				    static_cast<uint32_t>(image.numeric_class),
				    static_cast<uint32_t>(image_class.dimension),
				    static_cast<uint32_t>(image.dimension), image_class.cube, image.cube,
				    image_class.mip_count, image.mip_count,
				    static_cast<uint32_t>(image_class.conversion_format),
				    static_cast<uint32_t>(image.conversion_format), image_class.shader_swizzle,
				    image.shader_swizzle, format_of(exemplar), format_of(candidate),
				    root_base.resource_class == ImageResourceClass::Storage ? "storage" : "sampled",
				    static_cast<uint32_t>(root_base.dimension), root_base.table_access));
			}
		}
	}
	// Native filtering/border variants read their source sampler's snapshot entry
	// (SamplerResource::snapshot_index); the snapshot holds the program's samplers only.
	SamplerPlan sampler_plan;
	if (!BuildSamplerPlan(program.info, specialization.images, sampler_plan)) {
		return SpecializationFail("specialized sampler layout exceeds its resource limit");
	}
	ImageRemap(specialization).Apply(snapshot.images);
	return true;
}

template <typename Images>
bool BuildSamplerPlan(const ShaderInfo& base, const Images& images, SamplerPlan& plan) {
	if (base.samplers.size() > plan.mapping.size()) {
		return false;
	}
	std::array<uint8_t, ShaderInfo::MaxSamplers> usage {};
	plan.sampler_count = static_cast<uint32_t>(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		// (A specialization's images carry no depth_compare: the program's image has it.)
		const bool depth_compare = pair.image < base.images.size() && base.images[pair.image].depth_compare;
		usage[pair.sampler] |=
		    1u << static_cast<uint32_t>(ClassifySampler(images[pair.image], depth_compare));
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			const auto target = first ? index : plan.sampler_count++;
			if (target >= ShaderInfo::MaxSamplers) return false;
			mapping[type]         = target;
			plan.bindings[target] = {index, static_cast<SamplerClass>(type)};
			first                = false;
		}
	}
	return true;
}

static std::vector<ResourceBlock> ResourceControlFlow(const Program& program) {
	// Any shader write may alias a scalar predicate read, including on a later loop visit.
	if (program.blocks.size() != program.block_info.size() || HasShaderMemoryWrites(program)) {
		return {};
	}
	std::unordered_map<uint32_t, uint32_t> indices;
	for (uint32_t i = 0; i < program.block_info.size(); i++) {
		if (!indices.emplace(program.block_info[i].id, i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	// KYTY_SRT_IMAGE_REACH (upstream 9801fd862): images get activity slots of their own.
	const bool image_reach = GetCodegenOptions().srt_image_reach;
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&           info       = program.block_info[i];
		const auto&           terminator = info.terminator;
		std::vector<uint32_t> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors.push_back(terminator.true_block); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors = {terminator.true_block, terminator.false_block};
				if (ValidateRuntimeValue(program, info.condition, RuntimeValueType::Integer)) {
					block.condition = info.condition;
				}
				break;
			case CFG::TerminatorKind::IndirectBranch:
				successors = terminator.indirect_targets;
				break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only) {
				continue;
			}
			// KYTY_SRT_VARIANT_READS: an access through a V# the shader computes goes through BDA and
			// has no descriptor source (and no bound buffer to index).
			if (memory.kind == ResourceKind::IndirectBuffer &&
			    memory.resource == NoIndirectBufferResource) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				block.sources.push_back(program.info.buffers.at(memory.resource).source);
			} else {
				block.sources.push_back(program.info.images.at(memory.resource).source);
				if (image_reach) {
					block.sources.push_back(
					    ImageActiveSlot(program.descriptor_sources.size(), memory.resource));
				}
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		std::ranges::sort(block.sources);
		block.sources.erase(std::unique(block.sources.begin(), block.sources.end()),
		                    block.sources.end());
	}
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// ResolveComputeBufferFill proves complete workgroups and matching requested extents, so every
// dispatch-bound predicate accepted here is true for every invocation covered by the fill.
static bool IsFullDispatchPredicate(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value == Value(true)) return true;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) return false;
	if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		return IsFullDispatchPredicate(inst->Arg(0), depth + 1) &&
		       IsFullDispatchPredicate(inst->Arg(1), depth + 1);
	}
	if (inst->GetOpcode() != ValueOpcode::ULessThan32) return false;
	const auto* global = inst->Arg(0).Resolve().TryInstruction();
	const auto* extent = inst->Arg(1).Resolve().TryInstruction();
	if (global == nullptr || global->GetOpcode() != ValueOpcode::GetBuiltin ||
	    global->Arg(0).Resolve() != Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) ||
	    extent == nullptr || extent->GetOpcode() != ValueOpcode::GetDispatchThreadExtent)
		return false;
	const auto axis = global->Arg(1).Resolve();
	return axis.IsImmediate() && axis.U32() < 3 && extent->Arg(0).Resolve() == axis;
}

// Nonnegative affine coefficients for constant, local and workgroup coordinates. Reject modular
// arithmetic that could wrap; runtime coverage also bounds the largest invocation index.
static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard,
                                                      uint32_t depth = 0) {
	value = ResolveActiveU32(value, guard);
	if (depth > 32 || value.GetType() != Type::U32) {
		return {};
	}
	if (value.IsImmediate()) {
		return std::array<uint64_t, 3> {value.U32(), 0, 0};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return {};
	}
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::GetBuiltin && inst->Arg(1) == Value(axis)) {
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) {
			return std::array<uint64_t, 3> {0, 1, 0};
		}
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) {
			return std::array<uint64_t, 3> {0, 0, 1};
		}
	}
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) {
		return {};
	}
	auto left  = FillIndex(inst->Arg(0), axis, guard, depth + 1);
	auto right = FillIndex(inst->Arg(1), axis, guard, depth + 1);
	if (!left || !right) {
		return {};
	}
	if (op == ValueOpcode::IMul32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		std::swap(left, right);
	}
	if (op != ValueOpcode::IAdd32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		return {};
	}
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (uint32_t i = 0; i < left->size(); ++i) {
		(*left)[i] =
		    op == ValueOpcode::IAdd32 ? (*left)[i] + (*right)[i] : (*left)[i] * (*right)[0];
		if ((*left)[i] > UINT32_MAX) return {};
	}
	return left;
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() ||
	    program.blocks.size() != program.block_info.size() || program.info.uses_dma ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<uint32_t> visited;
	uint32_t                     index = 0;
	const Inst*                  store = nullptr;
	for (;;) {
		if (!visited.insert(index).second) return {};
		for (const auto& inst: *program.blocks[index]) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = program.block_info[index].terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		const auto next = std::ranges::find(program.block_info, term.true_block, &BlockInfo::id);
		if (next == program.block_info.end()) return {};
		index = static_cast<uint32_t>(next - program.block_info.begin());
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto& memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	Value guard(true);
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    store->Arg(3).Resolve() != Value(true)) return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis, guard);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u))
			return {};
		guard = store->Arg(5).Resolve();
		if (!IsFullDispatchPredicate(guard)) return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0, guard);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = ResolveActiveU32(data, guard);
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = ResolveActiveU32(result.fill.words == 1 ? data : vector->Arg(i), guard);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

// Flat slots the host never needs (port of senaxx 0845beadf), so a compute refresh on the command
// processor can leave their bytes to the GPU when GPU work wrote them (KYTY_SRT_GPU_FILL): the
// slot's value is a raw scalar read (S_LOAD / S_BUFFER_LOAD) whose address comes from user data
// and immediates alone, and no value the host evaluates refers to it: descriptor dwords and
// indirect-descriptor keys, branch conditions, uniform fills and other reads' addresses. A slot
// that only the index/offset/soffset operands of buffer stores and atomics use (the write-range
// bounds, WriteRangeAnalysis.cpp, which read the flat buffer on the host) is marked 2: when it is
// filled, the renderer takes the whole binding as written for write ranges that read it.
// Clean (specialization) slots are never filled. Indexed like clean_flat_slots; empty when no
// slot qualifies or when the plan's flat offsets are not its read indices.
static std::vector<uint8_t> GpuFillSlots(const Program& program, const ResourcePlan& plan) {
	const auto count = plan.srt_reads.size();
	for (size_t slot = 0; slot < count; slot++) {
		if (plan.srt_reads[slot].flat_offset != slot) {
			return {};
		}
	}
	// A raw read instruction of either form (the plan's clone or the program's own) -> its slot.
	std::unordered_map<const Inst*, uint32_t> read_slot;
	for (uint32_t slot = 0; slot < count; slot++) {
		if (const auto* inst = plan.srt_reads[slot].value.Resolve().TryInstruction(); inst != nullptr) {
			read_slot.emplace(inst, slot);
		}
		if (slot < program.srt_reads.size()) {
			if (const auto* inst = program.srt_reads[slot].value.Resolve().TryInstruction();
			    inst != nullptr) {
				read_slot.emplace(inst, slot);
			}
		}
	}
	// Why a slot is a host value (KYTY_SRT_GPU_FILL_DIAG): the first root kind that reached it.
	std::vector<uint8_t>            host(count, 0u);
	std::vector<Value>              pending;
	std::unordered_set<const Inst*> visited;
	uint8_t                         why = 1;
	const auto mark = [&](Value root) {
		if (root.IsEmpty()) {
			return;
		}
		pending.push_back(root);
		while (!pending.empty()) {
			const auto value = pending.back().Resolve();
			pending.pop_back();
			if (value.IsEmpty() || value.IsImmediate()) {
				continue;
			}
			const auto* inst = value.TryInstruction();
			if (inst == nullptr || !visited.insert(inst).second) {
				continue;
			}
			if (const auto found = read_slot.find(inst); found != read_slot.end()) {
				host[found->second] = host[found->second] != 0u ? host[found->second] : why;
			}
			if (inst->GetOpcode() == ValueOpcode::ReadConst) {
				const auto slot = inst->Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 || slot.U32() >= count) {
					// An unknown slot: any of them may be a host value.
					for (auto& each: host) {
						each = each != 0u ? each : why;
					}
					continue;
				}
				host[slot.U32()] = host[slot.U32()] != 0u ? host[slot.U32()] : why;
				pending.push_back(plan.srt_reads[slot.U32()].value);
				continue;
			}
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				pending.push_back(inst->Arg(arg));
			}
		}
	};
	why = 1; // descriptor
	for (const auto& source: plan.descriptor_sources) {
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			mark(source.dwords[dword]);
		}
		if (source.indirect_descriptor.has_value()) {
			mark(source.indirect_descriptor->key_count);
			mark(source.indirect_descriptor->selector_first);
			mark(source.indirect_descriptor->selector_mask);
		}
	}
	why = 2; // condition
	for (const auto& block: plan.control_flow) {
		mark(block.condition);
	}
	why = 3; // uniform fill
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		mark(plan.uniform_fill.values[i]);
	}
	why = 4; // another read's address
	// Another read's address operands (the read itself is not a host use of its own slot).
	for (const auto& read: plan.srt_reads) {
		if (const auto* inst = read.value.Resolve().TryInstruction(); inst != nullptr) {
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				mark(inst->Arg(arg));
			}
		}
	}
	why = 5; // buffer store address
	// Buffer stores' and atomics' address operands: the write-range program evaluates them from
	// the snapshot's flat buffer on the host (AnalyzeBufferWriteRanges visits the same operands).
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = BufferAccessOf(inst.GetOpcode());
			if (access != BufferAccess::Write && access != BufferAccess::Atomic) {
				continue;
			}
			for (size_t arg = 1; arg < std::min<size_t>(inst.NumArgs(), 4); arg++) {
				mark(inst.Arg(arg));
			}
		}
	}
	// The read's own address must not depend on memory: one host read per slot, the slot's own.
	const auto memory_free = [&](const Inst& read) {
		std::vector<Value>              stack;
		std::unordered_set<const Inst*> seen;
		for (size_t arg = 0; arg < read.NumArgs(); arg++) {
			stack.push_back(read.Arg(arg));
		}
		while (!stack.empty()) {
			const auto value = stack.back().Resolve();
			stack.pop_back();
			if (value.IsEmpty() || value.IsImmediate()) {
				continue;
			}
			const auto* inst = value.TryInstruction();
			if (inst == nullptr || !seen.insert(inst).second) {
				continue;
			}
			switch (inst->GetOpcode()) {
				case ValueOpcode::ReadConst:
				case ValueOpcode::LoadAddressU32:
				case ValueOpcode::ReadConstBuffer:
				case ValueOpcode::LoadBufferU32:
				case ValueOpcode::Phi: return false;
				default: break;
			}
			if (read_slot.contains(inst)) {
				return false;
			}
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				stack.push_back(inst->Arg(arg));
			}
		}
		return true;
	};
	static const bool diag = [] {
		const auto* value = std::getenv("KYTY_SRT_GPU_FILL_DIAG");
		return value != nullptr && value[0] == '1';
	}();
	std::vector<uint8_t> fill(count, 0u);
	bool                 any = false;
	std::string          reasons;
	for (size_t slot = 0; slot < count; slot++) {
		const auto* inst   = plan.srt_reads[slot].value.Resolve().TryInstruction();
		const char* reason = nullptr;
		if (inst == nullptr) {
			reason = "no-inst";
		} else if (host[slot] != 0u && host[slot] != 5u) {
			static constexpr const char* kinds[] = {"?", "descriptor", "condition", "uniform-fill",
			                                        "read-address", "store-address"};
			reason = kinds[host[slot] < 6u ? host[slot] : 0u];
		} else if (slot < plan.clean_flat_slots.size() && plan.clean_flat_slots[slot] != 0u) {
			reason = "clean";
		}
		if (reason == nullptr) {
			const auto op    = inst->GetOpcode();
			const auto index = inst->Flags<MemoryFlags>().index;
			const auto kind  = index < plan.memory_info.size() ? plan.memory_info[index].kind
			                                                  : ResourceKind::None;
			if (!((op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
			      (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer))) {
				reason = "not-raw";
			} else if (!memory_free(*inst)) {
				reason = "address-from-memory";
			} else {
				// 2: only buffer store/atomic address operands use it; the renderer then
				// treats the dispatch's write ranges that read it as the whole binding.
				fill[slot] = host[slot] == 5u ? 2u : 1u;
				any        = true;
			}
		}
		if (diag && reason != nullptr && slot < 16) {
			reasons += fmt::format(" slot{}={}", slot, reason);
		}
	}
	if (diag) {
		std::printf("SrtGpuFill plan %016" PRIx64 ": %zu slots, %zu fillable;%s\n", plan.shader_hash,
		            count, count - static_cast<size_t>(std::count(fill.begin(), fill.end(), uint8_t {0})),
		            reasons.c_str());
	}
	if (!any) {
		return {};
	}
	return fill;
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;
	plan.has_address_writes         = program.has_address_writes;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		target.bindless_sampler    = source.bindless_sampler;
		if (target.indirect_descriptor.has_value()) {
			target.indirect_descriptor->key_count = Clone(target.indirect_descriptor->key_count);
			target.indirect_descriptor->selector_first =
			    Clone(target.indirect_descriptor->selector_first);
			target.indirect_descriptor->selector_mask =
			    Clone(target.indirect_descriptor->selector_mask);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	plan.control_flow = ResourceControlFlow(program);
	for (auto& block: plan.control_flow) {
		block.condition = Clone(block.condition);
	}
	plan.uniform_fill = AnalyzeUniformFill(program);
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		plan.uniform_fill.values[i] = Clone(plan.uniform_fill.values[i]);
	}
	plan.clean_flat_slots.resize(plan.srt_reads.size());
	for (const auto& image: plan.info.images) {
		const auto* source = Source(plan, image.source);
		if (source == nullptr || !source->indirect_descriptor.has_value()) {
			continue;
		}
		const auto& indirect = *source->indirect_descriptor;
		if (indirect.runtime) {
			continue; // KYTY_RUNTIME_IMAGES: the T# is the shader's own value
		}
		if (indirect.bindless) {
			// The host reads the heap's entries itself; the plan only needs the heap's V#.
			MarkCleanFlatSlots(plan, Source(plan, indirect.table_source), plan.clean_flat_slots);
			continue;
		}
		plan.requires_specialization_memory = true;
		MarkCleanFlatSlots(plan, indirect.selector ? Source(plan, indirect.selector->source) : nullptr,
		                   plan.clean_flat_slots, indirect.selector_mask);
		MarkCleanFlatSlots(plan, Source(plan, indirect.table_source), plan.clean_flat_slots);
		// The KYTY_SELECTOR_* plans' mask buffer, counts and first keys are read by the host too.
		if (indirect.selector && indirect.selector->ranks) {
			MarkCleanFlatSlots(plan, Source(plan, indirect.selector->ranks->source),
			                   plan.clean_flat_slots);
		}
		if (GetCodegenOptions().selector_keys != 0u) {
			MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.key_count);
			MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.selector_first);
		}
	}
	// KYTY_BINDLESS_SAMPLERS: the host reads the sampler heap itself; the plan needs its V#.
	for (const auto& sampler: plan.info.samplers) {
		const auto* source = Source(plan, sampler.source);
		if (source != nullptr && source->bindless_sampler.has_value()) {
			MarkCleanFlatSlots(plan, source, plan.clean_flat_slots);
		}
	}
	BuildSrtReadRuns(plan);
	BuildSrtEvaluationRecipes(plan);
	BuildSrtArithmeticTapes(plan);
	if (Common::RendererBatchEnabled() && !plan.control_flow.empty()) {
		const auto count = plan.control_flow.size();
		plan.flow_aliases.resize(count);
		plan.flow_initial_sources.assign(ActiveSlotCount(plan), 1u);
		std::vector<uint8_t> resolved(count, 0u);
		std::vector<uint32_t> path;
		for (const auto& block: plan.control_flow) {
			for (const auto source: block.sources) plan.flow_initial_sources.at(source) = 0u;
		}
		for (uint32_t start = 0; start < count; ++start) {
			if (resolved[start] == 2u) continue;
			path.clear();
			auto index = start;
			for (;;) {
				if (resolved[index] == 2u) { index = plan.flow_aliases[index]; break; }
				if (resolved[index] == 1u) break; // retain one anchor for an empty cycle
				const auto& block = plan.control_flow[index];
				if (!block.sources.empty() || !block.condition.IsEmpty() ||
				    block.successors.size() != 1u || block.successors.front() >= count) break;
				resolved[index] = 1u;
				path.push_back(index);
				index = block.successors.front();
			}
			plan.flow_aliases[index] = index;
			resolved[index] = 2u;
			for (const auto skipped: path) {
				plan.flow_aliases[skipped] = index;
				resolved[skipped] = 2u;
			}
		}
	}
	if (plan.stage == ShaderType::Compute) {
		plan.gpu_fill_slots = GpuFillSlots(program, plan);
	}
	// Last: the builders above may still assign memo slots. From here the plan is read-only.
	SealEvaluationIndices(plan);
	return plan;
}

template <bool Optimize>
static bool MaterializeResourcesImpl(const ResourcePlan& program, const SrtRuntime& runtime,
                                    EvaluationScratch& scratch, ResourceSnapshot& snapshot,
                                    ResourceSpecialization& specialization, bool& trace_used) {
	trace_used = false;
	if (!program.resource_tracking_complete ||
	    (program.requires_specialization_memory && runtime.read_specialization_memory == nullptr)) {
		return false;
	}
	const bool masked_image = std::ranges::any_of(program.info.images, [&](const auto& image) {
		const auto* source = Source(program, image.source);
		return source != nullptr && source->indirect_descriptor.has_value() &&
		       !source->indirect_descriptor->selector_mask.IsEmpty();
	});
	if (masked_image &&
	    (program.has_address_writes ||
	     std::ranges::any_of(program.info.images, &ImageResource::written))) {
		return false;
	}
	const bool capture_reads = masked_image &&
	    std::ranges::any_of(program.info.buffers, &BufferResource::written);
	const bool fuse_buffers = g_buffer_refresh_fusion.On() && !capture_reads;
	auto& reads = scratch.specialization_reads;
	ReadCapture capture {runtime, reads};
	SrtRuntime observed = runtime;
	if (capture_reads) {
		reads.clear();
		observed.userdata = &capture;
		observed.read_specialization_memory = CaptureStrictRead;
		if (observed.read_memory != nullptr) observed.read_memory = CaptureOrdinaryRead;
		if (runtime.try_read_clean_backing != nullptr) {
			observed.try_read_clean_backing = ProbeOrdinaryBacking;
		}
		if (runtime.try_read_specialization_backing != nullptr ||
		    runtime.try_read_clean_backing != nullptr) {
			observed.try_read_specialization_backing = ProbeStrictBacking;
		}
	}
	// KYTY_SRT_GPU_FILL: the slots this refresh leaves to the GPU go to the snapshot. Not with
	// captured reads (their certificate covers every read), and without trace replays (a replay
	// reads every slot on the host).
	snapshot.gpu_fills.clear();
	snapshot.bindless_heaps.clear();
	snapshot.bindless_sampler_heaps.clear();
	observed.gpu_fills = nullptr;
	const bool gpu_fill = runtime.gpu_written != nullptr && !capture_reads &&
	                      !program.gpu_fill_slots.empty();
	if (gpu_fill) {
		observed.gpu_fills = &snapshot.gpu_fills;
	}
	SrtWalker clean(program, scratch, CleanRuntime(observed));
	SrtWalker walker(program, scratch, observed, program.clean_flat_slots, &clean);
	// KYTY_SRT_TRACE (SrtWalker.h): replays of this plan's earlier refreshes on this thread.
	SrtTraceSession trace(program, clean, walker, observed, !capture_reads && !gpu_fill);
	std::span<const uint8_t> active;
	namespace Phases = Common::PrepPhases; // KYTY_DRAW_PREP_PHASES sub-buckets
	auto phase_start = Phases::SubStart();
	{
		KYTY_PROFILER_DETAIL_BLOCK("Resources::SRT refresh");
		active = clean.FindActiveSources();
		if (!walker.RefreshFlatBuffer(snapshot.flattened_srt)) {
			return false;
		}
		snapshot.uniform_fill = {};
		const auto& fill = program.uniform_fill;
		const auto words = fill.fill.words;
		std::array<uint32_t, 4> stored {};
		bool uniform_fill = words != 0;
		for (uint32_t i = 0; i < words && uniform_fill; ++i) {
			uniform_fill = clean.Evaluate(fill.values[i], stored[i]) && stored[i] == stored[0];
		}
		if (uniform_fill) {
			snapshot.uniform_fill = fill.fill;
			snapshot.uniform_fill.value = stored[0];
		}
	}
	// KYTY_BINDLESS_SAMPLERS: two words per bindless sampler (region base, record count) right
	// after the SRT slots, before any image's words (BindlessSamplerMappingOffset); zeros select
	// the stand-in sampler.
	if (const auto bindless_samplers = BindlessSamplerCount(program.info); bindless_samplers != 0u) {
		if (snapshot.flattened_srt.size() != program.srt_reads.size()) {
			return false;
		}
		snapshot.flattened_srt.resize(snapshot.flattened_srt.size() + 2u * bindless_samplers, 0u);
	}
	const auto evaluate = [&](uint32_t source, DescriptorValue& value) {
		if (source >= program.descriptor_sources.size()) {
			return false;
		}
		if (active.empty() || active[source]) {
			return walker.EvaluateDescriptor(source, value);
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	Phases::SubAdd(Phases::Sub::SrtRefresh, phase_start);
	phase_start = Phases::SubStart();
	{
		KYTY_PROFILER_DETAIL_BLOCK("Resources::Buffers");
		snapshot.buffers.resize(program.info.buffers.size());
		if (fuse_buffers) {
			specialization.buffers.clear();
			specialization.buffers.reserve(program.info.buffers.size());
		}
		for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
			if (!evaluate(program.info.buffers[i].source, snapshot.buffers[i])) {
				return false;
			}
			if (fuse_buffers && !SpecializeBuffer(program, program.info.buffers[i], snapshot.buffers[i], specialization, i)) return false;
		}
		if (capture_reads) {
			for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
				const auto& buffer = program.info.buffers[i];
				if (!buffer.written || (!active.empty() && !active[buffer.source])) continue;
				DescriptorValue strict;
				if (!clean.EvaluateDescriptor(buffer.source, strict)) return false;
				if (fuse_buffers) {
					// SpecializeBuffer nulled the snapshot's copy the same way.
					ShaderBufferResource decoded;
					if (DecodeBufferDescriptor(strict, decoded) && NullBufferDescriptor(decoded, false)) {
						strict.dwords.fill(0);
					}
				}
				if (strict != snapshot.buffers[i]) return false;
			}
		}
	}
	Phases::SubAdd(Phases::Sub::Buffers, phase_start);
	phase_start = Phases::SubStart();
	{
		KYTY_PROFILER_DETAIL_BLOCK("Resources::Images");
		snapshot.images.resize(program.info.images.size());
		const auto initial_image = [](const ImageResource& image) {
			return ResourceSpecialization::Image {
			    .numeric_class = image.numeric_class,
			    .dimension = image.dimension,
			    .mip_count = image.mip_count,
			    .conversion_format = image.conversion_format,
			    .shader_swizzle = image.shader_swizzle,
			    .indirect_root = image.indirect_root,
			    .indirect_mapping_offset = image.indirect_mapping_offset,
			    .indirect_search_iterations = image.indirect_search_iterations,
			    .cube = image.cube,
			    .bindless = image.bindless,
			    .runtime = image.runtime,
			};
		};
		if constexpr (Optimize) {
			specialization.images.resize(program.info.images.size());
		} else {
			specialization.images.clear();
			specialization.images.reserve(program.info.images.size());
			for (const auto& image: program.info.images) {
				specialization.images.push_back(initial_image(image));
			}
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& image = program.info.images[i];
			if constexpr (Optimize) specialization.images[i] = initial_image(image);
			const auto* source = Source(program, image.source);
			if (source == nullptr) {
				return false;
			}
			// KYTY_SRT_IMAGE_REACH: an image only unreached blocks use reads zero.
			const bool image_reached =
			    active.empty() || active[ImageActiveSlot(program.descriptor_sources.size(), i)];
			if (source->indirect_descriptor.has_value()) {
				snapshot.images[i] = {.dword_count = 8u};
				if (!active.empty() && (!active[image.source] || !image_reached)) {
					continue;
				}
				const auto& indirect = *source->indirect_descriptor;
				DescriptorValue material;
				DescriptorValue table;
				if (indirect.runtime) {
					// KYTY_RUNTIME_IMAGES: bound to set 1 like a bindless image, through the runtime
					// image heap of its binding (nothing to patch); set 0 sees a null descriptor.
					auto& root         = specialization.images[i];
					root.bindless      = true;
					root.runtime       = true;
					root.indirect_root = ImageResource::NoIndirectImage;
					snapshot.bindless_heaps.push_back({.image = i, .runtime = true});
					continue;
				}
				if (indirect.bindless) {
					if (!clean.EvaluateDescriptor(indirect.table_source, table) ||
					    !MaterializeBindlessImage(indirect, table, i, snapshot, specialization)) {
						return false;
					}
					continue;
				}
				if ((indirect.selector &&
				     !clean.EvaluateDescriptor(indirect.selector->source, material)) ||
				    !clean.EvaluateDescriptor(indirect.table_source, table) ||
				    !MaterializeIndirectImage<Optimize>(program, indirect, material, table, i, observed,
				                                         clean, scratch, snapshot, specialization)) {
					return false;
				}
			} else {
				if (!evaluate(image.source, snapshot.images[i])) {
					return false;
				}
				if (!image_reached) {
					snapshot.images[i].dwords.fill(0);
				} else if (!ValidImageDescriptor(snapshot.images[i], image.r128)) {
					if (!NullImageDescriptor(snapshot.images[i])) {
						ReportInvalidImageDescriptor(program.shader_hash, i, snapshot.images[i]);
						NullTraceNote(NullTraceKind::NullImage, program.shader_hash);
					}
					snapshot.images[i].dwords.fill(0);
				}
			}
		}
	}
	Phases::SubAdd(Phases::Sub::Images, phase_start);
	phase_start = Phases::SubStart();
	{
		KYTY_PROFILER_DETAIL_BLOCK("Resources::Specialization");
		snapshot.samplers.resize(program.info.samplers.size());
		snapshot.bindless_sampler_heaps.clear();
		uint32_t bindless_sampler = 0;
		for (uint32_t i = 0; i < program.info.samplers.size(); ++i) {
			const auto  source_index = program.info.samplers[i].source;
			const auto* source       = Source(program, source_index);
			if (source != nullptr && source->bindless_sampler.has_value()) {
				// KYTY_BINDLESS_SAMPLERS: the shader indexes the host's mirror of the heap; set 0
				// binds the stand-in sampler for this resource.
				snapshot.samplers[i].dword_count = 4u;
				std::copy_n(BindlessDefaultSampler, 4u, snapshot.samplers[i].dwords.begin());
				const auto mapping_offset =
				    BindlessSamplerMappingOffset(program.srt_reads.size(), bindless_sampler++);
				DescriptorValue      table;
				ShaderBufferResource heap;
				if ((active.empty() || active[source_index]) &&
				    clean.EvaluateDescriptor(source_index, table) &&
				    DecodeBufferDescriptor(table, heap) && heap.Base48() != 0) {
					snapshot.bindless_sampler_heaps.push_back(
					    {.base           = heap.Base48(),
					     .size           = heap.GetSize(),
					     .table_offset   = source->bindless_sampler->table_offset,
					     .stride         = source->bindless_sampler->stride,
					     .sampler        = i,
					     .mapping_offset = mapping_offset});
				}
				continue;
			}
			if (!evaluate(source_index, snapshot.samplers[i])) {
				return false;
			}
			if (program.info.samplers[i].gather_lod) {
				const auto control = snapshot.samplers[i].dwords[2];
				const auto filter  = (control >> 26u) & 3u;
				// MipNone always selects the base level and Point the nearest mip. Linear mip
				// selection and nonzero LOD biases also use the nearest mip: upstream rejects the
				// stage here, which drops draws that earlier builds rendered from mip 0.
				if (filter > 1u || (filter == 1u && (control & 0xfffffu) != 0u)) {
					static std::atomic_flag warned = ATOMIC_FLAG_INIT;
					if (!warned.test_and_set(std::memory_order_relaxed)) {
						std::fputs("Warning: explicit-LOD gather with linear mip filtering or a LOD bias "
						           "reads the nearest mip.\n",
						           stderr);
					}
				}
			}
		}
		if (capture_reads && !WrittenBuffersDisjoint(program, snapshot, reads)) return false;
		snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
		const bool built = BuildResourceSpecialization(program, snapshot, specialization, fuse_buffers);
		Phases::SubAdd(Phases::Sub::Samplers, phase_start);
		if (built) {
			trace.Succeeded();
			trace_used = trace.UsedTrace();
		}
		return built;
	}
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          EvaluationScratch& scratch, ResourceSnapshot& snapshot,
                          ResourceSpecialization& specialization) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::ResourceMaterialization);
	NullTraceSetShader(program.shader_hash);
	static const bool optimized = [] {
		const auto* setting = std::getenv("KYTY_RESOURCE_MATERIALIZATION");
		return setting != nullptr && std::strcmp(setting, "optimized") == 0;
	}();
	bool       trace_used = false;
	const bool ok =
	    optimized
	        ? MaterializeResourcesImpl<true>(program, runtime, scratch, snapshot, specialization, trace_used)
	        : MaterializeResourcesImpl<false>(program, runtime, scratch, snapshot, specialization, trace_used);
	if (ok && trace_used && SrtTraceSession::Mode() >= 2) [[unlikely]] {
		// KYTY_SRT_TRACE=verify: the refresh again without traces, compared field by field.
		ResourceSnapshot       reference;
		ResourceSpecialization reference_specialization;
		bool                   unused     = false;
		auto&                  suppressed = SrtTraceSession::Suppressed();
		suppressed                        = true;
		const bool reference_ok =
		    optimized ? MaterializeResourcesImpl<true>(program, runtime, scratch, reference,
		                                               reference_specialization, unused)
		              : MaterializeResourcesImpl<false>(program, runtime, scratch, reference,
		                                                reference_specialization, unused);
		suppressed = false;
		const char* field = nullptr;
		if (!reference_ok) field = "reference refresh failed";
		else if (snapshot.flattened_srt != reference.flattened_srt) field = "flattened_srt";
		else if (snapshot.buffers != reference.buffers) field = "buffers";
		else if (snapshot.images != reference.images) field = "images";
		else if (snapshot.samplers != reference.samplers) field = "samplers";
		else if (snapshot.user_data != reference.user_data) field = "user_data";
		else if (!(snapshot.uniform_fill == reference.uniform_fill)) field = "uniform_fill";
		else if (!(specialization == reference_specialization)) field = "specialization";
		SrtTraceSession::NoteVerified(program, field);
	}
	return ok;
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	return MaterializeResources(program, runtime, ThreadEvaluationScratch(), snapshot,
	                            specialization);
}

const char* UnsupportedResourceSpecialization(const Program&                program,
                                              const ResourceSpecialization& specialization) {
	// ApplyResourceSpecialization's check, before it changes anything (adapted from
	// usa-ratt/KytyPS5-GT7 c3d3e76e8 / b70de4ef9): enumerated candidates only sample, or are
	// stored to (KYTY_WORKGROUP_IMAGES). A load or query through an enumerated material table
	// ended the emulator in GT7's cafe; only that specialization is skipped instead.
	// The exemptions are the ones ApplyResourceSpecialization accepts: storage reads, and with
	// KYTY_SAMPLED_TABLE_READS loads of the candidate the key selects (Ghost of Yotei's Update Oct
	// Maps CS e8e383355f93ab53: skipping it left its partner 0b0ec2ba writing alone, magenta light).
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (ImageOpcodeInfoOf(op).access == ImageAccess::None || op == ValueOpcode::ImageSampleRaw ||
			    op == ValueOpcode::ImageWrite || op == ValueOpcode::ImageReadStorage ||
			    (op == ValueOpcode::ImageRead && GetCodegenOptions().sampled_table_reads)) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				continue;
			}
			const auto resource = program.memory_info[index].resource;
			if (resource >= specialization.images.size()) {
				continue;
			}
			const auto& image = specialization.images[resource];
			if (!image.fmask && !image.bindless && image.indirect_root == resource) {
				return "an image of an enumerated material table is read other than by a sample";
			}
		}
	}
	return nullptr;
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete ||
	        program.binding_layout_complete);
	EXIT_IF(program.info.buffers.size() != specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size());

	auto buffers = program.info.buffers;
	for (size_t index = 0; index < buffers.size(); index++) {
		buffers[index].packed_stride      = specialization.buffers[index].packed_stride;
		buffers[index].descriptor_format  = specialization.buffers[index].descriptor_format;
		buffers[index].descriptor_swizzle = specialization.buffers[index].descriptor_swizzle;
	}
	auto images = program.info.images;
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= program.info.images.size());
			images.push_back(program.info.images[source.indirect_root]);
		}
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.cube                       = source.cube;
		image.bindless                   = source.bindless;
		image.runtime                    = source.runtime;
		image.clamp_half_float           = source.clamp_half_float;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, images, sampler_plan));
	auto samplers      = program.info.samplers;
	auto sampled_pairs = program.info.sampled_pairs;
	// KYTY_BINDLESS_SAMPLERS: where MaterializeResources put each bindless sampler's two words. A
	// class copy below shares its source's words (and so its mirror of the heap).
	for (uint32_t index = 0, bindless = 0; index < samplers.size(); index++) {
		if (samplers[index].bindless) {
			samplers[index].bindless_mapping_offset =
			    BindlessSamplerMappingOffset(program.srt_reads.size(), bindless++);
		}
	}
	samplers.reserve(sampler_plan.sampler_count);
	for (uint32_t index = 0; index < sampler_plan.sampler_count; index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= program.info.samplers.size()) {
			const auto copy = samplers[binding.source];
			samplers.push_back(copy);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::PointInteger;
		samplers[index].integer_border        = binding.type != SamplerClass::Float &&
		                                        binding.type != SamplerClass::FloatCompare;
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
	}

	auto memory_info = program.memory_info;
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				if (memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob) {
					// Bounds mode 0 checks offset >= stride, so zero stride
					// makes every vector read out of bounds regardless of its address.
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						const bool d16 = memory.data_bits == 16u;
						const auto components = d16 ? memory.component_count : count;
						std::array<uint32_t, 4> constants {};
						for (uint32_t component = 0; component < components; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							if (d16 && one == 0x3f800000u) one = 0x3c00u;
							constants[d16 ? component / 2u : component] |=
							    one << (d16 ? (component % 2u) * 16u : 0u);
						}
						for (uint32_t component = 0; component < count; component++) {
							if (constants[component] == 0u) continue;
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(constants[component]), Value(0u)}));
						}
					}
					Value result = values[0];
					switch (inst.GetType()) {
						case Type::U8: result = Value(uint8_t {0}); break;
						case Type::U16: result = Value(uint16_t {0}); break;
						case Type::U32x2:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x2,
							                                      {values[0], values[1]}));
							break;
						case Type::U32x3:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x3,
							                                      {values[0], values[1], values[2]}));
							break;
						case Type::U32x4:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
							                                      {values[0], values[1], values[2], values[3]}));
							break;
						default: break;
					}
					inst.ReplaceUsesWith(result);
				}
				continue;
			}
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF((inst.GetOpcode() != ValueOpcode::ImageRead &&
				         inst.GetOpcode() != ValueOpcode::ImageReadStorage) ||
				        memory.data_bits != 32u);
				// The host image holds one sample per guest fragment. FMASK's four-bit fragment
				// indices map coverage sample i to host sample i, or with EQAA (fewer fragments
				// than samples, KYTY_EQAA_MIXED) to i mod fragments, so every index names a
				// sample that exists.
				const uint32_t fragments_per_pixel =
				    specialization.images[memory.resource].fmask_fragments;
				const uint32_t indices[] = {FmaskSampleMapWord(fragments_per_pixel, 0),
				                            FmaskSampleMapWord(fragments_per_pixel, 8)};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result = block->PrependNewInst(
				    it, ValueOpcode::CompositeConstructU32x4,
				    {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler &&
			    memory.sampler < program.info.samplers.size()) {
				const auto type = static_cast<uint32_t>(ClassifySampler(image));
				memory.sampler = sampler_plan.mapping[memory.sampler][type];
				EXIT_IF(memory.sampler == UINT32_MAX);
			}
			// Enumerated candidates only sample, or are stored to (KYTY_WORKGROUP_IMAGES); a
			// bindless image is loaded per instruction and may take any sampled operation.
			// KYTY_SAMPLED_TABLE_READS: loads read the candidate the key selects too.
			EXIT_IF(image.indirect_root == memory.resource && !image.bindless &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageWrite &&
			        inst.GetOpcode() != ValueOpcode::ImageReadStorage &&
			        !(inst.GetOpcode() == ValueOpcode::ImageRead &&
			          GetCodegenOptions().sampled_table_reads));
		}
	}
	// Dead-code elimination can remove an image operation (an unused IMAGE_GET_LOD result) before
	// resource tracking, which leaves its memory entry behind with the frontend's descriptor
	// register as the resource: no image was tracked for it, so it is not remapped.
	std::vector<bool> memory_used(memory_info.size(), false);
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
			} else if (ImageOpcodeInfoOf(inst.GetOpcode()).access != ImageAccess::None) {
				const auto index = inst.Flags<MemoryFlags>().index;
				EXIT_IF(index >= memory_info.size());
				memory_used[index] = true;
			}
		}
	}
	for (size_t index = 0; index < memory_info.size(); index++) {
		auto& memory = memory_info[index];
		if (memory.kind == ResourceKind::Image && !memory.planning_only && memory_used[index]) {
			memory.resource = image_remap[memory.resource];
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap[buffer.image_alias];
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap[pair.image];
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap[image.indirect_root];
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap[resource];
		}
	}
	image_remap.Apply(images);
	program.info.buffers       = std::move(buffers);
	program.info.images        = std::move(images);
	program.info.samplers      = std::move(samplers);
	program.info.sampled_pairs = std::move(sampled_pairs);
	program.memory_info        = std::move(memory_info);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
