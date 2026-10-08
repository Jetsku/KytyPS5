#include "graphics/shader/recompiler/ir/ProgramCodec.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <utility>

// See ProgramCodec.h. Field order follows the declarations; bump nothing by hand: the persistent
// program cache keys its files by a hash of the recompiler sources (this file included), so any
// change here, like any other codegen change, starts a new cache file.

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

#if defined(_MSC_VER) && defined(_WIN64) && defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL == 0
// Update the encoders and decoders below, then these sizes, when one of these types changes.
// Members: Inst 7; Value 2 (type and one union member); MemoryInfo 25; BufferResource 12;
// ImageResource 20; SamplerResource 7; SampledResourcePair 3; StageInput 5; StageOutput 4;
// ShaderInfo 10; DescriptorBinding 2; BindingLayout 6; WriteRangeNode 6; WriteRangeAccess 5;
// BufferWriteRange 4; WriteRangeProgram 2; CompiledShaderInfo 11; DescriptorSource 3 (IndirectImage
// 7); SrtRead 2; ResourceBlock 3; EvaluationOperand 3; EvaluationRecipe 6; ArithmeticTapeOperand 2;
// ArithmeticTapeInstruction 5; ArithmeticTape 2; UniformFill 5; UniformFillPlan 2; ResourcePlan 28;
// ResourceSpecialization 2 (Buffer 3, Image 10).
static_assert(sizeof(Inst) == 104, "IR::Inst changed: update ProgramCodec");
static_assert(sizeof(Value) == 16, "IR::Value changed: update ProgramCodec");
static_assert(sizeof(MemoryInfo) == 72, "IR::MemoryInfo changed: update ProgramCodec");
static_assert(sizeof(BufferResource) == 36, "IR::BufferResource changed: update ProgramCodec");
static_assert(sizeof(ImageResource) == 80, "IR::ImageResource changed: update ProgramCodec");
static_assert(sizeof(SamplerResource) == 16, "IR::SamplerResource changed: update ProgramCodec");
static_assert(sizeof(SampledResourcePair) == 12, "IR::SampledResourcePair changed: update ProgramCodec");
static_assert(sizeof(StageInput) == 56, "IR::StageInput changed: update ProgramCodec");
static_assert(sizeof(StageOutput) == 48, "IR::StageOutput changed: update ProgramCodec");
static_assert(sizeof(ShaderInfo) == 192, "IR::ShaderInfo changed: update ProgramCodec");
static_assert(sizeof(DescriptorBinding) == 32, "IR::DescriptorBinding changed: update ProgramCodec");
static_assert(sizeof(BindingLayout) == 64, "IR::BindingLayout changed: update ProgramCodec");
static_assert(sizeof(WriteRangeNode) == 24, "IR::WriteRangeNode changed: update ProgramCodec");
static_assert(sizeof(WriteRangeAccess) == 20, "IR::WriteRangeAccess changed: update ProgramCodec");
static_assert(sizeof(BufferWriteRange) == 40, "IR::BufferWriteRange changed: update ProgramCodec");
static_assert(sizeof(WriteRangeProgram) == 48, "IR::WriteRangeProgram changed: update ProgramCodec");
static_assert(sizeof(CompiledShaderInfo) == 344, "IR::CompiledShaderInfo changed: update ProgramCodec");
static_assert(sizeof(DescriptorSource) == 200, "IR::DescriptorSource changed: update ProgramCodec");
static_assert(sizeof(DescriptorSource::IndirectImage) == 56,
              "IR::DescriptorSource::IndirectImage changed: update ProgramCodec");
static_assert(sizeof(SrtRead) == 24, "IR::SrtRead changed: update ProgramCodec");
static_assert(sizeof(ResourceBlock) == 64, "IR::ResourceBlock changed: update ProgramCodec");
static_assert(sizeof(ResourcePlan::EvaluationOperand) == 16,
              "IR::ResourcePlan::EvaluationOperand changed: update ProgramCodec");
static_assert(sizeof(ResourcePlan::EvaluationRecipe) == 120,
              "IR::ResourcePlan::EvaluationRecipe changed: update ProgramCodec");
static_assert(sizeof(ResourcePlan::ArithmeticTapeOperand) == 16,
              "IR::ResourcePlan::ArithmeticTapeOperand changed: update ProgramCodec");
static_assert(sizeof(ResourcePlan::ArithmeticTapeInstruction) == 80,
              "IR::ResourcePlan::ArithmeticTapeInstruction changed: update ProgramCodec");
static_assert(sizeof(ResourcePlan::ArithmeticTape) == 8,
              "IR::ResourcePlan::ArithmeticTape changed: update ProgramCodec");
static_assert(sizeof(UniformFill) == 28, "IR::UniformFill changed: update ProgramCodec");
static_assert(sizeof(UniformFillPlan) == 96, "IR::UniformFillPlan changed: update ProgramCodec");
static_assert(sizeof(ResourcePlan) == 704, "IR::ResourcePlan changed: update ProgramCodec");
static_assert(sizeof(ResourceSpecialization) == 48,
              "IR::ResourceSpecialization changed: update ProgramCodec");
static_assert(sizeof(ResourceSpecialization::Buffer) == 16,
              "IR::ResourceSpecialization::Buffer changed: update ProgramCodec");
static_assert(sizeof(ResourceSpecialization::Image) == 36,
              "IR::ResourceSpecialization::Image changed: update ProgramCodec");
#endif

constexpr uint32_t NullIndex = UINT32_MAX;

template <typename Enum>
uint32_t EnumBits(Enum value) {
	return static_cast<uint32_t>(value);
}

template <typename Enum>
Enum EnumFrom(uint32_t bits) {
	return static_cast<Enum>(bits);
}

} // namespace

// Private members of Value and Inst (friend of both).
struct ProgramCodecAccess {
	static Type  RawType(const Value& value) { return value.type; }
	static Inst* RawInst(const Value& value) { return value.inst; }

	// The active union member of an immediate, zero-extended; false for types a plan cannot hold.
	static bool ImmediateBits(const Value& value, uint64_t& bits) {
		switch (value.type) {
			case Type::ScalarReg: bits = static_cast<uint16_t>(value.scalar_reg); return true;
			case Type::VectorReg: bits = static_cast<uint16_t>(value.vector_reg); return true;
			case Type::U1: bits = value.imm_u1 ? 1u : 0u; return true;
			case Type::U8: bits = value.imm_u8; return true;
			case Type::U16:
			case Type::F16: bits = value.imm_u16; return true;
			case Type::U32:
			case Type::F32: bits = value.imm_u32; return true;
			case Type::U64: bits = value.imm_u64; return true;
			default: return false;
		}
	}

	static bool MakeImmediate(Type type, uint64_t bits, Value& value) {
		switch (type) {
			case Type::ScalarReg:
				if (bits > 0xffffu) return false;
				value = Value(static_cast<ScalarReg>(bits));
				return true;
			case Type::VectorReg:
				if (bits > 0xffffu) return false;
				value = Value(static_cast<VectorReg>(bits));
				return true;
			case Type::U1:
				if (bits > 1u) return false;
				value = Value(bits != 0u);
				return true;
			case Type::U8:
				if (bits > 0xffu) return false;
				value = Value(static_cast<uint8_t>(bits));
				return true;
			case Type::U16:
				if (bits > 0xffffu) return false;
				value = Value(static_cast<uint16_t>(bits));
				return true;
			case Type::F16:
				if (bits > 0xffffu) return false;
				value = Value::F16(static_cast<uint16_t>(bits));
				return true;
			case Type::U32:
				if (bits > 0xffffffffu) return false;
				value = Value(static_cast<uint32_t>(bits));
				return true;
			case Type::F32:
				if (bits > 0xffffffffu) return false;
				// The raw bits, never through a float (NaN payloads stay as they are).
				value = Value(Type::F32, bits);
				return true;
			case Type::U64: value = Value(bits); return true;
			default: return false;
		}
	}

	static ValueOpcode                Opcode(const Inst& inst) { return inst.opcode; }
	static uint64_t                   Flags(const Inst& inst) { return inst.flags; }
	static const Block*               Parent(const Inst& inst) { return inst.parent; }
	static const std::vector<Value>&  Args(const Inst& inst) { return inst.args; }
	static const std::vector<Block*>& PhiBlocks(const Inst& inst) { return inst.phi_blocks; }
	static const std::vector<Use>&    Uses(const Inst& inst) { return inst.uses; }
	static uint32_t                   EvaluationIndex(const Inst& inst) {
		return inst.evaluation_index;
	}

	static void Assign(Inst& inst, std::vector<Value> args, size_t phi_blocks,
	                   std::vector<Use> uses, uint32_t evaluation_index) {
		inst.parent           = nullptr;
		inst.args             = std::move(args);
		inst.phi_blocks.assign(phi_blocks, nullptr);
		inst.uses             = std::move(uses);
		inst.evaluation_index = evaluation_index;
	}

	// Drops every link without touching the instructions it names (a rejected decode).
	static void Detach(Inst& inst) {
		inst.args.clear();
		inst.phi_blocks.clear();
		inst.uses.clear();
	}
};

// ---------------------------------------------------------------------------------------------
// Writer and reader.

void CodecWriter::Raw(const void* data, size_t size) {
	const auto* bytes = static_cast<const uint8_t*>(data);
	m_out.insert(m_out.end(), bytes, bytes + size);
}

void CodecWriter::String(std::string_view text) {
	U32(static_cast<uint32_t>(text.size()));
	Raw(text.data(), text.size());
}

void CodecWriter::Words(std::span<const uint32_t> words) {
	U32(static_cast<uint32_t>(words.size()));
	Raw(words.data(), words.size_bytes());
}

bool CodecReader::Read(void* data, size_t size) {
	if (m_failed || size > m_input.size() - m_position) {
		m_failed = true;
		std::memset(data, 0, size);
		return false;
	}
	std::memcpy(data, m_input.data() + m_position, size);
	m_position += size;
	return true;
}

uint8_t CodecReader::U8() {
	uint8_t value = 0;
	Read(&value, sizeof(value));
	return value;
}

uint16_t CodecReader::U16() {
	uint16_t value = 0;
	Read(&value, sizeof(value));
	return value;
}

uint32_t CodecReader::U32() {
	uint32_t value = 0;
	Read(&value, sizeof(value));
	return value;
}

uint64_t CodecReader::U64() {
	uint64_t value = 0;
	Read(&value, sizeof(value));
	return value;
}

int32_t CodecReader::I32() {
	int32_t value = 0;
	Read(&value, sizeof(value));
	return value;
}

int64_t CodecReader::I64() {
	int64_t value = 0;
	Read(&value, sizeof(value));
	return value;
}

bool CodecReader::Bool() {
	const auto value = U8();
	if (value > 1u) {
		m_failed = true;
		return false;
	}
	return value != 0u;
}

uint32_t CodecReader::Count(size_t min_bytes) {
	const auto count = U32();
	if (m_failed || (min_bytes != 0 && count > Remaining() / min_bytes)) {
		m_failed = true;
		return 0;
	}
	return count;
}

std::span<const uint8_t> CodecReader::Take(size_t size) {
	if (m_failed || size > m_input.size() - m_position) {
		m_failed = true;
		return {};
	}
	const auto result = m_input.subspan(m_position, size);
	m_position += size;
	return result;
}

std::string CodecReader::String() {
	const auto size  = Count(1);
	const auto bytes = Take(size);
	return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::vector<uint32_t> CodecReader::Words() {
	const auto count = Count(sizeof(uint32_t));
	std::vector<uint32_t> words(count);
	if (count != 0) {
		const auto bytes = Take(count * sizeof(uint32_t));
		if (!bytes.empty()) std::memcpy(words.data(), bytes.data(), bytes.size());
	}
	return words;
}

namespace {

// ---------------------------------------------------------------------------------------------
// Plain data shared by plans and compiled metadata.

void Write(CodecWriter& w, const MemoryInfo& v) {
	w.U32(EnumBits(v.kind));
	w.U32(v.resource);
	w.U32(v.sampler);
	w.U32(v.offset);
	w.U32(v.secondary_offset);
	w.U32(v.dmask);
	w.U32(v.data_dwords);
	w.U32(v.data_bits);
	w.U32(v.component_index);
	w.U32(v.component_count);
	w.U32(v.data_format);
	w.U32(v.number_format);
	w.U32(v.image_sample_flags);
	w.U32(EnumBits(v.image_dimension));
	w.U32(v.image_address_components);
	w.Bool(v.address_is_full);
	w.Bool(v.data_signed);
	w.Bool(v.typed);
	w.Bool(v.formatted);
	w.Bool(v.image_has_mip);
	w.Bool(v.image_r128);
	w.Bool(v.idxen);
	w.Bool(v.offen);
	w.Bool(v.coherent);
	w.Bool(v.planning_only);
}

void Read(CodecReader& r, MemoryInfo& v) {
	v.kind                     = EnumFrom<ResourceKind>(r.U32());
	v.resource                 = r.U32();
	v.sampler                  = r.U32();
	v.offset                   = r.U32();
	v.secondary_offset         = r.U32();
	v.dmask                    = r.U32();
	v.data_dwords              = r.U32();
	v.data_bits                = r.U32();
	v.component_index          = r.U32();
	v.component_count          = r.U32();
	v.data_format              = r.U32();
	v.number_format            = r.U32();
	v.image_sample_flags       = r.U32();
	v.image_dimension          = EnumFrom<Decoder::ImageDimension>(r.U32());
	v.image_address_components = r.U32();
	v.address_is_full          = r.Bool();
	v.data_signed              = r.Bool();
	v.typed                    = r.Bool();
	v.formatted                = r.Bool();
	v.image_has_mip            = r.Bool();
	v.image_r128               = r.Bool();
	v.idxen                    = r.Bool();
	v.offen                    = r.Bool();
	v.coherent                 = r.Bool();
	v.planning_only            = r.Bool();
}

void Write(CodecWriter& w, const BufferResource& v) {
	w.U32(v.source);
	w.U32(v.first_use_pc);
	w.U32(v.max_byte_extent);
	w.U32(v.packed_stride);
	w.U32(EnumBits(v.descriptor_format));
	w.U32(v.descriptor_swizzle);
	w.U32(v.image_alias);
	w.Bool(v.read);
	w.Bool(v.written);
	w.Bool(v.atomic);
	w.Bool(v.formatted);
	w.Bool(v.scalar);
}

void Read(CodecReader& r, BufferResource& v) {
	v.source             = r.U32();
	v.first_use_pc       = r.U32();
	v.max_byte_extent    = r.U32();
	v.packed_stride      = r.U32();
	v.descriptor_format  = EnumFrom<Prospero::BufferFormat>(r.U32());
	v.descriptor_swizzle = r.U32();
	v.image_alias        = r.U32();
	v.read               = r.Bool();
	v.written            = r.Bool();
	v.atomic             = r.Bool();
	v.formatted          = r.Bool();
	v.scalar             = r.Bool();
}

void Write(CodecWriter& w, const ImageResource& v) {
	w.U32(v.source);
	w.U32(v.first_use_pc);
	w.U32(EnumBits(v.resource_class));
	w.U32(EnumBits(v.numeric_class));
	w.U32(EnumBits(v.dimension));
	w.U32(EnumBits(v.mip_mode));
	w.U32(v.mip_count);
	w.U32(EnumBits(v.conversion_format));
	w.U32(v.shader_swizzle);
	w.Bool(v.read);
	w.Bool(v.written);
	w.Bool(v.atomic);
	w.Bool(v.atomic64);
	w.Bool(v.depth_compare);
	w.Bool(v.cube);
	w.Bool(v.r128);
	w.U32(v.indirect_root);
	w.U32(v.indirect_mapping_offset);
	w.U32(v.indirect_search_iterations);
	w.Words(v.indirect_resources);
}

void Read(CodecReader& r, ImageResource& v) {
	v.source                     = r.U32();
	v.first_use_pc               = r.U32();
	v.resource_class             = EnumFrom<ImageResourceClass>(r.U32());
	v.numeric_class              = EnumFrom<Prospero::TextureNumericClass>(r.U32());
	v.dimension                  = EnumFrom<Decoder::ImageDimension>(r.U32());
	v.mip_mode                   = EnumFrom<ImageMipMode>(r.U32());
	v.mip_count                  = r.U32();
	v.conversion_format          = EnumFrom<Prospero::BufferFormat>(r.U32());
	v.shader_swizzle             = r.U32();
	v.read                       = r.Bool();
	v.written                    = r.Bool();
	v.atomic                     = r.Bool();
	v.atomic64                   = r.Bool();
	v.depth_compare              = r.Bool();
	v.cube                       = r.Bool();
	v.r128                       = r.Bool();
	v.indirect_root              = r.U32();
	v.indirect_mapping_offset    = r.U32();
	v.indirect_search_iterations = r.U32();
	v.indirect_resources         = r.Words();
}

void Write(CodecWriter& w, const SamplerResource& v) {
	w.U32(v.source);
	w.U32(v.first_use_pc);
	w.U32(v.snapshot_index);
	w.Bool(v.force_point_filtering);
	w.Bool(v.depth_compare);
	w.Bool(v.integer_border);
	w.Bool(v.gather_lod);
}

void Read(CodecReader& r, SamplerResource& v) {
	v.source                = r.U32();
	v.first_use_pc          = r.U32();
	v.snapshot_index        = r.U32();
	v.force_point_filtering = r.Bool();
	v.depth_compare         = r.Bool();
	v.integer_border        = r.Bool();
	v.gather_lod            = r.Bool();
}

void Write(CodecWriter& w, const SampledResourcePair& v) {
	w.U32(v.image);
	w.U32(v.sampler);
	w.U32(v.first_use_pc);
}

void Read(CodecReader& r, SampledResourcePair& v) {
	v.image        = r.U32();
	v.sampler      = r.U32();
	v.first_use_pc = r.U32();
}

void Write(CodecWriter& w, const StageInput& v) {
	w.U32(EnumBits(v.kind));
	w.U32(v.location);
	w.U32(v.component_count);
	w.String(v.debug_name);
	w.Bool(v.per_vertex);
}

void Read(CodecReader& r, StageInput& v) {
	v.kind            = EnumFrom<StageInputKind>(r.U32());
	v.location        = r.U32();
	v.component_count = r.U32();
	v.debug_name      = r.String();
	v.per_vertex      = r.Bool();
}

void Write(CodecWriter& w, const StageOutput& v) {
	w.U32(EnumBits(v.kind));
	w.U32(v.index);
	w.U32(v.location);
	w.String(v.debug_name);
}

void Read(CodecReader& r, StageOutput& v) {
	v.kind       = EnumFrom<StageOutputKind>(r.U32());
	v.index      = r.U32();
	v.location   = r.U32();
	v.debug_name = r.String();
}

void Write(CodecWriter& w, const WriteRangeNode& v) {
	w.U8(static_cast<uint8_t>(v.op));
	w.U32(v.a);
	w.U32(v.b);
	w.U32(v.c);
	w.U32(v.lo);
	w.U32(v.hi);
}

void Read(CodecReader& r, WriteRangeNode& v) {
	v.op = static_cast<WriteRangeNode::Op>(r.U8());
	v.a  = r.U32();
	v.b  = r.U32();
	v.c  = r.U32();
	v.lo = r.U32();
	v.hi = r.U32();
}

void Write(CodecWriter& w, const WriteRangeAccess& v) {
	w.U32(v.index);
	w.U32(v.offset);
	w.U32(v.soffset);
	w.U32(v.immediate);
	w.U32(v.extent);
}

void Read(CodecReader& r, WriteRangeAccess& v) {
	v.index     = r.U32();
	v.offset    = r.U32();
	v.soffset   = r.U32();
	v.immediate = r.U32();
	v.extent    = r.U32();
}

void Write(CodecWriter& w, const DescriptorBinding& v) {
	w.U32(EnumBits(v.kind));
	w.Words(v.resources);
}

void Read(CodecReader& r, DescriptorBinding& v) {
	v.kind      = EnumFrom<DescriptorBindingKind>(r.U32());
	v.resources = r.Words();
}

void Write(CodecWriter& w, const ResourceSpecialization::Buffer& v) {
	w.U32(v.packed_stride);
	w.U32(EnumBits(v.descriptor_format));
	w.U32(v.descriptor_swizzle);
	w.Bool(v.zero_stride_oob);
}

void Read(CodecReader& r, ResourceSpecialization::Buffer& v) {
	v.packed_stride      = r.U32();
	v.descriptor_format  = EnumFrom<Prospero::BufferFormat>(r.U32());
	v.descriptor_swizzle = r.U32();
	v.zero_stride_oob    = r.Bool();
}

void Write(CodecWriter& w, const ResourceSpecialization::Image& v) {
	w.U32(EnumBits(v.numeric_class));
	w.U32(EnumBits(v.dimension));
	w.U32(v.mip_count);
	w.U32(EnumBits(v.conversion_format));
	w.U32(v.shader_swizzle);
	w.U32(v.indirect_root);
	w.U32(v.indirect_mapping_offset);
	w.U32(v.indirect_search_iterations);
	w.Bool(v.cube);
	w.Bool(v.fmask);
}

void Read(CodecReader& r, ResourceSpecialization::Image& v) {
	v.numeric_class              = EnumFrom<Prospero::TextureNumericClass>(r.U32());
	v.dimension                  = EnumFrom<Decoder::ImageDimension>(r.U32());
	v.mip_count                  = r.U32();
	v.conversion_format          = EnumFrom<Prospero::BufferFormat>(r.U32());
	v.shader_swizzle             = r.U32();
	v.indirect_root              = r.U32();
	v.indirect_mapping_offset    = r.U32();
	v.indirect_search_iterations = r.U32();
	v.cube                       = r.Bool();
	v.fmask                      = r.Bool();
}

void Write(CodecWriter& w, const ResourcePlan::EvaluationOperand& v) {
	w.U64(v.value);
	w.U32(v.index);
	w.U32(EnumBits(v.kind));
}

void Read(CodecReader& r, ResourcePlan::EvaluationOperand& v) {
	v.value = r.U64();
	v.index = r.U32();
	v.kind  = EnumFrom<ResourcePlan::EvaluationOperand::Kind>(r.U32());
}

void Write(CodecWriter& w, const ResourcePlan::ArithmeticTape& v) {
	w.U32(v.first);
	w.U32(v.count);
}

void Read(CodecReader& r, ResourcePlan::ArithmeticTape& v) {
	v.first = r.U32();
	v.count = r.U32();
}

void Write(CodecWriter& w, uint8_t value) {
	w.U8(value);
}

void Read(CodecReader& r, uint8_t& value) {
	value = r.U8();
}

// Element types defined after the vector helpers (template lookup needs them declared first).
void Write(CodecWriter& w, const BufferWriteRange& v);
void Read(CodecReader& r, BufferWriteRange& v);

template <typename T>
void WriteVector(CodecWriter& w, const std::vector<T>& values) {
	w.U32(static_cast<uint32_t>(values.size()));
	for (const auto& value: values) {
		Write(w, value);
	}
}

template <typename T>
void ReadVector(CodecReader& r, std::vector<T>& values) {
	// Every element occupies at least one byte.
	const auto count = r.Count(1);
	values.clear();
	values.resize(count);
	for (auto& value: values) {
		Read(r, value);
		if (r.Failed()) {
			values.clear();
			return;
		}
	}
}

void Write(CodecWriter& w, const BufferWriteRange& v) {
	w.U32(v.buffer);
	w.U32(v.packed_stride);
	w.Bool(v.bounded);
	WriteVector(w, v.accesses);
}

void Read(CodecReader& r, BufferWriteRange& v) {
	v.buffer        = r.U32();
	v.packed_stride = r.U32();
	v.bounded       = r.Bool();
	ReadVector(r, v.accesses);
}

void Write(CodecWriter& w, const ShaderInfo& v) {
	WriteVector(w, v.buffers);
	WriteVector(w, v.images);
	WriteVector(w, v.samplers);
	WriteVector(w, v.sampled_pairs);
	WriteVector(w, v.inputs);
	WriteVector(w, v.outputs);
	w.Raw(v.vertex_fetch_components.data(), v.vertex_fetch_components.size());
	w.I32(v.vertex_offset_sgpr);
	w.I32(v.instance_offset_sgpr);
	w.Bool(v.has_bitwise_xor);
	w.Bool(v.uses_dma);
	w.Bool(v.uses_bvh);
	w.Bool(v.bda_writes);
}

void Read(CodecReader& r, ShaderInfo& v) {
	ReadVector(r, v.buffers);
	ReadVector(r, v.images);
	ReadVector(r, v.samplers);
	ReadVector(r, v.sampled_pairs);
	ReadVector(r, v.inputs);
	ReadVector(r, v.outputs);
	const auto fetch = r.Take(v.vertex_fetch_components.size());
	if (!fetch.empty()) {
		std::memcpy(v.vertex_fetch_components.data(), fetch.data(), fetch.size());
	}
	v.vertex_offset_sgpr   = r.I32();
	v.instance_offset_sgpr = r.I32();
	v.has_bitwise_xor      = r.Bool();
	v.uses_dma             = r.Bool();
	v.uses_bvh             = r.Bool();
	v.bda_writes           = r.Bool();
}

void Write(CodecWriter& w, const BindingLayout& v) {
	w.U32(v.push_data_start_dword);
	w.U32(v.memory_offset_dword);
	w.U32(v.memory_offset_count);
	w.U32(v.mip_stats_count);
	w.Words(v.user_data_registers);
	WriteVector(w, v.descriptors);
}

void Read(CodecReader& r, BindingLayout& v) {
	v.push_data_start_dword = r.U32();
	v.memory_offset_dword   = r.U32();
	v.memory_offset_count   = r.U32();
	v.mip_stats_count       = r.U32();
	v.user_data_registers   = r.Words();
	ReadVector(r, v.descriptors);
}

void Write(CodecWriter& w, const WriteRangeProgram& v) {
	WriteVector(w, v.nodes);
	WriteVector(w, v.buffers);
}

void Read(CodecReader& r, WriteRangeProgram& v) {
	ReadVector(r, v.nodes);
	ReadVector(r, v.buffers);
}

// ---------------------------------------------------------------------------------------------
// Resource plans: instructions become indices into value_storage.

class PlanEncoder {
public:
	PlanEncoder(const ResourcePlan& plan, CodecWriter& writer): m_plan(plan), w(writer) {}

	bool Encode() {
		uint32_t index = 0;
		for (const auto& inst: m_plan.value_storage) {
			m_indices.emplace(&inst, index++);
		}
		w.U32(index);
		for (const auto& inst: m_plan.value_storage) {
			if (!Instruction(inst)) return false;
		}
		const auto& p = m_plan;
		w.U32(EnumBits(p.stage));
		w.U64(p.shader_hash);
		w.U32(p.user_data_base);
		w.U32(p.user_data_count);
		WriteVector(w, p.memory_info);
		w.U32(static_cast<uint32_t>(p.descriptor_sources.size()));
		for (const auto& source: p.descriptor_sources) {
			for (const auto& dword: source.dwords) {
				if (!ValueOf(dword)) return false;
			}
			w.U32(source.dword_count);
			w.Bool(source.indirect_image.has_value());
			if (source.indirect_image) {
				const auto& image = *source.indirect_image;
				w.U32(image.material_source);
				w.U32(image.table_source);
				w.U32(image.selector_stride);
				w.U32(image.selector_offset);
				w.U32(image.table_offset);
				if (!ValueOf(image.key_count) || !ValueOf(image.selector_mask)) return false;
			}
		}
		w.U32(static_cast<uint32_t>(p.control_flow.size()));
		for (const auto& block: p.control_flow) {
			if (!ValueOf(block.condition)) return false;
			w.Words(block.successors);
			w.Words(block.sources);
		}
		w.U32(static_cast<uint32_t>(p.srt_reads.size()));
		for (const auto& read: p.srt_reads) {
			if (!ValueOf(read.value)) return false;
			w.U32(read.flat_offset);
		}
		w.Words(p.srt_read_run_ends);
		w.U32(static_cast<uint32_t>(p.evaluation_recipes.size()));
		for (const auto& recipe: p.evaluation_recipes) {
			if (!InstructionIndex(recipe.instruction)) return false;
			for (const auto& operand: recipe.operands) {
				Write(w, operand);
			}
			if (!ValueOf(recipe.selection_mask)) return false;
			w.I64(recipe.offset);
			w.U32(recipe.parameter);
			w.U32(EnumBits(recipe.kind));
		}
		w.U32(static_cast<uint32_t>(p.descriptor_roots.size()));
		for (const auto& roots: p.descriptor_roots) {
			for (const auto& operand: roots) {
				Write(w, operand);
			}
		}
		WriteVector(w, p.flat_read_roots);
		WriteVector(w, p.condition_roots);
		WriteVector(w, p.initial_active_sources);
		w.Words(p.flow_aliases);
		WriteVector(w, p.flow_initial_sources);
		WriteVector(w, p.arithmetic_tapes);
		w.U32(static_cast<uint32_t>(p.arithmetic_tape_instructions.size()));
		for (const auto& instruction: p.arithmetic_tape_instructions) {
			for (const auto& operand: instruction.operands) {
				w.U64(operand.value);
				w.Bool(operand.immediate);
			}
			if (!InstructionIndex(instruction.instruction)) return false;
			w.U32(instruction.parameter);
			w.U8(static_cast<uint8_t>(instruction.kind));
			w.U8(instruction.operand_count);
		}
		WriteVector(w, p.clean_flat_slots);
		w.Bool(p.requires_specialization_memory);
		w.Bool(p.has_address_writes);
		w.Bool(p.srt_plan_complete);
		w.Bool(p.resource_tracking_complete);
		Write(w, p.info);
		const auto& fill = p.uniform_fill.fill;
		w.U32(EnumBits(fill.kind));
		w.U32(fill.resource);
		for (const auto stride: fill.group_stride) {
			w.U32(stride);
		}
		w.U32(fill.words);
		w.U32(fill.value);
		for (const auto& value: p.uniform_fill.values) {
			if (!ValueOf(value)) return false;
		}
		w.U32(p.evaluation_value_count);
		w.Bool(p.evaluation_sealed);
		return true;
	}

private:
	bool InstructionIndex(const Inst* inst) {
		if (inst == nullptr) {
			w.U32(NullIndex);
			return true;
		}
		const auto found = m_indices.find(inst);
		if (found == m_indices.end()) return false;
		w.U32(found->second);
		return true;
	}

	bool ValueOf(const Value& value) {
		const auto type = ProgramCodecAccess::RawType(value);
		w.U32(EnumBits(type));
		if (type == Type::Void) return true;
		if (type == Type::Opaque) return InstructionIndex(ProgramCodecAccess::RawInst(value));
		uint64_t bits = 0;
		if (!ProgramCodecAccess::ImmediateBits(value, bits)) return false;
		w.U64(bits);
		return true;
	}

	bool Instruction(const Inst& inst) {
		// Plan instructions belong to no block (ExtractResourcePlan).
		if (ProgramCodecAccess::Parent(inst) != nullptr) return false;
		w.U32(EnumBits(ProgramCodecAccess::Opcode(inst)));
		w.U64(ProgramCodecAccess::Flags(inst));
		const auto& args = ProgramCodecAccess::Args(inst);
		w.U32(static_cast<uint32_t>(args.size()));
		for (const auto& arg: args) {
			if (!ValueOf(arg)) return false;
		}
		const auto& phi_blocks = ProgramCodecAccess::PhiBlocks(inst);
		if (std::ranges::any_of(phi_blocks, [](const Block* block) { return block != nullptr; })) {
			return false;
		}
		w.U32(static_cast<uint32_t>(phi_blocks.size()));
		const auto& uses = ProgramCodecAccess::Uses(inst);
		w.U32(static_cast<uint32_t>(uses.size()));
		for (const auto& use: uses) {
			if (use.user == nullptr || !InstructionIndex(use.user)) return false;
			w.U64(static_cast<uint64_t>(use.operand));
		}
		w.U32(ProgramCodecAccess::EvaluationIndex(inst));
		return true;
	}

	const ResourcePlan&                          m_plan;
	CodecWriter&                                 w;
	std::unordered_map<const Inst*, uint32_t>    m_indices;
};

class PlanDecoder {
public:
	PlanDecoder(CodecReader& reader, ResourcePlan& plan): r(reader), m_plan(plan) {}

	bool Decode() {
		// Instructions first: every later reference resolves against the complete list.
		const auto count = r.Count(4 + 8 + 4 + 4 + 4 + 4);
		if (r.Failed()) return false;
		struct PendingInst {
			std::vector<std::pair<Type, uint64_t>> args; // Opaque: index
			uint32_t                               phi_blocks = 0;
			std::vector<std::pair<uint32_t, uint64_t>> uses;
			uint32_t                               evaluation_index = 0;
		};
		std::vector<PendingInst> pending(count);
		m_insts.reserve(count);
		for (auto& inst: pending) {
			const auto opcode = r.U32();
			const auto flags  = r.U64();
			if (r.Failed() || opcode >= static_cast<uint32_t>(ValueOpcode::Count)) return false;
			m_insts.push_back(
			    &m_plan.value_storage.emplace_back(EnumFrom<ValueOpcode>(opcode), flags));
			const auto args = r.Count(4);
			inst.args.resize(args);
			for (auto& arg: inst.args) {
				if (!RawValue(arg)) return false;
			}
			inst.phi_blocks = r.U32();
			const auto uses = r.Count(12);
			inst.uses.resize(uses);
			for (auto& use: inst.uses) {
				use.first  = r.U32();
				use.second = r.U64();
			}
			inst.evaluation_index = r.U32();
			if (r.Failed()) return false;
		}
		for (uint32_t index = 0; index < count; ++index) {
			m_index_of.emplace(m_insts[index], index);
		}
		// Links, then their consistency: each (user, operand) use must name exactly one argument
		// slot holding this instruction, and every such slot must be listed once. Destruction
		// (Inst::Invalidate) and every later SetArg rely on it.
		std::vector<std::vector<std::pair<const Inst*, uint64_t>>> expected(count);
		for (uint32_t index = 0; index < count; ++index) {
			const auto& source = pending[index];
			if (source.phi_blocks != 0 && source.phi_blocks != source.args.size()) return false;
			std::vector<Value> args(source.args.size());
			for (size_t operand = 0; operand < args.size(); ++operand) {
				if (!Resolve(source.args[operand], args[operand])) return false;
				if (auto* target = args[operand].TryInstruction(); target != nullptr) {
					expected[Index(target)].emplace_back(m_insts[index], operand);
				}
			}
			std::vector<Use> uses(source.uses.size());
			for (size_t i = 0; i < uses.size(); ++i) {
				const auto [user, operand] = source.uses[i];
				if (user >= count) return false;
				uses[i] = {m_insts[user], static_cast<size_t>(operand)};
			}
			ProgramCodecAccess::Assign(*m_insts[index], std::move(args), source.phi_blocks,
			                           std::move(uses), source.evaluation_index);
		}
		for (uint32_t index = 0; index < count; ++index) {
			std::vector<std::pair<const Inst*, uint64_t>> actual;
			for (const auto& use: ProgramCodecAccess::Uses(*m_insts[index])) {
				actual.emplace_back(use.user, use.operand);
			}
			auto& wanted = expected[index];
			std::ranges::sort(actual);
			std::ranges::sort(wanted);
			if (actual != wanted || std::adjacent_find(actual.begin(), actual.end()) != actual.end()) {
				return false;
			}
		}

		auto& p = m_plan;
		p.stage           = EnumFrom<ShaderType>(r.U32());
		p.shader_hash     = r.U64();
		p.user_data_base  = r.U32();
		p.user_data_count = r.U32();
		ReadVector(r, p.memory_info);
		const auto descriptors = r.Count(1);
		p.descriptor_sources.resize(descriptors);
		for (auto& source: p.descriptor_sources) {
			for (auto& dword: source.dwords) {
				if (!ValueOf(dword)) return false;
			}
			source.dword_count = r.U32();
			if (r.Bool()) {
				auto& image = source.indirect_image.emplace(DescriptorSource::IndirectImage {});
				image.material_source = r.U32();
				image.table_source    = r.U32();
				image.selector_stride = r.U32();
				image.selector_offset = r.U32();
				image.table_offset    = r.U32();
				if (!ValueOf(image.key_count) || !ValueOf(image.selector_mask)) return false;
			}
			if (r.Failed()) return false;
		}
		const auto blocks = r.Count(1);
		p.control_flow.resize(blocks);
		for (auto& block: p.control_flow) {
			if (!ValueOf(block.condition)) return false;
			block.successors = r.Words();
			block.sources    = r.Words();
			if (r.Failed()) return false;
		}
		const auto reads = r.Count(1);
		p.srt_reads.resize(reads);
		for (auto& read: p.srt_reads) {
			if (!ValueOf(read.value)) return false;
			read.flat_offset = r.U32();
		}
		p.srt_read_run_ends = r.Words();
		const auto recipes = r.Count(1);
		p.evaluation_recipes.resize(recipes);
		for (auto& recipe: p.evaluation_recipes) {
			if (!InstructionOf(recipe.instruction)) return false;
			for (auto& operand: recipe.operands) {
				Read(r, operand);
			}
			if (!ValueOf(recipe.selection_mask)) return false;
			recipe.offset    = r.I64();
			recipe.parameter = r.U32();
			recipe.kind      = EnumFrom<ResourcePlan::EvaluationRecipe::Kind>(r.U32());
			if (r.Failed()) return false;
		}
		const auto roots = r.Count(1);
		p.descriptor_roots.resize(roots);
		for (auto& operands: p.descriptor_roots) {
			for (auto& operand: operands) {
				Read(r, operand);
			}
			if (r.Failed()) return false;
		}
		ReadVector(r, p.flat_read_roots);
		ReadVector(r, p.condition_roots);
		ReadVector(r, p.initial_active_sources);
		p.flow_aliases = r.Words();
		ReadVector(r, p.flow_initial_sources);
		ReadVector(r, p.arithmetic_tapes);
		const auto tape = r.Count(1);
		p.arithmetic_tape_instructions.resize(tape);
		for (auto& instruction: p.arithmetic_tape_instructions) {
			for (auto& operand: instruction.operands) {
				operand.value     = r.U64();
				operand.immediate = r.Bool();
			}
			if (!InstructionOf(instruction.instruction)) return false;
			instruction.parameter     = r.U32();
			instruction.kind          = static_cast<ResourcePlan::ArithmeticTapeInstruction::Kind>(r.U8());
			instruction.operand_count = r.U8();
			if (r.Failed()) return false;
		}
		ReadVector(r, p.clean_flat_slots);
		p.requires_specialization_memory = r.Bool();
		p.has_address_writes             = r.Bool();
		p.srt_plan_complete              = r.Bool();
		p.resource_tracking_complete     = r.Bool();
		Read(r, p.info);
		auto& fill    = p.uniform_fill.fill;
		fill.kind     = EnumFrom<UniformFillKind>(r.U32());
		fill.resource = r.U32();
		for (auto& stride: fill.group_stride) {
			stride = r.U32();
		}
		fill.words = r.U32();
		fill.value = r.U32();
		for (auto& value: p.uniform_fill.values) {
			if (!ValueOf(value)) return false;
		}
		p.evaluation_value_count = r.U32();
		p.evaluation_sealed      = r.Bool();
		return r.AtEnd();
	}

private:
	uint32_t Index(const Inst* inst) const {
		// Only called for instructions this decoder created (Resolve maps indices to them).
		return m_index_of.at(inst);
	}

	bool RawValue(std::pair<Type, uint64_t>& value) {
		value.first = EnumFrom<Type>(r.U32());
		if (value.first == Type::Void) return !r.Failed();
		value.second = value.first == Type::Opaque ? r.U32() : r.U64();
		return !r.Failed();
	}

	bool Resolve(const std::pair<Type, uint64_t>& raw, Value& value) {
		if (raw.first == Type::Void) {
			value = Value();
			return true;
		}
		if (raw.first == Type::Opaque) {
			if (raw.second == NullIndex) {
				value = Value(static_cast<Inst*>(nullptr));
				return true;
			}
			if (raw.second >= m_insts.size()) return false;
			value = Value(m_insts[raw.second]);
			return true;
		}
		return ProgramCodecAccess::MakeImmediate(raw.first, raw.second, value);
	}

	bool ValueOf(Value& value) {
		std::pair<Type, uint64_t> raw;
		return RawValue(raw) && Resolve(raw, value);
	}

	bool InstructionOf(const Inst*& inst) {
		const auto index = r.U32();
		if (r.Failed()) return false;
		if (index == NullIndex) {
			inst = nullptr;
			return true;
		}
		if (index >= m_insts.size()) return false;
		inst = m_insts[index];
		return true;
	}

	CodecReader&                              r;
	ResourcePlan&                             m_plan;
	std::vector<Inst*>                        m_insts;
	std::unordered_map<const Inst*, uint32_t> m_index_of;
};

} // namespace

bool EncodeResourcePlan(const ResourcePlan& plan, std::vector<uint8_t>& out) {
	std::vector<uint8_t> bytes;
	CodecWriter          writer(bytes);
	PlanEncoder          encoder(plan, writer);
	if (!encoder.Encode()) return false;
	out.insert(out.end(), bytes.begin(), bytes.end());
	return true;
}

bool DecodeResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan) {
	plan = ResourcePlan {};
	CodecReader reader(bytes);
	PlanDecoder decoder(reader, plan);
	if (decoder.Decode()) return true;
	// Unlink everything before the plan's destructor walks the instructions.
	for (auto& inst: plan.value_storage) {
		ProgramCodecAccess::Detach(inst);
	}
	plan = ResourcePlan {};
	return false;
}

void EncodeSpecialization(const ResourceSpecialization& value, std::vector<uint8_t>& out) {
	CodecWriter writer(out);
	WriteVector(writer, value.buffers);
	WriteVector(writer, value.images);
}

bool DecodeSpecialization(std::span<const uint8_t> bytes, ResourceSpecialization& value) {
	CodecReader reader(bytes);
	ReadVector(reader, value.buffers);
	ReadVector(reader, value.images);
	return reader.AtEnd();
}

void EncodeCompiledShaderInfo(const CompiledShaderInfo& value, std::vector<uint8_t>& out) {
	CodecWriter w(out);
	w.U32(EnumBits(value.stage));
	w.U64(value.shader_hash);
	w.U32(value.wave_size);
	w.U32(value.user_data_base);
	w.U32(value.user_data_count);
	w.U32(value.scratch_dwords);
	w.U32(value.param_export_mask);
	w.Bool(value.has_address_writes);
	Write(w, value.info);
	Write(w, value.bindings);
	Write(w, value.write_ranges);
}

bool DecodeCompiledShaderInfo(std::span<const uint8_t> bytes, CompiledShaderInfo& value) {
	CodecReader r(bytes);
	value.stage              = EnumFrom<ShaderType>(r.U32());
	value.shader_hash        = r.U64();
	value.wave_size          = r.U32();
	value.user_data_base     = r.U32();
	value.user_data_count    = r.U32();
	value.scratch_dwords     = r.U32();
	value.param_export_mask  = r.U32();
	value.has_address_writes = r.Bool();
	Read(r, value.info);
	Read(r, value.bindings);
	Read(r, value.write_ranges);
	return r.AtEnd();
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
