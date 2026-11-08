#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PROGRAMCODEC_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PROGRAMCODEC_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Byte encodings of what a translation leaves the pipeline cache, for the persistent program
// cache (KYTY_PROGRAM_CACHE, host_gpu/renderer/pipeline/programDiskCache.h): a program source's
// sealed resource plan (ExtractResourcePlan), and a permutation's resource specialization and
// compiled shader metadata (Program::TakeCompiledInfo).
//
// Every member of every encoded type is written, in declaration order, as fixed-width
// little-endian fields; the layout checks in ProgramCodec.cpp break the build when a member is
// added to one of them. Pointers become indices into the plan's value storage. Decoding rebuilds
// the object member for member: the same instructions in the same order with the same opcode,
// flags, arguments, phi operands, use lists (in their exact order) and evaluation indices, and
// every other member, so a decoded plan evaluates exactly as the plan it was encoded from, and
// encoding it again yields the same bytes. The encoding is a deterministic function of the members
// (no padding, no pointer values), so equal bytes mean equal objects: the specialization bytes
// are the permutation's key.

class CodecWriter {
public:
	explicit CodecWriter(std::vector<uint8_t>& out): m_out(out) {}

	void U8(uint8_t value) { m_out.push_back(value); }
	void U16(uint16_t value) { Raw(&value, sizeof(value)); }
	void U32(uint32_t value) { Raw(&value, sizeof(value)); }
	void U64(uint64_t value) { Raw(&value, sizeof(value)); }
	void I32(int32_t value) { Raw(&value, sizeof(value)); }
	void I64(int64_t value) { Raw(&value, sizeof(value)); }
	void Bool(bool value) { U8(value ? 1u : 0u); }
	void String(std::string_view text);
	void Words(std::span<const uint32_t> words);
	void Raw(const void* data, size_t size);

private:
	std::vector<uint8_t>& m_out;
};

// Bounds-checked reader: a read past the end, a boolean other than 0/1 or an implausible count
// sets the failure flag and returns zero, so decoders can read straight through and check once.
class CodecReader {
public:
	explicit CodecReader(std::span<const uint8_t> input): m_input(input) {}

	uint8_t  U8();
	uint16_t U16();
	uint32_t U32();
	uint64_t U64();
	int32_t  I32();
	int64_t  I64();
	bool     Bool();
	std::string String();
	std::vector<uint32_t> Words();
	// A count of elements that each occupy at least `min_bytes` further bytes; fails when fewer
	// bytes remain (a corrupt count cannot make a decoder allocate beyond the input).
	uint32_t Count(size_t min_bytes);
	std::span<const uint8_t> Take(size_t size);

	void Fail() { m_failed = true; }
	[[nodiscard]] bool   Failed() const { return m_failed; }
	[[nodiscard]] bool   AtEnd() const { return !m_failed && m_position == m_input.size(); }
	[[nodiscard]] size_t Remaining() const { return m_failed ? 0 : m_input.size() - m_position; }

private:
	bool Read(void* data, size_t size);

	std::span<const uint8_t> m_input;
	size_t                   m_position = 0;
	bool                     m_failed   = false;
};

// False (and `out` unchanged) when the plan references an instruction it does not own, or holds
// a value of a type an encoded plan cannot contain; such a program is not persisted.
[[nodiscard]] bool EncodeResourcePlan(const ResourcePlan& plan, std::vector<uint8_t>& out);
// Replaces `plan`; false (plan empty) for malformed input. Checks that every reference resolves
// inside the plan and that use lists match the arguments exactly, so that even an inconsistent
// input cannot leave a plan whose destruction would fail.
[[nodiscard]] bool DecodeResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan);

void EncodeSpecialization(const ResourceSpecialization& value, std::vector<uint8_t>& out);
[[nodiscard]] bool DecodeSpecialization(std::span<const uint8_t> bytes,
                                        ResourceSpecialization& value);
// A specialization as int21 (744705ee1) journals stored it: image records end after `fmask`.
[[nodiscard]] bool DecodeSpecializationInt21(std::span<const uint8_t> bytes,
                                             ResourceSpecialization& value);

void EncodeCompiledShaderInfo(const CompiledShaderInfo& value, std::vector<uint8_t>& out);
[[nodiscard]] bool DecodeCompiledShaderInfo(std::span<const uint8_t> bytes,
                                            CompiledShaderInfo& value);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PROGRAMCODEC_H_
