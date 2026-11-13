#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_SEGMENT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_SEGMENT_H_

// KYTY_SPEC_SEGMENTS (stage 1 of a parallel commit of continuation segments; design: the spec
// track's DESIGN.md, not in the repository).
//
// The preparing thread encodes, next to its preparation, the part of a DrawIndex continuation's
// commands that is a pure function of the slot (its draw arguments, registers and prepared vertex
// stage): the draw command's arguments and what a continuation must share with its run's record
// (structure key, index element size, primitive class). The commit thread, at a continuation the
// continuation kernel takes, validates the encoded draw against the run it continues (the same
// conditions the kernel checks) and splices it: the encoded command is emitted instead of computing
// the draw offsets and the draw command there. A consecutive series of spliced continuations of one
// run is a segment.
//
// Stage 1 moves only the draw command (offsets and arguments). Buffer work, descriptors and push
// constants stay on the commit thread: a worker cannot resolve buffers yet (no read-only buffer
// view). The stage carries the hand-over, the validation, the splice point, the verify mode and
// the counters that later stages extend.
//
// Everything in this header is free of the renderer runtime (tested in tests/DrawPrepTests.cpp).

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>

namespace Libs::Graphics::DrawPrep::Segment {

// KYTY_SPEC_SEGMENTS: 0 off, 1 on (splice), 2 verify (encode and validate, then run the serial
// computation, compare field by field, use the serial result).
enum class Mode : uint8_t { Off = 0, On = 1, Verify = 2 };

[[nodiscard]] inline Mode ParseMode(const char* value) noexcept {
	if (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) {
		return Mode::Off;
	}
	if ((value[0] == '2' && value[1] == '\0') || std::string(value) == "verify") {
		return Mode::Verify;
	}
	return (value[0] == '1' && value[1] == '\0') ? Mode::On : Mode::Off;
}

// Why the preparing thread encoded no draw, or why the commit thread did not splice one.
enum class Miss : uint8_t {
	None,
	// Encode (preparing thread).
	NotIndexed,      // not a DrawIndex slot
	NotPrepared,     // the preparation failed
	NoKey,           // no run structure key (KYTY_DRAW_RUN off, or ineligible)
	OffsetSource,    // offsets not from the draw state (indirect offsets)
	GpuInstances,    // KYTY_GPU_INSTANCE_COUNT: the instance count is GPU data
	PrimitiveType,   // a primitive type the plain draw command does not cover
	ElementSize,     // not a 16- or 32-bit index type
	NoVertexStage,   // the prepared vertex stage has no program
	// Validate (commit thread).
	NotEncoded,      // the slot carried no encoded draw (switched on after it was prepared)
	Key,             // its structure key is not the run's
	RecordElement,   // its element size is not the run record's
	Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Miss::Count)> kMissNames {
    "none",          "not_indexed",  "not_prepared", "no_key",    "offset_source", "gpu_instances",
    "primitive_type", "element_size", "no_vertex_stage", "not_encoded", "key",     "record_element"};

// The draw a continuation emits: vkCmdDrawIndexed's arguments, and what it must share with the run.
struct EncodedDraw {
	Miss     miss           = Miss::NotEncoded; // None: encoded
	uint64_t run_key        = 0;
	uint32_t element_size   = 0;
	uint32_t index_count    = 0;
	uint32_t instance_count = 0;
	int32_t  vertex_offset  = 0;
	uint32_t first_instance = 0;
	uint64_t index_address  = 0;
	uint64_t index_bytes    = 0;
};

// The inputs of the encode, as the slot holds them (DrawIndexArgs, user config, prepared VS).
struct EncodeInputs {
	bool     indexed              = true;
	bool     prepared_ok          = true;
	uint64_t run_key              = 0;
	bool     offsets_from_state   = true;  // DrawOffsetSource::DrawState
	uint64_t gpu_instance_address = 0;
	bool     plain_primitive      = true;  // EmitDrawPrimitives' one-command primitive types
	uint32_t index_type           = 0;     // Prospero::IndexType: 0 16-bit, 1 32-bit
	bool     vertex_stage         = true;  // the prepared vertex stage has its program
	uint32_t index_count          = 0;
	uint32_t instance_count       = 0;
	int32_t  base_vertex          = 0;
	uint64_t index_address        = 0;
	// ResolveDrawOffsets(index offset, prepared vertex info): the stage's vertex and instance offsets.
	int32_t  resolved_vertex_offset   = 0;
	uint32_t resolved_instance_offset = 0;
};

// The pure encode (preparing thread). The order of the checks gives the reported reason.
[[nodiscard]] inline EncodedDraw Encode(const EncodeInputs& in) noexcept {
	EncodedDraw draw;
	const auto  fail = [&](Miss why) {
        draw.miss = why;
        return draw;
	};
	if (!in.indexed) return fail(Miss::NotIndexed);
	if (!in.prepared_ok) return fail(Miss::NotPrepared);
	if (in.run_key == 0) return fail(Miss::NoKey);
	if (!in.offsets_from_state) return fail(Miss::OffsetSource);
	if (in.gpu_instance_address != 0) return fail(Miss::GpuInstances);
	if (!in.plain_primitive) return fail(Miss::PrimitiveType);
	if (in.index_type > 1) return fail(Miss::ElementSize);
	if (!in.vertex_stage) return fail(Miss::NoVertexStage);
	draw.miss           = Miss::None;
	draw.run_key        = in.run_key;
	draw.element_size   = in.index_type == 0 ? 2u : 4u;
	draw.index_count    = in.index_count;
	draw.instance_count = in.instance_count;
	draw.vertex_offset  = in.resolved_vertex_offset + in.base_vertex;
	draw.first_instance = in.resolved_instance_offset;
	draw.index_address  = in.index_address;
	draw.index_bytes    = uint64_t {in.index_count} * draw.element_size;
	return draw;
}

// The commit thread's validation against the run the continuation continues (beyond what the kernel
// entry already proved: the slot's key equals the run's, the element size the record's).
[[nodiscard]] inline Miss Validate(const EncodedDraw& draw, uint64_t run_key,
                                   uint32_t record_element_size) noexcept {
	if (draw.miss != Miss::None) return Miss::NotEncoded;
	if (draw.run_key != run_key) return Miss::Key;
	if (draw.element_size != record_element_size) return Miss::RecordElement;
	return Miss::None;
}

// Verify: the fields the serial path computes for the same draw.
enum class Field : uint8_t { IndexCount, InstanceCount, VertexOffset, FirstInstance, IndexAddress, IndexBytes, Count };
inline constexpr std::array<const char*, static_cast<size_t>(Field::Count)> kFieldNames {
    "index_count", "instance_count", "vertex_offset", "first_instance", "index_address", "index_bytes"};

// A mask of the fields where `encoded` differs from `serial` (0: equal).
[[nodiscard]] inline uint32_t Compare(const EncodedDraw& encoded, const EncodedDraw& serial) noexcept {
	uint32_t mask = 0;
	const auto note = [&](bool differs, Field field) {
		if (differs) mask |= 1u << static_cast<uint32_t>(field);
	};
	note(encoded.index_count != serial.index_count, Field::IndexCount);
	note(encoded.instance_count != serial.instance_count, Field::InstanceCount);
	note(encoded.vertex_offset != serial.vertex_offset, Field::VertexOffset);
	note(encoded.first_instance != serial.first_instance, Field::FirstInstance);
	note(encoded.index_address != serial.index_address, Field::IndexAddress);
	note(encoded.index_bytes != serial.index_bytes, Field::IndexBytes);
	return mask;
}

// Totals (process-wide; encodes on the preparing threads, the rest on the commit thread). Relaxed.
struct Counters {
	std::atomic<uint64_t> encoded {0};      // drawn up by a preparing thread
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Miss::Count)> misses {}; // encode + validate
	std::atomic<uint64_t> kernel_taken {0}; // continuations the kernel took while the switch was on
	std::atomic<uint64_t> validated {0};    // validated against their run
	std::atomic<uint64_t> spliced {0};      // emitted from the encoded draw (mode 1)
	std::atomic<uint64_t> segments {0};     // series of consecutive validated continuations of one run
	std::atomic<uint64_t> verify_draws {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Field::Count)> mismatches {};
	std::atomic<uint64_t> validate_ns {0};  // commit thread, inside Validate (sampled clock)
	std::atomic<uint64_t> splice_ns {0};    // commit thread, emitting a spliced draw (sampled clock)
	std::atomic<uint64_t> timed {0};        // validations that were timed
	std::atomic<uint64_t> splice_timed {0}; // splices that were timed
};

// The series bookkeeping of the commit thread: a validated continuation whose run serial is the
// previous validated one's continues its segment; otherwise it starts one.
class SegmentTracker {
public:
	// Returns true when this draw starts a new segment.
	bool Note(uint64_t run_serial) noexcept {
		const bool starts = !m_open || run_serial != m_serial;
		m_open            = true;
		m_serial          = run_serial;
		return starts;
	}
	// Anything else committed in between (a start draw, a miss, another op) ends the segment.
	void Break() noexcept { m_open = false; }

private:
	bool     m_open   = false;
	uint64_t m_serial = 0;
};

// The 10 s line: everything a count boot needs to read the coverage and the verify result.
[[nodiscard]] inline std::string FormatLine(const Counters& c, const char* mode) {
	char head[512];
	const auto taken     = c.kernel_taken.load(std::memory_order_relaxed);
	const auto validated = c.validated.load(std::memory_order_relaxed);
	const auto segments  = c.segments.load(std::memory_order_relaxed);
	const auto timed     = c.timed.load(std::memory_order_relaxed);
	const auto s_timed   = c.splice_timed.load(std::memory_order_relaxed);
	std::snprintf(head, sizeof(head),
	              "SpecSegments (%s): encoded %" PRIu64 " | kernel-taken %" PRIu64 " validated %" PRIu64
	              " (%.1f%%) spliced %" PRIu64 " segments %" PRIu64 " (%.1f draws/segment) | validate %.0f ns"
	              " splice %.0f ns (per timed draw) | verify draws %" PRIu64 " mismatches",
	              mode, c.encoded.load(std::memory_order_relaxed), taken, validated,
	              taken != 0 ? 100.0 * static_cast<double>(validated) / static_cast<double>(taken) : 0.0,
	              c.spliced.load(std::memory_order_relaxed), segments,
	              segments != 0 ? static_cast<double>(validated) / static_cast<double>(segments) : 0.0,
	              timed != 0 ? static_cast<double>(c.validate_ns.load(std::memory_order_relaxed)) / timed : 0.0,
	              s_timed != 0 ? static_cast<double>(c.splice_ns.load(std::memory_order_relaxed)) / s_timed : 0.0,
	              c.verify_draws.load(std::memory_order_relaxed));
	std::string line = head;
	uint64_t    total = 0;
	for (size_t i = 0; i < kFieldNames.size(); i++) {
		const auto n = c.mismatches[i].load(std::memory_order_relaxed);
		total += n;
		line += " ";
		line += kFieldNames[i];
		line += "=" + std::to_string(n);
	}
	line += " total=" + std::to_string(total) + " | misses";
	for (size_t i = 1; i < kMissNames.size(); i++) {
		line += " ";
		line += kMissNames[i];
		line += "=" + std::to_string(c.misses[i].load(std::memory_order_relaxed));
	}
	return line;
}

} // namespace Libs::Graphics::DrawPrep::Segment

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_SEGMENT_H_
