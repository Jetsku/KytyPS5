// KYTY_BDA_REACH: the reach of device-address reads (WriteRangeAnalysis: CollectDmaReads,
// WriteRangeEvaluator::DmaSpans) and the span logic the renderer synchronizes (NormalizeDmaSpans).
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/WriteRangeAnalysis.h"
#include "graphics/shader/shader.h"

#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderStageInputInfo;
using Libs::Graphics::ShaderType;
using Spans = std::vector<std::pair<uint64_t, uint64_t>>;

void Check(bool condition, const char* message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

struct Fixture {
	Program program;
	Block*  block = nullptr;

	Fixture() {
		program.stage           = ShaderType::Compute;
		program.user_data_count = 64;
		program.info.uses_dma   = true;
		auto storage            = std::make_unique<Block>();
		block                   = storage.get();
		program.block_storage.push_back(std::move(storage));
		program.blocks.push_back(block);
		program.block_info.push_back({.id = 0});
	}

	Value Emit(ValueOpcode opcode, std::initializer_list<Value> args, MemoryFlags flags = {}) {
		uint64_t bits = 0;
		std::memcpy(&bits, &flags, sizeof(flags));
		return Value(&block->AppendNewInst(opcode, args, bits));
	}
	Value UserData(uint32_t index) { return Emit(ValueOpcode::GetUserData, {Value(static_cast<Libs::Graphics::ShaderRecompiler::IR::ScalarReg>(index))}); }
	MemoryFlags Memory(ResourceKind kind, uint32_t offset, bool planning_only = false) {
		MemoryInfo memory;
		memory.kind          = kind;
		memory.offset        = offset;
		memory.data_bits     = 32;
		memory.data_dwords   = 1;
		memory.planning_only = planning_only;
		program.memory_info.push_back(memory);
		return {static_cast<uint32_t>(program.memory_info.size() - 1u), 0};
	}
	Value Address(Value low, Value high) { return Emit(ValueOpcode::GetAddressResource, {low, high}); }
	Value Load(ResourceKind kind, Value address, Value offset, uint32_t immediate, bool planning_only = false) {
		return Emit(ValueOpcode::LoadAddressU32, {address, offset, Value(0u), Value(true)},
		            Memory(kind, immediate, planning_only));
	}
	void Analyze() { AnalyzeBufferWriteRanges(program, ShaderStageInputInfo {}); }
};

// Guest memory for V# tables: dword address -> value; reads of unknown dwords fail.
struct Memory {
	std::map<uint64_t, uint32_t> words;
	uint32_t                     reads = 0;
	void Put(uint64_t address, std::array<uint32_t, 4> vsharp) {
		for (uint32_t k = 0; k < 4u; ++k) words[address + 4u * k] = vsharp[k];
	}
	static bool Read(void* user, uint64_t address, std::span<uint32_t> out) {
		auto* memory = static_cast<Memory*>(user);
		memory->reads++;
		for (size_t k = 0; k < out.size(); ++k) {
			const auto found = memory->words.find(address + 4u * k);
			if (found == memory->words.end()) return false;
			out[k] = found->second;
		}
		return true;
	}
};

bool Evaluate(const Program& program, std::vector<uint32_t> user_data, Spans& spans, DmaReachFailure& why,
              Memory* memory = nullptr, std::vector<uint32_t> flat = {}) {
	WriteRangeEvaluator evaluator;
	evaluator.Evaluate(program.write_ranges, {.user_data = user_data, .flattened_srt = flat});
	DmaReachInputs inputs;
	if (memory != nullptr) {
		inputs.read = &Memory::Read;
		inputs.user = memory;
	}
	spans.clear();
	return evaluator.DmaSpans(program.write_ranges, inputs, spans, why);
}

bool Covers(const Spans& spans, uint64_t begin, uint64_t end) {
	for (const auto& [first, last]: spans) {
		if (first <= begin && end <= last) return true;
	}
	return false;
}

// A global load at base(s[0:1]) + (s[2] & 0xff0) + 0x20: the window is that offset range.
void TestWindow() {
	Fixture f;
	const auto address = f.Address(f.UserData(0), f.UserData(1));
	const auto offset  = f.Emit(ValueOpcode::BitwiseAnd32, {f.UserData(2), Value(0xff0u)});
	f.Load(ResourceKind::Global, address, offset, 0x20);
	f.Analyze();
	Check(f.program.write_ranges.dma.bounded && f.program.write_ranges.dma.accesses.size() == 1u,
	      "a user-data window is bounded");
	Spans           spans;
	DmaReachFailure why {};
	Check(Evaluate(f.program, {0x10000u, 0x2u, 0xffffffffu}, spans, why), "window evaluates");
	const uint64_t base = 0x2'0001'0000ull;
	Check(spans.size() == 1u && Covers(spans, base + 0x20, base + 0xff0 + 0x20 + 4) && spans[0].first >= base + 0x20 - 4 &&
	          spans[0].second <= base + 0xff0 + 0x20 + 16,
	      "window covers exactly the offset range plus one access");
}

// Loads at the same base with different immediates share one hull; a scalar read clears bits 1:0.
void TestWindowMergeAndScalar() {
	Fixture f;
	const auto address = f.Address(f.UserData(0), f.UserData(1));
	f.Load(ResourceKind::ScalarAddress, address, Value(0u), 0x40);
	f.Load(ResourceKind::ScalarAddress, address, Value(0u), 0x7);
	f.Load(ResourceKind::ScalarAddress, address, Value(0u), 0x100);
	f.Analyze();
	Check(f.program.write_ranges.dma.bounded && f.program.write_ranges.dma.accesses.size() == 1u,
	      "same-base reads merge");
	Spans           spans;
	DmaReachFailure why {};
	Check(Evaluate(f.program, {0x1003u, 0u}, spans, why), "merged window evaluates");
	// base & ~3 = 0x1000; immediates 4..0x100 (0x7 & ~3 = 4).
	Check(spans.size() == 1u && Covers(spans, 0x1004, 0x1104) && spans[0].first >= 0x1000 - 4,
	      "scalar window from the masked base");
}

// An offset loaded from memory is unknown: bounded program, unknown reach at run time.
void TestUnknownOffset() {
	Fixture f;
	const auto address = f.Address(f.UserData(0), f.UserData(1));
	const auto loaded  = f.Load(ResourceKind::Global, address, Value(0u), 0);
	f.Load(ResourceKind::Global, address, loaded, 0);
	f.Analyze();
	Check(f.program.write_ranges.dma.bounded, "unknown offset still collects");
	Spans           spans;
	DmaReachFailure why {};
	Check(!Evaluate(f.program, {0x1000u, 0u}, spans, why) && why == DmaReachFailure::Size,
	      "an unbounded offset fails the reach");
}

// A base the shader loads from memory is not one value: the reach fails at run time.
void TestLoadedBase() {
	Fixture f;
	const auto pointer = f.Address(f.UserData(0), f.UserData(1));
	const auto low     = f.Load(ResourceKind::ScalarAddress, pointer, Value(0u), 0);
	const auto inner   = f.Address(low, f.UserData(1));
	f.Load(ResourceKind::Global, inner, Value(0u), 0);
	f.Analyze();
	Spans           spans;
	DmaReachFailure why {};
	Check(!Evaluate(f.program, {0x1000u, 0u}, spans, why) && why == DmaReachFailure::Base,
	      "a loaded base fails the reach");
}

// Address writes, 64-bit addresses and planning-only reads.
void TestUnboundedKinds() {
	{
		Fixture f;
		const auto address = f.Address(f.UserData(0), f.UserData(1));
		f.Emit(ValueOpcode::StoreAddressU32, {address, Value(0u), Value(0u), Value(1u), Value(true)},
		       f.Memory(ResourceKind::Global, 0));
		f.program.has_address_writes = true;
		f.Analyze();
		Check(!f.program.write_ranges.dma.bounded &&
		          f.program.write_ranges.dma.reason == static_cast<uint32_t>(DmaUnbounded::AddressWrites),
		      "address writes are unbounded");
	}
	{
		Fixture f;
		const auto address = f.Address(f.UserData(0), f.UserData(1));
		f.Load(ResourceKind::Global, address, f.UserData(2), 0);
		f.program.memory_info.back().address_is_full = true;
		f.Analyze();
		Check(!f.program.write_ranges.dma.bounded &&
		          f.program.write_ranges.dma.reason == static_cast<uint32_t>(DmaUnbounded::FullAddress),
		      "a full 64-bit address is unbounded");
	}
	{
		Fixture f;
		const auto address = f.Address(f.UserData(0), f.UserData(1));
		f.Load(ResourceKind::ScalarAddress, address, f.UserData(9), 0, true);
		f.Analyze();
		Check(f.program.write_ranges.dma.bounded && f.program.write_ranges.dma.accesses.empty(),
		      "planning-only reads are not emitted");
	}
	{
		Fixture f;
		f.program.info.uses_bvh = true;
		f.Analyze();
		Check(!f.program.write_ranges.dma.bounded &&
		          f.program.write_ranges.dma.reason == static_cast<uint32_t>(DmaUnbounded::RayTracing),
		      "BVH walks are unbounded");
	}
}

// A V# table: entries at base(s[0:1]) + ((s[2] & 3) << 4); S_BUFFER_LOADs through each V#.
void MakeTable(Fixture& f, bool vector) {
	const auto table  = f.Address(f.UserData(0), f.UserData(1));
	const auto offset = f.Emit(ValueOpcode::ShiftLeftLogical32,
	                           {f.Emit(ValueOpcode::BitwiseAnd32, {f.UserData(2), Value(3u)}), Value(4u)});
	std::array<Value, 4> words;
	for (uint32_t k = 0; k < 4u; ++k) {
		words[k] = f.Load(ResourceKind::ScalarAddress, table, offset, 0x100 + 4u * k);
	}
	const auto vsharp = f.Emit(ValueOpcode::GetBufferResource, {words[0], words[1], words[2], words[3]});
	if (vector) {
		f.Emit(ValueOpcode::LoadBufferU32, {vsharp, f.UserData(3), f.UserData(4), Value(0u), Value(true)},
		       f.Memory(ResourceKind::IndirectBuffer, 0));
	} else {
		f.Emit(ValueOpcode::ReadConstBuffer, {vsharp, f.UserData(3)}, f.Memory(ResourceKind::IndirectBuffer, 0));
	}
}

void TestScalarTable() {
	Fixture f;
	MakeTable(f, false);
	f.Analyze();
	const auto& dma = f.program.write_ranges.dma;
	Check(dma.bounded && dma.accesses.size() == 2u, "table window + indirect scalar access");
	const auto* indirect = &dma.accesses[0];
	if (indirect->kind != DmaReadAccess::Kind::IndirectScalar) indirect = &dma.accesses[1];
	Check(indirect->kind == DmaReadAccess::Kind::IndirectScalar && indirect->table_words == 0xfu &&
	          indirect->table_bits == 0x30u && indirect->table_immediate == 0x100,
	      "the V# comes from one table entry");
	Memory memory;
	const uint64_t table = 0x5000;
	for (uint32_t entry = 0; entry < 4u; ++entry) {
		// base 0x100000 * (entry + 1), stride 0, 0x200 bytes.
		memory.Put(table + 0x100 + 16u * entry, {0x100000u * (entry + 1u), 0u, 0x200u, 0x20000u});
	}
	Spans           spans;
	DmaReachFailure why {};
	Check(Evaluate(f.program, {0x5000u, 0u, 0xffffffffu, 0u}, spans, why, &memory), "table evaluates");
	Check(memory.reads == 4u, "every possible entry is read");
	for (uint32_t entry = 0; entry < 4u; ++entry) {
		const uint64_t base = 0x100000ull * (entry + 1u);
		Check(Covers(spans, base, base + 0x200), "each V# range is in the reach");
	}
	Check(Covers(spans, table + 0x100, table + 0x140), "the table entries are in the reach");
	uint64_t total = 0;
	for (const auto& [first, last]: spans) total += last - first;
	Check(total < 4u * 0x200u + 0x200u, "nothing else");
	// An unreadable entry (GPU-written) fails the reach.
	memory.words.erase(table + 0x100 + 16u * 2u);
	Check(!Evaluate(f.program, {0x5000u, 0u, 0xffffffffu, 0u}, spans, why, &memory) && why == DmaReachFailure::TableRead,
	      "an unreadable entry fails");
}

void TestVectorTable() {
	Fixture f;
	MakeTable(f, true);
	f.Analyze();
	Check(f.program.write_ranges.dma.bounded, "vector table bounded");
	Memory         memory;
	const uint64_t table = 0x5000;
	// Entry 0: raw (OOB_SELECT 3), stride 0, 0x400 bytes, FORMAT 4: offsets are bounded by records.
	memory.Put(table + 0x100, {0x200000u, 0u, 0x400u, (3u << 28u) | (4u << 12u)});
	// Entry 1: FORMAT 0: reads nothing.
	memory.Put(table + 0x110, {0x300000u, 0u, 0x400u, 0u});
	// Entry 2: structured (OOB_SELECT 0), stride 16, 8 records: index < 8 and offset < 16.
	memory.Put(table + 0x120, {0x400000u, 16u << 16u, 8u, (0u << 28u) | (4u << 12u)});
	// Entry 3: OOB_SELECT 2 (records only): offsets from the user data intervals (exact here).
	memory.Put(table + 0x130, {0x500000u, 0u, 0x10u, (2u << 28u) | (4u << 12u)});
	Spans           spans;
	DmaReachFailure why {};
	Check(Evaluate(f.program, {0x5000u, 0u, 0xffffffffu, 7u, 0x40u}, spans, why, &memory), "vector table evaluates");
	Check(Covers(spans, 0x200000, 0x200400), "raw V# reach is its records");
	Check(!Covers(spans, 0x300000, 0x300004), "FORMAT 0 reads nothing");
	Check(Covers(spans, 0x400000, 0x400000 + 8u * 16u), "structured V# reach is records * stride");
	Check(Covers(spans, 0x500040, 0x500044), "mode 2 at the exact offset");
	Check(!Covers(spans, 0x500000 + 0x1000, 0x500000 + 0x1004), "mode 2 not past the offset");
}

void TestVectorUnknownIndex() {
	Fixture    f;
	const auto vsharp = f.Emit(ValueOpcode::GetBufferResource, {f.UserData(0), f.UserData(1), f.UserData(2), f.UserData(3)});
	const auto index  = f.Load(ResourceKind::Global, f.Address(f.UserData(4), f.UserData(5)), Value(0u), 0);
	f.Emit(ValueOpcode::LoadBufferU32, {vsharp, index, Value(0u), Value(0u), Value(true)},
	       f.Memory(ResourceKind::IndirectBuffer, 0));
	f.Analyze();
	Check(f.program.write_ranges.dma.bounded, "direct V# bounded");
	Spans           spans;
	DmaReachFailure why {};
	// Structured, stride 32, 100 records: index < 100 bounds an unknown index.
	Check(Evaluate(f.program, {0x10000u, 32u << 16u, 100u, (4u << 12u), 0x9000u, 0u}, spans, why),
	      "structured V# with an unknown index");
	Check(Covers(spans, 0x10000, 0x10000 + 99u * 32u + 4u) && !Covers(spans, 0x10000, 0x10000 + 0x10000),
	      "records bound the index");
	// OOB_SELECT 2: nothing bounds the unknown index.
	Check(!Evaluate(f.program, {0x10000u, 32u << 16u, 100u, (2u << 28u) | (4u << 12u), 0x9000u, 0u}, spans, why) &&
	          why == DmaReachFailure::Offset,
	      "mode 2 with an unknown index fails");
	// Raw (mode 3) with a stride: the index is not checked.
	Check(!Evaluate(f.program, {0x10000u, 32u << 16u, 100u, (3u << 28u) | (4u << 12u), 0x9000u, 0u}, spans, why),
	      "raw strided V# with an unknown index fails");
	// Raw with stride 0: records bound everything.
	Check(Evaluate(f.program, {0x10000u, 0u, 0x300u, (3u << 28u) | (4u << 12u), 0x9000u, 0u}, spans, why) &&
	          Covers(spans, 0x10000, 0x10300),
	      "raw unstrided V#");
	// The flattened SRT: an exact base read from the flat buffer.
	Fixture    g;
	const auto low  = g.Emit(ValueOpcode::ReadConst, {g.Emit(ValueOpcode::GetSrtResource, {}), Value(1u)});
	const auto high = g.Emit(ValueOpcode::ReadConst, {g.Emit(ValueOpcode::GetSrtResource, {}), Value(2u)});
	g.Load(ResourceKind::Global, g.Address(low, high), Value(0u), 8);
	g.Analyze();
	Check(Evaluate(g.program, {}, spans, why, nullptr, {0u, 0x7000u, 0x1u}) && Covers(spans, 0x1'0000'7008ull, 0x1'0000'700cull),
	      "a flattened-SRT base");
}

// The mesh prologue's index fetch: the reach is the draw's index range from the host.
void TestMeshIndex() {
	Fixture    f;
	const auto low  = f.Emit(ValueOpcode::MeshDrawParameter, {Value(4u)});
	const auto high = f.Emit(ValueOpcode::MeshDrawParameter, {Value(5u)});
	const auto base = f.Address(f.Emit(ValueOpcode::BitwiseAnd32, {low, Value(~3u)}), high);
	f.Load(ResourceKind::Global, base, f.UserData(0), 0);
	f.Analyze();
	const auto& dma = f.program.write_ranges.dma;
	Check(dma.bounded && dma.accesses.size() == 1u && dma.accesses[0].kind == DmaReadAccess::Kind::MeshIndex,
	      "the index fetch is a mesh-index access");
	WriteRangeEvaluator evaluator;
	evaluator.Evaluate(f.program.write_ranges, {});
	DmaReachInputs  inputs;
	Spans           spans;
	DmaReachFailure why {};
	Check(!evaluator.DmaSpans(f.program.write_ranges, inputs, spans, why) && why == DmaReachFailure::Base,
	      "without the draw's index range the reach is unknown");
	inputs.mesh_index_known = true;
	inputs.mesh_index_begin = 0x7002;
	inputs.mesh_index_end   = 0x7802;
	spans.clear();
	Check(evaluator.DmaSpans(f.program.write_ranges, inputs, spans, why) && Covers(spans, 0x7000, 0x7806),
	      "the index range, dword aligned");
	// A non-indexed draw (direct, or a native indirect one): known, empty.
	inputs.mesh_index_begin = 0;
	inputs.mesh_index_end   = 0;
	spans.clear();
	Check(evaluator.DmaSpans(f.program.write_ranges, inputs, spans, why) && spans.empty(),
	      "a known empty index range reads nothing");
	// A native indexed indirect draw: the bound [INDEX_BASE, +INDEX_BUFFER_SIZE) range.
	inputs.mesh_index_begin = 0x9000;
	inputs.mesh_index_end   = 0x9000 + 3 * 2;
	spans.clear();
	Check(evaluator.DmaSpans(f.program.write_ranges, inputs, spans, why) && Covers(spans, 0x9000, 0x900a),
	      "the bound index range of an indirect draw");
}
void TestNormalize() {
	Spans spans {{0x5010, 0x5020}, {0x1000, 0x1001}, {0x4ff0, 0x5000}, {0x9000, 0x9000}, {0xc010, 0xc011}};
	NormalizeDmaSpans(spans, 0x4000);
	Check(spans.size() == 2u && spans[0] == std::pair<uint64_t, uint64_t>(0x0, 0x8000) &&
	          spans[1] == std::pair<uint64_t, uint64_t>(0xc000, 0x10000),
	      "page-aligned, merged, empty dropped");
	Spans adjacent {{0x4000, 0x8000}, {0x8000, 0xc000}};
	NormalizeDmaSpans(adjacent, 0x4000);
	Check(adjacent.size() == 1u && adjacent[0].second == 0xc000, "touching spans merge");
	Spans top {{UINT64_MAX - 10, UINT64_MAX}};
	NormalizeDmaSpans(top, 0x4000);
	Check(top.size() == 1u && top[0].second >= top[0].first, "no wrap at the top");
}

} // namespace

int main() {
	int failures = 0;
	const auto run = [&](const char* name, void (*test)()) {
		try {
			test();
			std::printf("PASS %s\n", name);
		} catch (const std::exception& error) {
			std::printf("FAIL %s: %s\n", name, error.what());
			failures++;
		}
	};
	run("window", TestWindow);
	run("window merge and scalar", TestWindowMergeAndScalar);
	run("unknown offset", TestUnknownOffset);
	run("loaded base", TestLoadedBase);
	run("unbounded kinds", TestUnboundedKinds);
	run("scalar table", TestScalarTable);
	run("vector table", TestVectorTable);
	run("vector unknown index", TestVectorUnknownIndex);
	run("mesh index", TestMeshIndex);
	run("normalize", TestNormalize);
	return failures == 0 ? 0 : 1;
}
