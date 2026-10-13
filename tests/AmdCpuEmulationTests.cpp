// Windows: the two ways Kyty runs the PS5's AMD-only instructions on a host without them.
//  - The runtime path: the instruction raises #UD, the vectored handler calls
//    X64InstructionEmulator::TryEmulate on the exception CONTEXT, and Windows restores it.
//  - The load-time path (--amd-cpu): the patcher replaces EXTRQ and VRSQRTPS with native
//    trampolines and guards the instructions that still trap against the guest red zone.
// Both must produce what the AMD CPU produces: every register bit, the flags and the RIP advance.
// On an SSE4a host (AMD) the CPU itself is the reference; elsewhere a model of the AMD manual.
// The red-zone tests show why the runtime path is unsafe on Windows: exception dispatch writes
// its frame directly below RSP, over the 128 bytes the SysV guest code may still be using.

#include "common/hostException.h"
#include "loader/guestInstructionPatcher.h"
#include "loader/x64InstructionEmulator.h"

#include <windows.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <intrin.h>
#include <random>
#include <vector>

namespace {

using Common::HostException::ExceptionInfo;
using Common::HostException::ExceptionType;

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "AmdCpuEmulationTests: failed: %s\n", text);
		g_failures++;
	}
}

void Require(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "AmdCpuEmulationTests: fatal: %s\n", text);
		std::exit(1);
	}
}

std::atomic<int>  g_illegal_hits {0};
std::atomic<bool> g_skip_ud2 {false};

// The runtime path as runtimeLinker installs it. With g_skip_ud2 armed, a UD2 in front of an
// instruction the CPU supports stands in for that instruction's own #UD: the handler steps over the
// UD2 and emulates the next instruction in the real exception CONTEXT, which Windows then restores.
bool Handler(const ExceptionInfo& info) {
	if (info.type != ExceptionType::IllegalInstruction) {
		return false;
	}
	g_illegal_hits.fetch_add(1, std::memory_order_relaxed);
	auto*       context = static_cast<PCONTEXT>(info.native_context);
	const auto* rip     = reinterpret_cast<const uint8_t*>(context->Rip);
	if (g_skip_ud2.load(std::memory_order_relaxed) && rip[0] == 0x0f && rip[1] == 0x0b) {
		context->Rip += 2;
	}
	return Loader::X64InstructionEmulator::TryEmulate(context);
}

bool HostHasSse4a() {
	int regs[4] {};
	__cpuid(regs, 0x80000000);
	if (static_cast<uint32_t>(regs[0]) < 0x80000001u) {
		return false;
	}
	__cpuid(regs, 0x80000001);
	return (regs[2] & (1 << 6)) != 0;
}

// ---- Code emission -------------------------------------------------------------------------------

constexpr int RCX = 1;
constexpr int RDX = 2;
constexpr int RSP = 4;
constexpr int R8  = 8;

struct Emitter {
	std::vector<uint8_t> bytes;

	void Raw(std::initializer_list<uint8_t> list) { bytes.insert(bytes.end(), list); }
	void Raw(const std::vector<uint8_t>& list) { bytes.insert(bytes.end(), list.begin(), list.end()); }

	// movdqu xmm, [base + disp32] (load) or movdqu [base + disp32], xmm (store).
	void Movdqu(bool load, int xmm, int base, int32_t disp) {
		bytes.push_back(0xf3);
		const uint8_t rex = 0x40 | ((xmm >> 3) & 1) << 2 | ((base >> 3) & 1);
		if (rex != 0x40) {
			bytes.push_back(rex);
		}
		bytes.push_back(0x0f);
		bytes.push_back(load ? 0x6f : 0x7f);
		bytes.push_back(static_cast<uint8_t>(0x80 | (xmm & 7) << 3 | (base & 7)));
		if ((base & 7) == RSP) {
			bytes.push_back(0x24);
		}
		uint8_t d[4];
		std::memcpy(d, &disp, 4);
		bytes.insert(bytes.end(), d, d + 4);
	}
	// mov r10, r11: a flag-neutral neighbour the patcher can relocate next to a short instruction.
	void Filler() { Raw({0x4d, 0x89, 0xda}); }
};

struct CodePage {
	uint8_t* base = nullptr;
	size_t   size = 0;

	explicit CodePage(size_t bytes): size(bytes) {
		base = static_cast<uint8_t*>(
		    VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
		Require(base != nullptr, "allocate an executable page");
	}
	~CodePage() { VirtualFree(base, 0, MEM_RELEASE); }
	CodePage(const CodePage&)            = delete;
	CodePage& operator=(const CodePage&) = delete;

	void* Load(const std::vector<uint8_t>& code) {
		Require(code.size() <= size, "code fits its page");
		std::memset(base, 0xcc, size);
		std::memcpy(base, code.data(), code.size());
		FlushInstructionCache(GetCurrentProcess(), base, size);
		return base;
	}
};

// ---- The instruction forms -----------------------------------------------------------------------

struct Xmm {
	uint64_t q[2];
	bool     operator==(const Xmm& other) const { return q[0] == other.q[0] && q[1] == other.q[1]; }
};

struct RegisterFile {
	Xmm      xmm[16];
	uint64_t flags;
};

enum class Form { ExtrqImm, ExtrqReg, InsertqImm, InsertqReg };

struct Insn {
	Form                 form {};
	int                  dest = 0;   // the register written
	int                  source = 0; // the other operand (controls for EXTRQ reg)
	uint8_t              length = 0; // immediates
	uint8_t              index  = 0;
	uint8_t              rex_wx = 0; // REX.W/REX.X bits, which the instruction ignores
	std::vector<uint8_t> bytes;
};

void Encode(Insn& insn) {
	const bool insert = insn.form == Form::InsertqImm || insn.form == Form::InsertqReg;
	const bool imm    = insn.form == Form::ExtrqImm || insn.form == Form::InsertqImm;
	int        reg    = 0;
	int        rm     = 0;
	switch (insn.form) {
		case Form::ExtrqImm: rm = insn.dest; break; // /0, r/m is the register
		case Form::ExtrqReg:
		case Form::InsertqImm:
		case Form::InsertqReg:
			reg = insn.dest;
			rm  = insn.source;
			break;
	}
	insn.bytes.clear();
	insn.bytes.push_back(insert ? 0xf2 : 0x66);
	const uint8_t rex = 0x40 | insn.rex_wx | ((reg >> 3) & 1) << 2 | ((rm >> 3) & 1);
	if (rex != 0x40) {
		insn.bytes.push_back(rex);
	}
	insn.bytes.push_back(0x0f);
	insn.bytes.push_back(imm ? 0x78 : 0x79);
	insn.bytes.push_back(static_cast<uint8_t>(0xc0 | (reg & 7) << 3 | (rm & 7)));
	if (imm) {
		insn.bytes.push_back(insn.length);
		insn.bytes.push_back(insn.index);
	}
}

// The AMD manual (APM vol. 4, EXTRQ/INSERTQ): a zero length means 64; length + index > 64 and bits
// 127:64 of the destination are undefined. What a Zen 4 does, and the model below: only the low 6
// bits of length and index count, a field reaching past bit 63 is cut at bit 63, and bits 127:64
// are zeroed by both instructions.
bool Defined(uint32_t length, uint32_t index) {
	length &= 63;
	index &= 63;
	return (length == 0 ? 64 : length) + index <= 64;
}

void Model(const Insn& insn, RegisterFile& regs) {
	Xmm&       dest = regs.xmm[insn.dest];
	const Xmm& src  = regs.xmm[insn.source];
	uint32_t   length = insn.length;
	uint32_t   index  = insn.index;
	if (insn.form == Form::ExtrqReg) {
		length = static_cast<uint32_t>(src.q[0]);
		index  = static_cast<uint32_t>(src.q[0] >> 8);
	} else if (insn.form == Form::InsertqReg) {
		length = static_cast<uint32_t>(src.q[1]);
		index  = static_cast<uint32_t>(src.q[1] >> 8);
	}
	length &= 63;
	index &= 63;
	const uint32_t bits = length == 0 ? 64 : length;
	const uint64_t mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
	if (insn.form == Form::ExtrqImm || insn.form == Form::ExtrqReg) {
		dest.q[0] = (dest.q[0] >> index) & mask;
		dest.q[1] = 0;
	} else {
		const uint64_t field = (src.q[0] & mask) << index;
		const uint64_t place = mask << index;
		dest.q[0]            = (dest.q[0] & ~place) | field;
		dest.q[1]            = 0;
	}
}

// A routine that loads all 16 XMM registers and the flags, runs `body`, and stores them back.
// Windows ABI: rcx = RegisterFile in, rdx = RegisterFile out, r8 = save area for xmm6-15,
// r9 = flags to load.
std::vector<uint8_t> RegisterFileRoutine(const std::vector<uint8_t>& body) {
	Emitter e;
	for (int i = 6; i < 16; i++) {
		e.Movdqu(false, i, R8, 16 * (i - 6));
	}
	for (int i = 0; i < 16; i++) {
		e.Movdqu(true, i, RCX, 16 * i);
	}
	e.Raw({0x41, 0x51}); // push r9
	e.Raw({0x9d});       // popfq
	e.Filler();
	e.Raw(body);
	e.Filler();
	e.Raw({0x9c});                                     // pushfq
	e.Raw({0x58});                                     // pop rax
	e.Raw({0x48, 0x89, 0x82, 0x00, 0x01, 0x00, 0x00}); // mov [rdx + 256], rax
	for (int i = 0; i < 16; i++) {
		e.Movdqu(false, i, RDX, 16 * i);
	}
	for (int i = 6; i < 16; i++) {
		e.Movdqu(true, i, R8, 16 * (i - 6));
	}
	e.Raw({0xc3});
	return e.bytes;
}

using RegisterFileFn = void (*)(const RegisterFile* in, RegisterFile* out, Xmm* save, uint64_t flags);

constexpr uint64_t FlagMask = 0x8d5; // CF PF AF ZF SF OF

RegisterFile RunRoutine(void* code, const RegisterFile& in) {
	alignas(16) RegisterFile out {};
	alignas(16) Xmm          save[10] {};
	reinterpret_cast<RegisterFileFn>(code)(&in, &out, save, (in.flags & FlagMask) | 0x202);
	out.flags &= FlagMask;
	return out;
}

// The runtime path without an exception: a CONTEXT as Windows hands it to the handler.
RegisterFile RunEmulatorOnContext(const Insn& insn, const RegisterFile& in, bool& handled,
                                  uint64_t& advance) {
	alignas(16) CONTEXT context {};
	context.ContextFlags = CONTEXT_FULL;
	context.EFlags       = static_cast<DWORD>((in.flags & FlagMask) | 0x202);
	auto* xmm            = &context.Xmm0;
	for (int i = 0; i < 16; i++) {
		std::memcpy(&xmm[i], &in.xmm[i], 16);
	}
	std::array<uint8_t, 32> code {};
	std::memcpy(code.data(), insn.bytes.data(), insn.bytes.size());
	context.Rip = reinterpret_cast<DWORD64>(code.data());
	handled     = Loader::X64InstructionEmulator::TryEmulate(&context);
	advance     = context.Rip - reinterpret_cast<DWORD64>(code.data());
	RegisterFile out {};
	for (int i = 0; i < 16; i++) {
		std::memcpy(&out.xmm[i], &xmm[i], 16);
	}
	out.flags = context.EFlags & FlagMask;
	return out;
}

Insn RandomInsn(std::mt19937_64& rng, Form form, bool defined_only) {
	Insn insn;
	insn.form   = form;
	insn.dest   = static_cast<int>(rng() % 16);
	insn.source = static_cast<int>(rng() % 16);
	// REX.W and REX.X do not change these instructions; REX.R on the /0 EXTRQ form would.
	insn.rex_wx = static_cast<uint8_t>((rng() % 4) == 0 ? (rng() & 0x0a) : 0);
	do {
		insn.length = static_cast<uint8_t>(rng());
		insn.index  = static_cast<uint8_t>(rng());
	} while (defined_only && !Defined(insn.length, insn.index));
	if (form == Form::ExtrqImm) {
		insn.source = insn.dest;
	}
	Encode(insn);
	return insn;
}

void RandomRegisters(std::mt19937_64& rng, const Insn& insn, bool defined_only, RegisterFile& regs) {
	for (auto& x: regs.xmm) {
		x.q[0] = rng();
		x.q[1] = rng();
	}
	regs.flags = rng() & FlagMask;
	if (insn.form == Form::ExtrqReg || insn.form == Form::InsertqReg) {
		auto& controls = regs.xmm[insn.source].q[insn.form == Form::ExtrqReg ? 0 : 1];
		while (defined_only && !Defined(static_cast<uint32_t>(controls), static_cast<uint32_t>(controls >> 8))) {
			controls = rng();
		}
	}
}

bool ControlsDefined(const Insn& insn, const RegisterFile& regs) {
	if (insn.form == Form::ExtrqReg || insn.form == Form::InsertqReg) {
		const uint64_t c = regs.xmm[insn.source].q[insn.form == Form::ExtrqReg ? 0 : 1];
		return Defined(static_cast<uint32_t>(c), static_cast<uint32_t>(c >> 8));
	}
	return Defined(insn.length, insn.index);
}

const char* FormName(Form form) {
	switch (form) {
		case Form::ExtrqImm: return "EXTRQ xmm, imm8, imm8";
		case Form::ExtrqReg: return "EXTRQ xmm, xmm";
		case Form::InsertqImm: return "INSERTQ xmm, xmm, imm8, imm8";
		default: return "INSERTQ xmm, xmm";
	}
}

void PrintMismatch(const char* what, const Insn& insn, const RegisterFile& in, const RegisterFile& expected,
                   const RegisterFile& got) {
	std::fprintf(stderr, "  %s: %s dest=xmm%d src=xmm%d len=%u idx=%u bytes=", what, FormName(insn.form),
	             insn.dest, insn.source, insn.length, insn.index);
	for (auto b: insn.bytes) {
		std::fprintf(stderr, "%02x ", b);
	}
	std::fprintf(stderr, "\n");
	for (int i = 0; i < 16; i++) {
		if (!(expected.xmm[i] == got.xmm[i])) {
			std::fprintf(stderr, "    xmm%d in %016llx:%016llx expected %016llx:%016llx got %016llx:%016llx\n", i,
			             in.xmm[i].q[1], in.xmm[i].q[0], expected.xmm[i].q[1], expected.xmm[i].q[0],
			             got.xmm[i].q[1], got.xmm[i].q[0]);
		}
	}
	if (expected.flags != got.flags) {
		std::fprintf(stderr, "    flags expected %llx got %llx\n", expected.flags, got.flags);
	}
}

bool Same(const RegisterFile& a, const RegisterFile& b) {
	for (int i = 0; i < 16; i++) {
		if (!(a.xmm[i] == b.xmm[i])) {
			return false;
		}
	}
	return a.flags == b.flags;
}

// ---- 1. Runtime emulator on a CONTEXT vs the model and the AMD CPU --------------------------------

void TestEmulatorRandomized(bool sse4a) {
	std::mt19937_64 rng(0x5eed4a);
	CodePage        page(4096);
	int             cases = 0;
	int             model_mismatch = 0;
	int             cpu_mismatch = 0;
	int             undefined_cpu_mismatch = 0;
	int             undefined_cases = 0;
	for (Form form: {Form::ExtrqImm, Form::ExtrqReg, Form::InsertqImm, Form::InsertqReg}) {
		for (int n = 0; n < 6000; n++) {
			const bool   defined_only = n % 2 == 0;
			const Insn   insn         = RandomInsn(rng, form, defined_only);
			RegisterFile in {};
			RandomRegisters(rng, insn, defined_only, in);
			const bool defined = ControlsDefined(insn, in);

			bool         handled = false;
			uint64_t     advance = 0;
			RegisterFile emulated = RunEmulatorOnContext(insn, in, handled, advance);
			cases++;
			if (!handled || advance != insn.bytes.size()) {
				Check(false, "TryEmulate handles the instruction and advances RIP by its length");
				PrintMismatch("not handled / wrong advance", insn, in, in, emulated);
				continue;
			}
			RegisterFile model = in;
			Model(insn, model);
			if (!Same(model, emulated)) {
				if (model_mismatch++ < 5) {
					PrintMismatch("emulator vs AMD model", insn, in, model, emulated);
				}
			}
			if (!sse4a) {
				continue;
			}
			const RegisterFile cpu = RunRoutine(page.Load(RegisterFileRoutine(insn.bytes)), in);
			undefined_cases += !defined;
			if (!Same(cpu, emulated)) {
				undefined_cpu_mismatch += !defined;
				if (cpu_mismatch++ < 5) {
					PrintMismatch("emulator vs AMD CPU", insn, in, cpu, emulated);
				}
			}
		}
	}
	Check(model_mismatch == 0, "runtime emulator matches the AMD model (all 16 XMM registers, flags)");
	Check(cpu_mismatch == 0, "runtime emulator matches the AMD CPU bit for bit (all 16 XMM registers, flags)");
	std::printf("emulator on CONTEXT: %d cases, %d model mismatches, %d CPU mismatches%s "
	            "(length+index>64: %d CPU mismatches in %d cases)\n",
	            cases, model_mismatch, cpu_mismatch, sse4a ? "" : " (no SSE4a: CPU not compared)",
	            undefined_cpu_mismatch, undefined_cases);
}

// ---- 2. Runtime emulator through a real exception (CONTEXT write-back by Windows) ------------------

void TestEmulatorThroughException(bool sse4a) {
	std::mt19937_64 rng(0xe8ce97);
	CodePage        page(4096);
	int             mismatches = 0;
	int             cases      = 0;
	g_skip_ud2.store(true);
	for (Form form: {Form::ExtrqImm, Form::ExtrqReg, Form::InsertqImm, Form::InsertqReg}) {
		for (int n = 0; n < 400; n++) {
			const bool   defined_only = n % 2 == 0;
			const Insn   insn         = RandomInsn(rng, form, defined_only);
			RegisterFile in {};
			RandomRegisters(rng, insn, defined_only, in);
			std::vector<uint8_t> body {0x0f, 0x0b};
			body.insert(body.end(), insn.bytes.begin(), insn.bytes.end());
			const int          before = g_illegal_hits.load();
			const RegisterFile got    = RunRoutine(page.Load(RegisterFileRoutine(body)), in);
			Check(g_illegal_hits.load() == before + 1, "one #UD per emulated instruction");
			RegisterFile expected = in;
			if (sse4a) {
				expected = RunRoutine(page.Load(RegisterFileRoutine(insn.bytes)), in);
			} else {
				Model(insn, expected);
			}
			cases++;
			if (!Same(expected, got) && mismatches++ < 5) {
				PrintMismatch("emulated through #UD", insn, in, expected, got);
			}
		}
	}
	g_skip_ud2.store(false);
	Check(mismatches == 0, "Windows restores every emulated XMM register and the flags from the CONTEXT");
	std::printf("emulator through a real #UD: %d cases, %d mismatches\n", cases, mismatches);
}

// ---- 3. Load-time patch (native trampolines) vs the CPU --------------------------------------------

struct Patched {
	uint8_t*                            area = nullptr;
	Loader::GuestInstructionPatchResult result {};
};

constexpr size_t PatchAreaCode       = 64 * 1024;
constexpr size_t PatchAreaTrampoline = 64 * 1024;

// The patcher as the loader runs it for a host without SSE4a/RDPID/CLWB (an Intel CPU).
Patched PatchCode(const std::vector<uint8_t>& code, uint8_t* area) {
	std::memset(area, 0xcc, PatchAreaCode + PatchAreaTrampoline);
	std::memcpy(area, code.data(), code.size());
	Loader::RegisterGuestInstructionPatchModule(area, PatchAreaCode, area + PatchAreaCode, PatchAreaTrampoline);
	const std::array<uintptr_t, 1>            starts {reinterpret_cast<uintptr_t>(area)};
	const Loader::GuestInstructionHostFeatures intel {false, false, false};
	Patched                                   patched;
	patched.area   = area;
	patched.result = Loader::PatchGuestInstructions(reinterpret_cast<uintptr_t>(area), code.size(), starts,
	                                                false, true, intel);
	Loader::UnregisterGuestInstructionPatchModule(area);
	FlushInstructionCache(GetCurrentProcess(), area, PatchAreaCode + PatchAreaTrampoline);
	return patched;
}

void TestPatchedExtrqRandomized(bool sse4a) {
	std::mt19937_64 rng(0x7a7c4ed);
	auto*           area = static_cast<uint8_t*>(VirtualAlloc(nullptr, PatchAreaCode + PatchAreaTrampoline,
	                                                          MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	Require(area != nullptr, "allocate the patch area");
	CodePage page(4096);
	int      cases = 0;
	int      mismatches = 0;
	int      not_native = 0;
	for (Form form: {Form::ExtrqImm, Form::ExtrqReg}) {
		for (int n = 0; n < 300; n++) {
			const bool   defined_only = n % 2 == 0;
			const Insn   insn         = RandomInsn(rng, form, defined_only);
			RegisterFile in {};
			RandomRegisters(rng, insn, defined_only, in);
			const auto patched = PatchCode(RegisterFileRoutine(insn.bytes), area);
			if (patched.result.extrq.found != 1 || patched.result.extrq.native != 1) {
				not_native++;
				continue;
			}
			const int          before = g_illegal_hits.load();
			const RegisterFile got    = RunRoutine(patched.area, in);
			Check(g_illegal_hits.load() == before, "a native EXTRQ trampoline raises no #UD");
			RegisterFile expected = in;
			if (sse4a) {
				expected = RunRoutine(page.Load(RegisterFileRoutine(insn.bytes)), in);
			} else {
				Model(insn, expected);
			}
			bool     handled = false;
			uint64_t advance = 0;
			const RegisterFile emulated = RunEmulatorOnContext(insn, in, handled, advance);
			cases++;
			if ((!Same(expected, got) || !Same(emulated, got)) && mismatches++ < 5) {
				PrintMismatch("patched EXTRQ vs CPU", insn, in, expected, got);
			}
		}
	}
	VirtualFree(area, 0, MEM_RELEASE);
	Check(not_native == 0, "the patcher gives every EXTRQ a native trampoline");
	Check(mismatches == 0, "patched EXTRQ equals the CPU and the runtime emulator (16 XMM registers, flags)");
	std::printf("patched EXTRQ trampolines: %d cases, %d mismatches, %d without trampoline\n", cases, mismatches,
	            not_native);
}

// INSERTQ stays the original instruction (it traps by design) inside a red-zone guarded span.
void TestPatchedInsertqKeepsTrap() {
	auto* area = static_cast<uint8_t*>(VirtualAlloc(nullptr, PatchAreaCode + PatchAreaTrampoline,
	                                                MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	Require(area != nullptr, "allocate the patch area");
	Insn insn;
	insn.form   = Form::InsertqReg;
	insn.dest   = 10;
	insn.source = 9;
	Encode(insn);
	const auto patched = PatchCode(RegisterFileRoutine(insn.bytes), area);
	Check(patched.result.insertq.found == 1 && patched.result.insertq.trapped == 1,
	      "INSERTQ is kept as a trap by the patcher");
	VirtualFree(area, 0, MEM_RELEASE);
}

// ---- 4. VRSQRTPS: trap emulation vs the patcher's trampoline ---------------------------------------

// rcx = 8 input floats (only the low 4 are used by the 128-bit form), rdx = 8 output floats:
// vmovups ymm1, [rcx]; vmovups xmm0, [rcx]; body; vmovups [rdx], ymm1; vzeroupper; ret.
std::vector<uint8_t> RsqrtRoutine(const std::vector<uint8_t>& body) {
	Emitter e;
	e.Raw({0xc5, 0xfc, 0x10, 0x09}); // vmovups ymm1, [rcx]: dirty the upper half
	e.Raw({0xc5, 0xf8, 0x10, 0x01}); // vmovups xmm0, [rcx]
	e.Filler();
	e.Raw(body);
	e.Filler();
	e.Raw({0xc5, 0xfc, 0x11, 0x0a}); // vmovups [rdx], ymm1
	e.Raw({0xc5, 0xf8, 0x77});       // vzeroupper
	e.Raw({0xc3});
	return e.bytes;
}

using RsqrtFn = void (*)(const uint32_t* in, uint32_t* out);

uint32_t RandomFloatBits(std::mt19937_64& rng) {
	static constexpr uint32_t specials[] {0x00000000u, 0x80000000u, 0x00000001u, 0x807fffffu, 0x00800000u,
	                                      0x7f7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00001u,
	                                      0x7f800001u, 0x3f800000u, 0xbf800000u, 0x40800000u};
	switch (rng() % 4) {
		case 0: return specials[rng() % std::size(specials)];
		case 1: return static_cast<uint32_t>(rng()) & 0x807fffffu; // denormal
		default: return static_cast<uint32_t>(rng());
	}
}

void TestReciprocalSquareRoot() {
	auto* area = static_cast<uint8_t*>(VirtualAlloc(nullptr, PatchAreaCode + PatchAreaTrampoline,
	                                                MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	Require(area != nullptr, "allocate the patch area");
	CodePage trap_page(4096);
	// vrsqrtps xmm1, xmm0 with the patcher's trap mark (VEX.vvvv bit cleared): #UD on every CPU.
	const auto trap = reinterpret_cast<RsqrtFn>(trap_page.Load(RsqrtRoutine({0xc5, 0xf0, 0x52, 0xc8})));
	const auto patched = PatchCode(RsqrtRoutine({0xc5, 0xf8, 0x52, 0xc8}), area);
	Check(patched.result.reciprocal_sqrt.found == 1 && patched.result.reciprocal_sqrt.native == 1,
	      "VRSQRTPS gets a native trampoline");
	const auto      native = reinterpret_cast<RsqrtFn>(patched.area);
	std::mt19937_64 rng(0x125a77);
	int             mismatches = 0;
	int             upper_dirty = 0;
	for (int n = 0; n < 20000; n++) {
		alignas(32) uint32_t in[8];
		for (auto& v: in) {
			v = RandomFloatBits(rng);
		}
		alignas(32) uint32_t a[8] {};
		alignas(32) uint32_t b[8] {};
		trap(in, a);
		native(in, b);
		if (std::memcmp(a, b, 16) != 0 && mismatches++ < 5) {
			std::fprintf(stderr, "  VRSQRTPS in %08x %08x %08x %08x trap %08x %08x %08x %08x trampoline %08x %08x %08x %08x\n",
			             in[0], in[1], in[2], in[3], a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
		}
		upper_dirty += (a[4] | a[5] | a[6] | a[7] | b[4] | b[5] | b[6] | b[7]) != 0;
	}
	VirtualFree(area, 0, MEM_RELEASE);
	Check(mismatches == 0, "VRSQRTPS: the trap emulator and the trampoline give the same bits");
	Check(upper_dirty == 0, "VRSQRTPS (VEX.128) zeroes YMM bits 255:128 on both paths");
	std::printf("VRSQRTPS trap vs trampoline: 20000 cases, %d mismatches\n", mismatches);
}

// ---- 5. The guest red zone ----------------------------------------------------------------------

// rcx = 128-byte pattern, rdx = out (128 bytes of red zone + xmm1 + xmm2), r8 = xmm0..2 inputs.
// Fills [rsp-128, rsp), runs `body`, reads the red zone back: the red zone is live across `body`.
std::vector<uint8_t> RedZoneRoutine(const std::vector<uint8_t>& body) {
	Emitter e;
	for (int k = 0; k < 8; k++) {
		e.Movdqu(true, 0, RCX, 16 * k);
		e.Movdqu(false, 0, RSP, -128 + 16 * k);
	}
	e.Movdqu(true, 0, R8, 0);
	e.Movdqu(true, 1, R8, 16);
	e.Movdqu(true, 2, R8, 32);
	e.Filler();
	e.Raw(body);
	e.Filler();
	e.Movdqu(false, 1, RDX, 128);
	e.Movdqu(false, 2, RDX, 144);
	for (int k = 0; k < 8; k++) {
		e.Movdqu(true, 0, RSP, -128 + 16 * k);
		e.Movdqu(false, 0, RDX, 16 * k);
	}
	e.Raw({0xc3});
	return e.bytes;
}

using RedZoneFn = void (*)(const uint8_t* pattern, uint8_t* out, const Xmm* inputs);

int RedZoneBytesChanged(void* code, Xmm* result = nullptr) {
	alignas(16) uint8_t pattern[128];
	for (int i = 0; i < 128; i++) {
		pattern[i] = static_cast<uint8_t>(0xa5 ^ i);
	}
	alignas(16) uint8_t out[160] {};
	alignas(16) Xmm     inputs[3] {{{0x3f8000003f800000ull, 0x4080000040800000ull}},
	                               {{0x1122334455667788ull, 0x99aabbccddeeff00ull}},
	                               {{0x0000000000001810ull, 0ull}}};
	reinterpret_cast<RedZoneFn>(code)(pattern, out, inputs);
	if (result != nullptr) {
		std::memcpy(result, out + 128, sizeof(Xmm));
	}
	int changed = 0;
	for (int i = 0; i < 128; i++) {
		changed += out[i] != pattern[i];
	}
	return changed;
}

// Windows builds the exception frame from RSP down with the XSAVE area on top, ending about 40-60
// bytes below RSP, and does not write state components that are in their initial state. On a CPU
// with AVX-512 the top components are AVX-512 state, idle in guest code, so the red zone survives;
// on one without (Intel Core 12th-14th gen) the top component is the YMM upper halves, live in AVX
// code. ZMM16 made live here stands in for that.
bool HostHasAvx512() {
	int regs[4] {};
	__cpuidex(regs, 7, 0);
	if ((regs[1] & (1 << 16)) == 0) {
		return false;
	}
	__cpuid(regs, 1);
	return (regs[2] & (1 << 27)) != 0 && (_xgetbv(0) & 0xe0) == 0xe0;
}

void TestRedZone(bool sse4a) {
	auto* area = static_cast<uint8_t*>(VirtualAlloc(nullptr, PatchAreaCode + PatchAreaTrampoline,
	                                                MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	Require(area != nullptr, "allocate the patch area");
	CodePage page(4096);
	// extrq xmm1, xmm2: controls 0x1810 = length 16, index 24.
	const std::vector<uint8_t> extrq {0x66, 0x0f, 0x79, 0xca};
	const bool                 avx512 = HostHasAvx512();
	// vpternlogd zmm16, zmm16, zmm16, 0xff
	const std::vector<uint8_t> live_zmm16 {0x62, 0xa3, 0x7d, 0x40, 0x25, 0xc0, 0xff};

	// Without the patch, on an Intel CPU: EXTRQ raises #UD and the handler emulates it. The UD2 in
	// front stands in for that #UD on this CPU.
	const auto trap_changed = [&](bool top_component_live) {
		std::vector<uint8_t> body;
		if (top_component_live) {
			body = live_zmm16;
		}
		body.insert(body.end(), {0x0f, 0x0b});
		body.insert(body.end(), extrq.begin(), extrq.end());
		g_skip_ud2.store(true);
		Xmm       trapped {};
		const int changed = RedZoneBytesChanged(page.Load(RedZoneRoutine(body)), &trapped);
		g_skip_ud2.store(false);
		Check(trapped.q[0] == 0x4455 && trapped.q[1] == 0, "EXTRQ emulated through #UD gives the field");
		return changed;
	};
	const int idle_changed = trap_changed(false);
	const int live_changed = avx512 ? trap_changed(true) : -1;

	// With the patch (--amd-cpu): the native trampoline keeps the red zone.
	for (bool top_component_live: {false, true}) {
		if (top_component_live && !avx512) {
			continue;
		}
		std::vector<uint8_t> body;
		if (top_component_live) {
			body = live_zmm16;
		}
		body.insert(body.end(), extrq.begin(), extrq.end());
		const auto patched = PatchCode(RedZoneRoutine(body), area);
		Check(patched.result.extrq.found == 1 && patched.result.extrq.native == 1,
		      "EXTRQ with a live red zone gets a native trampoline");
		Xmm       native {};
		const int before = g_illegal_hits.load();
		Check(RedZoneBytesChanged(patched.area, &native) == 0,
		      "the EXTRQ trampoline preserves all 128 red-zone bytes");
		Check(g_illegal_hits.load() == before, "the EXTRQ trampoline raises no #UD");
		Check(native.q[0] == 0x4455 && native.q[1] == 0, "the EXTRQ trampoline gives the field");
	}

	if (sse4a) {
		Check(RedZoneBytesChanged(page.Load(RedZoneRoutine(extrq))) == 0,
		      "native EXTRQ on an AMD CPU keeps the red zone");
	}
	VirtualFree(area, 0, MEM_RELEASE);
	// Windows' behaviour, not Kyty's: recorded, not required. It is why the patch guards every
	// instruction that still traps, and why --amd-cpu is on by default without an AMD CPU.
	std::printf("red zone: EXTRQ emulated through #UD changed %d of the 128 bytes below RSP with the top "
	            "XSAVE component idle, %d with it live%s; the patched EXTRQ changed none\n",
	            idle_changed, live_changed, avx512 ? "" : " (no AVX-512: not measured)");
}

// ---- 6. The automatic --amd-cpu decision -------------------------------------------------------

void TestVendorDecision() {
	using Loader::X64InstructionEmulator::IsAmdCpuVendor;
	Check(IsAmdCpuVendor("AuthenticAMD"), "AuthenticAMD is AMD");
	Check(IsAmdCpuVendor("HygonGenuine"), "Hygon (Zen) counts as AMD");
	Check(!IsAmdCpuVendor("GenuineIntel"), "GenuineIntel is not AMD");
	Check(!IsAmdCpuVendor("CentaurHauls"), "other vendors are not AMD");
	Check(!IsAmdCpuVendor(""), "an empty vendor is not AMD");
	const auto vendor = Loader::X64InstructionEmulator::HostCpuVendor();
	Check(vendor.size() == 12, "CPUID leaf 0 gives a 12-character vendor");
	std::printf("host CPU vendor: %s (SSE4a: %s)\n", vendor.c_str(), HostHasSse4a() ? "yes" : "no");
}

} // namespace

int main() {
	Require(Common::HostException::InstallHandler(Handler), "install the host exception handler");
	const bool sse4a = HostHasSse4a();
	TestVendorDecision();
	TestEmulatorRandomized(sse4a);
	TestEmulatorThroughException(sse4a);
	TestPatchedExtrqRandomized(sse4a);
	TestPatchedInsertqKeepsTrap();
	TestReciprocalSquareRoot();
	TestRedZone(sse4a);
	if (g_failures != 0) {
		std::fprintf(stderr, "AmdCpuEmulationTests: %d checks failed\n", g_failures);
		return 1;
	}
	std::printf("AmdCpuEmulationTests: all passed\n");
	return 0;
}
