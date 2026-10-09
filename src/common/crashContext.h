#ifndef EMULATOR_SRC_COMMON_CRASHCONTEXT_H_
#define EMULATOR_SRC_COMMON_CRASHCONTEXT_H_

#include <cinttypes>
#include <cstdint>
#include <cstdio>

// What the current thread is doing that a host crash report should name: the driver call in
// progress (a pipeline build), with up to two guest shader hashes. The unhandled-exception report
// (loader/runtimeLinker.cpp) prints the faulting thread's note, so a crash inside a driver or a
// Vulkan layer names the pipeline it was building. Kinds are string literals.
namespace Common::CrashContext {

struct Note {
	const char* kind = nullptr;
	uint64_t    a    = 0;
	uint64_t    b    = 0;
};

[[nodiscard]] inline Note& Current() noexcept {
	static thread_local Note note;
	return note;
}

// Sets the thread's note for its lifetime and restores the previous one.
class Scope {
public:
	Scope(const char* kind, uint64_t a, uint64_t b = 0) noexcept: m_saved(Current()) {
		Current() = {kind, a, b};
	}
	~Scope() { Current() = m_saved; }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Note m_saved;
};

// "graphics pipeline (optimized) VS 0x... PS 0x..." for the report, or an empty string.
inline void Describe(const Note& note, char* out, size_t size) {
	if (size == 0) return;
	if (note.kind == nullptr) {
		out[0] = '\0';
		return;
	}
	std::snprintf(out, size, "%s 0x%016" PRIx64 " 0x%016" PRIx64, note.kind, note.a, note.b);
}

} // namespace Common::CrashContext

#endif // EMULATOR_SRC_COMMON_CRASHCONTEXT_H_
