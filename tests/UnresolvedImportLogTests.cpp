#include "loader/unresolvedImportLog.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

namespace L = Loader::UnresolvedImportLog;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "UnresolvedImportLogTests: failed: %s\n", message);
		std::abort();
	}
}

static_assert(L::ReportFirst(1) && !L::ReportFirst(0) && !L::ReportFirst(2));
static_assert(!L::ReportCount(1) && !L::ReportCount(10) && !L::ReportCount(100));
static_assert(L::ReportCount(1000) && L::ReportCount(10000) && L::ReportCount(1000000));
static_assert(!L::ReportCount(1001) && !L::ReportCount(2000) && !L::ReportCount(999));

} // namespace

int main() {
	// One import called a million times: one console line, then count lines at 10^3..10^6.
	uint64_t first  = 0;
	uint64_t counts = 0;
	for (uint64_t call = 1; call <= 1000000; call++) {
		first += L::ReportFirst(call) ? 1 : 0;
		counts += L::ReportCount(call) ? 1 : 0;
	}
	Check(first == 1, "only the first call reaches the console");
	Check(counts == 4, "count lines at 1e3, 1e4, 1e5 and 1e6");
	Check(!L::ReportCount(UINT64_MAX), "no count line for a non-power of ten");
	std::printf("UnresolvedImportLogTests: all passed\n");
	return 0;
}
