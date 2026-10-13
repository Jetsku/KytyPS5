#ifndef EMULATOR_SRC_LOADER_UNRESOLVEDIMPORTLOG_H_
#define EMULATOR_SRC_LOADER_UNRESOLVEDIMPORTLOG_H_

#include <cstdint>

// Which calls of one unresolved import (runtimeLinker.cpp UnresolvedImportStub) are reported:
// the first on the console and in the log, so a frequent import no longer hides the others, then
// a count line in the log at 1,000, 10,000, 100,000 ... calls of that import.
namespace Loader::UnresolvedImportLog {

[[nodiscard]] constexpr bool ReportFirst(uint64_t call) {
	return call == 1;
}

[[nodiscard]] constexpr bool ReportCount(uint64_t call) {
	if (call < 1000) {
		return false;
	}
	while (call % 10 == 0) {
		call /= 10;
	}
	return call == 1;
}

} // namespace Loader::UnresolvedImportLog

#endif // EMULATOR_SRC_LOADER_UNRESOLVEDIMPORTLOG_H_
