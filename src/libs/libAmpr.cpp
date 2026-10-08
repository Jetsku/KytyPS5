#include "common/abi.h"
#include "common/dateTime.h"
#include "common/file.h"
#include "common/hangTrace.h"
#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "kernel/eventQueue.h"
#include "kernel/fileSystem.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/amprCounterBank.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs {

namespace AprShared {

struct ResultBuffer {
	int32_t  result      = 0;
	uint32_t errorOffset = 0;
};

class LoadingSubmissionDiagnostic {
public:
	LoadingSubmissionDiagnostic(uint64_t kind, uint64_t command_buffer, uint64_t argument,
	                            uint64_t result_address = 0, uint64_t out_id_address = 0,
	                            uint64_t argument2 = 0) {
		m_enabled = Profiler::BeginLoadingAprSubmission(&m_record);
		if (m_enabled) {
			m_record.api_kind = kind;
			m_record.command_buffer = command_buffer;
			m_record.argument1 = argument;
			m_record.argument2 = argument2;
			m_record.result_address = result_address;
			m_record.out_id_address = out_id_address;
		}
	}
	~LoadingSubmissionDiagnostic() {
		if (m_enabled) Profiler::RecordLoadingAprSubmission(m_record);
	}
	Profiler::LoadingAprSubmission* Get() { return m_enabled ? &m_record : nullptr; }
	int Return(int result) {
		if (m_enabled) m_record.api_kernel_result = static_cast<uint32_t>(result);
		return result;
	}
	void SetId(uint32_t id) {
		if (m_enabled) m_record.submission_id = id;
	}
private:
	Profiler::LoadingAprSubmission m_record {};
	bool m_enabled = false;
};

struct SubmissionState {
	uint64_t command_buffer   = 0;
	uint64_t result           = 0;
	int32_t  execution_result = 0;
	uint32_t error_offset     = 0;
};

struct ResolvedPathInfo {
	int         result    = OK;
	uint32_t    file_id   = 0xffffffffu;
	uint64_t    file_size = 0;
	bool        is_dir    = false;
	std::string host_path;
};

static std::mutex                                        g_mutex;
static uint32_t                                          g_next_submission_id = 1;
static std::unordered_map<uint32_t, SubmissionState>     g_submissions;
static std::unordered_map<uint32_t, std::string>         g_files;
static std::unordered_map<uint32_t, uint64_t>            g_file_sizes;
static std::unordered_map<std::string, ResolvedPathInfo> g_resolved_paths;

static bool IsValidGuestRange(uint64_t addr, uint64_t size, bool write = false) {
	(void)write;
	if (size == 0) {
		return true;
	}
	return addr != 0 && addr <= std::numeric_limits<uint64_t>::max() - size;
}

static bool ReadGuestBytes(uint64_t addr, void* out, uint64_t size) {
	if (out == nullptr || !IsValidGuestRange(addr, size)) {
		return false;
	}
	std::memcpy(out, reinterpret_cast<const void*>(addr), static_cast<size_t>(size));
	return true;
}

static bool WriteGuestBytes(uint64_t addr, const void* in, uint64_t size) {
	if (in == nullptr || !IsValidGuestRange(addr, size, true)) {
		return false;
	}
	std::memcpy(reinterpret_cast<void*>(addr), in, static_cast<size_t>(size));
	return true;
}

template <typename T>
static bool ReadGuest(uint64_t addr, T* value) {
	return ReadGuestBytes(addr, value, sizeof(T));
}

template <typename T>
static bool WriteGuest(uint64_t addr, const T& value) {
	return WriteGuestBytes(addr, &value, sizeof(T));
}

// APR file ids are given one per resolved path (from chenxiao07 3248e3726). The FNV-1a hash of the
// guest path collided: Astro Bot's 168,875 /app0 entries hold 6 colliding pairs (one is the galaxy
// map's nx_ui_worldmap_bot_icon~~159.odxb with enemy_hand_ghost_anim_mat.anim), Demon's Souls' 13
// pairs drew "?" models; whichever path of a pair was resolved last served the reads of both.
// KYTY_APR_HASHED_IDS=1 restores the hashed ids.
static bool UseHashedFileIds() {
	static const bool hashed = [] {
		const char* v = std::getenv("KYTY_APR_HASHED_IDS");
		return v != nullptr && v[0] != '\0' && v[0] != '0';
	}();
	return hashed;
}

static uint32_t g_next_file_id = 1;

static uint32_t ComputeFileId(const char* guest_path) {
	uint32_t hash = 2166136261u;
	for (auto* p = reinterpret_cast<const uint8_t*>(guest_path); p != nullptr && *p != 0; ++p) {
		hash ^= *p;
		hash *= 16777619u;
	}
	return hash & static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
}

static int ReadGuestCString(uint64_t addr, char* out, size_t out_size) {
	if (addr == 0) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	const auto* src = reinterpret_cast<const char*>(addr);
	for (size_t pos = 0; pos < out_size; pos++) {
		if (addr > std::numeric_limits<uint64_t>::max() - pos) {
			return LibKernel::KERNEL_ERROR_EFAULT;
		}
		out[pos] = src[pos];
		if (out[pos] == '\0') {
			return OK;
		}
	}

	return LibKernel::KERNEL_ERROR_ENAMETOOLONG;
}

static void RegisterHostPathLocked(uint32_t file_id, const std::string& host_path,
                                   uint64_t file_size, bool is_dir) {
	g_files[file_id]      = host_path;
	g_file_sizes[file_id] = is_dir ? 0 : file_size;
}

static void RegisterHostPath(uint32_t file_id, const std::string& host_path, uint64_t file_size,
                             bool is_dir) {
	std::scoped_lock lock(g_mutex);
	RegisterHostPathLocked(file_id, host_path, file_size, is_dir);
}

static bool TryGetHostPath(uint32_t file_id, std::string* out) {
	std::scoped_lock lock(g_mutex);
	const auto       it = g_files.find(file_id);
	if (it == g_files.end()) {
		return false;
	}
	*out = it->second;
	return true;
}

static bool TryGetHostFileSize(uint32_t file_id, uint64_t* out) {
	std::scoped_lock lock(g_mutex);
	const auto       it = g_file_sizes.find(file_id);
	if (it == g_file_sizes.end()) {
		return false;
	}
	*out = it->second;
	return true;
}

static int GetHostPathStat(const std::string& host_path, LibKernel::FileSystem::FileStat* st) {
	if (st == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	const bool is_dir  = Common::File::IsDirectoryExisting(host_path);
	const bool is_file = Common::File::IsFileExisting(host_path);
	if (!is_dir && !is_file) {
		return LibKernel::KERNEL_ERROR_ENOENT;
	}

	LibKernel::FileSystem::FileStat stat {};
	stat.st_mode = 0000777u | (is_dir ? 0040000u : 0100000u);

	auto at = Common::DateTime::FromSystemUTC();
	auto wt = at;

	if (is_dir) {
		stat.st_size    = 0;
		stat.st_blksize = 512;
		stat.st_blocks  = 0;
	} else {
		stat.st_size    = static_cast<int64_t>(Common::File::Size(host_path));
		stat.st_blksize = 512;
		stat.st_blocks  = (stat.st_size + 511) / 512;

		Common::File::GetLastAccessAndWriteTimeUTC(host_path, &at, &wt);
	}

	stat.st_atim.tv_sec  = static_cast<int64_t>(at.ToUnix());
	stat.st_atim.tv_nsec = static_cast<int64_t>(
	    (at.ToUnix() - static_cast<double>(stat.st_atim.tv_sec)) * 1000000000.0);
	stat.st_mtim.tv_sec  = static_cast<int64_t>(wt.ToUnix());
	stat.st_mtim.tv_nsec = static_cast<int64_t>(
	    (wt.ToUnix() - static_cast<double>(stat.st_mtim.tv_sec)) * 1000000000.0);
	stat.st_ctim     = stat.st_atim;
	stat.st_birthtim = stat.st_mtim;
	*st              = stat;

	return OK;
}

static int ResolveOnePath(const char* guest_path, uint32_t* id, uint64_t* size) {
	if (guest_path == nullptr || guest_path[0] == '\0') {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	ResolvedPathInfo  info {};
	const std::string path(guest_path);
	bool              found = false;
	{
		std::scoped_lock lock(g_mutex);
		const auto       it = g_resolved_paths.find(path);
		if (it != g_resolved_paths.end()) {
			info  = it->second;
			found = true;
		}
	}

	if (!found) {
		const auto real_path = LibKernel::FileSystem::GetRealFilename(path);
		info.file_id         = AprShared::ComputeFileId(guest_path);
		info.host_path       = Common::PathToString(real_path);

		if (Common::File::IsDirectoryExisting(real_path)) {
			info.is_dir    = true;
			info.file_size = 0x10000;
		} else if (Common::File::IsFileExisting(real_path)) {
			info.file_size = Common::File::Size(real_path);
		} else {
			info.result = LibKernel::KERNEL_ERROR_ENOENT;
		}

		bool log_missing = false;
		{
			std::scoped_lock lock(g_mutex);
			auto [it, inserted] = g_resolved_paths.emplace(path, info);
			if (!inserted) {
				info = it->second;
			} else if (info.result == OK) {
				if (!AprShared::UseHashedFileIds()) {
					info.file_id = it->second.file_id = AprShared::g_next_file_id++;
				}
				RegisterHostPathLocked(info.file_id, info.host_path, info.file_size, info.is_dir);
			} else if (info.result == LibKernel::KERNEL_ERROR_ENOENT) {
				log_missing = true;
			}
		}
		if (log_missing) {
			LOGF("\tAPR resolve missing path: %s -> %s\n", guest_path, info.host_path.c_str());
		}
	} else if (info.result == OK) {
		AprShared::RegisterHostPath(info.file_id, info.host_path, info.file_size, info.is_dir);
	}

	if (info.result != OK) {
		return info.result;
	}

	if (id != nullptr) {
		if (!AprShared::IsValidGuestRange(reinterpret_cast<uint64_t>(id), sizeof(*id), true)) {
			return LibKernel::KERNEL_ERROR_EFAULT;
		}
		*id = info.file_id;
	}

	if (size != nullptr) {
		if (!AprShared::IsValidGuestRange(reinterpret_cast<uint64_t>(size), sizeof(*size), true)) {
			return LibKernel::KERNEL_ERROR_EFAULT;
		}
		*size = info.file_size;
	}

	return OK;
}

static int ResolvePathsCommon(const char* const* path_list, uint32_t count, uint32_t* ids,
                              uint64_t* sizes, uint32_t* error_index, const char* prefix, int* results) {
	if (path_list == nullptr || count == 0 || count > 1024 ||
	    (ids == nullptr && sizes == nullptr && results == nullptr)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	if (error_index != nullptr &&
	    !AprShared::IsValidGuestRange(reinterpret_cast<uint64_t>(error_index), sizeof(*error_index),
	                                  true)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	if (ids != nullptr && !AprShared::IsValidGuestRange(reinterpret_cast<uint64_t>(ids),
	                                                    count * sizeof(*ids), true)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	if (sizes != nullptr && !AprShared::IsValidGuestRange(reinterpret_cast<uint64_t>(sizes),
	                                                      count * sizeof(*sizes), true)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	if (results != nullptr && !AprShared::IsValidGuestRange(reinterpret_cast<uint64_t>(results),
	                                                        count * sizeof(*results), true)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	if (!IsValidGuestRange(reinterpret_cast<uint64_t>(path_list), count * sizeof(*path_list))) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	char prefix_buf[1024] {};
	if (prefix != nullptr) {
		const int result = ReadGuestCString(reinterpret_cast<uint64_t>(prefix), prefix_buf,
		                                   sizeof(prefix_buf));
		if (result != OK) {
			return result;
		}
	}
	const auto prefix_size   = std::strlen(prefix_buf);
	uint32_t   success_count = 0;
	for (uint32_t i = 0; i < count; i++) {
		char resolved_path[1024] {};
		// Concatenate the prefix literally, including an empty prefix.
		std::memcpy(resolved_path, prefix_buf, prefix_size);
		int result = ReadGuestCString(reinterpret_cast<uint64_t>(path_list[i]),
		                             resolved_path + prefix_size, sizeof(resolved_path) - prefix_size);
		if (result == OK) {
			result = ResolveOnePath(resolved_path, ids != nullptr ? &ids[i] : nullptr,
			                        sizes != nullptr ? &sizes[i] : nullptr);
		}

		if (results != nullptr) {
			results[i] = result;
		}
		if (result == OK) {
			success_count++;
		} else {
			if (ids != nullptr) {
				ids[i] = 0xffffffffu;
			}
			if (sizes != nullptr) {
				sizes[i] = 0;
			}
			if (error_index != nullptr) {
				*error_index = static_cast<uint32_t>(i);
			}
			if (results == nullptr) {
				return result;
			}
		}
	}

	return results != nullptr ? static_cast<int>(success_count) : OK;
}

static uint32_t AllocateSubmissionId(uint64_t command_buffer, uint64_t result) {
	std::scoped_lock lock(g_mutex);

	auto id = g_next_submission_id++;
	if (id == 0) {
		id = g_next_submission_id++;
	}
	g_submissions[id] = SubmissionState {command_buffer, result};
	return id;
}

static void SetSubmissionResult(uint32_t submission_id, int32_t execution_result,
                                uint32_t error_offset) {
	std::scoped_lock lock(g_mutex);
	auto             it = g_submissions.find(submission_id);
	if (it != g_submissions.end()) {
		it->second.execution_result = execution_result;
		it->second.error_offset     = error_offset;
	}
}

static bool CompleteSubmission(uint32_t submission_id, SubmissionState* state) {
	std::scoped_lock lock(g_mutex);
	auto             it = g_submissions.find(submission_id);
	if (it == g_submissions.end()) {
		return false;
	}
	if (state != nullptr) {
		*state = it->second;
	}
	g_submissions.erase(it);
	return true;
}

static int WriteResult(void* result, int32_t execution_result = 0, uint32_t error_offset = 0) {
	if (result == nullptr) {
		return OK;
	}
	ResultBuffer res {};
	res.result      = execution_result;
	res.errorOffset = error_offset;
	if (!WriteGuest(reinterpret_cast<uint64_t>(result), res)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	return OK;
}

} // namespace AprShared

namespace LibAmpr::Ampr {
static int ExecuteAprCommandBuffer(uint64_t command_buffer, int32_t* execution_result,
                                   uint32_t* error_offset,
                                   Profiler::LoadingAprSubmission* diagnostic = nullptr);
}

namespace LibKernelApr {

LIB_VERSION("libkernel", 1, "libkernel", 1, 1);

namespace Apr {

// TODO: change helper name
static int KernelSyscallResult(int result) {
	if (result >= OK) {
		return result;
	}

	// TODO: actually __error() thread local
	*Posix::GetErrorAddr() = LibKernel::KernelToPosix(result);
	return -1;
}

static int KYTY_SYSV_ABI ResolveFilepathsToIds(const char* const* path_list, uint32_t count, uint32_t* ids,
                                               uint32_t* error_index) {
	PRINT_NAME();

	return KernelSyscallResult(AprShared::ResolvePathsCommon(path_list, count, ids, nullptr,
	                                                         error_index, nullptr, nullptr));
}

static int KYTY_SYSV_ABI GetFileStat(uint32_t file_id, LibKernel::FileSystem::FileStat* st) {
	PRINT_NAME();

	if (st == nullptr) {
		return KernelSyscallResult(LibKernel::KERNEL_ERROR_EINVAL);
	}

	std::string host_path;
	if (!AprShared::TryGetHostPath(file_id, &host_path)) {
		LOGF("\tAPR stat failed for unknown file id: 0x%08" PRIx32 "\n", file_id);
		return KernelSyscallResult(LibKernel::KERNEL_ERROR_ENOENT);
	}
	if (!AprShared::IsValidGuestRange(reinterpret_cast<uint64_t>(st), sizeof(*st), true)) {
		return KernelSyscallResult(LibKernel::KERNEL_ERROR_EFAULT);
	}

	return KernelSyscallResult(AprShared::GetHostPathStat(host_path, st));
}

static int KYTY_SYSV_ABI GetFileSize(uint32_t file_id, uint64_t* size) {
	PRINT_NAME();

	if (size == nullptr) {
		return KernelSyscallResult(LibKernel::KERNEL_ERROR_EINVAL);
	}
	if (!AprShared::IsValidGuestRange(reinterpret_cast<uint64_t>(size), sizeof(*size), true)) {
		return KernelSyscallResult(LibKernel::KERNEL_ERROR_EFAULT);
	}

	std::string host_path;
	if (!AprShared::TryGetHostPath(file_id, &host_path)) {
		LOGF("\tAPR size failed for unknown file id: 0x%08" PRIx32 "\n", file_id);
		return KernelSyscallResult(LibKernel::KERNEL_ERROR_ENOENT);
	}

	uint64_t cached_size = 0;
	if (AprShared::TryGetHostFileSize(file_id, &cached_size)) {
		*size = cached_size;
		return OK;
	}

	if (Common::File::IsDirectoryExisting(host_path)) {
		*size = 0;
		return OK;
	}
	if (!Common::File::IsFileExisting(host_path)) {
		return KernelSyscallResult(LibKernel::KERNEL_ERROR_ENOENT);
	}

	*size = Common::File::Size(host_path);
	return OK;
}

static int KYTY_SYSV_ABI ResolveFilepathsToIdsAndFileSizes(const char* const* path_list, uint32_t count,
                                                           uint32_t* ids, uint64_t* sizes,
                                                           uint32_t* error_index) {
	PRINT_NAME();

	return KernelSyscallResult(
	    AprShared::ResolvePathsCommon(path_list, count, ids, sizes, error_index, nullptr, nullptr));
}

static int KYTY_SYSV_ABI ResolveFilepathsWithPrefixToIds(const char* prefix, const char* const* path_list,
                                                         uint32_t count, uint32_t* ids,
                                                         uint32_t* error_index) {
	PRINT_NAME();

	return KernelSyscallResult(AprShared::ResolvePathsCommon(path_list, count, ids, nullptr,
	                                                         error_index, prefix, nullptr));
}

static int KYTY_SYSV_ABI ResolveFilepathsWithPrefixToIdsAndFileSizes(const char* prefix,
                                                                     const char* const* path_list,
                                                                     uint32_t count, uint32_t* ids,
                                                                     uint64_t* sizes,
                                                                     uint32_t* error_index) {
	PRINT_NAME();

	return KernelSyscallResult(AprShared::ResolvePathsCommon(path_list, count, ids, sizes,
	                                                         error_index, prefix, nullptr));
}

static int KYTY_SYSV_ABI ResolveFilepathsToIdsForEach(const char* const* path_list, uint32_t count,
                                                      uint32_t* ids, int* results) {
	PRINT_NAME();

	return KernelSyscallResult(
	    AprShared::ResolvePathsCommon(path_list, count, ids, nullptr, nullptr, nullptr, results));
}

static int KYTY_SYSV_ABI ResolveFilepathsToIdsAndFileSizesForEach(const char* const* path_list,
                                                                  uint32_t count, uint32_t* ids,
                                                                  uint64_t* sizes, int* results) {
	PRINT_NAME();

	return KernelSyscallResult(
	    AprShared::ResolvePathsCommon(path_list, count, ids, sizes, nullptr, nullptr, results));
}

static int KYTY_SYSV_ABI ResolveFilepathsWithPrefixToIdsForEach(const char* prefix,
                                                                const char* const* path_list,
                                                                uint32_t count, uint32_t* ids,
                                                                int* results) {
	PRINT_NAME();

	return KernelSyscallResult(AprShared::ResolvePathsCommon(path_list, count, ids, nullptr,
	                                                         nullptr, prefix, results));
}

static int KYTY_SYSV_ABI ResolveFilepathsWithPrefixToIdsAndFileSizesForEach(
    const char* prefix, const char* const* path_list, uint32_t count, uint32_t* ids, uint64_t* sizes,
    int* results) {
	PRINT_NAME();

	return KernelSyscallResult(
	    AprShared::ResolvePathsCommon(path_list, count, ids, sizes, nullptr, prefix, results));
}

static int KYTY_SYSV_ABI SubmitCommandBufferAndGetResult(void*     command_buffer, uint64_t argument,
                                                         void*     result,
                                                         uint32_t* out_submission_id) {
	PRINT_NAME();
	HangTrace::SetGuestCaller(reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprSubmitAndGetResult);
	AprShared::LoadingSubmissionDiagnostic diagnostic(
	    1, reinterpret_cast<uint64_t>(command_buffer), argument, reinterpret_cast<uint64_t>(result),
	    reinterpret_cast<uint64_t>(out_submission_id));

	if (command_buffer == nullptr) {
		return KernelSyscallResult(diagnostic.Return(LibKernel::KERNEL_ERROR_EINVAL));
	}

	const auto command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	const auto id =
	    AprShared::AllocateSubmissionId(command_buffer_addr, reinterpret_cast<uint64_t>(result));
	diagnostic.SetId(id);

	int32_t  execution_result = OK;
	uint32_t error_offset     = 0;
	auto submit_result = LibAmpr::Ampr::ExecuteAprCommandBuffer(command_buffer_addr,
	                                                            &execution_result, &error_offset,
	                                                            diagnostic.Get());
	if (submit_result != OK) {
		return KernelSyscallResult(diagnostic.Return(submit_result));
	}
	AprShared::SetSubmissionResult(id, execution_result, error_offset);

	if (out_submission_id != nullptr) {
		if (!AprShared::WriteGuest(reinterpret_cast<uint64_t>(out_submission_id), id)) {
			return KernelSyscallResult(diagnostic.Return(LibKernel::KERNEL_ERROR_EFAULT));
		}
	}

	return KernelSyscallResult(
	    diagnostic.Return(AprShared::WriteResult(result, execution_result, error_offset)));
}

static int KYTY_SYSV_ABI SubmitCommandBuffer(void* command_buffer, uint64_t argument) {
	PRINT_NAME();
	HangTrace::SetGuestCaller(reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprSubmitPlain);
	AprShared::LoadingSubmissionDiagnostic diagnostic(
	    0, reinterpret_cast<uint64_t>(command_buffer), argument);

	if (command_buffer == nullptr) {
		return KernelSyscallResult(diagnostic.Return(LibKernel::KERNEL_ERROR_EINVAL));
	}

	int32_t  execution_result = OK;
	uint32_t error_offset     = 0;
	return KernelSyscallResult(diagnostic.Return(LibAmpr::Ampr::ExecuteAprCommandBuffer(
	    reinterpret_cast<uint64_t>(command_buffer), &execution_result, &error_offset, diagnostic.Get())));
}

static int KYTY_SYSV_ABI SubmitCommandBufferAndGetId(void*     command_buffer, uint64_t argument,
                                                     uint32_t* out_submission_id) {
	PRINT_NAME();
	HangTrace::SetGuestCaller(reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprSubmitAndGetId);
	AprShared::LoadingSubmissionDiagnostic diagnostic(
	    2, reinterpret_cast<uint64_t>(command_buffer), argument, 0,
	    reinterpret_cast<uint64_t>(out_submission_id));

	if (command_buffer == nullptr || out_submission_id == nullptr) {
		return KernelSyscallResult(diagnostic.Return(LibKernel::KERNEL_ERROR_EINVAL));
	}

	const auto command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	const auto id                  = AprShared::AllocateSubmissionId(command_buffer_addr, 0);
	diagnostic.SetId(id);

	int32_t  execution_result = OK;
	uint32_t error_offset     = 0;
	auto submit_result = LibAmpr::Ampr::ExecuteAprCommandBuffer(command_buffer_addr,
	                                                            &execution_result, &error_offset,
	                                                            diagnostic.Get());
	if (submit_result != OK) {
		return KernelSyscallResult(diagnostic.Return(submit_result));
	}
	AprShared::SetSubmissionResult(id, execution_result, error_offset);

	if (!AprShared::WriteGuest(reinterpret_cast<uint64_t>(out_submission_id), id)) {
		return KernelSyscallResult(diagnostic.Return(LibKernel::KERNEL_ERROR_EFAULT));
	}

	return diagnostic.Return(OK);
}

static int KYTY_SYSV_ABI WaitCommandBuffer(uint32_t submission_id) {
	PRINT_NAME();

	AprShared::SubmissionState state {};
	if (!AprShared::CompleteSubmission(submission_id, &state)) {
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprWaitUnknownId);
		return KernelSyscallResult(LibKernel::KERNEL_ERROR_ESRCH);
	}
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprWaitCompleted);
	if (state.result != 0) {
		return KernelSyscallResult(AprShared::WriteResult(
		    reinterpret_cast<void*>(state.result), state.execution_result, state.error_offset));
	}
	return OK;
}

} // namespace Apr

LIB_DEFINE(InitLibKernel_1_Apr) {
	LIB_FUNC("WT-5NKy42fw", Apr::ResolveFilepathsToIds);
	LIB_FUNC("ApkYaHb8Sek", Apr::GetFileStat);
	LIB_FUNC("WvEu7yl3Ivg", Apr::GetFileSize);
	LIB_FUNC("gEpBkcwxUjw", Apr::ResolveFilepathsToIdsAndFileSizes);
	LIB_FUNC("i3HWvW35jao", Apr::ResolveFilepathsWithPrefixToIds);
	LIB_FUNC("w5fcCG+t31g", Apr::ResolveFilepathsWithPrefixToIdsAndFileSizes);
	LIB_FUNC("eYAh2vlCY-U", Apr::ResolveFilepathsToIdsForEach);
	LIB_FUNC("QzB4O+bJQyA", Apr::ResolveFilepathsToIdsAndFileSizesForEach);
	LIB_FUNC("VB-BtuIW8Xc", Apr::ResolveFilepathsWithPrefixToIdsForEach);
	LIB_FUNC("C+Khtbbx2g8", Apr::ResolveFilepathsWithPrefixToIdsAndFileSizesForEach);
	LIB_FUNC("ASoW5WE-UPo", Apr::SubmitCommandBufferAndGetResult);
	LIB_FUNC("rqwFKI4PAiM", Apr::WaitCommandBuffer);
	LIB_FUNC("eE4Szl8sil8", Apr::SubmitCommandBuffer);
	LIB_FUNC("qvMUCyyaCSI", Apr::SubmitCommandBufferAndGetId);
}

} // namespace LibKernelApr

namespace LibAmpr {

LIB_VERSION("Ampr", 1, "Ampr", 1, 1);

namespace Ampr {

constexpr uint64_t COMMAND_BUFFER_TYPE_OFFSET    = 0x00;
constexpr uint64_t COMMAND_BUFFER_OFFSET_OFFSET  = 0x04;
constexpr uint64_t COMMAND_BUFFER_NUM_OFFSET     = 0x08;
constexpr uint64_t COMMAND_BUFFER_SIZE_OFFSET    = 0x0c;
constexpr uint64_t COMMAND_BUFFER_DATA_OFFSET    = 0x10;
constexpr uint64_t COMMAND_BUFFER_SIZE           = 0x18;
constexpr uint64_t APR_COMMAND_BUFFER_MAP_OFFSET = 0x18;
constexpr uint64_t APR_COMMAND_BUFFER_SG_OFFSET  = 0x20;
constexpr uint32_t COMMAND_BUFFER_SIZE_MAX       = 64 * 1024 * 1024;
constexpr uint64_t READ_FILE_RECORD_SIZE         = 0x14;
constexpr uint64_t READ_FILE_RECORD_SIZE_EXT     = 0x18;
constexpr uint64_t READ_GATHER_RECORD_SIZE       = 0x08;
constexpr uint64_t READ_GATHER_RECORD_SIZE_EXT   = 0x0c;
constexpr uint64_t READ_SCATTER_RECORD_SIZE      = 0x0c;
constexpr uint64_t READ_GATHER_SCATTER_SIZE      = 0x10;
constexpr uint64_t READ_GATHER_SCATTER_SIZE_EXT  = 0x14;
constexpr uint64_t RESET_GATHER_SCATTER_SIZE     = 0x04;
constexpr uint64_t KERNEL_EVENT_RECORD_SIZE      = 0x20;
constexpr uint64_t APR_MAP_BEGIN_RECORD_SIZE     = 0x0c;
constexpr uint64_t APR_MAP_DIRECT_BEGIN_SIZE     = 0x10;
constexpr uint64_t APR_MAP_END_RECORD_SIZE       = 0x04;
constexpr uint64_t AMM_MAP_RECORD_SIZE           = 0x20;
constexpr uint64_t AMM_MAP_DIRECT_RECORD_SIZE    = 0x30;
constexpr uint64_t AMM_UNMAP_RECORD_SIZE         = 0x20;
constexpr uint64_t AMM_PAGE_SIZE                 = 0x4000;
constexpr int      AMM_MAP_FIXED                 = 0x10;
constexpr int      AMM_USAGE_DIRECT              = 0;
constexpr int      AMM_USAGE_AUTO                = 1;
constexpr int      PROT_CPU_READ                 = 0x01;
constexpr int      PROT_CPU_WRITE                = 0x02;
constexpr int      PROT_CPU_EXEC                 = 0x04;
constexpr int      PROT_GPU_READ                 = 0x10;
constexpr int      PROT_GPU_WRITE                = 0x20;
constexpr int      PROT_AMPR_READ                = 0x40;
constexpr int      PROT_AMPR_WRITE               = 0x80;
constexpr int      PROT_ACP_READ                 = 0x100;
constexpr int      PROT_ACP_WRITE                = 0x200;
constexpr uint64_t AMM_VA_START                  = 0x0000001000000000ull;
constexpr uint64_t AMM_VA_SIZE                   = 0x0000001000000000ull;
constexpr uint64_t APR_MAX_READ_LENGTH           = 0x0000000100000000ull;
constexpr uint64_t APR_MAX_FILE_OFFSET           = 0x0000010000000000ull;
constexpr uint64_t APR_MAX_APP_ADDRESS           = 0x0000f00000000000ull;
constexpr uint64_t APR_HOST_READ_CHUNK_SIZE      = 4 * 1024 * 1024;
constexpr uint32_t APR_TYPE_GATHER_SCATTER_VALID = 0x00010000;
constexpr uint32_t APR_TYPE_MAP_ACTIVE           = 0x00020000;

enum class AmmCommandKind : uint32_t {
	MapAuto,
	MapDirect,
	Unmap,
};

struct CommandBufferState {
	uint64_t buffer           = 0;
	uint64_t size             = 0;
	uint64_t write_offset     = 0;
	bool     header_validated = false;
	bool     buffer_validated = false;
	// Diagnostic identity only. Mutated under the existing command-buffer mutex;
	// these values never participate in command execution or guest-visible state.
	uint64_t diagnostic_generation = 0;
	uint64_t diagnostic_append_serial = 0;
	uint64_t diagnostic_submits = 0;
	uint64_t diagnostic_last_submit_serial = 0;
	uint64_t diagnostic_unchanged_submits = 0;
	struct ReadFileCommand {
		uint64_t record_offset = 0;
		uint32_t file_id       = 0;
		uint64_t destination   = 0;
		uint64_t size          = 0;
		uint64_t file_offset   = 0;
	};
	struct KernelEventCommand {
		uint64_t record_offset = 0;
		uint64_t eq            = 0;
		int32_t  id            = 0;
		uint64_t data          = 0;
	};
	struct WriteAddressCommand {
		uint64_t record_offset = 0;
		uint64_t address       = 0;
		uint64_t value         = 0;
	};
	struct AmmMapCommand {
		uint64_t       record_offset = 0;
		AmmCommandKind kind          = AmmCommandKind::MapAuto;
		uint64_t       va            = 0;
		uint64_t       dmem_offset   = 0;
		uint64_t       size          = 0;
		int32_t        type          = 0;
		int32_t        prot          = 0;
		uint8_t        gpu_mask_id   = 0;
	};
	// Counter-bank, timestamp and wait commands. Executed in record order together
	// with every other command kind; see ExecuteSyncCommand().
	enum class SyncKind : uint32_t {
		WriteCounter,
		WaitOnCounter,
		WaitOnAddress,
		WriteAddressFromCounter,
		WriteAddressFromCounterPair,
		WriteAddressFromTimeCounter,
	};
	struct SyncCommand {
		uint64_t record_offset = 0;
		SyncKind kind          = SyncKind::WriteCounter;
		uint8_t  counter_index = 0;
		uint8_t  access        = 0; // CounterBank::ACCESS_*
		uint8_t  op            = 0; // CounterBank::WRITE_* or CounterBank::COMPARE_*
		uint8_t  mask_op       = 0; // CounterBank::MASK_*
		uint64_t value         = 0; // write operand or wait reference
		uint64_t mask          = 0; // wait-on-counter AND mask
		uint64_t address       = 0; // guest address read (wait) or written (write-address)
	};
	std::vector<ReadFileCommand>     read_file_commands;
	std::vector<KernelEventCommand>  kernel_event_commands;
	std::vector<WriteAddressCommand> write_address_commands;
	std::vector<AmmMapCommand>       amm_map_commands;
	std::vector<SyncCommand>         sync_commands;
	bool                             gather_scatter_valid       = false;
	uint32_t                         gather_scatter_file_id     = 0;
	uint64_t                         gather_scatter_destination = 0;
	uint64_t                         gather_scatter_file_offset = 0;
};

struct AmmAutoPoolRange {
	uint64_t start = 0;
	uint64_t size  = 0;
	uint64_t used  = 0;
};

struct AmmVirtualAddressRanges {
	uint64_t va_start          = 0;
	uint64_t va_end            = 0;
	uint64_t multimap_va_start = 0;
	uint64_t multimap_va_end   = 0;
};

struct AmmUsageStatsData {
	uint64_t size_in_bytes                                    = 0;
	uint16_t num_page_table_pool_entries                      = 0;
	uint16_t snapshot_page_table_pool_allocated_entries       = 0;
	uint16_t high_watermark_allocated_page_table_pool_entries = 0;
	uint16_t reserved1                                        = 0;
	uint32_t ring_idle_flags                                  = 0;
};
static_assert(sizeof(AmmUsageStatsData) == 0x18);

static std::mutex                                       g_command_buffer_mutex;
static std::unordered_map<uint64_t, CommandBufferState> g_command_buffers;
static std::unordered_map<uint64_t, uint64_t>           g_command_buffer_aliases;
static std::mutex                                       g_amm_auto_pool_mutex;
static std::vector<AmmAutoPoolRange>                    g_amm_auto_pool;
using CommandBufferIterator = std::unordered_map<uint64_t, CommandBufferState>::iterator;
static std::atomic<uint64_t> g_diagnostic_generation {0};

static uint64_t NewDiagnosticGeneration() {
	return Profiler::LoadingEnabled()
	           ? g_diagnostic_generation.fetch_add(1, std::memory_order_relaxed) + 1
	           : 0;
}

static bool HasQueuedCommands(const CommandBufferState& state) {
	return !state.read_file_commands.empty() || !state.kernel_event_commands.empty() ||
	       !state.write_address_commands.empty() || !state.amm_map_commands.empty() ||
	       !state.sync_commands.empty();
}

static void RegisterCommandBufferAliasLocked(uint64_t command_buffer, uint64_t buffer) {
	for (auto it = g_command_buffer_aliases.begin(); it != g_command_buffer_aliases.end();) {
		if (it->second == command_buffer) {
			it = g_command_buffer_aliases.erase(it);
		} else {
			++it;
		}
	}

	if (buffer != 0 && buffer != command_buffer) {
		g_command_buffer_aliases[buffer] = command_buffer;
	}
}

static void EraseCommandBufferStateLocked(uint64_t command_buffer) {
	g_command_buffers.erase(command_buffer);
	for (auto it = g_command_buffer_aliases.begin(); it != g_command_buffer_aliases.end();) {
		if (it->first == command_buffer || it->second == command_buffer) {
			it = g_command_buffer_aliases.erase(it);
		} else {
			++it;
		}
	}
}

static bool AddU64(uint64_t a, uint64_t b, uint64_t* out) {
	if (out == nullptr || std::numeric_limits<uint64_t>::max() - a < b) {
		return false;
	}

	*out = a + b;
	return true;
}

static bool IsValidAprFileOffset(uint64_t file_offset) {
	return file_offset < APR_MAX_FILE_OFFSET;
}

static bool IsValidAprReadSize(uint64_t size) {
	return size != 0 && size <= APR_MAX_READ_LENGTH;
}

static bool IsValidAprReadRange(uint64_t destination, uint64_t size) {
	return IsValidAprReadSize(size) && destination <= APR_MAX_APP_ADDRESS &&
	       APR_MAX_APP_ADDRESS - destination >= size;
}

static uint64_t KernelErrorU64(int error) {
	return static_cast<uint32_t>(error);
}

template <typename T>
static T ReadCommandBufferUnchecked(uint64_t addr) {
	T value {};
	std::memcpy(&value, reinterpret_cast<const void*>(addr), sizeof(value));
	return value;
}

template <typename T>
static void WriteCommandBufferUnchecked(uint64_t addr, const T& value) {
	std::memcpy(reinterpret_cast<void*>(addr), &value, sizeof(value));
}

static bool ValidateCommandBufferHeader(uint64_t            command_buffer,
                                        CommandBufferState* state = nullptr) {
	if (state != nullptr && state->header_validated) {
		return true;
	}
	if (!AprShared::IsValidGuestRange(command_buffer, COMMAND_BUFFER_SIZE, true)) {
		return false;
	}
	if (state != nullptr) {
		state->header_validated = true;
	}
	return true;
}

static bool ValidateCommandBufferBacking(CommandBufferState* state) {
	if (state == nullptr || state->buffer == 0 || state->size == 0) {
		return false;
	}
	if (state->buffer_validated) {
		return true;
	}
	if (!AprShared::IsValidGuestRange(state->buffer, state->size, true)) {
		return false;
	}
	state->buffer_validated = true;
	return true;
}

static bool LoadCommandBufferStateFromGuest(uint64_t command_buffer, CommandBufferState* state) {
	if (state == nullptr || !ValidateCommandBufferHeader(command_buffer)) {
		return false;
	}

	CommandBufferState loaded {};
	loaded.buffer =
	    ReadCommandBufferUnchecked<uint64_t>(command_buffer + COMMAND_BUFFER_DATA_OFFSET);
	loaded.size = ReadCommandBufferUnchecked<uint32_t>(command_buffer + COMMAND_BUFFER_SIZE_OFFSET);
	loaded.write_offset =
	    ReadCommandBufferUnchecked<uint32_t>(command_buffer + COMMAND_BUFFER_OFFSET_OFFSET);
	loaded.header_validated = true;
	loaded.diagnostic_generation = NewDiagnosticGeneration();
	*state                  = loaded;
	return true;
}

static bool GetOrCreateCommandBufferStateLocked(uint64_t               command_buffer,
                                                CommandBufferIterator* out) {
	if (out == nullptr) {
		return false;
	}

	auto it = g_command_buffers.find(command_buffer);
	if (it == g_command_buffers.end()) {
		CommandBufferState state {};
		if (!LoadCommandBufferStateFromGuest(command_buffer, &state)) {
			return false;
		}
		it = g_command_buffers.emplace(command_buffer, state).first;
	}

	*out = it;
	return true;
}

static bool EnsureCommandBufferRecordSpace(uint64_t command_buffer, CommandBufferState* state,
                                           uint64_t record_size) {
	if (state == nullptr || record_size == 0 ||
	    !ValidateCommandBufferHeader(command_buffer, state) || state->buffer == 0 ||
	    record_size > state->size || state->write_offset > state->size - record_size) {
		return false;
	}

	return ValidateCommandBufferBacking(state);
}

static bool CommitCommandBufferRecord(uint64_t command_buffer, CommandBufferState* state,
                                      uint64_t record_size) {
	if (state == nullptr) {
		return false;
	}

	state->write_offset += record_size;
	if (state->write_offset > std::numeric_limits<uint32_t>::max()) {
		return false;
	}

	WriteCommandBufferUnchecked(command_buffer + COMMAND_BUFFER_OFFSET_OFFSET,
	                            static_cast<uint32_t>(state->write_offset));
	const auto num =
	    ReadCommandBufferUnchecked<int32_t>(command_buffer + COMMAND_BUFFER_NUM_OFFSET);
	WriteCommandBufferUnchecked(command_buffer + COMMAND_BUFFER_NUM_OFFSET, num + 1);
	if (Profiler::LoadingEnabled()) ++state->diagnostic_append_serial;
	return true;
}

static void UpdateCommandBufferTypeFlagsUnchecked(uint64_t command_buffer, uint32_t set_bits,
                                                  uint32_t clear_bits) {
	auto type = ReadCommandBufferUnchecked<uint32_t>(command_buffer + COMMAND_BUFFER_TYPE_OFFSET);
	type |= set_bits;
	type &= ~clear_bits;
	WriteCommandBufferUnchecked(command_buffer + COMMAND_BUFFER_TYPE_OFFSET, type);
}

static uint64_t AprReadFileRecordSize(uint64_t file_offset) {
	return (file_offset >> 32u) != 0 ? READ_FILE_RECORD_SIZE_EXT : READ_FILE_RECORD_SIZE;
}

static uint64_t AprReadGatherRecordSize(uint64_t file_offset) {
	return file_offset > 0x3ffffu ? READ_GATHER_RECORD_SIZE_EXT : READ_GATHER_RECORD_SIZE;
}

static uint64_t AprReadGatherScatterRecordSize(uint64_t file_offset) {
	return (file_offset >> 32u) != 0 ? READ_GATHER_SCATTER_SIZE_EXT : READ_GATHER_SCATTER_SIZE;
}

static bool UpdateCommandBufferTypeFlags(uint64_t command_buffer, uint32_t set_bits,
                                         uint32_t clear_bits) {
	if (!ValidateCommandBufferHeader(command_buffer)) {
		return false;
	}
	UpdateCommandBufferTypeFlagsUnchecked(command_buffer, set_bits, clear_bits);
	return true;
}

static std::unordered_map<uint64_t, CommandBufferState>::iterator
ResolveCommandBufferStateLocked(uint64_t command_buffer) {
	auto it = g_command_buffers.find(command_buffer);
	if (it != g_command_buffers.end() && HasQueuedCommands(it->second)) {
		return it;
	}

	const auto alias = g_command_buffer_aliases.find(command_buffer);
	if (alias != g_command_buffer_aliases.end()) {
		const auto alias_it = g_command_buffers.find(alias->second);
		if (alias_it != g_command_buffers.end()) {
			return alias_it;
		}
	}

	return it;
}

static bool WriteVisibleCommandBufferPointers(uint64_t command_buffer, uint64_t buffer,
                                              uint64_t size) {
	if (size > std::numeric_limits<uint32_t>::max() ||
	    !ValidateCommandBufferHeader(command_buffer)) {
		return false;
	}

	WriteCommandBufferUnchecked(command_buffer + COMMAND_BUFFER_DATA_OFFSET, buffer);
	WriteCommandBufferUnchecked(command_buffer + COMMAND_BUFFER_SIZE_OFFSET,
	                            static_cast<uint32_t>(size));
	return true;
}

static bool WriteCommandBufferPointers(uint64_t command_buffer, uint64_t buffer, uint64_t size,
                                       uint64_t write_offset = 0) {
	if (write_offset > std::numeric_limits<uint32_t>::max() ||
	    size > std::numeric_limits<uint32_t>::max() ||
	    !ValidateCommandBufferHeader(command_buffer)) {
		return false;
	}

	WriteCommandBufferUnchecked(command_buffer + COMMAND_BUFFER_DATA_OFFSET, buffer);
	WriteCommandBufferUnchecked(command_buffer + COMMAND_BUFFER_SIZE_OFFSET,
	                            static_cast<uint32_t>(size));
	WriteCommandBufferUnchecked(command_buffer + COMMAND_BUFFER_OFFSET_OFFSET,
	                            static_cast<uint32_t>(write_offset));

	std::scoped_lock lock(g_command_buffer_mutex);
	auto&            state = g_command_buffers[command_buffer];
	state.buffer           = buffer;
	state.size             = size;
	state.write_offset     = write_offset;
	state.header_validated = true;
	state.buffer_validated = false;
	state.diagnostic_generation = NewDiagnosticGeneration();
	state.diagnostic_append_serial = 0;
	state.diagnostic_submits = 0;
	state.diagnostic_last_submit_serial = 0;
	state.diagnostic_unchanged_submits = 0;
	state.read_file_commands.clear();
	state.kernel_event_commands.clear();
	state.write_address_commands.clear();
	state.amm_map_commands.clear();
	state.sync_commands.clear();
	state.gather_scatter_valid       = false;
	state.gather_scatter_file_id     = 0;
	state.gather_scatter_destination = 0;
	state.gather_scatter_file_offset = 0;
	RegisterCommandBufferAliasLocked(command_buffer, buffer);
	return true;
}

static bool TryGetCommandBufferState(uint64_t command_buffer, CommandBufferState* out,
                                    bool for_submission = false,
                                    Profiler::LoadingAprSubmission* diagnostic = nullptr) {
	if (out == nullptr) {
		return false;
	}
	const auto observe = [&](uint64_t object, CommandBufferState& state) {
		if (!for_submission || !Profiler::LoadingEnabled()) return;
		if (state.diagnostic_submits != 0 &&
		    state.diagnostic_last_submit_serial == state.diagnostic_append_serial) {
			++state.diagnostic_unchanged_submits;
		}
		++state.diagnostic_submits;
		state.diagnostic_last_submit_serial = state.diagnostic_append_serial;
		if (diagnostic != nullptr) {
			diagnostic->resolved_object = object;
			diagnostic->generation = state.diagnostic_generation;
			diagnostic->append_serial = state.diagnostic_append_serial;
			diagnostic->submits_in_generation = state.diagnostic_submits;
			diagnostic->unchanged_submits = state.diagnostic_unchanged_submits;
		}
	};

	{
		std::scoped_lock lock(g_command_buffer_mutex);
		const auto       it = ResolveCommandBufferStateLocked(command_buffer);
		if (it != g_command_buffers.end()) {
			observe(it->first, it->second);
			*out = it->second;
			return true;
		}
	}

	CommandBufferState state {};
	if (!LoadCommandBufferStateFromGuest(command_buffer, &state)) {
		return false;
	}

	std::scoped_lock lock(g_command_buffer_mutex);
	observe(command_buffer, state);
	g_command_buffers[command_buffer] = state;
	*out                              = state;
	return true;
}

static bool InitializeBaseCommandBuffer(uint64_t command_buffer) {
	if (!AprShared::IsValidGuestRange(command_buffer, COMMAND_BUFFER_SIZE, true)) {
		return false;
	}

	std::memset(reinterpret_cast<void*>(command_buffer), 0, COMMAND_BUFFER_SIZE);

	std::scoped_lock lock(g_command_buffer_mutex);
	EraseCommandBufferStateLocked(command_buffer);
	return true;
}

static bool WriteGuestZero64Validated(uint64_t addr) {
	if (!AprShared::IsValidGuestRange(addr, sizeof(uint64_t), true)) {
		return false;
	}
	WriteCommandBufferUnchecked(addr, uint64_t {0});
	return true;
}

static bool InitializeAprReservedState(uint64_t command_buffer, uint64_t reserved_state0,
                                       uint64_t reserved_state1) {
	if (reserved_state0 == 0) {
		reserved_state0 = command_buffer + APR_COMMAND_BUFFER_MAP_OFFSET;
	}
	if (reserved_state1 == 0) {
		reserved_state1 = command_buffer + APR_COMMAND_BUFFER_SG_OFFSET;
	}

	if (reserved_state0 == reserved_state1) {
		return WriteGuestZero64Validated(reserved_state0);
	}

	const auto word_size = static_cast<uint64_t>(sizeof(uint64_t));
	const auto span_size = word_size * 2u;
	const bool adjacent  = (reserved_state0 <= std::numeric_limits<uint64_t>::max() - word_size &&
	                        reserved_state0 + word_size == reserved_state1) ||
	                       (reserved_state1 <= std::numeric_limits<uint64_t>::max() - word_size &&
	                        reserved_state1 + word_size == reserved_state0);
	if (adjacent) {
		const auto first = std::min(reserved_state0, reserved_state1);
		if (first > std::numeric_limits<uint64_t>::max() - span_size ||
		    !AprShared::IsValidGuestRange(first, span_size, true)) {
			return false;
		}
		WriteCommandBufferUnchecked(reserved_state0, uint64_t {0});
		WriteCommandBufferUnchecked(reserved_state1, uint64_t {0});
		return true;
	}

	return WriteGuestZero64Validated(reserved_state0) && WriteGuestZero64Validated(reserved_state1);
}

static bool AppendReadFileRecord(uint64_t command_buffer, uint8_t opcode, uint32_t file_id,
                                 uint64_t destination, uint64_t size, uint64_t file_offset,
                                 uint64_t record_size) {
	std::scoped_lock      lock(g_command_buffer_mutex);
	CommandBufferIterator it;
	if (!GetOrCreateCommandBufferStateLocked(command_buffer, &it)) {
		return false;
	}

	auto& state = it->second;
	if (!EnsureCommandBufferRecordSpace(command_buffer, &state, record_size)) {
		return false;
	}

	const auto record_offset = state.write_offset;
	std::memset(reinterpret_cast<void*>(state.buffer + record_offset), 0,
	            static_cast<size_t>(record_size));
	std::memcpy(reinterpret_cast<void*>(state.buffer + record_offset), &opcode, sizeof(opcode));
	state.read_file_commands.push_back({record_offset, file_id, destination, size, file_offset});
	if (!CommitCommandBufferRecord(command_buffer, &state, record_size)) {
		return false;
	}
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprReadCommandsAppended);
	state.gather_scatter_valid   = true;
	state.gather_scatter_file_id = file_id;
	if (!AddU64(destination, size, &state.gather_scatter_destination) ||
	    !AddU64(file_offset, size, &state.gather_scatter_file_offset)) {
		state.gather_scatter_valid = false;
		return false;
	}
	UpdateCommandBufferTypeFlagsUnchecked(command_buffer, APR_TYPE_GATHER_SCATTER_VALID, 0);
	return true;
}

static bool AppendKernelEventRecord(uint64_t command_buffer, uint64_t eq, int32_t id,
                                    uint64_t data) {
	std::array<uint8_t, KERNEL_EVENT_RECORD_SIZE> record {};
	const uint32_t                                type = 2;
	std::memcpy(&record[0x00], &type, sizeof(type));
	std::memcpy(&record[0x08], &eq, sizeof(eq));
	std::memcpy(&record[0x10], &id, sizeof(id));
	std::memcpy(&record[0x18], &data, sizeof(data));

	std::scoped_lock      lock(g_command_buffer_mutex);
	CommandBufferIterator it;
	if (!GetOrCreateCommandBufferStateLocked(command_buffer, &it)) {
		return false;
	}

	auto& state = it->second;
	if (!EnsureCommandBufferRecordSpace(command_buffer, &state, KERNEL_EVENT_RECORD_SIZE)) {
		return false;
	}

	const auto record_offset = state.write_offset;
	std::memcpy(reinterpret_cast<void*>(state.buffer + record_offset), record.data(),
	            record.size());
	state.kernel_event_commands.push_back({record_offset, eq, id, data});
	return CommitCommandBufferRecord(command_buffer, &state, KERNEL_EVENT_RECORD_SIZE);
}

static bool AppendWriteAddressRecord(uint64_t command_buffer, uint64_t address, uint64_t value,
                                     uint64_t record_size = 0x20) {
	std::scoped_lock      lock(g_command_buffer_mutex);
	CommandBufferIterator it;
	if (!GetOrCreateCommandBufferStateLocked(command_buffer, &it)) {
		return false;
	}

	auto& state = it->second;
	if (!EnsureCommandBufferRecordSpace(command_buffer, &state, record_size)) {
		return false;
	}

	const auto record_offset = state.write_offset;
	std::memset(reinterpret_cast<void*>(state.buffer + record_offset), 0,
	            static_cast<size_t>(record_size));
	state.write_address_commands.push_back({record_offset, address, value});
	return CommitCommandBufferRecord(command_buffer, &state, record_size);
}

// All counter, timestamp and wait records are 0x20 bytes, matching the
// sceAmprMeasureCommandSize* results for these commands.
constexpr uint64_t SYNC_RECORD_SIZE = 0x20;

static bool AppendSyncRecord(uint64_t command_buffer, const CommandBufferState::SyncCommand& cmd) {
	std::scoped_lock      lock(g_command_buffer_mutex);
	CommandBufferIterator it;
	if (!GetOrCreateCommandBufferStateLocked(command_buffer, &it)) {
		return false;
	}

	auto& state = it->second;
	if (!EnsureCommandBufferRecordSpace(command_buffer, &state, SYNC_RECORD_SIZE)) {
		return false;
	}

	auto record          = cmd;
	record.record_offset = state.write_offset;
	std::memset(reinterpret_cast<void*>(state.buffer + state.write_offset), 0,
	            static_cast<size_t>(SYNC_RECORD_SIZE));
	state.sync_commands.push_back(record);
	return CommitCommandBufferRecord(command_buffer, &state, SYNC_RECORD_SIZE);
}

// Process-wide AMPR counter bank and wait machinery.
//
// Execution model: every submission runs synchronously on the submitting guest
// thread (ExecuteAprCommandBuffer), one record at a time, in record order. A
// command therefore only starts after every earlier command in the same buffer
// has fully completed, so the "OnCompletion" variants (which on hardware wait for
// all earlier reads to land before they take effect) and the immediate variants
// behave identically here, and both are exact with respect to the buffer's own
// reads.
//
// Waits: other guest threads submit their own buffers concurrently, and guest CPU
// code may store to a waited address, so an unsatisfied wait can legitimately be
// satisfied later. It is re-evaluated whenever any AMPR submission writes a counter
// or an address (condition variable + epoch), and polled with a backoff capped at
// AMPR_WAIT_MAX_SLICE to observe plain CPU stores. What a synchronous executor
// cannot do is let the *submitting* thread run ahead: on hardware the submit call
// returns at once, so a buffer may wait on a value that the same thread produces
// only after the submit returns (a later submission or a CPU store). Here that
// dependency can never resolve, so each wait is bounded by AMPR_WAIT_TIMEOUT;
// on expiry the wait is logged (rate-limited) and treated as satisfied so the
// buffer completes. Continuing is preferred to failing the submission: hardware
// never reports a wait as an error, and an error result would make streaming code
// retry or abandon the request, whereas a late-satisfied wait only reorders the
// following records relative to a producer that, on this thread, cannot run until
// the submit returns anyway.
constexpr auto AMPR_WAIT_TIMEOUT     = std::chrono::milliseconds(1000);
constexpr auto AMPR_WAIT_FIRST_SLICE = std::chrono::microseconds(50);
constexpr auto AMPR_WAIT_MAX_SLICE   = std::chrono::milliseconds(2);

static std::mutex              g_sync_mutex; // guards g_counter_bank and g_sync_epoch
static std::condition_variable g_sync_cv;
static CounterBank::Bank       g_counter_bank;
static uint64_t                g_sync_epoch = 0;
static std::atomic<uint64_t>   g_sync_wait_timeouts {0};
static std::atomic<uint64_t>   g_sync_unsupported_logged {0};

static void NotifySyncWaiters() {
	{
		std::scoped_lock lock(g_sync_mutex);
		++g_sync_epoch;
	}
	g_sync_cv.notify_all();
}

static uint64_t ReadCounter(const CounterBank::Lane& lane) {
	std::scoped_lock lock(g_sync_mutex);
	return g_counter_bank.Read(lane);
}

static void ApplyCounterWrite(const CounterBank::Lane& lane, uint64_t operand, uint8_t op) {
	{
		std::scoped_lock lock(g_sync_mutex);
		g_counter_bank.Apply(lane, operand, op);
		++g_sync_epoch;
	}
	g_sync_cv.notify_all();
}

static bool ReadGuestWaitValue(uint64_t address, uint64_t* out) {
	if (address != 0 && (address & (sizeof(uint64_t) - 1u)) == 0) {
		*out = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(address))
		           .load(std::memory_order_acquire);
		return true;
	}
	return AprShared::ReadGuest(address, out);
}

// Evaluates `satisfied` (called without g_sync_mutex held) until it returns true
// or AMPR_WAIT_TIMEOUT elapses. Returns whether it was satisfied.
template <typename Satisfied>
static bool WaitBounded(Satisfied&& satisfied, uint64_t* waited_us) {
	HangWatchdog::Scope wait("ampr-bounded-wait", reinterpret_cast<uint64_t>(waited_us), 1000000);
	using Clock = std::chrono::steady_clock;

	uint64_t seen = 0;
	{
		std::scoped_lock lock(g_sync_mutex);
		seen = g_sync_epoch;
	}
	*waited_us = 0;
	if (satisfied()) {
		return true;
	}

	const auto        start    = Clock::now();
	const auto        deadline = start + AMPR_WAIT_TIMEOUT;
	Clock::duration   slice    = AMPR_WAIT_FIRST_SLICE;
	bool              ok       = false;
	for (;;) {
		const auto now = Clock::now();
		if (now >= deadline) {
			break;
		}
		{
			std::unique_lock lock(g_sync_mutex);
			g_sync_cv.wait_for(lock, std::min<Clock::duration>(slice, deadline - now),
			                   [&] { return g_sync_epoch != seen; });
			seen = g_sync_epoch;
		}
		if (satisfied()) {
			ok = true;
			break;
		}
		slice = std::min<Clock::duration>(slice * 2, AMPR_WAIT_MAX_SLICE);
	}
	*waited_us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
	return ok;
}

static void ReportWaitTimeout(uint64_t command_buffer, const CommandBufferState::SyncCommand& command,
                              uint64_t observed, uint64_t waited_us) {
	const auto count = g_sync_wait_timeouts.fetch_add(1, std::memory_order_relaxed) + 1;
	if (count > 16 && (count & 255u) != 0) {
		return;
	}
	if (command.kind == CommandBufferState::SyncKind::WaitOnAddress) {
		LOGF("\tAMPR wait-on-address not satisfied after %" PRIu64
		     " us, continuing (synchronous executor; timeout #%" PRIu64 "): cb=0x%016" PRIx64
		     " record=0x%" PRIx64 " address=0x%016" PRIx64 " compare=%u ref=0x%016" PRIx64
		     " observed=0x%016" PRIx64 "\n",
		     waited_us, count, command_buffer, command.record_offset, command.address,
		     static_cast<uint32_t>(command.op), command.value, observed);
	} else {
		LOGF("\tAMPR wait-on-counter not satisfied after %" PRIu64
		     " us, continuing (synchronous executor; timeout #%" PRIu64 "): cb=0x%016" PRIx64
		     " record=0x%" PRIx64 " index=%u access=%u compare=%u mask_op=%u mask=0x%016" PRIx64
		     " ref=0x%016" PRIx64 " observed=0x%016" PRIx64 "\n",
		     waited_us, count, command_buffer, command.record_offset,
		     static_cast<uint32_t>(command.counter_index), static_cast<uint32_t>(command.access),
		     static_cast<uint32_t>(command.op), static_cast<uint32_t>(command.mask_op), command.mask,
		     command.value, observed);
	}
}

// Executes one counter/timestamp/wait record. Returns OK or the execution error
// to report for this record.
static int ExecuteSyncCommand(uint64_t command_buffer, const CommandBufferState::SyncCommand& command) {
	using Kind = CommandBufferState::SyncKind;

	CounterBank::Lane lane {};
	const bool        uses_counter = command.kind != Kind::WaitOnAddress &&
	                          command.kind != Kind::WriteAddressFromTimeCounter;
	if (uses_counter && !CounterBank::DecodeLane(command.counter_index, command.access, &lane)) {
		// Validated when the record was appended; unreachable unless state is corrupt.
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	switch (command.kind) {
		case Kind::WriteCounter: ApplyCounterWrite(lane, command.value, command.op); return OK;

		case Kind::WriteAddressFromCounter:
		case Kind::WriteAddressFromCounterPair:
		case Kind::WriteAddressFromTimeCounter: {
			// The time counter is the process time counter the guest reads through
			// sceKernelGetProcessTimeCounter(), sampled when the record executes.
			const uint64_t value = command.kind == Kind::WriteAddressFromTimeCounter
			                           ? LibKernel::KernelGetProcessTimeCounter()
			                           : ReadCounter(lane);
			if (!AprShared::WriteGuest(command.address, value)) {
				LOGF("\tAMPR submit write-address-from-counter failed: kind=%u address=0x%016" PRIx64
				     " index=%u\n",
				     static_cast<uint32_t>(command.kind), command.address,
				     static_cast<uint32_t>(command.counter_index));
				return LibKernel::KERNEL_ERROR_EFAULT;
			}
			NotifySyncWaiters();
			return OK;
		}

		case Kind::WaitOnCounter: {
			HangWatchdog::Scope scope("ampr-counter", command.counter_index, command.value, 0,
			                          command.mask, command.op);
			uint64_t   observed  = 0;
			const auto satisfied = [&]() {
				observed = ReadCounter(lane);
				scope.Observed(observed);
				if (command.mask_op == CounterBank::MASK_AND) {
					observed &= command.mask;
				}
				return CounterBank::CompareSatisfied(observed, command.value, command.op, lane.bytes);
			};
			uint64_t waited_us = 0;
			if (!WaitBounded(satisfied, &waited_us)) {
				ReportWaitTimeout(command_buffer, command, observed, waited_us);
			}
			return OK;
		}

		case Kind::WaitOnAddress: {
			HangWatchdog::Scope scope("ampr-address", command.address, command.value, 0, 0,
			                          command.op);
			uint64_t   observed  = 0;
			bool       readable  = true;
			const auto satisfied = [&]() {
				if (!ReadGuestWaitValue(command.address, &observed)) {
					readable = false;
					return true;
				}
				scope.Observed(observed);
				return CounterBank::CompareSatisfied(observed, command.value, command.op,
				                                     sizeof(uint64_t));
			};
			uint64_t waited_us = 0;
			if (!WaitBounded(satisfied, &waited_us)) {
				ReportWaitTimeout(command_buffer, command, observed, waited_us);
			}
			if (!readable) {
				LOGF("\tAMPR wait-on-address read failed: address=0x%016" PRIx64 "\n",
				     command.address);
				return LibKernel::KERNEL_ERROR_EFAULT;
			}
			return OK;
		}
	}
	return LibKernel::KERNEL_ERROR_EINVAL;
}

static bool ValidateAmmMapArgs(uint64_t va, uint64_t size) {
	return va != 0 && size != 0 && (va & (AMM_PAGE_SIZE - 1u)) == 0 &&
	       (size & (AMM_PAGE_SIZE - 1u)) == 0 && va + size >= va;
}

static int KYTY_SYSV_ABI AmmCommandBufferConstructor(void* command_buffer) {
	PRINT_NAME();

	(void)command_buffer;
	return OK;
}

static int KYTY_SYSV_ABI AmmCommandBufferDestructor(void* command_buffer) {
	PRINT_NAME();

	(void)command_buffer;
	return OK;
}

static int NormalizeAmmProtection(int prot) {
	constexpr int cpu_gpu_bits =
	    PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC | PROT_GPU_READ | PROT_GPU_WRITE;

	int normalized = prot & cpu_gpu_bits;

	if ((prot & PROT_AMPR_READ) != 0) {
		normalized |= PROT_CPU_READ;
	}
	if ((prot & PROT_AMPR_WRITE) != 0) {
		normalized |= PROT_CPU_READ | PROT_CPU_WRITE;
	}
	if ((prot & PROT_ACP_READ) != 0) {
		normalized |= PROT_CPU_READ;
	}
	if ((prot & PROT_ACP_WRITE) != 0) {
		normalized |= PROT_CPU_READ | PROT_CPU_WRITE;
	}

	return normalized;
}

static bool AllocateAmmAutoDirectMemory(uint64_t size, int memory_type, uint64_t* dmem_offset) {
	if (dmem_offset == nullptr || size == 0) {
		return false;
	}

	std::scoped_lock lock(g_amm_auto_pool_mutex);

	for (auto& range: g_amm_auto_pool) {
		const auto aligned_used = (range.used + (AMM_PAGE_SIZE - 1u)) & ~(AMM_PAGE_SIZE - 1u);
		if (aligned_used <= range.size && size <= range.size - aligned_used) {
			*dmem_offset = range.start + aligned_used;
			range.used   = aligned_used + size;
			return true;
		}
	}

	int64_t allocated = 0;
	if (LibKernel::Memory::KernelAllocateDirectMemory(
	        0, LibKernel::Memory::KernelGetDirectMemorySize(), size, AMM_PAGE_SIZE, memory_type,
	        &allocated) != OK) {
		return false;
	}

	*dmem_offset = static_cast<uint64_t>(allocated);
	return true;
}

static int ExecuteAmmMapCommand(const CommandBufferState::AmmMapCommand& command) {
	if (!ValidateAmmMapArgs(command.va, command.size)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	uint64_t dmem_offset = command.dmem_offset;
	if (command.kind == AmmCommandKind::MapAuto &&
	    !AllocateAmmAutoDirectMemory(command.size, command.type, &dmem_offset)) {
		return LibKernel::KERNEL_ERROR_EAGAIN;
	}

	void* addr = reinterpret_cast<void*>(command.va);
	return LibKernel::Memory::KernelMapDirectMemory2(
	    &addr, command.size, command.type, NormalizeAmmProtection(command.prot), AMM_MAP_FIXED,
	    static_cast<int64_t>(dmem_offset), AMM_PAGE_SIZE);
}

static bool AppendAmmMapRecord(uint64_t                                 command_buffer,
                               const CommandBufferState::AmmMapCommand& cmd, uint64_t record_size) {
	std::scoped_lock      lock(g_command_buffer_mutex);
	CommandBufferIterator it;
	if (!GetOrCreateCommandBufferStateLocked(command_buffer, &it)) {
		return false;
	}

	auto& state = it->second;
	if (!EnsureCommandBufferRecordSpace(command_buffer, &state, record_size)) {
		return false;
	}

	auto record          = cmd;
	record.record_offset = state.write_offset;
	auto* record_addr    = reinterpret_cast<void*>(state.buffer + state.write_offset);
	std::memset(record_addr, 0, static_cast<size_t>(record_size));
	state.amm_map_commands.push_back(record);
	return CommitCommandBufferRecord(command_buffer, &state, record_size);
}

static int ReadHostFileToGuest(const std::string& host_path, uint64_t file_offset,
                               uint64_t destination, uint64_t size, uint64_t* bytes_read);

static int ExecuteAprCommandBuffer(uint64_t command_buffer, int32_t* execution_result,
                                   uint32_t* error_offset,
                                   Profiler::LoadingAprSubmission* diagnostic) {
	Profiler::ScopedLoadingOperation loading(Profiler::LoadingOperation::AprSubmit);
	if (diagnostic != nullptr) diagnostic->execution_observed = 1;
	const auto finish = [&](int result) {
		if (diagnostic != nullptr) {
			diagnostic->submit_result = static_cast<uint32_t>(result);
			if (execution_result != nullptr && error_offset != nullptr) {
				diagnostic->execution_result = static_cast<uint32_t>(*execution_result);
				diagnostic->error_offset = *error_offset;
			}
		}
		return result;
	};
	if (execution_result == nullptr || error_offset == nullptr) {
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprErrors);
		return finish(LibKernel::KERNEL_ERROR_EINVAL);
	}

	*execution_result = OK;
	*error_offset     = 0;

	CommandBufferState state {};
	if (!TryGetCommandBufferState(command_buffer, &state, true, diagnostic)) {
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprErrors);
		return finish(LibKernel::KERNEL_ERROR_EFAULT);
	}
	if (diagnostic != nullptr) {
		diagnostic->cached_buffer = state.buffer;
		diagnostic->cached_size = state.size;
		diagnostic->cached_offset = state.write_offset;
		std::array<uint32_t, COMMAND_BUFFER_SIZE / sizeof(uint32_t)> header {};
		// Observe the resolved object's backing without unprotecting pages, invoking
		// fault service, or dereferencing a raw submitted command-stream pointer.
		// This is a nontransactional diagnostic snapshot, not a coherence decision.
		if (LibKernel::Memory::TryReadBacking(diagnostic->resolved_object, header.data(), sizeof(header))) {
			diagnostic->header_valid = 1;
			diagnostic->header_type = header[0];
			diagnostic->header_offset = header[1];
			diagnostic->header_commands = header[2];
			diagnostic->header_size = header[3];
			diagnostic->header_buffer = uint64_t {header[4]} | (uint64_t {header[5]} << 32u);
			diagnostic->header_mismatch = (diagnostic->header_buffer != state.buffer ? 1u : 0u) |
			                              (diagnostic->header_size != state.size ? 2u : 0u) |
			                              (diagnostic->header_offset != state.write_offset ? 4u : 0u);
		}
	}

	enum class CommandKind {
		ReadFile,
		KernelEvent,
		WriteAddress,
		AmmMap,
		Sync,
	};

	struct OrderedCommand {
		uint64_t    record_offset = 0;
		CommandKind kind          = CommandKind::ReadFile;
		size_t      index         = 0;
	};

	std::vector<OrderedCommand> ordered;
	ordered.reserve(state.read_file_commands.size() + state.kernel_event_commands.size() +
	                state.write_address_commands.size() + state.amm_map_commands.size() +
	                state.sync_commands.size());
	for (size_t i = 0; i < state.read_file_commands.size(); i++) {
		if (state.read_file_commands[i].record_offset < state.write_offset) {
			ordered.push_back(
			    {state.read_file_commands[i].record_offset, CommandKind::ReadFile, i});
		}
	}
	for (size_t i = 0; i < state.kernel_event_commands.size(); i++) {
		if (state.kernel_event_commands[i].record_offset < state.write_offset) {
			ordered.push_back(
			    {state.kernel_event_commands[i].record_offset, CommandKind::KernelEvent, i});
		}
	}
	for (size_t i = 0; i < state.write_address_commands.size(); i++) {
		if (state.write_address_commands[i].record_offset < state.write_offset) {
			ordered.push_back(
			    {state.write_address_commands[i].record_offset, CommandKind::WriteAddress, i});
		}
	}
	for (size_t i = 0; i < state.amm_map_commands.size(); i++) {
		if (state.amm_map_commands[i].record_offset < state.write_offset) {
			ordered.push_back({state.amm_map_commands[i].record_offset, CommandKind::AmmMap, i});
		}
	}
	for (size_t i = 0; i < state.sync_commands.size(); i++) {
		if (state.sync_commands[i].record_offset < state.write_offset) {
			ordered.push_back({state.sync_commands[i].record_offset, CommandKind::Sync, i});
		}
	}

	std::sort(ordered.begin(), ordered.end(), [](const OrderedCommand& a, const OrderedCommand& b) {
		return a.record_offset < b.record_offset;
	});
	if (diagnostic != nullptr) {
		uint64_t signature = 14695981039346656037ull;
		const auto hash = [&](uint64_t value) {
			for (unsigned byte = 0; byte < 8; ++byte) {
				signature ^= (value >> (byte * 8u)) & 0xffu;
				signature *= 1099511628211ull;
			}
		};
		for (const auto& entry: ordered) {
			hash(static_cast<uint64_t>(entry.kind));
			hash(entry.record_offset);
			switch (entry.kind) {
				case CommandKind::ReadFile: {
					const auto& command = state.read_file_commands[entry.index];
					if (diagnostic->read_count++ == 0) {
						diagnostic->first_read_id = command.file_id;
						diagnostic->first_read_destination = command.destination;
						diagnostic->first_read_size = command.size;
						diagnostic->first_read_offset = command.file_offset;
					}
					hash(command.file_id); hash(command.destination); hash(command.size); hash(command.file_offset);
					break;
				}
				case CommandKind::KernelEvent: {
					++diagnostic->event_count;
					const auto& command = state.kernel_event_commands[entry.index];
					hash(command.eq); hash(static_cast<uint32_t>(command.id)); hash(command.data);
					break;
				}
				case CommandKind::WriteAddress: {
					++diagnostic->write_count;
					const auto& command = state.write_address_commands[entry.index];
					hash(command.address); hash(command.value);
					break;
				}
				case CommandKind::AmmMap: {
					++diagnostic->map_count;
					const auto& command = state.amm_map_commands[entry.index];
					hash(static_cast<uint32_t>(command.kind)); hash(command.va); hash(command.dmem_offset);
					hash(command.size); hash(static_cast<uint32_t>(command.type));
					hash(static_cast<uint32_t>(command.prot)); hash(command.gpu_mask_id);
					break;
				}
				case CommandKind::Sync: {
					const auto& command = state.sync_commands[entry.index];
					if (command.kind == CommandBufferState::SyncKind::WriteAddressFromCounter ||
					    command.kind == CommandBufferState::SyncKind::WriteAddressFromCounterPair ||
					    command.kind == CommandBufferState::SyncKind::WriteAddressFromTimeCounter) {
						++diagnostic->write_count;
					}
					hash(static_cast<uint32_t>(command.kind));
					hash(command.counter_index | (uint32_t {command.access} << 8u) |
					     (uint32_t {command.op} << 16u) | (uint32_t {command.mask_op} << 24u));
					hash(command.value); hash(command.mask); hash(command.address);
					break;
				}
			}
		}
		diagnostic->commands_signature = signature;
	}

	for (const auto& entry: ordered) {
		switch (entry.kind) {
			case CommandKind::ReadFile: {
				const auto& command = state.read_file_commands[entry.index];
				std::string host_path;
				if (!AprShared::TryGetHostPath(command.file_id, &host_path)) {
					Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprErrors);
					LOGF("\tAPR submit failed for unknown file id: 0x%08" PRIx32 "\n",
					     command.file_id);
					*execution_result = LibKernel::KERNEL_ERROR_ENOENT;
					*error_offset     = static_cast<uint32_t>(command.record_offset);
					return finish(OK);
				}

				HangWatchdog::Scope read("ampr-file", command.destination, command.size,
				                         command.file_id, command.file_offset, command_buffer);
				uint64_t            bytes_read = 0;
				auto result = ReadHostFileToGuest(host_path, command.file_offset,
				                                  command.destination, command.size, &bytes_read);
				if (diagnostic != nullptr) diagnostic->read_bytes += bytes_read;
				if (HangTrace::Enabled()) {
					char thread_name[64] = "(host)";
					if (auto self = LibKernel::PthreadSelfOrNull(); self != nullptr) {
						if (LibKernel::PthreadGetname(self, thread_name) != 0) {
							std::snprintf(thread_name, sizeof(thread_name), "(unnamed)");
						}
					}
					HangTrace::RecordAprRead(command.file_id, host_path, command.file_offset,
					                         command.size, command.destination, bytes_read, result,
					                         thread_name);
				}
				if (result != OK) {
					Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprErrors);
					LOGF("\tAPR submit read failed: id=0x%08" PRIx32 ", result=0x%08" PRIx32
					     ", path=%s\n",
					     command.file_id, static_cast<uint32_t>(result), host_path.c_str());
					*execution_result = result;
					*error_offset     = static_cast<uint32_t>(command.record_offset);
					return finish(OK);
				}
			} break;
			case CommandKind::KernelEvent: {
				const auto& command = state.kernel_event_commands[entry.index];
				const auto  eq      = static_cast<LibKernel::EventQueue::KernelEqueue>(command.eq);
				auto        result  = LibKernel::EventQueue::KernelTriggerEvent(
				    eq, static_cast<uintptr_t>(command.id),
				    LibKernel::EventQueue::KERNEL_EVFILT_AMPR,
				    reinterpret_cast<void*>(command.data));
				if (result != OK) {
					Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprErrors);
					LOGF("\tAPR submit event failed: eq=0x%016" PRIx64 ", id=%" PRId32
					     ", result=0x%08" PRIx32 "\n",
					     command.eq, command.id, static_cast<uint32_t>(result));
					*execution_result = result;
					*error_offset     = static_cast<uint32_t>(command.record_offset);
					return finish(OK);
				}
			} break;
			case CommandKind::WriteAddress: {
				const auto& command = state.write_address_commands[entry.index];
				if (!AprShared::WriteGuest(command.address, command.value)) {
					Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprErrors);
					LOGF("\tAMPR submit write-address failed: address=0x%016" PRIx64
					     " value=0x%016" PRIx64 "\n",
					     command.address, command.value);
					*execution_result = LibKernel::KERNEL_ERROR_EFAULT;
					*error_offset     = static_cast<uint32_t>(command.record_offset);
					return finish(OK);
				}
				NotifySyncWaiters();
			} break;
			case CommandKind::Sync: {
				const auto& command = state.sync_commands[entry.index];
				const auto  result  = ExecuteSyncCommand(command_buffer, command);
				if (result != OK) {
					Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprErrors);
					*execution_result = result;
					*error_offset     = static_cast<uint32_t>(command.record_offset);
					return finish(OK);
				}
			} break;
			case CommandKind::AmmMap: {
				const auto& command = state.amm_map_commands[entry.index];
				int         result  = OK;
				if (command.kind == AmmCommandKind::Unmap) {
					Profiler::ScopedLoadingOperation unmap(Profiler::LoadingOperation::AmmUnmap);
					result = LibKernel::Memory::KernelMunmap(command.va, command.size);
				} else {
					Profiler::ScopedLoadingOperation map(Profiler::LoadingOperation::AmmMap);
					result = ExecuteAmmMapCommand(command);
				}

				if (result != OK) {
					Profiler::CountLoadingEvent(Profiler::LoadingEvent::AmmErrors);
					Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprErrors);
					LOGF("\tAMM submit command failed: kind=%u va=0x%016" PRIx64
					     " dmem=0x%016" PRIx64 " size=0x%016" PRIx64 " type=%" PRId32
					     " prot=0x%08" PRIx32 " result=0x%08" PRIx32 "\n",
					     static_cast<uint32_t>(command.kind), command.va, command.dmem_offset,
					     command.size, command.type, static_cast<uint32_t>(command.prot),
					     static_cast<uint32_t>(result));
					*execution_result = result;
					*error_offset     = static_cast<uint32_t>(command.record_offset);
					return finish(OK);
				}
			} break;
		}
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprRecordsCompleted);
	}

	return finish(OK);
}

static bool AdvanceCommandBuffer(uint64_t command_buffer, uint64_t record_size) {
	std::scoped_lock      lock(g_command_buffer_mutex);
	CommandBufferIterator it;
	if (!GetOrCreateCommandBufferStateLocked(command_buffer, &it)) {
		return false;
	}

	auto& state = it->second;
	if (!EnsureCommandBufferRecordSpace(command_buffer, &state, record_size)) {
		return false;
	}

	std::memset(reinterpret_cast<void*>(state.buffer + state.write_offset), 0,
	            static_cast<size_t>(record_size));
	return CommitCommandBufferRecord(command_buffer, &state, record_size);
}

static int ReadHostFileToGuest(const std::string& host_path, uint64_t file_offset,
                               uint64_t destination, uint64_t size, uint64_t* bytes_read) {
	Profiler::ScopedLoadingOperation loading(Profiler::LoadingOperation::AprRead);
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprRequestedBytes, size);
	if (bytes_read == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	*bytes_read = 0;
	if (size == 0) {
		return OK;
	}
	if (!AprShared::IsValidGuestRange(destination, size, true)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	Common::File file;
	if (!file.Open(host_path, Common::File::Mode::Read)) {
		LOGF("\tAPR read missing host file: %s\n", host_path.c_str());
		return LibKernel::KERNEL_ERROR_ENOENT;
	}
	const auto file_size = file.Size();
	if (file_offset >= file_size) {
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprShortReads);
		file.Close();
		return OK;
	}
	if (!file.Seek(file_offset)) {
		file.Close();
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	const auto readable = std::min<uint64_t>(size, file_size - file_offset);
	if (readable == 0) {
		file.Close();
		return OK;
	}

	std::vector<uint8_t> buffer(
	    static_cast<size_t>(std::min<uint64_t>(APR_HOST_READ_CHUNK_SIZE, readable)));
	while (*bytes_read < readable) {
		const auto request =
		    static_cast<uint32_t>(std::min<uint64_t>(buffer.size(), readable - *bytes_read));
		uint32_t read = 0;
		{
			Profiler::ScopedLoadingOperation host_read(Profiler::LoadingOperation::AprHostRead);
			HangWatchdog::Scope wait("ampr-host-read", destination + *bytes_read, request,
			                         *bytes_read, file_offset + *bytes_read);
			file.Read(buffer.data(), request, &read);
		}
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprHostReadBytes, read);
		if (read == 0) {
			break;
		}
		{
			Profiler::ScopedLoadingOperation guest_copy(Profiler::LoadingOperation::AprGuestCopy);
			HangWatchdog::Scope copy("ampr-guest-copy", destination + *bytes_read, read);
			std::memcpy(reinterpret_cast<void*>(destination + *bytes_read), buffer.data(), read);
		}
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprGuestCopiedBytes, read);
		*bytes_read += read;
	}

	file.Close();
	if (*bytes_read < size) {
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprShortReads);
	}
	return OK;
}

static int KYTY_SYSV_ABI CommandBufferConstructor(void* command_buffer) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return OK;
	}

	return InitializeBaseCommandBuffer(reinterpret_cast<uint64_t>(command_buffer))
	           ? OK
	           : LibKernel::KERNEL_ERROR_EFAULT;
}

static int KYTY_SYSV_ABI AprCommandBufferConstructor(void* command_buffer, void* reserved_state0,
                                                     void* reserved_state1) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return OK;
	}

	return InitializeAprReservedState(reinterpret_cast<uint64_t>(command_buffer),
	                                  reinterpret_cast<uint64_t>(reserved_state0),
	                                  reinterpret_cast<uint64_t>(reserved_state1))
	           ? OK
	           : LibKernel::KERNEL_ERROR_EFAULT;
}

static int KYTY_SYSV_ABI AprCommandBufferDestructor(void* command_buffer) {
	PRINT_NAME();

	(void)command_buffer;
	return OK;
}

static int KYTY_SYSV_ABI CommandBufferDestructor(void* command_buffer) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return OK;
	}

	const auto       command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	std::scoped_lock lock(g_command_buffer_mutex);
	EraseCommandBufferStateLocked(command_buffer_addr);
	return OK;
}

static int KYTY_SYSV_ABI CommandBufferSetBuffer(void* command_buffer, void* buffer, uint32_t size) {
	PRINT_NAME();
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprCommandBufferSets);

	if (command_buffer == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	const auto command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	if (!ValidateCommandBufferHeader(command_buffer_addr)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	if (buffer == nullptr || (reinterpret_cast<uint64_t>(buffer) & 0x3u) != 0 || size == 0 ||
	    size > COMMAND_BUFFER_SIZE_MAX || (size & 0x3u) != 0) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	const auto old_buffer =
	    ReadCommandBufferUnchecked<uint64_t>(command_buffer_addr + COMMAND_BUFFER_DATA_OFFSET);
	if (old_buffer != 0) {
		return LibKernel::KERNEL_ERROR_EBUSY;
	}

	if (!WriteCommandBufferPointers(command_buffer_addr, reinterpret_cast<uint64_t>(buffer),
	                                size)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	WriteCommandBufferUnchecked(command_buffer_addr + COMMAND_BUFFER_NUM_OFFSET, int32_t {0});
	return OK;
}

static int KYTY_SYSV_ABI CommandBufferReset(void* command_buffer) {
	PRINT_NAME();
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprCommandResets);

	if (command_buffer == nullptr) {
		return LibKernel::KERNEL_ERROR_EPERM;
	}

	const auto         command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	CommandBufferState state {};
	if (!TryGetCommandBufferState(command_buffer_addr, &state)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	if (state.buffer == 0) {
		return LibKernel::KERNEL_ERROR_EPERM;
	}

	if (!WriteCommandBufferPointers(command_buffer_addr, state.buffer, state.size)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	WriteCommandBufferUnchecked(command_buffer_addr + COMMAND_BUFFER_NUM_OFFSET, int32_t {0});
	return OK;
}

static void* KYTY_SYSV_ABI CommandBufferClearBuffer(void* command_buffer) {
	PRINT_NAME();
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprCommandBufferClears);

	if (command_buffer == nullptr) {
		return nullptr;
	}

	const auto command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	if (!ValidateCommandBufferHeader(command_buffer_addr)) {
		return nullptr;
	}
	const auto buffer =
	    ReadCommandBufferUnchecked<uint64_t>(command_buffer_addr + COMMAND_BUFFER_DATA_OFFSET);
	const auto size =
	    ReadCommandBufferUnchecked<uint32_t>(command_buffer_addr + COMMAND_BUFFER_SIZE_OFFSET);
	if (buffer == 0 || size == 0) {
		return nullptr;
	}

	if (!WriteVisibleCommandBufferPointers(command_buffer_addr, 0, 0)) {
		return nullptr;
	}

	std::scoped_lock lock(g_command_buffer_mutex);
	EraseCommandBufferStateLocked(command_buffer_addr);
	return reinterpret_cast<void*>(buffer);
}

static int KYTY_SYSV_ABI AprCommandBufferReadFile(void*    command_buffer, uint64_t, uint64_t,
                                                  uint32_t file_id, void* destination,
                                                  uint64_t size, uint64_t file_offset) {
	PRINT_NAME();

	const auto destination_addr = reinterpret_cast<uint64_t>(destination);
	if (command_buffer == nullptr || !IsValidAprReadRange(destination_addr, size) ||
	    !IsValidAprFileOffset(file_offset)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	const auto append_ok = AppendReadFileRecord(reinterpret_cast<uint64_t>(command_buffer), 0x17,
	                                            file_id, destination_addr, size, file_offset,
	                                            AprReadFileRecordSize(file_offset));
	return append_ok ? OK : LibKernel::KERNEL_ERROR_EFAULT;
}

static uint64_t AlignUp4(uint64_t value) {
	return (value + 3u) & ~uint64_t {3};
}

static uint64_t MarkerCommandSize(const char* msg, bool with_color) {
	const uint64_t msg_size = (msg == nullptr ? 0 : std::strlen(msg) + 1);
	return AlignUp4((with_color ? 8u : 4u) + msg_size);
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeReadFile(uint64_t, uint64_t destination,
                                                         uint64_t size, uint64_t file_offset) {
	PRINT_NAME();

	return IsValidAprReadRange(destination, size) && IsValidAprFileOffset(file_offset)
	           ? AprReadFileRecordSize(file_offset)
	           : KernelErrorU64(LibKernel::KERNEL_ERROR_EINVAL);
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeWriteKernelEventQueue(uint64_t, uint64_t, uint64_t,
                                                                      uint64_t, uint64_t,
                                                                      uint64_t) {
	PRINT_NAME();

	return KERNEL_EVENT_RECORD_SIZE;
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeFixed32(uint64_t, uint64_t, uint64_t, uint64_t,
                                                        uint64_t, uint64_t) {
	PRINT_NAME();

	return 0x20;
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeWriteAddressOnCompletion(volatile uint64_t*,
                                                                         uint64_t) {
	PRINT_NAME();

	return 0x20;
}

static uint64_t KYTY_SYSV_ABI
MeasureCommandSizeWriteAddressFromTimeCounterOnCompletion(volatile uint64_t*) {
	PRINT_NAME();

	return 0x20;
}

static uint64_t KYTY_SYSV_ABI
MeasureCommandSizeWriteAddressFromCounterOnCompletion(volatile uint64_t*, uint8_t) {
	PRINT_NAME();

	return 0x20;
}

static uint64_t KYTY_SYSV_ABI
MeasureCommandSizeWriteAddressFromCounterPairOnCompletion(volatile uint64_t*, uint8_t) {
	PRINT_NAME();

	return 0x20;
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeWriteCounterOnCompletion(uint8_t, uint8_t, uint64_t,
                                                                         uint8_t) {
	PRINT_NAME();

	return 0x20;
}

static int64_t KYTY_SYSV_ABI MeasureAmmCommandSizeMap(uint64_t, uint64_t, int32_t, int32_t) {
	PRINT_NAME();

	return AMM_MAP_RECORD_SIZE;
}

static int64_t KYTY_SYSV_ABI MeasureAmmCommandSizeMapWithGpuMaskId(uint64_t, uint64_t, int32_t,
                                                                   int32_t, uint8_t) {
	PRINT_NAME();

	return AMM_MAP_RECORD_SIZE;
}

static int64_t KYTY_SYSV_ABI MeasureAmmCommandSizeMapDirect(uint64_t, uint64_t, uint64_t, int32_t,
                                                            int32_t) {
	PRINT_NAME();

	return AMM_MAP_DIRECT_RECORD_SIZE;
}

static int64_t KYTY_SYSV_ABI MeasureAmmCommandSizeMapDirectWithGpuMaskId(uint64_t, uint64_t,
                                                                         uint64_t, int32_t, int32_t,
                                                                         uint8_t) {
	PRINT_NAME();

	return AMM_MAP_DIRECT_RECORD_SIZE;
}

static int64_t KYTY_SYSV_ABI MeasureAmmCommandSizeUnmap(uint64_t, uint64_t) {
	PRINT_NAME();

	return AMM_UNMAP_RECORD_SIZE;
}

static int64_t KYTY_SYSV_ABI MeasureAprCommandSizeMapBegin(uint64_t va, uint64_t size, int32_t,
                                                           int32_t) {
	PRINT_NAME();

	return ValidateAmmMapArgs(va, size) ? APR_MAP_BEGIN_RECORD_SIZE
	                                    : LibKernel::KERNEL_ERROR_EINVAL;
}

static int64_t KYTY_SYSV_ABI MeasureAprCommandSizeMapDirectBegin(uint64_t va, uint64_t dmem_offset,
                                                                 uint64_t size, int32_t, int32_t) {
	PRINT_NAME();

	return ValidateAmmMapArgs(va, size) && (dmem_offset & (AMM_PAGE_SIZE - 1u)) == 0
	           ? APR_MAP_DIRECT_BEGIN_SIZE
	           : LibKernel::KERNEL_ERROR_EINVAL;
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeNop(uint32_t num_u32) {
	PRINT_NAME();

	return num_u32 == 0 ? sizeof(uint32_t)
	                    : AlignUp4(static_cast<uint64_t>(num_u32) * sizeof(uint32_t));
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeNopWithData(uint32_t num_u32, const uint32_t*) {
	PRINT_NAME();

	return AlignUp4((static_cast<uint64_t>(num_u32) + 1u) * sizeof(uint32_t));
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeReadFileGather(uint64_t size,
                                                               uint64_t file_offset) {
	PRINT_NAME();

	return IsValidAprReadSize(size) && IsValidAprFileOffset(file_offset)
	           ? AprReadGatherRecordSize(file_offset)
	           : KernelErrorU64(LibKernel::KERNEL_ERROR_EINVAL);
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeReadFileScatter(uint64_t destination,
                                                                uint64_t size) {
	PRINT_NAME();

	return IsValidAprReadRange(destination, size) ? READ_SCATTER_RECORD_SIZE
	                                              : KernelErrorU64(LibKernel::KERNEL_ERROR_EINVAL);
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeReadFileGatherScatter(uint64_t destination,
                                                                      uint64_t size,
                                                                      uint64_t file_offset) {
	PRINT_NAME();

	return IsValidAprReadRange(destination, size) && IsValidAprFileOffset(file_offset)
	           ? AprReadGatherScatterRecordSize(file_offset)
	           : KernelErrorU64(LibKernel::KERNEL_ERROR_EINVAL);
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeMarkerWithColor(const char* msg, uint32_t) {
	PRINT_NAME();

	return MarkerCommandSize(msg, true);
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizeMarker(const char* msg) {
	PRINT_NAME();

	return MarkerCommandSize(msg, false);
}

static uint64_t KYTY_SYSV_ABI MeasureCommandSizePopMarker() {
	PRINT_NAME();

	return sizeof(uint32_t);
}

static uint64_t KYTY_SYSV_ABI CommandBufferGetSize(void* command_buffer) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return 0;
	}

	uint32_t size = 0;
	AprShared::ReadGuest(reinterpret_cast<uint64_t>(command_buffer) + COMMAND_BUFFER_SIZE_OFFSET,
	                     &size);
	return size;
}

static uint64_t KYTY_SYSV_ABI CommandBufferGetCurrentOffset(void* command_buffer) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return 0;
	}

	uint32_t offset = 0;
	AprShared::ReadGuest(reinterpret_cast<uint64_t>(command_buffer) + COMMAND_BUFFER_OFFSET_OFFSET,
	                     &offset);
	return offset;
}

static int KYTY_SYSV_ABI CommandBufferGetType(void* command_buffer) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return 0;
	}

	uint32_t type = 0;
	AprShared::ReadGuest(reinterpret_cast<uint64_t>(command_buffer) + COMMAND_BUFFER_TYPE_OFFSET,
	                     &type);
	return static_cast<int>(type);
}

static void* KYTY_SYSV_ABI CommandBufferGetBufferBaseAddress(void* command_buffer) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return nullptr;
	}

	uint64_t buffer = 0;
	AprShared::ReadGuest(reinterpret_cast<uint64_t>(command_buffer) + COMMAND_BUFFER_DATA_OFFSET,
	                     &buffer);
	return reinterpret_cast<void*>(buffer);
}

static uint32_t KYTY_SYSV_ABI CommandBufferGetNumCommands(void* command_buffer) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return 0;
	}

	uint32_t num = 0;
	AprShared::ReadGuest(reinterpret_cast<uint64_t>(command_buffer) + COMMAND_BUFFER_NUM_OFFSET,
	                     &num);
	return num;
}

static int AppendNoOpCommand(void* command_buffer, uint64_t size) {
	if (command_buffer == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	return AdvanceCommandBuffer(reinterpret_cast<uint64_t>(command_buffer), AlignUp4(size))
	           ? OK
	           : LibKernel::KERNEL_ERROR_EFAULT;
}

static int KYTY_SYSV_ABI CommandBufferWriteKernelEventQueue(void* command_buffer, uint64_t eq,
                                                            uint64_t id, uint64_t data, uint64_t,
                                                            uint64_t) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	return AppendKernelEventRecord(reinterpret_cast<uint64_t>(command_buffer), eq,
	                               static_cast<int32_t>(id), data)
	           ? OK
	           : LibKernel::KERNEL_ERROR_EFAULT;
}

static int KYTY_SYSV_ABI CommandBufferNop(void* command_buffer, uint32_t num_u32) {
	PRINT_NAME();

	if (num_u32 == 0 || num_u32 > 16) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}
	return AppendNoOpCommand(command_buffer, static_cast<uint64_t>(num_u32) * sizeof(uint32_t));
}

static int KYTY_SYSV_ABI CommandBufferNopWithData(void* command_buffer, uint32_t num_u32,
                                                  const uint32_t*) {
	PRINT_NAME();

	if (num_u32 > 15) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}
	return AppendNoOpCommand(command_buffer,
	                         (static_cast<uint64_t>(num_u32) + 1u) * sizeof(uint32_t));
}

static int KYTY_SYSV_ABI CommandBufferConstructNop(void*    command_buffer, uint32_t, const void*,
                                                   uint32_t bytes, const uint32_t*) {
	PRINT_NAME();

	return AppendNoOpCommand(command_buffer, sizeof(uint32_t) + bytes);
}

static int KYTY_SYSV_ABI CommandBufferConstructMarker(void*       command_buffer, uint32_t,
                                                      const char* msg, const uint32_t* color) {
	PRINT_NAME();

	return AppendNoOpCommand(command_buffer, MarkerCommandSize(msg, color != nullptr));
}

static int KYTY_SYSV_ABI CommandBufferSetMarker(void* command_buffer, const char* msg) {
	PRINT_NAME();

	return AppendNoOpCommand(command_buffer, MarkerCommandSize(msg, false));
}

static int KYTY_SYSV_ABI CommandBufferSetMarkerWithColor(void* command_buffer, const char* msg,
                                                         uint32_t) {
	PRINT_NAME();

	return AppendNoOpCommand(command_buffer, MarkerCommandSize(msg, true));
}

static int KYTY_SYSV_ABI CommandBufferPushMarker(void* command_buffer, const char* msg) {
	PRINT_NAME();

	return AppendNoOpCommand(command_buffer, MarkerCommandSize(msg, false));
}

static int KYTY_SYSV_ABI CommandBufferPushMarkerWithColor(void* command_buffer, const char* msg,
                                                          uint32_t) {
	PRINT_NAME();

	return AppendNoOpCommand(command_buffer, MarkerCommandSize(msg, true));
}

static int KYTY_SYSV_ABI CommandBufferPopMarker(void* command_buffer) {
	PRINT_NAME();

	return AppendNoOpCommand(command_buffer, sizeof(uint32_t));
}

// A counter/wait command whose operand encoding is outside the decoded tables in
// amprCounterBank.h. It keeps the historical behaviour (a 0x20-byte no-op record)
// so an unknown encoding never turns into a new guest-visible error, and is
// counted in AprUnsupportedWaitCommands / AprUnsupportedCounterCommands, which
// therefore only count encodings that are still not implemented.
static int AppendUnsupportedSyncCommand(void* command_buffer, bool is_wait, const char* name,
                                        uint32_t index, uint32_t access, uint32_t op,
                                        uint32_t mask_op) {
	Profiler::CountLoadingEvent(is_wait ? Profiler::LoadingEvent::AprUnsupportedWaitCommands
	                                    : Profiler::LoadingEvent::AprUnsupportedCounterCommands);
	if (g_sync_unsupported_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		LOGF("\tAMPR %s: unsupported encoding index=%u access=%u op=%u mask_op=%u; recorded as a "
		     "no-op\n",
		     name, index, access, op, mask_op);
	}
	return AppendNoOpCommand(command_buffer, SYNC_RECORD_SIZE);
}

static const char* SyncKindName(CommandBufferState::SyncKind kind) {
	switch (kind) {
		case CommandBufferState::SyncKind::WriteCounter: return "WriteCounter";
		case CommandBufferState::SyncKind::WaitOnCounter: return "WaitOnCounter";
		case CommandBufferState::SyncKind::WaitOnAddress: return "WaitOnAddress";
		case CommandBufferState::SyncKind::WriteAddressFromCounter: return "WriteAddressFromCounter";
		case CommandBufferState::SyncKind::WriteAddressFromCounterPair:
			return "WriteAddressFromCounterPair";
		case CommandBufferState::SyncKind::WriteAddressFromTimeCounter:
			return "WriteAddressFromTimeCounter";
	}
	return "?";
}

// Implemented (decoded) counter/wait commands are not CSV columns (that would
// need new Profiler::LoadingEvent entries); the first recording of each kind is
// logged so a run's log shows which of them a title actually uses.
static std::array<std::atomic<uint64_t>, 6> g_sync_recorded {};

static int AppendSyncCommand(void* command_buffer, const CommandBufferState::SyncCommand& command,
                             int append_error) {
	if (!AppendSyncRecord(reinterpret_cast<uint64_t>(command_buffer), command)) {
		return append_error;
	}
	const auto kind = static_cast<size_t>(command.kind);
	if (kind < g_sync_recorded.size() &&
	    g_sync_recorded[kind].fetch_add(1, std::memory_order_relaxed) == 0) {
		LOGF("\tAMPR first %s command recorded: cb=%p index=%u access=%u op=%u mask_op=%u "
		     "address=0x%016" PRIx64 "\n",
		     SyncKindName(command.kind), command_buffer,
		     static_cast<uint32_t>(command.counter_index), static_cast<uint32_t>(command.access),
		     static_cast<uint32_t>(command.op), static_cast<uint32_t>(command.mask_op),
		     command.address);
	}
	return OK;
}

static int KYTY_SYSV_ABI CommandBufferWaitOnAddress(void*              command_buffer,
                                                    volatile uint64_t* address, uint64_t reference,
                                                    uint8_t compare, uint8_t /*flush*/) {
	PRINT_NAME();

	if (command_buffer == nullptr || address == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}
	if (!CounterBank::IsValidCompare(compare)) {
		return AppendUnsupportedSyncCommand(command_buffer, true, "WaitOnAddress", 0, 0, compare, 0);
	}

	CommandBufferState::SyncCommand command {};
	command.kind    = CommandBufferState::SyncKind::WaitOnAddress;
	command.op      = compare;
	command.value   = reference;
	command.address = reinterpret_cast<uint64_t>(address);
	return AppendSyncCommand(command_buffer, command, LibKernel::KERNEL_ERROR_EFAULT);
}

static int KYTY_SYSV_ABI CommandBufferWaitOnCounter(void* command_buffer, uint8_t counter_index,
                                                    uint8_t access, uint64_t reference,
                                                    uint8_t compare, uint8_t mask_op, uint64_t mask,
                                                    uint8_t /*flush*/) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}
	if (!CounterBank::DecodeLane(counter_index, access, nullptr) ||
	    !CounterBank::IsValidCompare(compare) || !CounterBank::IsValidMaskOp(mask_op)) {
		return AppendUnsupportedSyncCommand(command_buffer, true, "WaitOnCounter", counter_index,
		                                    access, compare, mask_op);
	}

	CommandBufferState::SyncCommand command {};
	command.kind          = CommandBufferState::SyncKind::WaitOnCounter;
	command.counter_index = counter_index;
	command.access        = access;
	command.op            = compare;
	command.mask_op       = mask_op;
	command.value         = reference;
	command.mask          = mask;
	return AppendSyncCommand(command_buffer, command, LibKernel::KERNEL_ERROR_EFAULT);
}

static int AppendWriteCounterCommand(void* command_buffer, const char* name, uint8_t counter_index,
                                     uint8_t access, uint64_t value, uint8_t op) {
	if (command_buffer == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}
	if (!CounterBank::DecodeLane(counter_index, access, nullptr) ||
	    !CounterBank::IsValidWriteOp(op)) {
		return AppendUnsupportedSyncCommand(command_buffer, false, name, counter_index, access, op,
		                                    0);
	}

	CommandBufferState::SyncCommand command {};
	command.kind          = CommandBufferState::SyncKind::WriteCounter;
	command.counter_index = counter_index;
	command.access        = access;
	command.op            = op;
	command.value         = value;
	return AppendSyncCommand(command_buffer, command, LibKernel::KERNEL_ERROR_EFAULT);
}

// WriteAddressFromCounter writes the 32-bit counter zero-extended to 64 bits;
// the Pair form writes counter N in the low word and counter N+1 in the high word;
// the time-counter form writes the process time counter at execution time.
static int AppendWriteAddressFromSyncCommand(void* command_buffer, const char* name,
                                             volatile uint64_t* address,
                                             CommandBufferState::SyncKind kind,
                                             uint8_t counter_index) {
	if (command_buffer == nullptr || address == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	CommandBufferState::SyncCommand command {};
	command.kind          = kind;
	command.counter_index = counter_index;
	command.address       = reinterpret_cast<uint64_t>(address);
	if (kind != CommandBufferState::SyncKind::WriteAddressFromTimeCounter) {
		command.access = kind == CommandBufferState::SyncKind::WriteAddressFromCounterPair
		                     ? CounterBank::ACCESS_PAIR_64
		                     : CounterBank::ACCESS_32;
		if (!CounterBank::DecodeLane(counter_index, command.access, nullptr)) {
			return AppendUnsupportedSyncCommand(command_buffer, false, name, counter_index,
			                                    command.access, 0, 0);
		}
	}
	return AppendSyncCommand(command_buffer, command, LibKernel::KERNEL_ERROR_EBUSY);
}

static int AppendWriteAddressCommand(void* command_buffer, volatile uint64_t* address,
                                     uint64_t value) {
	if (command_buffer == nullptr || address == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}
	return AppendWriteAddressRecord(reinterpret_cast<uint64_t>(command_buffer),
	                                reinterpret_cast<uint64_t>(address), value)
	           ? OK
	           : LibKernel::KERNEL_ERROR_EBUSY;
}

static int KYTY_SYSV_ABI CommandBufferWriteAddress(void* command_buffer, volatile uint64_t* address,
                                                   uint64_t value, uint32_t) {
	PRINT_NAME();

	return AppendWriteAddressCommand(command_buffer, address, value);
}

static int KYTY_SYSV_ABI CommandBufferWriteAddressOnCompletion(void*              command_buffer,
                                                               volatile uint64_t* address,
                                                               uint64_t           value) {
	PRINT_NAME();

	return AppendWriteAddressCommand(command_buffer, address, value);
}

static int KYTY_SYSV_ABI CommandBufferWriteCounter(void* command_buffer, uint8_t counter_index,
                                                   uint8_t access, uint64_t value, uint8_t op,
                                                   uint32_t /*flags*/) {
	PRINT_NAME();

	return AppendWriteCounterCommand(command_buffer, "WriteCounter", counter_index, access, value,
	                                 op);
}

// Takes effect after every earlier record in the buffer has completed, which is
// what the synchronous executor does for every record (see the execution-model
// note above ExecuteSyncCommand).
static int KYTY_SYSV_ABI CommandBufferWriteCounterOnCompletion(void*    command_buffer,
                                                               uint8_t  counter_index,
                                                               uint8_t  access, uint64_t value,
                                                               uint8_t  op) {
	PRINT_NAME();

	return AppendWriteCounterCommand(command_buffer, "WriteCounterOnCompletion", counter_index,
	                                 access, value, op);
}

static int KYTY_SYSV_ABI CommandBufferWriteAddressFromTimeCounter(void*              command_buffer,
                                                                  volatile uint64_t* address,
                                                                  uint32_t /*flags*/) {
	PRINT_NAME();

	return AppendWriteAddressFromSyncCommand(
	    command_buffer, "WriteAddressFromTimeCounter", address,
	    CommandBufferState::SyncKind::WriteAddressFromTimeCounter, 0);
}

static int KYTY_SYSV_ABI CommandBufferWriteAddressFromTimeCounterOnCompletion(
    void* command_buffer, volatile uint64_t* address) {
	PRINT_NAME();

	return AppendWriteAddressFromSyncCommand(
	    command_buffer, "WriteAddressFromTimeCounterOnCompletion", address,
	    CommandBufferState::SyncKind::WriteAddressFromTimeCounter, 0);
}

static int KYTY_SYSV_ABI CommandBufferWriteAddressFromCounter(void*              command_buffer,
                                                              volatile uint64_t* address,
                                                              uint8_t            counter_index,
                                                              uint32_t /*flags*/) {
	PRINT_NAME();

	return AppendWriteAddressFromSyncCommand(command_buffer, "WriteAddressFromCounter", address,
	                                         CommandBufferState::SyncKind::WriteAddressFromCounter,
	                                         counter_index);
}

static int KYTY_SYSV_ABI CommandBufferWriteAddressFromCounterOnCompletion(
    void* command_buffer, volatile uint64_t* address, uint8_t counter_index) {
	PRINT_NAME();

	return AppendWriteAddressFromSyncCommand(
	    command_buffer, "WriteAddressFromCounterOnCompletion", address,
	    CommandBufferState::SyncKind::WriteAddressFromCounter, counter_index);
}

static int KYTY_SYSV_ABI CommandBufferWriteAddressFromCounterPair(void*              command_buffer,
                                                                  volatile uint64_t* address,
                                                                  uint8_t            counter_index,
                                                                  uint32_t /*flags*/) {
	PRINT_NAME();

	return AppendWriteAddressFromSyncCommand(
	    command_buffer, "WriteAddressFromCounterPair", address,
	    CommandBufferState::SyncKind::WriteAddressFromCounterPair, counter_index);
}

static int KYTY_SYSV_ABI CommandBufferWriteAddressFromCounterPairOnCompletion(
    void* command_buffer, volatile uint64_t* address, uint8_t counter_index) {
	PRINT_NAME();

	return AppendWriteAddressFromSyncCommand(
	    command_buffer, "WriteAddressFromCounterPairOnCompletion", address,
	    CommandBufferState::SyncKind::WriteAddressFromCounterPair, counter_index);
}

static int KYTY_SYSV_ABI AprCommandBufferReadFileGather(void*    command_buffer, uint64_t, uint64_t,
                                                        uint64_t size, uint64_t file_offset) {
	PRINT_NAME();

	if (command_buffer == nullptr || !IsValidAprReadSize(size) ||
	    !IsValidAprFileOffset(file_offset)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	CommandBufferState state {};
	const auto         command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	if (!TryGetCommandBufferState(command_buffer_addr, &state)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	if (!state.gather_scatter_valid ||
	    !IsValidAprReadRange(state.gather_scatter_destination, size)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	return AppendReadFileRecord(command_buffer_addr, 0x18, state.gather_scatter_file_id,
	                            state.gather_scatter_destination, size, file_offset,
	                            AprReadGatherRecordSize(file_offset))
	           ? OK
	           : LibKernel::KERNEL_ERROR_EFAULT;
}

static int KYTY_SYSV_ABI AprCommandBufferReadFileScatter(void* command_buffer, uint64_t, uint64_t,
                                                         void* destination, uint64_t size) {
	PRINT_NAME();

	const auto destination_addr = reinterpret_cast<uint64_t>(destination);
	if (command_buffer == nullptr || !IsValidAprReadRange(destination_addr, size)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	CommandBufferState state {};
	const auto         command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	if (!TryGetCommandBufferState(command_buffer_addr, &state)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	if (!state.gather_scatter_valid || !IsValidAprFileOffset(state.gather_scatter_file_offset)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	return AppendReadFileRecord(command_buffer_addr, 0x19, state.gather_scatter_file_id,
	                            destination_addr, size, state.gather_scatter_file_offset,
	                            READ_SCATTER_RECORD_SIZE)
	           ? OK
	           : LibKernel::KERNEL_ERROR_EFAULT;
}

static int KYTY_SYSV_ABI AprCommandBufferReadFileGatherScatter(void* command_buffer, uint64_t,
                                                               uint64_t, void* destination,
                                                               uint64_t size,
                                                               uint64_t file_offset) {
	PRINT_NAME();

	const auto destination_addr = reinterpret_cast<uint64_t>(destination);
	if (command_buffer == nullptr || !IsValidAprReadRange(destination_addr, size) ||
	    !IsValidAprFileOffset(file_offset)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	CommandBufferState state {};
	const auto         command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	if (!TryGetCommandBufferState(command_buffer_addr, &state)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	if (!state.gather_scatter_valid) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	return AppendReadFileRecord(command_buffer_addr, 0x1a, state.gather_scatter_file_id,
	                            destination_addr, size, file_offset,
	                            AprReadGatherScatterRecordSize(file_offset))
	           ? OK
	           : LibKernel::KERNEL_ERROR_EFAULT;
}

static int KYTY_SYSV_ABI AprCommandBufferResetGatherScatterState(void* command_buffer, uint64_t,
                                                                 uint64_t) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	const auto command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	if (AppendNoOpCommand(command_buffer, RESET_GATHER_SCATTER_SIZE) != OK) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	std::scoped_lock lock(g_command_buffer_mutex);
	auto             it = g_command_buffers.find(command_buffer_addr);
	if (it != g_command_buffers.end()) {
		it->second.gather_scatter_valid       = false;
		it->second.gather_scatter_file_id     = 0;
		it->second.gather_scatter_destination = 0;
		it->second.gather_scatter_file_offset = 0;
	}
	UpdateCommandBufferTypeFlags(command_buffer_addr, 0, APR_TYPE_GATHER_SCATTER_VALID);
	return OK;
}

static int KYTY_SYSV_ABI AprCommandBufferMapBegin(void* command_buffer, uint64_t va, uint64_t size,
                                                  int32_t, int32_t) {
	PRINT_NAME();

	if (command_buffer == nullptr || !ValidateAmmMapArgs(va, size)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	if (AppendNoOpCommand(command_buffer, APR_MAP_BEGIN_RECORD_SIZE) != OK) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	UpdateCommandBufferTypeFlags(reinterpret_cast<uint64_t>(command_buffer), APR_TYPE_MAP_ACTIVE,
	                             0);
	return OK;
}

static int KYTY_SYSV_ABI AprCommandBufferMapDirectBegin(void* command_buffer, uint64_t va,
                                                        uint64_t dmem_offset, uint64_t size,
                                                        int32_t, int32_t) {
	PRINT_NAME();

	if (command_buffer == nullptr || !ValidateAmmMapArgs(va, size) ||
	    (dmem_offset & (AMM_PAGE_SIZE - 1u)) != 0) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	if (AppendNoOpCommand(command_buffer, APR_MAP_DIRECT_BEGIN_SIZE) != OK) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	UpdateCommandBufferTypeFlags(reinterpret_cast<uint64_t>(command_buffer), APR_TYPE_MAP_ACTIVE,
	                             0);
	return OK;
}

static int KYTY_SYSV_ABI AprCommandBufferMapEnd(void* command_buffer) {
	PRINT_NAME();

	if (command_buffer == nullptr) {
		return LibKernel::KERNEL_ERROR_EPERM;
	}

	uint32_t   type                = 0;
	const auto command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer);
	if (!AprShared::ReadGuest(command_buffer_addr + COMMAND_BUFFER_TYPE_OFFSET, &type)) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}
	if ((type & APR_TYPE_MAP_ACTIVE) == 0) {
		return LibKernel::KERNEL_ERROR_EPERM;
	}

	if (AppendNoOpCommand(command_buffer, APR_MAP_END_RECORD_SIZE) != OK) {
		return LibKernel::KERNEL_ERROR_EFAULT;
	}

	UpdateCommandBufferTypeFlags(command_buffer_addr, 0, APR_TYPE_MAP_ACTIVE);
	return OK;
}

static int KYTY_SYSV_ABI AmmCommandBufferMap(void* command_buffer, uint64_t va, uint64_t size,
                                             int32_t type, int32_t prot) {
	PRINT_NAME();

	if (command_buffer == nullptr || !ValidateAmmMapArgs(va, size)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	CommandBufferState::AmmMapCommand command {};
	command.kind = AmmCommandKind::MapAuto;
	command.va   = va;
	command.size = size;
	command.type = type;
	command.prot = prot;

	return AppendAmmMapRecord(reinterpret_cast<uint64_t>(command_buffer), command,
	                          AMM_MAP_RECORD_SIZE)
	           ? OK
	           : LibKernel::KERNEL_ERROR_EBUSY;
}

static int KYTY_SYSV_ABI AmmCommandBufferMapWithGpuMaskId(void* command_buffer, uint64_t va,
                                                          uint64_t size, int32_t type, int32_t prot,
                                                          uint8_t gpu_mask_id) {
	PRINT_NAME();

	if (command_buffer == nullptr || !ValidateAmmMapArgs(va, size)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	CommandBufferState::AmmMapCommand command {};
	command.kind        = AmmCommandKind::MapAuto;
	command.va          = va;
	command.size        = size;
	command.type        = type;
	command.prot        = prot;
	command.gpu_mask_id = gpu_mask_id;

	return AppendAmmMapRecord(reinterpret_cast<uint64_t>(command_buffer), command,
	                          AMM_MAP_RECORD_SIZE)
	           ? OK
	           : LibKernel::KERNEL_ERROR_EBUSY;
}

static int KYTY_SYSV_ABI AmmCommandBufferMapDirect(void* command_buffer, uint64_t va,
                                                   uint64_t dmem_offset, uint64_t size,
                                                   int32_t type, int32_t prot) {
	PRINT_NAME();

	if (command_buffer == nullptr || !ValidateAmmMapArgs(va, size) ||
	    (dmem_offset & (AMM_PAGE_SIZE - 1u)) != 0) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	CommandBufferState::AmmMapCommand command {};
	command.kind        = AmmCommandKind::MapDirect;
	command.va          = va;
	command.dmem_offset = dmem_offset;
	command.size        = size;
	command.type        = type;
	command.prot        = prot;

	return AppendAmmMapRecord(reinterpret_cast<uint64_t>(command_buffer), command,
	                          AMM_MAP_DIRECT_RECORD_SIZE)
	           ? OK
	           : LibKernel::KERNEL_ERROR_EBUSY;
}

static int KYTY_SYSV_ABI AmmCommandBufferMapDirectWithGpuMaskId(void* command_buffer, uint64_t va,
                                                                uint64_t dmem_offset, uint64_t size,
                                                                int32_t type, int32_t prot,
                                                                uint8_t gpu_mask_id) {
	PRINT_NAME();

	if (command_buffer == nullptr || !ValidateAmmMapArgs(va, size) ||
	    (dmem_offset & (AMM_PAGE_SIZE - 1u)) != 0) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	CommandBufferState::AmmMapCommand command {};
	command.kind        = AmmCommandKind::MapDirect;
	command.va          = va;
	command.dmem_offset = dmem_offset;
	command.size        = size;
	command.type        = type;
	command.prot        = prot;
	command.gpu_mask_id = gpu_mask_id;

	return AppendAmmMapRecord(reinterpret_cast<uint64_t>(command_buffer), command,
	                          AMM_MAP_DIRECT_RECORD_SIZE)
	           ? OK
	           : LibKernel::KERNEL_ERROR_EBUSY;
}

static int KYTY_SYSV_ABI AmmCommandBufferUnmap(void* command_buffer, uint64_t va, uint64_t size) {
	PRINT_NAME();

	if (command_buffer == nullptr || !ValidateAmmMapArgs(va, size)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	CommandBufferState::AmmMapCommand command {};
	command.kind = AmmCommandKind::Unmap;
	command.va   = va;
	command.size = size;

	return AppendAmmMapRecord(reinterpret_cast<uint64_t>(command_buffer), command,
	                          AMM_UNMAP_RECORD_SIZE)
	           ? OK
	           : LibKernel::KERNEL_ERROR_EBUSY;
}

static int KYTY_SYSV_ABI AmmGiveDirectMemory(int64_t search_start, int64_t search_end,
                                             uint64_t size, uint64_t align, int32_t usage,
                                             int64_t* dmem_offset) {
	PRINT_NAME();

	if (dmem_offset == nullptr || size == 0 ||
	    (usage != AMM_USAGE_DIRECT && usage != AMM_USAGE_AUTO)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	int64_t allocated = 0;
	int     result = LibKernel::Memory::KernelAllocateDirectMemory(search_start, search_end, size,
	                                                               align, 0, &allocated);
	if (result != OK) {
		return result;
	}

	*dmem_offset = allocated;
	if (usage == AMM_USAGE_AUTO) {
		std::scoped_lock lock(g_amm_auto_pool_mutex);
		g_amm_auto_pool.push_back(AmmAutoPoolRange {static_cast<uint64_t>(allocated), size, 0});
	}

	return OK;
}

static void KYTY_SYSV_ABI AmmGetVirtualAddressRanges(uint64_t* va_start, uint64_t* va_end,
                                                     uint64_t* multimap_va_start,
                                                     uint64_t* multimap_va_end) {
	PRINT_NAME();

	if (va_start != nullptr) {
		AprShared::WriteGuest(reinterpret_cast<uint64_t>(va_start), AMM_VA_START);
	}
	if (va_end != nullptr) {
		AprShared::WriteGuest(reinterpret_cast<uint64_t>(va_end), AMM_VA_START + AMM_VA_SIZE);
	}
	if (multimap_va_start != nullptr) {
		AprShared::WriteGuest(reinterpret_cast<uint64_t>(multimap_va_start),
		                      AMM_VA_START + AMM_VA_SIZE / 2u);
	}
	if (multimap_va_end != nullptr) {
		AprShared::WriteGuest(reinterpret_cast<uint64_t>(multimap_va_end),
		                      AMM_VA_START + AMM_VA_SIZE);
	}
}

static int KYTY_SYSV_ABI AmmGetUsageStatsData(AmmUsageStatsData* stats) {
	PRINT_NAME();
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AmmUsageStatsQueries);

	if (stats == nullptr || stats->size_in_bytes > sizeof(AmmUsageStatsData)) {
		return LibKernel::KERNEL_ERROR_EINVAL;
	}

	AmmUsageStatsData out {};
	out.size_in_bytes               = stats->size_in_bytes;
	out.num_page_table_pool_entries = 512;
	out.ring_idle_flags             = 0x7;
	const auto write_size           = std::min<uint64_t>(stats->size_in_bytes, sizeof(out));
	return AprShared::WriteGuestBytes(reinterpret_cast<uint64_t>(stats), &out, write_size)
	           ? OK
	           : LibKernel::KERNEL_ERROR_EFAULT;
}

static int KYTY_SYSV_ABI AmmSetPageTablePoolOccupancyNotificationThreshold(uint32_t threshold) {
	PRINT_NAME();
	if (Profiler::LoadingEnabled()) {
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AmmOccupancyThresholdCalls);
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AmmOccupancyThresholdSum, threshold);
	}

	return OK;
}

static int KYTY_SYSV_ABI AmmSubmitCommandBuffer(void* command_buffer_base, uint32_t argument1,
                                               uint32_t argument2) {
	PRINT_NAME();
	HangTrace::SetGuestCaller(reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprSubmitPlain);
	AprShared::LoadingSubmissionDiagnostic diagnostic(
	    3, reinterpret_cast<uint64_t>(command_buffer_base), argument1, 0, 0, argument2);

	if (command_buffer_base == nullptr) {
		return diagnostic.Return(LibKernel::KERNEL_ERROR_EINVAL);
	}

	int32_t  execution_result = OK;
	uint32_t error_offset     = 0;
	return diagnostic.Return(ExecuteAprCommandBuffer(reinterpret_cast<uint64_t>(command_buffer_base),
	                                                &execution_result, &error_offset, diagnostic.Get()));
}

static int KYTY_SYSV_ABI AmmSubmitCommandBufferAndGetId(void* command_buffer_base, uint32_t argument1,
                                                        uint32_t argument2, uint32_t* out_submission_id) {
	PRINT_NAME();
	HangTrace::SetGuestCaller(reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprSubmitAndGetId);
	AprShared::LoadingSubmissionDiagnostic diagnostic(
	    5, reinterpret_cast<uint64_t>(command_buffer_base), argument1, 0,
	    reinterpret_cast<uint64_t>(out_submission_id), argument2);

	if (command_buffer_base == nullptr || out_submission_id == nullptr) {
		return diagnostic.Return(LibKernel::KERNEL_ERROR_EINVAL);
	}

	const auto command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer_base);
	const auto id                  = AprShared::AllocateSubmissionId(command_buffer_addr, 0);
	diagnostic.SetId(id);

	int32_t  execution_result = OK;
	uint32_t error_offset     = 0;
	auto     submit_result =
	    ExecuteAprCommandBuffer(command_buffer_addr, &execution_result, &error_offset, diagnostic.Get());
	if (submit_result != OK) {
		return diagnostic.Return(submit_result);
	}
	AprShared::SetSubmissionResult(id, execution_result, error_offset);

	if (!AprShared::WriteGuest(reinterpret_cast<uint64_t>(out_submission_id), id)) {
		return diagnostic.Return(LibKernel::KERNEL_ERROR_EFAULT);
	}

	return diagnostic.Return(OK);
}

static int KYTY_SYSV_ABI AmmSubmitCommandBufferAndGetResult(void* command_buffer_base, uint32_t argument1,
                                                            uint32_t argument2, void* result,
                                                            uint32_t* out_submission_id) {
	PRINT_NAME();
	HangTrace::SetGuestCaller(reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprSubmitAndGetResult);
	AprShared::LoadingSubmissionDiagnostic diagnostic(
	    4, reinterpret_cast<uint64_t>(command_buffer_base), argument1, reinterpret_cast<uint64_t>(result),
	    reinterpret_cast<uint64_t>(out_submission_id), argument2);

	if (command_buffer_base == nullptr) {
		return diagnostic.Return(LibKernel::KERNEL_ERROR_EINVAL);
	}

	const auto command_buffer_addr = reinterpret_cast<uint64_t>(command_buffer_base);
	const auto id =
	    AprShared::AllocateSubmissionId(command_buffer_addr, reinterpret_cast<uint64_t>(result));
	diagnostic.SetId(id);

	int32_t  execution_result = OK;
	uint32_t error_offset     = 0;
	auto     submit_result =
	    ExecuteAprCommandBuffer(command_buffer_addr, &execution_result, &error_offset, diagnostic.Get());
	if (submit_result != OK) {
		return diagnostic.Return(submit_result);
	}
	AprShared::SetSubmissionResult(id, execution_result, error_offset);

	if (out_submission_id != nullptr &&
	    !AprShared::WriteGuest(reinterpret_cast<uint64_t>(out_submission_id), id)) {
		return diagnostic.Return(LibKernel::KERNEL_ERROR_EFAULT);
	}

	return diagnostic.Return(AprShared::WriteResult(result, execution_result, error_offset));
}

static int KYTY_SYSV_ABI AmmWaitCommandBufferCompletion(uint32_t submission_id) {
	PRINT_NAME();

	AprShared::SubmissionState state {};
	if (!AprShared::CompleteSubmission(submission_id, &state)) {
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprWaitUnknownId);
		return LibKernel::KERNEL_ERROR_ESRCH;
	}
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprWaitCompleted);

	if (state.result != 0) {
		return AprShared::WriteResult(reinterpret_cast<void*>(state.result), state.execution_result,
		                              state.error_offset);
	}

	return OK;
}

} // namespace Ampr

LIB_DEFINE(InitAmpr_1) {
	LIB_FUNC("8aI7R7WaOlc", Ampr::CommandBufferConstructor);
	LIB_FUNC("a8uLzYY--tM", Ampr::AprCommandBufferConstructor);
	LIB_FUNC("Qs1xtplKo0U", Ampr::AprCommandBufferDestructor);
	LIB_FUNC("GuchCTefuZw", Ampr::CommandBufferDestructor);
	LIB_FUNC("N-FSPA4S3nI", Ampr::CommandBufferSetBuffer);
	LIB_FUNC("baQO9ez2gL4", Ampr::CommandBufferReset);
	LIB_FUNC("ULvXMDz56po", Ampr::CommandBufferClearBuffer);
	LIB_FUNC("VEDMaQmJZng", Ampr::CommandBufferGetType);
	LIB_FUNC("RPCAhx-aabE", Ampr::CommandBufferGetBufferBaseAddress);
	LIB_FUNC("gzndltBEzWc", Ampr::CommandBufferGetNumCommands);
	LIB_FUNC("GmOguNIsuKk", Ampr::CommandBufferConstructNop);
	LIB_FUNC("tNn5WBkta60", Ampr::CommandBufferNop);
	LIB_FUNC("pFQ9UHpO52s", Ampr::CommandBufferNopWithData);
	LIB_FUNC("4UkZbYKVF7c", Ampr::CommandBufferConstructMarker);
	LIB_FUNC("sWbST0oQKsc", Ampr::CommandBufferSetMarkerWithColor);
	LIB_FUNC("4quckD2y7Pg", Ampr::CommandBufferSetMarker);
	LIB_FUNC("f12ObAMEi9A", Ampr::CommandBufferPushMarkerWithColor);
	LIB_FUNC("dXPaz65HNmk", Ampr::CommandBufferPushMarker);
	LIB_FUNC("mv0O8Zg0woU", Ampr::CommandBufferPopMarker);
	LIB_FUNC("DLfoNxTFNVk", Ampr::CommandBufferWaitOnAddress);
	LIB_FUNC("V7GQTEeUfhw", Ampr::CommandBufferWaitOnAddress);
	LIB_FUNC("cQb8Zr8Q0Y0", Ampr::CommandBufferWaitOnCounter);
	LIB_FUNC("j0+3uJMxYJY", Ampr::CommandBufferWriteAddress);
	LIB_FUNC("sJXyWHjP-F8", Ampr::CommandBufferWriteAddressOnCompletion);
	LIB_FUNC("jK+yuYCI7MA", Ampr::CommandBufferWriteCounter);
	LIB_FUNC("3wn42MWTzTs", Ampr::CommandBufferWriteCounterOnCompletion);
	LIB_FUNC("bt3LHR9xjK4", Ampr::CommandBufferWriteAddressFromTimeCounter);
	LIB_FUNC("FI2JD0jAHCs", Ampr::CommandBufferWriteAddressFromTimeCounterOnCompletion);
	LIB_FUNC("t4ExS+SwLjs", Ampr::CommandBufferWriteAddressFromCounter);
	LIB_FUNC("gSF5OsXdfIg", Ampr::CommandBufferWriteAddressFromCounterOnCompletion);
	LIB_FUNC("enZm-6GjWqw", Ampr::CommandBufferWriteAddressFromCounterPair);
	LIB_FUNC("ZLWtNUP6R5E", Ampr::CommandBufferWriteAddressFromCounterPairOnCompletion);
	LIB_FUNC("vWU-odnS+fU", Ampr::MeasureCommandSizeReadFile);
	LIB_FUNC("sSAUCCU1dv4", Ampr::MeasureCommandSizeWriteKernelEventQueue);
	LIB_FUNC("Zi3dBUjgyXI", Ampr::MeasureCommandSizeWriteKernelEventQueue);
	LIB_FUNC("C+IEj+BsAFM", Ampr::MeasureCommandSizeWriteAddressOnCompletion);
	LIB_FUNC("x7SQEXfeovg", Ampr::MeasureCommandSizeWriteAddressFromTimeCounterOnCompletion);
	LIB_FUNC("32AcaTaBPSY", Ampr::MeasureCommandSizeWriteAddressFromCounterOnCompletion);
	LIB_FUNC("vxC58+DRk-U", Ampr::MeasureCommandSizeWriteAddressFromCounterPairOnCompletion);
	LIB_FUNC("4muPEJ-x5N8", Ampr::MeasureCommandSizeWriteCounterOnCompletion);
	LIB_FUNC("0BMj1hgG+kE", Ampr::MeasureCommandSizeFixed32);
	LIB_FUNC("ClnsFLLLcss", Ampr::MeasureCommandSizeFixed32);
	LIB_FUNC("4fgtGfXDrFc", Ampr::MeasureCommandSizeFixed32);
	LIB_FUNC("gAtc79UTt5E", Ampr::MeasureCommandSizeFixed32);
	LIB_FUNC("JYd9g9L+TmE", Ampr::MeasureCommandSizeFixed32);
	LIB_FUNC("2Hw8gjMdwSY", Ampr::MeasureCommandSizeFixed32);
	LIB_FUNC("I-Qm+MEso5c", Ampr::MeasureCommandSizeFixed32);
	LIB_FUNC("6hbai6KIXkk", Ampr::MeasureAmmCommandSizeMap);
	LIB_FUNC("m+fYyX8oFqw", Ampr::MeasureAmmCommandSizeMapWithGpuMaskId);
	LIB_FUNC("ZFDZoN9IbVU", Ampr::MeasureAmmCommandSizeMapDirect);
	LIB_FUNC("KUjtdPZJo5I", Ampr::MeasureAmmCommandSizeMapDirectWithGpuMaskId);
	LIB_FUNC("Ayg6PIon2wA", Ampr::MeasureAmmCommandSizeUnmap);
	LIB_FUNC("NNIZ-FMyz3M", Ampr::MeasureCommandSizeNop);
	LIB_FUNC("Xp85BP3+BBI", Ampr::MeasureCommandSizeNopWithData);
	LIB_FUNC("qesF88X4DRg", Ampr::MeasureCommandSizeReadFileGather);
	LIB_FUNC("7nXGDGMXSqo", Ampr::MeasureCommandSizeReadFileScatter);
	LIB_FUNC("DXmgc5op8Yw", Ampr::MeasureCommandSizeReadFileGatherScatter);
	LIB_FUNC("rddQYXM0CjM", Ampr::MeasureCommandSizePopMarker);
	LIB_FUNC("kdFImtTD0hc", Ampr::MeasureAprCommandSizeMapBegin);
	LIB_FUNC("qvbdJc7bG+s", Ampr::MeasureAprCommandSizeMapDirectBegin);
	LIB_FUNC("iwTNhyaemnw", Ampr::MeasureCommandSizePopMarker);
	LIB_FUNC("tmfr97+ED5I", Ampr::MeasureCommandSizeMarkerWithColor);
	LIB_FUNC("VGkEj4d6-Kg", Ampr::MeasureCommandSizeMarker);
	LIB_FUNC("3OfeY4pzDV0", Ampr::MeasureCommandSizeMarkerWithColor);
	LIB_FUNC("0RdLmAh7WVo", Ampr::MeasureCommandSizeMarker);
	LIB_FUNC("pbnNnahE8vk", Ampr::MeasureCommandSizePopMarker);
	LIB_FUNC("tZDDEo2tE5k", Ampr::CommandBufferGetSize);
	LIB_FUNC("GnxKOHEawhk", Ampr::CommandBufferGetCurrentOffset);
	LIB_FUNC("H896Pt-yB4I", Ampr::CommandBufferWriteKernelEventQueue);
	LIB_FUNC("o67gODLFpls", Ampr::CommandBufferWriteKernelEventQueue);
	LIB_FUNC("mQ16-QdKv7k", Ampr::AprCommandBufferReadFile);
	LIB_FUNC("mZSbNJVJpV8", Ampr::AprCommandBufferReadFileGather);
	LIB_FUNC("Jg-AgkdJHkk", Ampr::AprCommandBufferReadFileScatter);
	LIB_FUNC("BVmR1H8l+XI", Ampr::AprCommandBufferReadFileGatherScatter);
	LIB_FUNC("YPxkUDhgoNI", Ampr::AprCommandBufferResetGatherScatterState);
	LIB_FUNC("Eul7AGEpjLo", Ampr::AprCommandBufferMapBegin);
	LIB_FUNC("bFEs0Gs6D2A", Ampr::AprCommandBufferMapDirectBegin);
	LIB_FUNC("X169CE6G3Y4", Ampr::AprCommandBufferMapEnd);
	LIB_FUNC("Q07J7XpvhrU", Ampr::AmmGiveDirectMemory);
	LIB_FUNC("wkQR9+xTFKY", Ampr::AmmGetVirtualAddressRanges);
	LIB_FUNC("KqiWXLgCVe0", Ampr::AmmGetUsageStatsData);
	LIB_FUNC("touqMEt6qXQ", Ampr::AmmSetPageTablePoolOccupancyNotificationThreshold);
	LIB_FUNC("EDq5bqCqYpA", Ampr::AmmCommandBufferConstructor);
	LIB_FUNC("pvUFDOHilnE", Ampr::AmmCommandBufferDestructor);
	LIB_FUNC("JEVYGhDc97M", Ampr::AmmCommandBufferMap);
	LIB_FUNC("ojBkmG7+CgE", Ampr::AmmCommandBufferMapWithGpuMaskId);
	LIB_FUNC("8TBE+9XCZbI", Ampr::AmmCommandBufferMapDirect);
	LIB_FUNC("kOfZlhbVAkc", Ampr::AmmCommandBufferMapDirectWithGpuMaskId);
	LIB_FUNC("M-VFI2DJWQA", Ampr::AmmCommandBufferUnmap);
	LIB_FUNC("lwS-7y3jcBI", Ampr::AmmSubmitCommandBuffer);
	LIB_FUNC("NnKhlMJtIsI", Ampr::AmmSubmitCommandBufferAndGetId);
	LIB_FUNC("OJf3vCckPAM", Ampr::AmmSubmitCommandBufferAndGetResult);
	LIB_FUNC("HXymib4T8gc", Ampr::AmmWaitCommandBufferCompletion);
}

} // namespace LibAmpr

} // namespace Libs
