#ifndef KYTY_LIBS_APR_HOST_FILE_POOL_H_
#define KYTY_LIBS_APR_HOST_FILE_POOL_H_

// Keeps host files open between APR reads.
//
// An APR read used to open the host file, ask its size, seek, read and close it again. Astro Bot
// and Crash Bandicoot 4 issue thousands of reads of a few hundred bytes per second from pak files,
// so each read paid for a path lookup and a create/close pair. This pool hands out an idle open
// handle for a path and takes it back when the lease ends.
//
// - A handle belongs to one lease at a time, so the seek + read pair on it needs no extra locking
//   and its cursor is never shared. Concurrent reads of the same path each get their own handle
//   (the pool opens another one when none is idle), exactly as separate opens did before.
// - File I/O happens outside the pool mutex; the mutex only guards the maps.
// - At most `max_paths` paths keep idle handles (least recently used is dropped) and at most
//   `max_idle_per_path` idle handles are kept for one path. Extra handles are closed on release.
// - A failed read is the caller's to report; Lease::Discard() closes the handle instead of
//   returning it, so a handle that went bad is never reused.
//
// FileT needs: default construction, `bool Open(const std::filesystem::path&, Mode)` through the
// opener passed to Acquire, and `Close()`. It is not copied or moved.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::AprHostFiles {

template <typename FileT>
class HostFilePool {
public:
	struct Stats {
		uint64_t opens          = 0; // handles opened (misses)
		uint64_t reuses         = 0; // leases served from an idle handle (hits)
		uint64_t open_failures  = 0;
		uint64_t closed_on_full = 0; // released handles dropped because the idle list was full
		uint64_t evicted_paths  = 0;
	};

	explicit HostFilePool(std::size_t max_paths = 64, std::size_t max_idle_per_path = 8)
	    : m_max_paths(max_paths), m_max_idle(max_idle_per_path) {}

	HostFilePool(const HostFilePool&)            = delete;
	HostFilePool& operator=(const HostFilePool&) = delete;

	class Lease {
	public:
		Lease() = default;
		Lease(Lease&& other) noexcept { *this = std::move(other); }
		Lease& operator=(Lease&& other) noexcept {
			if (this != &other) {
				Release();
				m_pool  = other.m_pool;
				m_key   = std::move(other.m_key);
				m_file  = std::move(other.m_file);
				other.m_pool = nullptr;
			}
			return *this;
		}
		Lease(const Lease&)            = delete;
		Lease& operator=(const Lease&) = delete;
		~Lease() { Release(); }

		explicit operator bool() const { return m_file != nullptr; }
		FileT&   File() { return *m_file; }

		// Closes the handle instead of returning it to the pool.
		void Discard() {
			m_file.reset();
			m_pool = nullptr;
		}

	private:
		friend class HostFilePool;
		Lease(HostFilePool* pool, std::string key, std::unique_ptr<FileT> file)
		    : m_pool(pool), m_key(std::move(key)), m_file(std::move(file)) {}

		void Release() {
			if (m_file != nullptr && m_pool != nullptr) {
				m_pool->Return(m_key, std::move(m_file));
			}
			m_file.reset();
			m_pool = nullptr;
		}

		HostFilePool*      m_pool = nullptr;
		std::string        m_key;
		std::unique_ptr<FileT> m_file;
	};

	// `open` receives a default-constructed FileT and returns whether it is now open.
	// An empty lease means the open failed.
	template <typename OpenFn>
	Lease Acquire(const std::string& key, OpenFn&& open) {
		{
			std::scoped_lock lock(m_mutex);
			auto             it = m_entries.find(key);
			if (it != m_entries.end()) {
				it->second.last_use = ++m_tick;
				if (!it->second.idle.empty()) {
					auto file = std::move(it->second.idle.back());
					it->second.idle.pop_back();
					++m_stats.reuses;
					return Lease(this, key, std::move(file));
				}
			}
		}
		auto file = std::make_unique<FileT>();
		if (!open(*file)) {
			std::scoped_lock lock(m_mutex);
			++m_stats.open_failures;
			return Lease();
		}
		{
			std::scoped_lock lock(m_mutex);
			++m_stats.opens;
		}
		return Lease(this, key, std::move(file));
	}

	Stats GetStats() const {
		std::scoped_lock lock(m_mutex);
		return m_stats;
	}

	// Number of paths that currently hold idle handles (tests, diagnostics).
	std::size_t PathCount() const {
		std::scoped_lock lock(m_mutex);
		return m_entries.size();
	}

	std::size_t IdleCount(const std::string& key) const {
		std::scoped_lock lock(m_mutex);
		auto             it = m_entries.find(key);
		return it == m_entries.end() ? 0 : it->second.idle.size();
	}

private:
	struct Entry {
		std::vector<std::unique_ptr<FileT>> idle;
		uint64_t                            last_use = 0;
	};

	void Return(const std::string& key, std::unique_ptr<FileT> file) {
		std::unique_ptr<FileT>              dropped;
		std::vector<std::unique_ptr<FileT>> evicted_files;
		{
			std::scoped_lock lock(m_mutex);
			auto&            entry = m_entries[key];
			entry.last_use         = ++m_tick;
			if (entry.idle.size() < m_max_idle) {
				entry.idle.push_back(std::move(file));
			} else {
				dropped = std::move(file);
				++m_stats.closed_on_full;
			}
			while (m_entries.size() > m_max_paths) {
				auto oldest = m_entries.end();
				for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
					if (it->first == key) continue;
					if (oldest == m_entries.end() || it->second.last_use < oldest->second.last_use) {
						oldest = it;
					}
				}
				if (oldest == m_entries.end()) break;
				for (auto& idle: oldest->second.idle) evicted_files.push_back(std::move(idle));
				m_entries.erase(oldest);
				++m_stats.evicted_paths;
			}
		}
		// Handles are closed here, outside the mutex.
	}

	const std::size_t                      m_max_paths;
	const std::size_t                      m_max_idle;
	mutable std::mutex                     m_mutex;
	std::unordered_map<std::string, Entry> m_entries;
	uint64_t                               m_tick = 0;
	Stats                                  m_stats;
};

} // namespace Libs::AprHostFiles

#endif
