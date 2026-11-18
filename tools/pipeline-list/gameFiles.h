// The files of a game for kyty_pipeline_list and kyty_shaderlist_export: an unpacked game folder or
// a .zar archive of one (the launcher's "Compress game" makes <folder>.zar with the folder's
// contents at its root). Paths inside the game are relative, '/' separated, UTF-8.
#ifndef KYTY_TOOLS_PIPELINE_LIST_GAME_FILES_H_
#define KYTY_TOOLS_PIPELINE_LIST_GAME_FILES_H_

#include "common/stringUtils.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <zarchive/zarchivereader.h>

namespace PipelineListTool {

// A .zar file name (the extension in any case).
inline bool IsZarPath(const std::filesystem::path& path) {
	const auto extension = path.extension().native();
	if (extension.size() != 4 || extension[0] != '.') return false;
	const char want[] = "zar";
	for (size_t i = 0; i < 3; i++) {
		auto c = extension[i + 1];
		if (c >= 'A' && c <= 'Z') c = static_cast<decltype(c)>(c + ('a' - 'A'));
		if (c != static_cast<decltype(c)>(want[i])) return false;
	}
	return true;
}

class GameFiles {
public:
	struct Entry {
		uint64_t              size = 0;
		std::filesystem::path relative;
	};
	// Fills data with up to size bytes of the file in order; the bytes given (0 at the end).
	using Stream = std::function<size_t(uint8_t* data, size_t size)>;

	// A game folder, or a .zar archive when the name ends in .zar. Null when the archive cannot be
	// opened; a folder is not checked here (reads from a missing folder fail).
	static std::unique_ptr<GameFiles> Open(const std::filesystem::path& game) {
		std::unique_ptr<GameFiles> files(new GameFiles);
		files->m_root = game;
		if (IsZarPath(game)) {
			std::error_code error;
			if (!std::filesystem::is_regular_file(game, error)) return nullptr;
			files->m_archive.reset(ZArchiveReader::OpenFromFile(game));
			if (files->m_archive == nullptr) return nullptr;
		}
		return files;
	}

	[[nodiscard]] bool IsArchive() const { return m_archive != nullptr; }

	// Whether the game has this file.
	bool HasFile(const std::filesystem::path& relative) {
		if (m_archive == nullptr) {
			std::error_code error;
			return std::filesystem::is_regular_file(m_root / relative, error);
		}
		return Node(relative) != ZARCHIVE_INVALID_NODE;
	}

	// The whole file.
	bool Read(const std::filesystem::path& relative, std::vector<uint8_t>& bytes) {
		if (m_archive == nullptr) {
			const auto      path = m_root / relative;
			std::ifstream   in(path, std::ios::binary);
			std::error_code error;
			const auto      size = std::filesystem::file_size(path, error);
			if (!in || error) return false;
			bytes.resize(static_cast<size_t>(size));
			return bytes.empty() ||
			       static_cast<bool>(in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)));
		}
		const auto node = Node(relative);
		if (node == ZARCHIVE_INVALID_NODE) return false;
		const auto size = m_archive->GetFileSize(node);
		bytes.resize(static_cast<size_t>(size));
		return size == 0 || m_archive->ReadFromFile(node, 0, size, bytes.data()) == size;
	}

	// `out.size()` bytes at `offset` of the file; false when the file does not hold them. For the many
	// small reads of a list resolve (a data file can be many gigabytes): a folder's file stays open
	// until another file is read.
	bool ReadAt(const std::filesystem::path& relative, uint64_t offset, std::span<uint8_t> out) {
		if (m_archive == nullptr) {
			if (!m_at.is_open() || m_at_path != relative) {
				m_at.close();
				m_at.clear();
				m_at.open(m_root / relative, std::ios::binary);
				m_at_path = relative;
			}
			if (!m_at.is_open()) return false;
			m_at.clear();
			m_at.seekg(static_cast<std::streamoff>(offset));
			return out.empty() ||
			       static_cast<bool>(m_at.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size())));
		}
		const auto node = Node(relative);
		if (node == ZARCHIVE_INVALID_NODE) return false;
		return out.empty() || m_archive->ReadFromFile(node, offset, out.size(), out.data()) == out.size();
	}

	// The file read in order; an empty function when it cannot be opened.
	Stream OpenStream(const std::filesystem::path& relative) {
		if (m_archive == nullptr) {
			auto in = std::make_shared<std::ifstream>(m_root / relative, std::ios::binary);
			if (!*in) return {};
			return [in](uint8_t* data, size_t size) -> size_t {
				in->read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
				return static_cast<size_t>(in->gcount());
			};
		}
		const auto node = Node(relative);
		if (node == ZARCHIVE_INVALID_NODE) return {};
		auto position = std::make_shared<uint64_t>(0);
		return [archive = m_archive.get(), node, position](uint8_t* data, size_t size) -> size_t {
			const auto got = archive->ReadFromFile(node, *position, size, data);
			*position += got;
			return static_cast<size_t>(got);
		};
	}

	// Every regular file of the game, by size, then by relative path (the order kyty_pipeline_list
	// --scan reads them in, the same for a folder and its archive).
	std::vector<Entry> Files() {
		std::vector<Entry> files;
		if (m_archive == nullptr) {
			std::error_code error;
			for (auto it = std::filesystem::recursive_directory_iterator(m_root, error);
			     !error && it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
				if (!it->is_regular_file(error)) continue;
				files.push_back({it->file_size(error), std::filesystem::relative(it->path(), m_root, error)});
			}
		} else {
			// Directories to list: archive path ("" is the root) and its path as a host path.
			std::vector<std::pair<std::string, std::filesystem::path>> pending {{std::string(), std::filesystem::path()}};
			while (!pending.empty()) {
				const auto [directory, relative] = std::move(pending.back());
				pending.pop_back();
				const auto node = m_archive->LookUp(directory, false, true);
				if (node == ZARCHIVE_INVALID_NODE) continue;
				const auto count = m_archive->GetDirEntryCount(node);
				for (uint32_t index = 0; index < count; index++) {
					ZArchiveReader::DirEntry entry {};
					if (!m_archive->GetDirEntry(node, index, entry)) continue;
					const std::string name(entry.name);
					const auto        path   = directory.empty() ? name : directory + "/" + name;
					const auto        member = relative / Common::PathFromUtf8(name);
					if (entry.isDirectory) {
						pending.emplace_back(path, member);
					} else if (entry.isFile) {
						files.push_back({entry.size, member}); // GetDirEntry sets the size of files
					}
				}
			}
		}
		std::sort(files.begin(), files.end(),
		          [](const Entry& a, const Entry& b) { return a.size != b.size ? a.size < b.size : a.relative < b.relative; });
		return files;
	}

private:
	GameFiles() = default;

	ZArchiveNodeHandle Node(const std::filesystem::path& relative) {
		// LookUp with allowDirectory false still returns directories: files only.
		const auto node = m_archive->LookUp(Common::PathToGenericString(relative), true, false);
		return node != ZARCHIVE_INVALID_NODE && m_archive->IsFile(node) ? node : ZARCHIVE_INVALID_NODE;
	}

	std::filesystem::path           m_root;
	std::unique_ptr<ZArchiveReader> m_archive;
	// ReadAt: the folder file it read last.
	std::ifstream         m_at;
	std::filesystem::path m_at_path;
};

// Title and version as the emulator reads them (loader/systemContent.cpp: titleId, contentVersion).
inline bool ReadParam(GameFiles& game, std::string& title, std::string& version) {
	std::vector<uint8_t> bytes;
	if (!game.Read(std::filesystem::path("sce_sys") / "param.json", bytes)) return false;
	const auto json = nlohmann::json::parse(bytes.begin(), bytes.end(), nullptr, false);
	if (json.is_discarded() || !json.is_object()) return false;
	if (json.contains("titleId") && json["titleId"].is_string()) title = json["titleId"].get<std::string>();
	if (json.contains("contentVersion") && json["contentVersion"].is_string()) version = json["contentVersion"].get<std::string>();
	if (json.contains("appVersion") && json["appVersion"].is_string()) version = json["appVersion"].get<std::string>();
	return !title.empty() && !version.empty();
}

} // namespace PipelineListTool

#endif /* KYTY_TOOLS_PIPELINE_LIST_GAME_FILES_H_ */
