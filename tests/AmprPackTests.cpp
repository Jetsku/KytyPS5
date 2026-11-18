#include "common/amprPack.h"
#include "common/archive.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "kernel/fileSystem.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void Check(bool value, const char* text) {
  if (!value) {
    std::fprintf(stderr, "AmprPackTests: failed: %s\n", text);
    std::abort();
  }
}

// CRC-32/IEEE, matching the oracle produced by zlib.crc32.
uint32_t Crc32(const uint8_t* data, size_t size) {
  uint32_t crc = 0xffffffffu;
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
  }
  return ~crc;
}

uint32_t RotateRight(uint32_t value, int bits) {
  return (value >> bits) | (value << (32 - bits));
}

std::string Sha256Hex(const uint8_t* data, size_t size) {
  static constexpr uint32_t kConstant[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t block[64] = {};
  size_t used = 0;
  const auto compress = [&](const uint8_t* p) {
    uint32_t w[64] = {};
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
             (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = RotateRight(w[i - 15], 7) ^ RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = RotateRight(w[i - 2], 17) ^ RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + s1 + ch + kConstant[i] + w[i];
      const uint32_t s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  };
  uint64_t bits = 0;
  for (size_t i = 0; i < size; ++i) {
    block[used++] = data[i];
    bits += 8;
    if (used == sizeof(block)) {
      compress(block);
      used = 0;
    }
  }
  block[used++] = 0x80;
  if (used > 56) {
    while (used < sizeof(block)) block[used++] = 0;
    compress(block);
    used = 0;
  }
  while (used < 56) block[used++] = 0;
  for (int i = 7; i >= 0; --i) block[used++] = uint8_t(bits >> (8 * i));
  compress(block);
  char hex[65] = {};
  for (int i = 0; i < 8; ++i) {
    std::snprintf(hex + 8 * i, 9, "%08x", h[i]);
  }
  return std::string(hex, 64);
}

class TempDirectory {
public:
  TempDirectory() {
    m_path = std::filesystem::temp_directory_path() /
             ("kyty_ampr_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Check(std::filesystem::create_directories(m_path), "create temporary directory");
  }
  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(m_path, error);
  }
  [[nodiscard]] const std::filesystem::path& Path() const { return m_path; }
  KYTY_CLASS_NO_COPY(TempDirectory);
private:
  std::filesystem::path m_path;
};

}  // namespace

int main() {
  // Golden oracle: the same font read out of /mnt/hdd/roms/ps5/ASTRO_BOT.zar via gamefs.Zar.
  constexpr uint32_t kFontCrc = 0xfcd88cddu;
  constexpr const char* kFontSha256 = "215c175ce595d0f42bc1f703ef94ecbf6f45dc875cdbfd7ea59d9d8b28467cf9";
  constexpr uint64_t kChunkSize = 262144;
  constexpr uint64_t kChunksOff = 5335424;

  // The golden checks need a real AMPR dump; point KYTY_AMPR_TEST_DUMP at one to run them.
  const char* dumpRoot = std::getenv("KYTY_AMPR_TEST_DUMP");
  if (dumpRoot == nullptr || *dumpRoot == '\0') {
    std::printf("AmprPackTests: SKIP - set KYTY_AMPR_TEST_DUMP to an AMPR dump root\n");
    return 0;
  }
  const std::filesystem::path root = dumpRoot;
  const auto indexPath = root / "ampr_zstd.index";
  if (!std::filesystem::is_regular_file(indexPath)) {
    std::printf("AmprPackTests: SKIP - %s has no ampr_zstd.index\n", dumpRoot);
    return 0;
  }

  Common::AmprPack pack;
  Check(pack.Init(root), "Init on real dump");
  Common::AmprPack::FileInfo font{};
  Check(pack.Lookup("/app0/data/common/font/FuturaStd-Bold.otf", font), "exact-case lookup");
  Check(font.size == 28280 && font.firstChunk == 587370, "font metadata");
  Check(pack.Exists("/app0/data/common/font/FuturaStd-Bold.otf"), "Exists wrapper");

  Common::AmprPack::FileInfo folded{};
  Check(pack.Lookup("/app0/DATA/COMMON/FONT/FUTURASTD-BOLD.OTF", folded), "mixed-case lookup");
  Check(folded.size == font.size && folded.firstChunk == font.firstChunk, "mixed-case resolves to same file");

  std::vector<uint8_t> bytes(font.size);
  Check(pack.Read(font, 0, bytes.size(), bytes.data()), "read whole font");
  Check(Crc32(bytes.data(), bytes.size()) == kFontCrc, "font CRC-32 matches oracle");
  Check(Sha256Hex(bytes.data(), bytes.size()) == kFontSha256, "font SHA-256 matches oracle");
  uint8_t last = 0;
  Check(pack.Read(font, font.size - 1, 1, &last) && last == bytes.back(), "read last byte");

  Common::AmprPack::FileInfo big{};
  Check(pack.Lookup("/app0/data/prein/bg/gfx/big_tornado_scifi_parts_a_05_1v.jxm", big), "large file lookup");
  Check(big.size == 899952 && big.firstChunk == 478528, "large file metadata");
  Check(big.chunkCount == (big.size + kChunkSize - 1) / kChunkSize, "large file chunk count");
  std::vector<uint8_t> whole(big.size);
  std::vector<uint8_t> chunk(kChunkSize);
  for (uint32_t k = 0; k < big.chunkCount; ++k) {
    const uint64_t base = uint64_t(k) * kChunkSize;
    const size_t decoded = size_t(std::min<uint64_t>(kChunkSize, big.size - base));
    Check(pack.Read(big, base, decoded, chunk.data()), "read one full decoded chunk");
    Check(pack.Read(big, base, decoded, whole.data() + base), "read chunk into range buffer");
  }
  constexpr uint64_t crossOffset = kChunkSize - 32;
  constexpr size_t crossLength = 64;
  std::vector<uint8_t> cross(crossLength);
  Check(pack.Read(big, crossOffset, crossLength, cross.data()), "read across chunk boundary");
  Check(std::memcmp(cross.data(), whole.data() + crossOffset, crossLength) == 0,
        "cross-boundary range matches per-chunk reads");

  TempDirectory temporary;
  const auto truncatedRoot = temporary.Path() / "truncated";
  Check(std::filesystem::create_directories(truncatedRoot), "create truncated fixture root");
  {
    std::ifstream source(indexPath, std::ios::binary);
    std::ofstream dest(truncatedRoot / "ampr_zstd.index", std::ios::binary);
    char head[4096] = {};
    source.read(head, sizeof(head));
    dest.write(head, source.gcount());
  }
  Common::AmprPack truncated;
  Check(!truncated.Init(truncatedRoot), "Init rejects truncated index");

  const auto badOffsetRoot = temporary.Path() / "bad-offset";
  Check(std::filesystem::create_directories(badOffsetRoot), "create bad-offset fixture root");
  {
    std::ifstream source(indexPath, std::ios::binary);
    std::ofstream dest(badOffsetRoot / "ampr_zstd.index", std::ios::binary);
    char header[128] = {};
    source.read(header, sizeof(header));
    const uint64_t badOffset = UINT64_MAX;
    std::memcpy(header + 64, &badOffset, sizeof(badOffset));  // chunks_off
    dest.write(header, sizeof(header));
  }
  Common::AmprPack badOffset;
  Check(!badOffset.Init(badOffsetRoot), "Init rejects out-of-range table offset");

  const auto corruptRoot = temporary.Path() / "corrupt";
  Check(std::filesystem::create_directories(corruptRoot), "create corrupt fixture root");
  const auto corruptIndex = corruptRoot / "ampr_zstd.index";
  Check(std::filesystem::copy_file(indexPath, corruptIndex), "copy index for corruption");
  {
    std::fstream file(corruptIndex, std::ios::in | std::ios::out | std::ios::binary);
    file.seekg(static_cast<std::streamoff>(kChunksOff + uint64_t(font.firstChunk) * 8));
    uint64_t word = 0;
    file.read(reinterpret_cast<char*>(&word), sizeof(word));
    Check(file.good(), "read chunk word");
    word = (word & ~(0xfffuLL << 32)) | (0xfffuLL << 32);  // volume id past the volume table
    file.seekp(static_cast<std::streamoff>(kChunksOff + uint64_t(font.firstChunk) * 8));
    file.write(reinterpret_cast<const char*>(&word), sizeof(word));
    Check(file.good(), "write corrupt chunk word");
  }
  Common::AmprPack corrupt;
  Check(corrupt.Init(corruptRoot), "corrupt index still parses");
  Common::AmprPack::FileInfo corruptFont{};
  Check(corrupt.Lookup("/app0/data/common/font/FuturaStd-Bold.otf", corruptFont), "corrupt index lookup");
  std::vector<uint8_t> ignored(font.size);
  Check(!corrupt.Read(corruptFont, 0, ignored.size(), ignored.data()), "corrupt chunk word fails Read");

  // Task 2: the same index is served through the registered archive backend and virtual paths.
  {
    const auto archive = root / "__ampr__.amprz";
    const auto fontVirtual = Common::MakeArchivePath(archive, "data/common/font/FuturaStd-Bold.otf");
    Check(Common::IsArchivePath(fontVirtual), "amprz virtual path is recognized");
    Check(Common::File::IsFileExisting(fontVirtual), "virtual font exists");
    Check(Common::File::Size(fontVirtual) == 28280, "virtual font size");

    Common::File file;
    Check(file.Open(fontVirtual, Common::File::Mode::Read), "virtual font opens");
    Check(file.Size() == 28280, "opened virtual font size");
    std::vector<uint8_t> virtualBytes(28280);
    uint32_t got = 0;
    file.Read(virtualBytes.data(), static_cast<uint32_t>(virtualBytes.size()), &got);
    Check(got == virtualBytes.size(), "virtual font full read");
    Check(Crc32(virtualBytes.data(), virtualBytes.size()) == kFontCrc, "virtual font CRC matches oracle");
    Check(std::memcmp(virtualBytes.data(), bytes.data(), bytes.size()) == 0,
          "virtual font matches direct reader");
    file.Close();

    const auto missing = Common::MakeArchivePath(archive, "data/common/font/NoSuchFont.otf");
    Check(!Common::File::IsFileExisting(missing), "non-index member does not exist");

    // Fix round 1: index-only directories must be detectable and enumerable.
    const std::string fontDir = "data/common/font";
    Check(pack.IsDirectory(fontDir), "known index-only directory is detected");
    const auto children = pack.ListDirectory(fontDir);
    bool foundFontChild = false;
    for (const auto& entry : children) {
      if (entry.name == "FuturaStd-Bold.otf") {
        foundFontChild = true;
        Check(entry.is_file, "font child is a file");
      }
    }
    Check(foundFontChild, "ListDirectory includes the known font");
    Check(!pack.IsDirectory("data/common/no_such_dir"), "unknown directory is not detected");
    Check(pack.ListDirectory("data/common/no_such_dir").empty(), "unknown directory lists nothing");

    const auto dirVirtual = Common::MakeArchivePath(archive, fontDir);
    Check(Common::File::IsDirectoryExisting(dirVirtual), "virtual directory stats as a directory");
    const auto virtualChildren = Common::File::GetDirEntries(dirVirtual);
    bool foundVirtualFont = false;
    for (const auto& entry : virtualChildren) {
      if (entry.name == "FuturaStd-Bold.otf") {
        foundVirtualFont = true;
        Check(entry.is_file, "virtual font child is a file");
      }
    }
    Check(foundVirtualFont, "virtual directory listing includes the font");
    Check(!Common::File::IsDirectoryExisting(Common::MakeArchivePath(archive, "data/common/no_such_dir")),
          "non-existent virtual directory is not detected");
  }

  // Task 3: mount-time index load and the ResolvePath hook. These go through the real
  // LibKernel::FileSystem API, so the test target links the full emulator (see CMakeLists.txt).
  {
    namespace FS = Libs::LibKernel::FileSystem;

    Common::InitializeThreads();
    Common::Subsystems subsystems;
    subsystems.Initialize<Config::Lifecycle>();
    Config::ConfigOptions options;
    options.printf_direction = Config::LogDirection::Silent;
    Config::Load(options);
    subsystems.Initialize<Log::Lifecycle>();
    FS::Initialize();

    constexpr const char* kFontGuest = "/app0/data/common/font/FuturaStd-Bold.otf";

    // Real dump: the loose font is absent, so the guest path must resolve into the
    // synthetic .amprz archive and still stat as the 28,280-byte font.
    FS::Mount(root, "/app0");
    FS::ResetResolvePathIgnoringCaseCount();
    const auto packedResolved = FS::GetRealFilename(kFontGuest);
    Check(FS::ResolvePathIgnoringCaseCount() == 0,
          "packed file resolves without a case-insensitive host walk");
    const auto packedStr = Common::PathToGenericString(packedResolved);
    Check(packedStr.find("!/") != std::string::npos, "packed guest path resolves to an archive path");
    Check(packedStr.find("__ampr__.amprz") != std::string::npos, "resolved path uses the amprz backend");
    Check(Common::File::Size(packedResolved) == 28280, "resolved virtual font size is 28280");

    // Case-folded guest paths must resolve to a readable virtual path too.
    FS::ResetResolvePathIgnoringCaseCount();
    const auto packedFolded = FS::GetRealFilename("/app0/DATA/COMMON/FONT/FUTURASTD-BOLD.OTF");
    Check(FS::ResolvePathIgnoringCaseCount() == 0,
          "mixed-case packed file resolves without a case-insensitive host walk");
    const auto packedFoldedStr = Common::PathToGenericString(packedFolded);
    Check(packedFoldedStr.find("!/") != std::string::npos, "mixed-case guest path resolves to a virtual path");
    Check(Common::File::Size(packedFolded) == 28280, "mixed-case guest path resolves to a readable 28280-byte font");

    // Task 5: index-only DIRECTORIES that are missing on the host must redirect to the
    // synthetic .amprz archive too, so directory stat/enumeration succeeds.
    const std::string packedDirRel = "data/prein/spx";
    Check(pack.IsDirectory(packedDirRel), "spx is an index-only directory");
    const auto packedDirResolved = FS::GetRealFilename("/app0/" + packedDirRel);
    const auto packedDirStr = Common::PathToGenericString(packedDirResolved);
    Check(packedDirStr.find("!/") != std::string::npos, "missing guest directory resolves to an archive path");
    Check(packedDirStr.find("__ampr__.amprz") != std::string::npos, "resolved directory uses the amprz backend");
    Check(Common::File::IsDirectoryExisting(packedDirResolved), "resolved directory stats as a directory");
    Check(!Common::File::GetDirEntries(packedDirResolved).empty(), "resolved directory has non-empty listing");

    const std::string animRel =
        "data/prein/levels/hero_locoroco/cinematics/power_up_cinematics_locoroco_01/anim";
    Check(pack.IsDirectory(animRel), "anim is an index-only directory");
    const auto animResolved = FS::GetRealFilename("/app0/" + animRel);
    const auto animStr = Common::PathToGenericString(animResolved);
    Check(animStr.find("!/") != std::string::npos, "missing anim directory resolves to an archive path");
    Check(Common::File::IsDirectoryExisting(animResolved), "resolved anim directory stats as a directory");
    Check(!Common::File::GetDirEntries(animResolved).empty(), "resolved anim directory has non-empty listing");

    Check(FS::GetRealFilename("/unmounted/font.otf").empty(), "unmounted guest path stays empty");

    // Mutating operations on the virtual path are rejected and create no host file.
    Check(FS::KernelOpen(kFontGuest, 0x0200u | 0x1u, 0) == Libs::LibKernel::KERNEL_ERROR_EROFS,
          "O_CREAT|O_WRONLY on virtual path rejected");
    Check(FS::KernelTruncate(kFontGuest, 0) == Libs::LibKernel::KERNEL_ERROR_EROFS,
          "truncate on virtual path rejected");
    Check(FS::KernelUnlink(kFontGuest) == Libs::LibKernel::KERNEL_ERROR_EROFS,
          "unlink on virtual path rejected");
    Check(!std::filesystem::exists(root / "data/common/font/FuturaStd-Bold.otf"),
          "no host file created by mutating ops");
    FS::Umount("/app0");

    // Review Focus 1: no ampr_zstd.index must mean no behavior change. A loose folder
    // with the same font resolves to a plain host path (no "!/") and still opens.
    TempDirectory loose;
    const auto looseFont = loose.Path() / "data/common/font/FuturaStd-Bold.otf";
    Check(std::filesystem::create_directories(looseFont.parent_path()), "create loose font directory");
    {
      std::ofstream out(looseFont, std::ios::binary);
      out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    FS::Mount(loose.Path(), "/app0");
    FS::ResetResolvePathIgnoringCaseCount();
    const auto looseResolved = FS::GetRealFilename(kFontGuest);
    Check(FS::ResolvePathIgnoringCaseCount() >= 1,
          "loose file still uses the case-insensitive host walk (counter is live)");
    const auto looseStr = Common::PathToGenericString(looseResolved);
    Check(looseStr.find("!/") == std::string::npos, "loose folder resolves to a normal path");
    Check(looseResolved == looseFont, "loose font resolves to the real host file");
    Check(Common::File::Size(looseResolved) == 28280, "loose font still opens");
    FS::Umount("/app0");

    // The working old-ROM path is untouched: build a small .zar containing the font and mount it
    // unconditionally. The guest path must resolve to that archive's own member path — never the
    // synthetic .amprz one. No dependency on /mnt/hdd, so this guard runs on any machine.
    TempDirectory zarTemp;
    const auto zarSrc = zarTemp.Path() / "src";
    const auto zarFont = zarSrc / "data/common/font/FuturaStd-Bold.otf";
    Check(std::filesystem::create_directories(zarFont.parent_path()), "create .zar fixture source dir");
    {
      std::ofstream out(zarFont, std::ios::binary);
      out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    const auto fixtureZar = zarTemp.Path() / "ASTRO_BOT.zar";
    Check(Common::PackArchive(zarSrc, fixtureZar, [](uint64_t, uint64_t) { return true; }),
          "pack .zar fixture containing the font");
    const auto expectedZarPath = Common::MakeArchivePath(fixtureZar, "data/common/font/FuturaStd-Bold.otf");
    FS::Mount(Common::MakeArchivePath(fixtureZar), "/app0");
    const auto zarResolved = FS::GetRealFilename(kFontGuest);
    const auto zarStr = Common::PathToGenericString(zarResolved);
    Check(zarStr.find("__ampr__.amprz") == std::string::npos, "zar mount does not use the amprz backend");
    Check(zarResolved == expectedZarPath, "zar mount resolves to its unchanged member path");
    Check(Common::File::Size(zarResolved) == 28280, "zar font still opens at 28280 bytes");
    FS::Umount("/app0");

    FS::Shutdown();
    subsystems.Destroy();
  }

  // Task 6: repeated opens of one root must reuse a single AmprPack. OpenArchive keeps readers as
  // weak_ptr only, so without a strong owner every stat re-runs the 22 MB parse + string sort.
  // A fresh root keeps this independent of earlier opens of the real dump in this process.
  {
    TempDirectory cacheTemp;
    const auto cacheRoot = cacheTemp.Path() / "root";
    Check(std::filesystem::create_directories(cacheRoot), "create fresh ampr root");
    Check(std::filesystem::copy_file(indexPath, cacheRoot / "ampr_zstd.index"),
          "copy index into fresh ampr root");

    const auto cacheArchive = cacheRoot / "__ampr__.amprz";
    const auto firstMember = Common::MakeArchivePath(cacheArchive, "data/common/font/FuturaStd-Bold.otf");
    const auto secondMember = Common::MakeArchivePath(
        cacheArchive, "data/prein/bg/gfx/big_tornado_scifi_parts_a_05_1v.jxm");

    Common::AmprPack::ResetInitCount();
    Check(Common::File::IsFileExisting(firstMember), "first fresh-root member resolves");
    Check(Common::File::IsFileExisting(secondMember), "second fresh-root member resolves");
    Check(Common::AmprPack::InitCount() == 1,
          "index parsed exactly once for two opens of the same root");
  }

  std::printf("AmprPackTests: all cases passed\n");
  return 0;
}
