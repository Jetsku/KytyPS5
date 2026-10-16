#include "graphics/host_gpu/rangeSet.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <new>
#include <random>
#include <utility>
#include <vector>

// Count only Add itself, not construction, assertions, or result collection.
// These replacements are local to this standalone test executable.
namespace {
thread_local bool g_count_allocations = false;
thread_local size_t g_allocations = 0;
thread_local size_t g_frees = 0;
}

void* operator new(size_t size) {
    void* result = std::malloc(size == 0 ? 1 : size);
    if (result == nullptr) {
        throw std::bad_alloc();
    }
    if (g_count_allocations) {
        ++g_allocations;
    }
    return result;
}

void operator delete(void* pointer) noexcept {
    if (pointer != nullptr && g_count_allocations) {
        ++g_frees;
    }
    std::free(pointer);
}

void* operator new[](size_t size) { return ::operator new(size); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, size_t) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, size_t) noexcept { ::operator delete(pointer); }

namespace {
using Libs::Graphics::RangeSet;
using Interval = std::pair<uint64_t, uint64_t>;
int g_failures = 0;

void Check(bool value, const char* text) {
    if (!value) {
        std::fprintf(stderr, "RangeSetTests: failed: %s\n", text);
        ++g_failures;
    }
}

std::vector<Interval> Collect(const RangeSet& ranges) {
    std::vector<Interval> result;
    ranges.ForEach([&](uint64_t begin, uint64_t end) { result.emplace_back(begin, end); });
    return result;
}

void AllocationCase(const char* name, std::initializer_list<Interval> seed,
                    uint64_t address, uint64_t size,
                    std::initializer_list<Interval> expected, size_t expected_frees) {
    RangeSet ranges;
    for (const auto& [begin, end]: seed) {
        ranges.Add(begin, end - begin);
    }
    g_allocations = 0;
    g_frees = 0;
    g_count_allocations = true;
    ranges.Add(address, size);
    g_count_allocations = false;
    std::printf("%s: allocations=%zu frees=%zu expected_frees=%zu\n",
                name, g_allocations, g_frees, expected_frees);
    Check(g_allocations == 0, "unchanged-start merge must not allocate a replacement node");
    Check(g_frees == expected_frees, "only absorbed successors may be freed");
    Check(Collect(ranges) == std::vector<Interval>(expected), "allocation case changed union");
    for (const auto& [begin, end]: expected) {
        Check(ranges.Contains(begin, end - begin), "merged interval not fully contained");
        Check(ranges.Intersects(end - 1, 1), "last covered byte missing");
        if (end != UINT64_MAX) {
            Check(!ranges.Intersects(end, 1), "half-open end became covered");
        }
    }
}

void TestAllocationContract() {
    AllocationCase("touching-right", {{0x1000, 0x1080}}, 0x1080, 0x80,
                   {{0x1000, 0x1100}}, 0);
    AllocationCase("overlapping-right", {{0x1000, 0x1080}}, 0x1040, 0xc0,
                   {{0x1000, 0x1100}}, 0);
    AllocationCase("same-start-growth", {{0x1000, 0x1080}}, 0x1000, 0x100,
                   {{0x1000, 0x1100}}, 0);
    AllocationCase("contained", {{0x1000, 0x1100}}, 0x1040, 0x20,
                   {{0x1000, 0x1100}}, 0);
    AllocationCase("absorb-touching-successor", {{0x1000, 0x1080}, {0x1100, 0x1200}},
                   0x1080, 0x80, {{0x1000, 0x1200}}, 1);
    AllocationCase("absorb-many-with-sentinel", {{0x1000, 0x1080}, {0x1100, 0x1140},
                                                {0x1180, 0x1200}, {0x2000, 0x2100}},
                   0x1040, 0x180, {{0x1000, 0x1200}, {0x2000, 0x2100}}, 2);
    AllocationCase("maximum-end", {{UINT64_MAX - 0x100, UINT64_MAX - 0x80}},
                   UINT64_MAX - 0x80, 0x80, {{UINT64_MAX - 0x100, UINT64_MAX}}, 0);
}

std::vector<Interval> Oracle(const std::array<bool, 256>& bytes, uint64_t base,
                             size_t from = 0, size_t to = 256) {
    std::vector<Interval> result;
    for (size_t i = from; i < to;) {
        if (!bytes[i]) {
            ++i;
            continue;
        }
        const auto begin = i++;
        while (i < to && bytes[i]) {
            ++i;
        }
        result.emplace_back(base + begin, base + i);
    }
    return result;
}

void TestByteModel() {
    constexpr std::array<uint64_t, 3> bases{0, 0x200000000ull, UINT64_MAX - 256};
    for (uint64_t base: bases) {
        RangeSet ranges;
        std::array<bool, 256> bytes{};
        std::mt19937 random(0x528b73);
        for (size_t step = 0; step < 20000; ++step) {
            const size_t begin = random() % bytes.size();
            const size_t size = 1 + random() % (bytes.size() - begin);
            const auto operation = random() % 20;
            if (operation == 0) {
                ranges.Clear();
                bytes.fill(false);
            } else if (operation < 8) {
                ranges.Subtract(base + begin, size);
                std::fill_n(bytes.begin() + begin, size, false);
            } else {
                ranges.Add(base + begin, size);
                std::fill_n(bytes.begin() + begin, size, true);
            }
            const auto expected = Oracle(bytes, base);
            Check(Collect(ranges) == expected, "randomized exact interval union disagrees with byte model");
            Check(ranges.Size() == expected.size() && ranges.Empty() == expected.empty(),
                  "range count/emptiness disagrees with byte model");
            for (size_t query = 0; query < 4; ++query) {
                const size_t first = random() % bytes.size();
                const size_t count = 1 + random() % (bytes.size() - first);
                const bool contains = std::all_of(bytes.begin() + first, bytes.begin() + first + count,
                                                   [](bool set) { return set; });
                const bool intersects = std::any_of(bytes.begin() + first, bytes.begin() + first + count,
                                                    [](bool set) { return set; });
                Check(ranges.Contains(base + first, count) == contains, "Contains disagrees with byte model");
                Check(ranges.Intersects(base + first, count) == intersects, "Intersects disagrees with byte model");
                std::vector<Interval> actual;
                ranges.ForEachInRange(base + first, count, [&](uint64_t b, uint64_t e) { actual.emplace_back(b, e); });
                Check(actual == Oracle(bytes, base, first, first + count), "clipped enumeration disagrees with byte model");
            }
            if (g_failures != 0) {
                return;
            }
        }
    }
    std::puts("RangeSetTests: byte model passed 60000 operations and 240000 range queries");
}

void TestBoundaries() {
    RangeSet ranges;
    ranges.Add(100, 50);
    ranges.Add(75, 30); // left growth must retain the fallback semantics
    ranges.Add(50, 25); // touching from the left
    ranges.Add(200, 50);
    ranges.Add(0, 250); // covers all nodes from a new start
    Check(Collect(ranges) == std::vector<Interval>{{0, 250}}, "left growth/covers-all union incorrect");
    ranges.Subtract(25, 200);
    Check(Collect(ranges) == std::vector<Interval>{{0, 25}, {225, 250}}, "subtraction split incorrect");
    ranges.Add(25, 200);
    Check(Collect(ranges) == std::vector<Interval>{{0, 250}}, "bridge after subtraction incorrect");
    ranges.Add(251, 1);
    Check(!ranges.Intersects(250, 1) && !ranges.Contains(249, 3), "one-byte hole disappeared");
    ranges.Add(250, 1);
    Check(ranges.Contains(0, 252) && ranges.Size() == 1, "one-byte bridge failed");
    RangeSet copy = ranges;
    ranges.Clear();
    Check(copy.Contains(0, 252) && ranges.Empty(), "copy/clear semantics changed");
}

enum class Workload { Right, SameStart, Bridge, Contained, Left, Disjoint };
constexpr size_t kSteps = 32;
constexpr uint64_t kStride = 0x10000;
constexpr uint64_t kWidth = 16;

uint64_t Begin(size_t index) { return (index + 1) * kStride; }

void Seed(RangeSet& ranges, Workload workload, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const auto begin = Begin(i);
        if (workload == Workload::Bridge) {
            for (size_t j = 0; j <= kSteps; ++j) {
                ranges.Add(begin + j * kWidth * 2, kWidth);
            }
        } else {
            ranges.Add(begin, workload == Workload::Contained ? (kSteps + 1) * kWidth : kWidth);
        }
    }
}

void Work(RangeSet& ranges, Workload workload, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const auto begin = Begin(i);
        for (size_t j = 0; j < kSteps; ++j) {
            switch (workload) {
                case Workload::Right:
                case Workload::Contained: ranges.Add(begin + (j + 1) * kWidth, kWidth); break;
                case Workload::SameStart: ranges.Add(begin, (j + 2) * kWidth); break;
                case Workload::Bridge: ranges.Add(begin + (j * 2 + 1) * kWidth, kWidth); break;
                case Workload::Left: ranges.Add(begin - (j + 1) * kWidth, kWidth); break;
                case Workload::Disjoint: ranges.Add(begin + (j + 1) * 2 * kWidth, kWidth); break;
            }
        }
    }
}

uint64_t ValidateWork(const RangeSet& ranges, Workload workload, size_t count) {
    std::vector<Interval> expected;
    for (size_t i = 0; i < count; ++i) {
        const auto begin = Begin(i);
        if (workload == Workload::Disjoint) {
            for (size_t j = 0; j <= kSteps; ++j) {
                expected.emplace_back(begin + j * 2 * kWidth, begin + j * 2 * kWidth + kWidth);
            }
        } else if (workload == Workload::Left) {
            expected.emplace_back(begin - kSteps * kWidth, begin + kWidth);
        } else {
            const auto width = (workload == Workload::Bridge ? kSteps * 2 + 1 : kSteps + 1) * kWidth;
            expected.emplace_back(begin, begin + width);
        }
    }
    Check(Collect(ranges) == expected, "benchmark final union differs from independent expected intervals");
    uint64_t checksum = 0;
    ranges.ForEach([&](uint64_t begin, uint64_t end) { checksum += begin ^ end; });
    return checksum;
}

void Benchmark() {
    constexpr std::array<const char*, 6> names{"right", "same-start", "bridge", "contained", "left", "disjoint"};
    std::puts("workload,intervals,operations,ns_per_add,allocations,frees,checksum");
    for (size_t w = 0; w < names.size(); ++w) {
        const auto workload = static_cast<Workload>(w);
        for (size_t count: {size_t{1}, size_t{64}, size_t{1024}}) {
            RangeSet counted;
            Seed(counted, workload, count);
            g_allocations = g_frees = 0;
            g_count_allocations = true;
            Work(counted, workload, count);
            g_count_allocations = false;
            const auto allocations = g_allocations;
            const auto frees = g_frees;
            const auto checksum = ValidateWork(counted, workload, count);
            // Setup/destruction and result validation are outside every timed region.
            // Use the same work and allocator wrappers for both compiled variants.
            const size_t repeats = std::max(size_t{8}, size_t{262144} / (count * kSteps));
            double nanoseconds = 0;
            for (size_t repeat = 0; repeat < repeats; ++repeat) {
                RangeSet ranges;
                Seed(ranges, workload, count);
                const auto start = std::chrono::steady_clock::now();
                Work(ranges, workload, count);
                const auto stop = std::chrono::steady_clock::now();
                nanoseconds += std::chrono::duration<double, std::nano>(stop - start).count();
                Check(ValidateWork(ranges, workload, count) == checksum, "benchmark checksum changed between repeats");
            }
            std::printf("%s,%zu,%zu,%.6f,%zu,%zu,%llu\n", names[w], count, repeats * count * kSteps,
                        nanoseconds / static_cast<double>(repeats * count * kSteps), allocations, frees,
                        static_cast<unsigned long long>(checksum));
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        TestBoundaries();
        TestByteModel();
        TestAllocationContract();
    } else if (argc == 2 && std::strcmp(argv[1], "--semantics-only") == 0) {
        TestBoundaries();
        TestByteModel();
    } else if (argc == 2 && std::strcmp(argv[1], "--allocations-only") == 0) {
        TestAllocationContract();
    } else if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        Benchmark();
    } else {
        std::fprintf(stderr, "usage: range_set_tests [--semantics-only|--allocations-only|--bench]\n");
        return 2;
    }
    return g_failures == 0 ? 0 : 1;
}
