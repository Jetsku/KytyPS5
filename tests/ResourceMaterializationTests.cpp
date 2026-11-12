#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/liveSwitch.h"
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  program.block_info.push_back({.id = 0});
  program.block_storage.push_back(std::move(block));
  return *result;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan SrtPlan(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  program.info.uses_dma = true;
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::Program MixedSamplerProgram() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count) {
    DescriptorSource source;
    source.dword_count = dword_count;
    for (uint32_t i = 0; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8);
  const auto image1 = AddSource(8);
  const auto sampler0 = AddSource(4);
  const auto sampler1 = AddSource(4);
  program.descriptor_sources[image1].dwords[0] = Value(1u);
  program.descriptor_sources[image1].dwords[1] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::BufferFormat::k11_11_10UInt) << 20u);
  program.descriptor_sources[image1].dwords[3] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  for (uint32_t index = 0; index < 2; ++index) {
    auto &value = block.AppendNewInst(
        ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))});
    program.descriptor_sources[sampler0 + index].dwords[0] = Value(&value);
  }
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return program;
}

// Four adjacent raw SRT reads feed flat slots, and a buffer descriptor mixes them with
// user data, so each walk exercises nested memo contexts, flat reads and user data.
Libs::Graphics::ShaderRecompiler::IR::ResourcePlan
SharedEvaluationPlan(const uint32_t *table) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto address = reinterpret_cast<uint64_t>(table);
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  auto &handle = value_block.AppendNewInst(
      ValueOpcode::GetAddressResource,
      {Value(static_cast<uint32_t>(address)),
       Value(static_cast<uint32_t>(address >> 32u))});
  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  std::array<Value, 4> flat;
  for (uint32_t i = 0; i < flat.size(); i++) {
    auto &raw = value_block.AppendNewInst(
        ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(i * 4u), Value(0u), Value(true)});
    raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40 + i * 4u});
    program.srt_reads.push_back({Value(&raw), i});
    flat[i] = Value(&value_block.AppendNewInst(ValueOpcode::ReadConst,
                                               {Value(&srt), Value(i)}));
  }
  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  auto &sum = value_block.AppendNewInst(ValueOpcode::IAdd32,
                                        {flat[0], Value(&user_data)});
  auto &shifted = value_block.AppendNewInst(
      ValueOpcode::ShiftLeftLogical32, {Value(&user_data), Value(4u)});
  auto &records =
      value_block.AppendNewInst(ValueOpcode::IAdd32, {flat[2], Value(&shifted)});
  DescriptorSource source;
  source.dwords[0] = Value(&sum);
  source.dwords[1] = flat[1];
  source.dwords[2] = Value(&records);
  source.dwords[3] = flat[3];
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

void TestSealedPlanEvaluatesConcurrently() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const std::array<uint32_t, 4> first_table{0x10000u, 0x00100000u, 0x40u,
                                            0x00027facu};
  const std::array<uint32_t, 4> second_table{0x20000u, 0x00200000u, 0x80u,
                                             0x00027facu};
  const std::array<ResourcePlan, 2> plans{
      SharedEvaluationPlan(first_table.data()),
      SharedEvaluationPlan(second_table.data())};
  const std::array<uint32_t, 2> counts{plans[0].evaluation_value_count,
                                       plans[1].evaluation_value_count};
  Check(plans[0].evaluation_sealed && plans[1].evaluation_sealed,
        "extracted resource plan was not sealed");
  Check(counts[0] == plans[0].value_storage.size(),
        "sealing did not assign every memo slot");

  constexpr std::array<uint32_t, 3> user_values{7u, 0x100u, 0xfffffff0u};
  struct Result {
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
  };
  const auto run = [&](size_t plan, uint32_t user, EvaluationScratch *scratch,
                       Result &result) {
    const std::array<uint32_t, 1> user_data{user_values[user]};
    const SrtRuntime runtime{.user_data = user_data};
    return scratch != nullptr
               ? MaterializeResources(plans[plan], runtime, *scratch,
                                      result.snapshot, result.specialization)
               : MaterializeResources(plans[plan], runtime, result.snapshot,
                                      result.specialization);
  };
  const auto same = [](const Result &left, const Result &right) {
    return left.snapshot.flattened_srt == right.snapshot.flattened_srt &&
           left.snapshot.buffers == right.snapshot.buffers &&
           left.snapshot.user_data == right.snapshot.user_data &&
           left.specialization == right.specialization;
  };

  // Serial reference results, each from a fresh scratch.
  std::array<std::array<Result, user_values.size()>, 2> expected;
  for (size_t plan = 0; plan < plans.size(); plan++) {
    for (size_t user = 0; user < user_values.size(); user++) {
      EvaluationScratch scratch;
      Check(run(plan, static_cast<uint32_t>(user), &scratch,
                expected[plan][user]),
            "serial shared-plan materialization failed");
    }
  }
  // Every case differs, so a stale memo entry from another plan or user-data
  // set in a reused scratch would produce a detectable mismatch.
  Check(expected[0][0].snapshot.flattened_srt !=
                expected[1][0].snapshot.flattened_srt &&
            expected[0][0].snapshot.buffers != expected[0][1].snapshot.buffers &&
            expected[0][1].snapshot.buffers != expected[0][2].snapshot.buffers,
        "shared-plan cases are not distinguishable");

  // Half the workers own a scratch; the others use their thread's default one.
  // Each worker alternates plans and user data on the same scratch.
  constexpr uint32_t ThreadCount = 6;
  constexpr uint32_t Iterations = 2000;
  std::atomic<uint32_t> failures{0};
  std::vector<std::thread> threads;
  for (uint32_t thread = 0; thread < ThreadCount; thread++) {
    threads.emplace_back([&, thread] {
      EvaluationScratch owned;
      auto *scratch = thread % 2u == 0u ? &owned : nullptr;
      for (uint32_t i = 0; i < Iterations; i++) {
        const auto plan = (i + thread) % plans.size();
        const auto user = (i / 2u + thread) % user_values.size();
        Result result;
        if (!run(plan, static_cast<uint32_t>(user), scratch, result) ||
            !same(result, expected[plan][user])) {
          failures.fetch_add(1u, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  Check(failures.load() == 0,
        "concurrent shared-plan materialization differed from serial");
  Check(plans[0].evaluation_value_count == counts[0] &&
            plans[1].evaluation_value_count == counts[1],
        "evaluation assigned memo slots on a sealed plan");
}

// Draw-prep S3 (pipelineCache.cpp MakeSpeculativeRuntime): a materialization whose every read
// goes through the silent clean probe must equal the serial runtime's result on clean memory,
// and must fail, without reading the unclean word, when any read is not provably clean.
struct ProbeMemory {
  uint64_t dirty_address = 0;
  uint32_t strict_failures = 0;
  uint32_t probe_failures = 0;
};
ProbeMemory g_probe_memory;

bool CleanProbe(void *, uint64_t address, std::span<uint32_t> values) {
  const auto end = address + values.size_bytes();
  if (g_probe_memory.dirty_address != 0 &&
      g_probe_memory.dirty_address >= address &&
      g_probe_memory.dirty_address < end) {
    ++g_probe_memory.probe_failures;
    return false;
  }
  std::memcpy(values.data(), reinterpret_cast<const void *>(address),
              values.size_bytes());
  return true;
}

bool StrictRead(void *userdata, uint64_t address, std::span<uint32_t> values) {
  if (CleanProbe(userdata, address, values)) {
    return true;
  }
  ++g_probe_memory.strict_failures;
  return false;
}

void TestSpeculativeRuntimeMatchesSerial() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const std::array<uint32_t, 4> table{0x30000u, 0x00300000u, 0xc0u, 0x00027facu};
  const auto plan = SharedEvaluationPlan(table.data());
  const std::array<uint32_t, 1> user_data{0x55u};
  const SrtRuntime serial{.user_data = user_data,
                          .read_specialization_memory = StrictRead,
                          .try_read_clean_backing = CleanProbe,
                          .share_clean_values = true};
  const SrtRuntime speculative{.user_data = user_data,
                               .read_memory = StrictRead,
                               .read_specialization_memory = StrictRead,
                               .try_read_clean_backing = CleanProbe,
                               .share_clean_values = false};
  g_probe_memory = {};
  ResourceSnapshot serial_snapshot;
  ResourceSpecialization serial_specialization;
  Check(MaterializeResources(plan, serial, serial_snapshot, serial_specialization),
        "serial clean materialization failed");
  ResourceSnapshot speculative_snapshot;
  ResourceSpecialization speculative_specialization;
  Check(MaterializeResources(plan, speculative, speculative_snapshot,
                             speculative_specialization),
        "speculative clean materialization failed");
  Check(serial_snapshot.flattened_srt == speculative_snapshot.flattened_srt &&
            serial_snapshot.buffers == speculative_snapshot.buffers &&
            serial_snapshot.images == speculative_snapshot.images &&
            serial_snapshot.samplers == speculative_snapshot.samplers &&
            serial_snapshot.user_data == speculative_snapshot.user_data &&
            serial_snapshot.uniform_fill == speculative_snapshot.uniform_fill &&
            serial_specialization == speculative_specialization,
        "speculative materialization differed from the serial runtime");
  Check(g_probe_memory.strict_failures == 0 && g_probe_memory.probe_failures == 0,
        "clean materializations reported failed reads");

  // One unclean word (the base address, which no specialization depends on): the serial
  // runtime reads it through its ordinary fallback, the speculative one must refuse the stage.
  g_probe_memory = {};
  g_probe_memory.dirty_address = reinterpret_cast<uint64_t>(&table[0]);
  ResourceSnapshot dirty_serial;
  ResourceSpecialization dirty_serial_specialization;
  Check(MaterializeResources(plan, serial, dirty_serial, dirty_serial_specialization),
        "serial materialization did not fall back for an unclean word");
  Check(dirty_serial.buffers == serial_snapshot.buffers,
        "serial fallback read different bytes");
  g_probe_memory.strict_failures = 0;
  ResourceSnapshot dirty_speculative;
  ResourceSpecialization dirty_speculative_specialization;
  Check(!MaterializeResources(plan, speculative, dirty_speculative,
                              dirty_speculative_specialization),
        "speculative materialization accepted an unclean word");
  Check(g_probe_memory.strict_failures != 0,
        "speculative failure did not come from the silent reader");
  g_probe_memory = {};
}

// Draw-prep S5/S6 (readSet.h): a preparation that records every read certifies itself. While the
// recorded ranges hold the recorded bytes, the recorded preparation equals the serial one; a
// change to a read byte fails the certificate and a change elsewhere does not. Preparations run
// concurrently on several threads, each with its own read set and scratch, as DrawPrep workers do.
bool RecordingRead(void *userdata, uint64_t address, std::span<uint32_t> values) {
  auto &reads = *static_cast<Libs::Graphics::DrawPrep::ReadSet *>(userdata);
  if (values.empty()) {
    return false;
  }
  std::memcpy(values.data(), reinterpret_cast<const void *>(address),
              values.size_bytes());
  return reads.Record(address, values.data(), values.size_bytes());
}

void TestRecordedPreparationCertifies() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::DrawPrep::ReadSet;
  using Libs::Graphics::DrawPrep::ValidateResult;
  std::array<uint32_t, 4> table{0x40000u, 0x00400000u, 0x100u, 0x00027facu};
  const auto plan = SharedEvaluationPlan(table.data());
  const std::array<uint32_t, 1> user_data{0x66u};
  const SrtRuntime serial{.user_data = user_data,
                          .read_specialization_memory = StrictRead,
                          .try_read_clean_backing = CleanProbe,
                          .share_clean_values = true};
  const auto recording_runtime = [&](ReadSet &reads) {
    return SrtRuntime{.user_data = user_data,
                      .read_memory = RecordingRead,
                      .userdata = &reads,
                      .read_specialization_memory = RecordingRead,
                      .try_read_clean_backing = RecordingRead,
                      .share_clean_values = false};
  };
  const auto read_now = [](uint64_t address, void *data, uint64_t size) {
    std::memcpy(data, reinterpret_cast<const void *>(address), size);
    return true;
  };
  struct Output {
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
  };
  const auto same = [](const Output &a, const Output &b) {
    return a.snapshot.flattened_srt == b.snapshot.flattened_srt &&
           a.snapshot.buffers == b.snapshot.buffers &&
           a.snapshot.images == b.snapshot.images &&
           a.snapshot.samplers == b.snapshot.samplers &&
           a.snapshot.user_data == b.snapshot.user_data &&
           a.snapshot.uniform_fill == b.snapshot.uniform_fill &&
           a.specialization == b.specialization;
  };
  g_probe_memory = {};
  Output expected;
  Check(MaterializeResources(plan, serial, expected.snapshot,
                             expected.specialization),
        "serial materialization for the certificate test failed");

  ReadSet reads;
  Output recorded;
  EvaluationScratch scratch;
  Check(MaterializeResources(plan, recording_runtime(reads), scratch,
                             recorded.snapshot, recorded.specialization),
        "recording materialization failed");
  Check(same(expected, recorded), "recorded preparation differs from serial");
  Check(reads.Finish() && !reads.Ranges().empty(), "read set did not finish");
  bool covers_table = false;
  for (const auto &range : reads.Ranges()) {
    covers_table |= range.begin <= reinterpret_cast<uint64_t>(&table[0]) &&
                    reinterpret_cast<uint64_t>(&table[0]) < range.end;
  }
  Check(covers_table, "the descriptor table read is not in the certificate");
  std::vector<uint8_t> validation;
  Check(reads.Validate(read_now, validation) == ValidateResult::Ok,
        "unchanged memory did not validate");

  // A read byte changes: the certificate fails (the serial result differs too).
  table[0] ^= 0x1000u;
  Check(reads.Validate(read_now, validation) == ValidateResult::Changed,
        "a changed read byte validated");
  Output changed;
  Check(MaterializeResources(plan, serial, changed.snapshot,
                             changed.specialization),
        "serial materialization of changed memory failed");
  Check(!same(changed, recorded),
        "test premise: the changed byte must change the serial result");
  table[0] ^= 0x1000u;
  Check(reads.Validate(read_now, validation) == ValidateResult::Ok,
        "restored memory did not validate");

  // Concurrent preparations on shared memory, each certified and compared with the serial
  // result. Plans are sealed (immutable); scratch and read sets are per thread.
  constexpr uint32_t ThreadCount = 8;
  constexpr uint32_t Iterations = 1000;
  std::atomic<uint32_t> failures{0};
  std::vector<std::thread> threads;
  for (uint32_t thread = 0; thread < ThreadCount; thread++) {
    threads.emplace_back([&] {
      ReadSet thread_reads;
      EvaluationScratch thread_scratch;
      Output output;
      std::vector<uint8_t> thread_validation;
      for (uint32_t i = 0; i < Iterations; i++) {
        thread_reads.Reset();
        const bool ok = MaterializeResources(plan, recording_runtime(thread_reads),
                                             thread_scratch, output.snapshot,
                                             output.specialization) &&
                        thread_reads.Finish() &&
                        thread_reads.Validate(read_now, thread_validation) ==
                            ValidateResult::Ok &&
                        same(output, expected);
        if (!ok) {
          failures.fetch_add(1u, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  Check(failures.load() == 0,
        "concurrent recorded preparations differed from serial or failed to certify");
  g_probe_memory = {};
}

// KYTY_NULL_BUFFER_SHARE: a formatted buffer slot read from user data specializes a null V# (all
// zeros) like the slot's last live V#; without the switch the null V# is its own specialization.
void TestNullBufferShare() {
  using namespace Libs::Graphics::ShaderRecompiler;
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto saved = GetCodegenOptions();
  for (const bool share : {false, true}) {
    auto options = saved;
    options.null_buffer_share = share;
    SetCodegenOptions(options);
    Program program;
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.srt_plan_complete = true;
    program.resource_tracking_complete = true;
    auto &value_block = AddValueBlock(program);
    DescriptorSource source;
    for (uint32_t i = 0; i < 4; ++i) {
      auto &user_data = value_block.AppendNewInst(
          ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(i))});
      source.dwords[i] = Value(&user_data);
    }
    source.dword_count = 4;
    program.descriptor_sources.push_back(source);
    program.info.buffers.push_back({.source = 0, .read = true, .formatted = true});
    const auto plan = ExtractResourcePlan(program);
    // A live V#: base 0x10000, stride 16, 64 records, DST_SEL X,Y,Z,1, format 77, OOB_SELECT 3.
    const std::array<uint32_t, 4> live{0x10000u, 16u << 16u, 64u,
                                       (3u << 28u) | (77u << 12u) | 4u | (5u << 3u) | (6u << 6u) | (1u << 9u)};
    const std::array<uint32_t, 4> null{};
    ResourceSnapshot live_snapshot, null_snapshot;
    ResourceSpecialization live_spec, null_spec;
    Check(MaterializeResources(plan, SrtRuntime{.user_data = live}, live_snapshot, live_spec),
          "null share: live materialization failed");
    Check(MaterializeResources(plan, SrtRuntime{.user_data = null}, null_snapshot, null_spec),
          "null share: null materialization failed");
    Check(live_spec.buffers.size() == 1 && null_spec.buffers.size() == 1,
          "null share: one buffer expected");
    if (share) {
      Check(null_spec == live_spec, "null share: the null V# did not take the live specialization");
      Check(std::ranges::all_of(null_snapshot.buffers[0].dwords, [](uint32_t w) { return w == 0u; }),
            "null share: the null V#'s snapshot must stay null (the renderer binds the null buffer)");
    } else {
      Check(!(null_spec == live_spec) && null_spec.buffers[0].zero_stride_oob,
            "null share off: the null V# must keep its own specialization");
    }
    // A raw slot: a live V# with stride 0 and OOB_SELECT 0 (every access out of range) is shared
    // and bound like a null V#.
    Program raw_program;
    raw_program.stage = Libs::Graphics::ShaderType::Compute;
    raw_program.shader_hash = 0x5ca1ab1e00000001ull;
    raw_program.srt_plan_complete = true;
    raw_program.resource_tracking_complete = true;
    auto &raw_block = AddValueBlock(raw_program);
    DescriptorSource raw_source;
    for (uint32_t i = 0; i < 4; ++i) {
      auto &user_data = raw_block.AppendNewInst(
          ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(i))});
      raw_source.dwords[i] = Value(&user_data);
    }
    raw_source.dword_count = 4;
    raw_program.descriptor_sources.push_back(raw_source);
    raw_program.info.buffers.push_back({.source = 0, .read = true});
    const auto raw_plan = ExtractResourcePlan(raw_program);
    const std::array<uint32_t, 4> raw_live{0x20000u, 12u << 16u, 64u, (77u << 12u) | 0xfacu};
    const std::array<uint32_t, 4> raw_oob{0x30000u, 0u, 64u, (77u << 12u) | 0xfacu};
    ResourceSnapshot raw_live_snapshot, raw_oob_snapshot;
    ResourceSpecialization raw_live_spec, raw_oob_spec;
    Check(MaterializeResources(raw_plan, SrtRuntime{.user_data = raw_live}, raw_live_snapshot,
                               raw_live_spec) &&
              MaterializeResources(raw_plan, SrtRuntime{.user_data = raw_oob}, raw_oob_snapshot,
                                   raw_oob_spec),
          "null share: raw materialization failed");
    if (share) {
      Check(raw_oob_spec == raw_live_spec &&
                std::ranges::all_of(raw_oob_snapshot.buffers[0].dwords, [](uint32_t w) { return w == 0u; }),
            "null share: a stride-0 OOB_SELECT-0 raw V# was not shared as null");
    } else {
      Check(raw_oob_spec.buffers[0].zero_stride_oob && raw_oob_snapshot.buffers[0].dwords[0] == 0x30000u,
            "null share off: the stride-0 raw V# must keep its own specialization and words");
    }
  }
  SetCodegenOptions(saved);
}

void TestMappedSrtUsesDirectReaderByDefault() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = SrtPlan(reinterpret_cast<uint64_t>(&dword));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "ordinary SRT read used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "cache rematerialization did not use the direct reader by default");
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtPlan(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestMixedSamplerVariantsShareRuntimeDescriptor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = MixedSamplerProgram();
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 2> user_data{0x11111111u, 0x22222222u};
  const SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed sampler materialization failed");
  ApplyResourceSpecialization(program, specialization);
  const auto &samplers = program.info.samplers;
  Check(snapshot.samplers.size() == 2 && samplers.size() == 3 &&
            samplers[0].snapshot_index == 0 && samplers[1].snapshot_index == 1 &&
            samplers[2].snapshot_index == 1 &&
            samplers[0].source == plan.info.samplers[0].source &&
            samplers[1].source == plan.info.samplers[1].source &&
            samplers[2].source == samplers[1].source &&
            !samplers[1].force_point_filtering && samplers[2].force_point_filtering &&
            program.info.sampled_pairs[2].sampler == 2,
        "native sampler variants lost their source identity or binding order");
  const auto capacity = snapshot.samplers.capacity();
  user_data[1] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.samplers.size() == 2 && snapshot.samplers.capacity() == capacity &&
            snapshot.samplers[samplers[0].snapshot_index].dwords[0] == user_data[0] &&
            snapshot.samplers[samplers[1].snapshot_index].dwords[0] == user_data[1] &&
            snapshot.samplers[samplers[2].snapshot_index].dwords[0] == user_data[1],
        "sampler variants retained stale or duplicated descriptors after refresh");
}

// KYTY_SRT_TRACE (SrtWalker.h): a plan whose handle comes from user data, with four adjacent
// flat reads (one batched run), user data arithmetic and a select on a guest word. Refreshes
// replayed from a trace must equal the walk for any user data and guest words, including when a
// guard fails: the flat run crossing a page (a shorter run) and the select taking the other arm.
Libs::Graphics::ShaderRecompiler::IR::ResourcePlan TracedPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  auto &low = value_block.AppendNewInst(ValueOpcode::GetUserData,
                                        {Value(static_cast<ScalarReg>(1))});
  auto &high = value_block.AppendNewInst(ValueOpcode::GetUserData,
                                         {Value(static_cast<ScalarReg>(2))});
  auto &handle = value_block.AppendNewInst(ValueOpcode::GetAddressResource,
                                           {Value(&low), Value(&high)});
  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  std::array<Value, 4> flat;
  for (uint32_t i = 0; i < flat.size(); i++) {
    auto &raw = value_block.AppendNewInst(
        ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(i * 4u), Value(0u), Value(true)});
    raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40 + i * 4u});
    program.srt_reads.push_back({Value(&raw), i});
    flat[i] = Value(&value_block.AppendNewInst(ValueOpcode::ReadConst,
                                               {Value(&srt), Value(i)}));
  }
  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  auto &sum = value_block.AppendNewInst(ValueOpcode::IAdd32,
                                        {flat[0], Value(&user_data)});
  auto &nonzero = value_block.AppendNewInst(ValueOpcode::INotEqual32,
                                            {flat[3], Value(0u)});
  auto &select = value_block.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&nonzero), flat[1], Value(&user_data)});
  DescriptorSource source;
  source.dwords[0] = Value(&sum);
  source.dwords[1] = Value(&select);
  source.dwords[2] = flat[2];
  source.dwords[3] = flat[3];
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

// KYTY_SRT_GPU_FILL (GpuFillSlots): of three flat reads through one user-data-free table, the
// one a descriptor uses stays a host value; the one the shader only stores as data may be left
// to the GPU (1), and so may the one only a buffer store's offset uses (2: the renderer then
// takes the binding as written whole). A slot is left when the runtime reports its dword
// GPU-written: the snapshot gets a placeholder and a fill, the other slots their values.
uint64_t g_gpu_written_address = 0;
bool GpuWrittenForTest(uint64_t address) { return address == g_gpu_written_address; }

void TestGpuFillSlotsLeaveShaderOnlyReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  alignas(16) static std::array<uint32_t, 4> table{0x1111u, 0x2222u, 0x3333u, 0u};
  const auto address = reinterpret_cast<uint64_t>(table.data());
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);
  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  auto &handle = value_block.AppendNewInst(
      ValueOpcode::GetAddressResource,
      {Value(static_cast<uint32_t>(address)), Value(static_cast<uint32_t>(address >> 32u))});
  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  std::array<Value, 3> flat;
  for (uint32_t i = 0; i < flat.size(); i++) {
    auto &raw = value_block.AppendNewInst(ValueOpcode::LoadAddressU32,
                                          {Value(&handle), Value(i * 4u), Value(0u), Value(true)});
    raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40 + i * 4u});
    program.srt_reads.push_back({Value(&raw), i});
    flat[i] = Value(&value_block.AppendNewInst(ValueOpcode::ReadConst, {Value(&srt), Value(i)}));
  }
  DescriptorSource source;
  source.dwords[0] = flat[0];
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  auto &buffer = value_block.AppendNewInst(ValueOpcode::GetBufferResource,
                                           {flat[0], Value(0u), Value(0u), Value(0u)});
  value_block.AppendNewInst(ValueOpcode::StoreBufferU32,
                            {Value(&buffer), Value(0u), flat[2], Value(0u), flat[1], Value(true)});
  const auto plan = ExtractResourcePlan(program);
  Check(plan.gpu_fill_slots.size() == 3 && plan.gpu_fill_slots[0] == 0u &&
            plan.gpu_fill_slots[1] == 1u && plan.gpu_fill_slots[2] == 2u,
        "GPU-fill slots: the descriptor read is a host value, the others qualify");

  for (const bool fill : {false, true}) {
    for (const uint32_t written : {0u, 1u, 2u}) {
      g_gpu_written_address = address + written * 4u;
      SrtRuntime runtime{};
      if (fill) {
        runtime.gpu_written = GpuWrittenForTest;
        runtime.gpu_fill_store_offsets = true;
      }
      ResourceSnapshot snapshot;
      ResourceSpecialization specialization;
      Check(MaterializeResources(plan, runtime, snapshot, specialization),
            "GPU-fill materialization failed");
      const bool left = fill && written != 0u;
      Check(snapshot.flattened_srt.size() == 3 && snapshot.flattened_srt[0] == 0x1111u &&
                snapshot.flattened_srt[1] == (left && written == 1u ? 0u : 0x2222u) &&
                snapshot.flattened_srt[2] == (left && written == 2u ? 0u : 0x3333u),
            "GPU-fill materialization changed a host slot or read a GPU slot");
      Check(left ? (snapshot.gpu_fills.size() == 1 && snapshot.gpu_fills[0].flat_offset == written &&
                    snapshot.gpu_fills[0].address == address + written * 4u)
                 : snapshot.gpu_fills.empty(),
            "GPU-fill list wrong");
    }
  }
}

void TestTraceReplayMatchesWalk() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  alignas(4096) static std::array<uint32_t, 2048> pages{};
  const auto plan = TracedPlan();
  struct Case {
    uint32_t offset; // in words: 1022 puts the four words across a page boundary
    uint32_t user;
    std::array<uint32_t, 4> words;
  };
  const std::array<Case, 6> cases{{
      {0, 7u, {0x10000u, 0x00100000u, 0x40u, 0x00027facu}},
      {16, 0x100u, {0x20000u, 0x00200000u, 0x80u, 0x00027facu}},
      {1022, 0x30u, {0x30000u, 0x00300000u, 0xc0u, 0x00027facu}},
      {32, 0x55u, {0x40000u, 0x00400000u, 0x100u, 0u}}, // the select's other arm
      {48, 0xfffffff0u, {0x50000u, 0x00500000u, 0x140u, 0x00027facu}},
      {64, 9u, {0x60000u, 0x00600000u, 0x180u, 0u}},
  }};
  struct Result {
    bool ok = false;
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
  };
  const auto run = [&](const Case &c, Result &result) {
    for (uint32_t i = 0; i < 4; i++) {
      pages[c.offset + i] = c.words[i];
    }
    const auto address = reinterpret_cast<uint64_t>(&pages[c.offset]);
    const std::array<uint32_t, 3> user_data{
        c.user, static_cast<uint32_t>(address),
        static_cast<uint32_t>(address >> 32u)};
    const SrtRuntime speculative{.user_data = user_data,
                                 .read_memory = StrictRead,
                                 .read_specialization_memory = StrictRead,
                                 .try_read_clean_backing = CleanProbe,
                                 .share_clean_values = false};
    result.ok = MaterializeResources(plan, speculative, result.snapshot,
                                     result.specialization);
  };
  const auto same = [](const Result &left, const Result &right) {
    return left.ok == right.ok &&
           left.snapshot.flattened_srt == right.snapshot.flattened_srt &&
           left.snapshot.buffers == right.snapshot.buffers &&
           left.snapshot.user_data == right.snapshot.user_data &&
           left.specialization == right.specialization;
  };
  g_probe_memory = {};
  Live::Testing::StageText("KYTY_SRT_TRACE=0\n");
  Live::OnCpFlip();
  std::array<Result, cases.size()> expected;
  for (size_t i = 0; i < cases.size(); i++) {
    run(cases[i], expected[i]);
    Check(expected[i].ok, "traced plan reference materialization failed");
  }
  Check(expected[0].snapshot.buffers[0].dwords[1] == 0x00100000u &&
            expected[3].snapshot.buffers[0].dwords[1] == 0x55u,
        "traced plan select did not pick both arms");
  // exit: every refresh that used a trace is evaluated again and any difference stops.
  const auto served_before = SrtTraceSession::ServedRefreshes();
  Live::Testing::StageText("KYTY_SRT_TRACE=exit\n");
  Live::OnCpFlip();
  // Repeats record, then serve; the page-crossing and other-arm cases fail guards.
  for (uint32_t round = 0; round < 6; round++) {
    for (size_t i = 0; i < cases.size(); i++) {
      const size_t index = round % 2u == 0u ? i : (i * 5u + round) % cases.size();
      for (uint32_t repeat = 0; repeat < 3; repeat++) {
        Result result;
        run(cases[index], result);
        Check(same(result, expected[index]), "traced refresh differed from the walk");
      }
    }
  }
  SrtTraceSession::PrintStats();
  Check(SrtTraceSession::ServedRefreshes() > served_before,
        "no refresh was served from a trace");
  Live::Testing::StageText("KYTY_SRT_TRACE=0\n");
  Live::OnCpFlip();
  Check(g_probe_memory.strict_failures == 0, "traced refreshes reported failed reads");
}

// KYTY_MS_DEPTH_SAMPLE0 (usa-ratt/KytyPS5-GT7 round 124): a multisample T# read by an
// instruction of a single-sample dimension. Depth tiling (a depth or stencil surface): the
// single-sample dimension the instruction reads (the host binds a copy of the first sample).
// Render-target tiling, an instruction of a multisample dimension, or the switch off: the
// multisample dimension, as before.
// One T# serves an image read in the entry block, an image read only behind a user-data branch,
// and an image no block lists. With KYTY_SRT_IMAGE_REACH the guarded image reads zero while its
// block is unreached; the others, and every image with the switch off, keep the T#.
bool NoSpecializationRead(void *, uint64_t, std::span<uint32_t>) { return false; }

void TestUnreachedImageIgnoresSharedDescriptor(bool image_reach) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::ImageType;
  using Libs::Graphics::Prospero::TileMode;
  auto options = Libs::Graphics::ShaderRecompiler::GetCodegenOptions();
  const auto saved = options;
  options.srt_image_reach = image_reach;
  Libs::Graphics::ShaderRecompiler::SetCodegenOptions(options);

  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 2u;
  auto &entry = AddValueBlock(program);
  auto &guarded = AddValueBlock(program);
  auto &done = AddValueBlock(program);
  for (uint32_t i = 0; i < 3u; ++i) program.block_info[i].id = i;
  const uint32_t words[8] = {
      0x0120fc00u, static_cast<uint32_t>(BufferFormat::k32Float) << 20u | 0xc0000000u,
      0x021bc1dfu,
      static_cast<uint32_t>(ImageType::kColor2D) << 28u |
          static_cast<uint32_t>(TileMode::kRenderTarget) << 20u | 0xfacu,
      0u, 0u, 0u, 0u};
  DescriptorSource source;
  source.dword_count = 8u;
  for (uint32_t i = 0; i < 8u; ++i) source.dwords[i] = Value(words[i]);
  program.descriptor_sources.push_back(source);
  for (uint32_t image = 0; image < 3u; ++image)
    program.info.images.push_back(
        {.source = 0u,
         .resource_class = ImageResourceClass::Sampled,
         .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
         .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  for (uint32_t image = 0; image < 2u; ++image)
    program.memory_info.push_back({.kind = ResourceKind::Image, .resource = image});
  entry.AppendNewInst(ValueOpcode::ImageRead, {Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 0});
  guarded.AppendNewInst(ValueOpcode::ImageRead, {Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  const auto user = Value(&entry.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(1))}));
  program.block_info[0].condition =
      Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {user, Value(0u)}));
  program.block_info[0].terminator.kind = TerminatorKind::ConditionalBranch;
  program.block_info[0].terminator.true_block = 1u;
  program.block_info[0].terminator.false_block = 2u;
  program.block_info[1].terminator.kind = TerminatorKind::Branch;
  program.block_info[1].terminator.true_block = 2u;
  program.block_info[2].terminator.kind = TerminatorKind::Return;
  auto plan = ExtractResourcePlan(program);
  Check(!plan.control_flow.empty(), "image reach test plan has no control flow");
  for (const uint32_t reached : {0u, 1u}) {
    const std::array<uint32_t, 2> data{0u, reached};
    const SrtRuntime runtime{.user_data = data,
                             .read_specialization_memory = NoSpecializationRead};
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.images.size() == 3u,
          "image reach test failed to materialize");
    for (uint32_t image = 0; image < 3u; ++image) {
      const bool zero = image_reach && image == 1u && reached == 0u;
      for (uint32_t i = 0; i < 8u; ++i) {
        Check(snapshot.images[image].dwords[i] == (zero ? 0u : words[i]),
              zero ? "unreached image kept the shared descriptor"
                   : "reached or unlisted image lost the shared descriptor");
      }
    }
  }
  Libs::Graphics::ShaderRecompiler::SetCodegenOptions(saved);
}

// KYTY_WORKGROUP_IMAGES with a runtime base and a constant division (upstream de3ed905c): the key
// (3 + workgroup ID z) * 0xaaaaaaab >> 33 = (3 + z) / 3 over four workgroups reaches rows 1 and 2
// only, and the key mapping that ends the flattened SRT lists exactly those.
struct WorkgroupTableMemory {
  uint64_t base = 0x10000u;
  std::vector<uint32_t> words;
};

bool ReadWorkgroupTable(void *userdata, uint64_t address, std::span<uint32_t> values) {
  const auto &memory = *static_cast<const WorkgroupTableMemory *>(userdata);
  for (size_t i = 0; i < values.size(); ++i) {
    const auto index = (address - memory.base) / 4u + i;
    if (address < memory.base || index >= memory.words.size()) return false;
    values[i] = memory.words[index];
  }
  return true;
}

void TestWorkgroupDividedImageKeys() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::ImageType;
  using Libs::Graphics::Prospero::TileMode;
  constexpr uint32_t table_offset = 0x100u;
  WorkgroupTableMemory memory;
  memory.words.resize((table_offset + 4u * 32u) / 4u);
  for (uint32_t row = 0; row < 4u; ++row) {
    const uint32_t words[8] = {
        row + 1u, static_cast<uint32_t>(BufferFormat::k32Float) << 20u, 0u,
        static_cast<uint32_t>(ImageType::kColor2D) << 28u |
            static_cast<uint32_t>(TileMode::kRenderTarget) << 20u | 0xfacu,
        0u, 0u, 0u, 0u};
    std::copy_n(words, 8, memory.words.begin() + (table_offset + row * 32u) / 4u);
  }
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  AddValueBlock(program);
  DescriptorSource table;
  table.dword_count = 2u;
  table.dwords.fill(Value(0u));
  table.dwords[0] = Value(static_cast<uint32_t>(memory.base));
  program.descriptor_sources.push_back(table);
  DescriptorSource image;
  image.dword_count = 8u;
  image.dwords.fill(Value(0u));
  auto &indirect = image.indirect_descriptor.emplace();
  indirect.table_source = 0u;
  indirect.table_offset = table_offset;
  indirect.table_stride = 32u;
  indirect.workgroup_axis = 2u;
  indirect.selector_first = Value(3u);
  indirect.workgroup_multiplier = 0xaaaaaaabu;
  indirect.workgroup_shift = 33u;
  program.descriptor_sources.push_back(image);
  program.info.images.push_back(
      {.source = 1u,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  auto plan = ExtractResourcePlan(program);
  const std::array<uint32_t, 3> groups{1u, 1u, 4u};
  const SrtRuntime runtime{.userdata = &memory,
                           .read_specialization_memory = ReadWorkgroupTable,
                           .workgroup_counts = groups};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "divided workgroup image keys failed to materialize");
  const auto &flat = snapshot.flattened_srt;
  const std::array<uint32_t, 5> expected{2u, 1u, 0u, 2u, 1u};
  Check(flat.size() >= expected.size() &&
            std::equal(expected.begin(), expected.end(), flat.end() - expected.size()) &&
            snapshot.images.size() == 2u && snapshot.images[0].dwords[0] == 2u &&
            snapshot.images[1].dwords[0] == 3u,
        "divided workgroup image keys enumerated other rows");
}

void TestMultisampleDepthReadAsSingleSample() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::ImageType;
  using Libs::Graphics::Prospero::TileMode;

  struct Case {
    bool enabled;
    ImageType type;
    TileMode tile;
    ImageDimension instruction;
    ImageDimension expected;
    const char *text;
  };
  const Case cases[] = {
      {true, ImageType::kColor2DMsaa, TileMode::kDepth, ImageDimension::Dim2D,
       ImageDimension::Dim2D, "a 2D read of a multisample depth T# is a single-sample 2D image"},
      {true, ImageType::kColor2DMsaa, TileMode::kDepth, ImageDimension::Dim2DArray,
       ImageDimension::Dim2D, "a 2D-array read of a multisample depth T# is a 2D image"},
      {true, ImageType::kColor2DMsaa, TileMode::kDepth, ImageDimension::Dim2DMsaa,
       ImageDimension::Dim2DMsaa, "a 2D-MSAA read of a multisample depth T# stays multisample"},
      {true, ImageType::kColor2DMsaaArray, TileMode::kDepth, ImageDimension::Dim2DArray,
       ImageDimension::Dim2DArray, "a 2D-array read of a multisample depth array is a 2D array"},
      {true, ImageType::kColor2DMsaaArray, TileMode::kDepth, ImageDimension::Dim2DMsaaArray,
       ImageDimension::Dim2DMsaaArray, "a 2D-MSAA-array read stays a multisample array"},
      {true, ImageType::kColor2DMsaa, TileMode::kRenderTarget, ImageDimension::Dim2D,
       ImageDimension::Dim2DMsaa, "a 2D read of a multisample colour T# keeps its samples"},
      {false, ImageType::kColor2DMsaa, TileMode::kDepth, ImageDimension::Dim2D,
       ImageDimension::Dim2DMsaa, "with the switch off a multisample depth T# stays multisample"},
  };
  for (const auto &test : cases) {
    SetMsDepthSample0ForTesting(test.enabled);
    Program program;
    program.stage = Libs::Graphics::ShaderType::Pixel;
    program.srt_plan_complete = true;
    program.resource_tracking_complete = true;
    AddValueBlock(program);
    DescriptorSource source;
    source.dword_count = 8;
    const uint32_t fragments = 1u; // two samples
    const uint32_t words[8] = {
        0x0120fc00u, // base address 0x120fc0000 >> 8
        static_cast<uint32_t>(BufferFormat::k32Float) << 20u | 0xc0000000u, // width - 1 low bits
        0x021bc1dfu, // width - 1 high bits (1919 >> 2), height - 1 (2159) << 14
        static_cast<uint32_t>(test.type) << 28u | static_cast<uint32_t>(test.tile) << 20u |
            fragments << 16u | 0xfacu,
        0u,
        fragments << 4u,
        0u,
        0u};
    for (uint32_t i = 0; i < 8; i++) {
      source.dwords[i] = Value(words[i]);
    }
    program.descriptor_sources.push_back(source);
    program.info.images.push_back(
        {.source = 0,
         .resource_class = ImageResourceClass::Sampled,
         .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
         .dimension = test.instruction});
    auto plan = ExtractResourcePlan(program);
    const SrtRuntime runtime{};
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    Check(MaterializeResources(plan, runtime, snapshot, specialization), test.text);
    Check(specialization.images.size() == 1 &&
              specialization.images[0].dimension == test.expected,
          test.text);
  }
  SetMsDepthSample0ForTesting(false);
}

// KYTY_RAW_SMALL_SINT_STORAGE: a write-only 16/8-bit SINT storage image is specialized as a uint
// storage image (the same low bits through an unsigned view); without the switch, or when the
// shader also reads it, the program fails as before (Ghost of Yotei's cloud shadow precompute).
void TestRawSmallSintStorage() {
  using namespace Libs::Graphics::ShaderRecompiler;
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::ImageType;
  using Libs::Graphics::Prospero::TextureNumericClass;
  using Libs::Graphics::Prospero::TileMode;
  const auto saved = GetCodegenOptions();
  struct Case {
    bool enabled;
    BufferFormat format;
    bool read;
    bool expected;
    const char *text;
  };
  const Case cases[] = {
      {false, BufferFormat::k16SInt, false, false, "16-bit SINT storage fails with the switch off"},
      {true, BufferFormat::k16SInt, false, true, "write-only 16-bit SINT storage is raw uint storage"},
      {true, BufferFormat::k8SInt, false, true, "write-only 8-bit SINT storage is raw uint storage"},
      {true, BufferFormat::k16SInt, true, false, "read-write 16-bit SINT storage still fails"},
  };
  for (const auto &test : cases) {
    auto options = saved;
    options.raw_small_sint_storage = test.enabled;
    SetCodegenOptions(options);
    Program program;
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.srt_plan_complete = true;
    program.resource_tracking_complete = true;
    AddValueBlock(program);
    DescriptorSource source;
    source.dword_count = 8;
    const uint32_t words[8] = {
        0x0120fc00u, // base address 0x120fc0000 >> 8
        static_cast<uint32_t>(test.format) << 20u | 0xc0000000u, // width - 1 low bits
        0x007fc07fu, // width 512, height 512
        static_cast<uint32_t>(ImageType::kColor2D) << 28u |
            static_cast<uint32_t>(TileMode::kRenderTarget) << 20u | 0xfacu,
        0u,
        0u,
        0u,
        0u};
    for (uint32_t i = 0; i < 8; i++) {
      source.dwords[i] = Value(words[i]);
    }
    program.descriptor_sources.push_back(source);
    program.info.images.push_back({.source = 0,
                                   .resource_class = ImageResourceClass::Storage,
                                   .numeric_class = TextureNumericClass::Uint,
                                   .dimension = ImageDimension::Dim2D,
                                   .read = test.read,
                                   .written = true});
    auto plan = ExtractResourcePlan(program);
    const SrtRuntime runtime{};
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    const bool materialized = MaterializeResources(plan, runtime, snapshot, specialization);
    Check(materialized == test.expected, test.text);
    if (materialized) {
      Check(specialization.images.size() == 1 &&
                specialization.images[0].numeric_class == TextureNumericClass::Uint,
            test.text);
    }
  }
  SetCodegenOptions(saved);
}

} // namespace

namespace Common {

void DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

void DbgExitIfHandler(const char *, const char *, int) { std::abort(); }

} // namespace Common

int main() {
  for (const char* state : {"KYTY_BUFFER_REFRESH_FUSION=0\n", "KYTY_BUFFER_REFRESH_FUSION=1\n", "KYTY_BUFFER_REFRESH_FUSION=0\n"}) {
    Live::Testing::StageText(state);
    Live::OnCpFlip();
  TestMappedSrtUsesDirectReaderByDefault();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestUnbasedFlatCacheHitMaterializes();
  TestFailedMaterializationRejectsStage();
  TestMixedSamplerVariantsShareRuntimeDescriptor();
  TestSealedPlanEvaluatesConcurrently();
  TestSpeculativeRuntimeMatchesSerial();
  TestRecordedPreparationCertifies();
  TestTraceReplayMatchesWalk();
  TestGpuFillSlotsLeaveShaderOnlyReads();
  }
  TestMultisampleDepthReadAsSingleSample();
  TestNullBufferShare();
  TestRawSmallSintStorage();
  for (const bool image_reach : {false, true}) TestUnreachedImageIgnoresSharedDescriptor(image_reach);
  TestWorkgroupDividedImageKeys();
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
