#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/liveSwitch.h"

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

} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

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
  }
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
