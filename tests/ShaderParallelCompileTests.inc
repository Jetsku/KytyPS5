// Parallel shader compilation (included by ShaderRecompilerComputeTests.cpp after
// ShaderIsaAuditTests.inc). CPU only.
//
// The shader precompile replay compiles journal entries on several threads at once, each from a
// fresh translation or from a copy of a translation the pipeline cache keeps per source
// (IR::CloneProgram, KYTY_TRANSLATION_CACHE). --parallel-compile-only does the same with every
// compute case: one single-threaded reference compile per case, then several threads that each
// compile every case several times, in their own order, half from fresh translations and half
// from copies of one shared translation, and compare the SPIR-V with the reference.
//
// Every compile also checks that the compiled program names no instruction it no longer has
// (IR::CountDeadPlanReferences). Specialization used to leave such references (the descriptor of
// a zero-stride buffer read or an FMASK read it replaced), and exec-select elimination read the
// freed instructions: harmless on one thread, a crash when another compile thread had reused
// their memory (Wolverine's parallel replay after a shader cache invalidation).
//
// --journal-compile <shaders.journal> [threads]: the same for every permutation of a shader
// journal (_PipelineCache/<title>.shaders.journal). Pass a copy: nothing is recorded, but the
// journal class may rewrite a file it finds damaged. Prints how many permutations name
// instructions specialization made dead before the clean-up, a digest of the SPIR-V (to compare
// builds), and whether parallel compiles agree with the single-threaded ones.

namespace ParallelCompileTests {

using ShaderRecompiler::IR::CloneProgram;
using ShaderRecompiler::IR::CountDeadPlanReferences;

// One program to compile repeatedly, with everything CompileProgram reads kept alive.
struct Job {
  std::string name;
  std::vector<u32> code;
  std::vector<u32> back_code;
  std::vector<u32> user_data;
  ShaderVertexInputInfo vertex{};
  ShaderPixelInputInfo pixel{};
  ShaderComputeInputInfo compute{};
  ShaderRecompiler::CompileOptions options;
  ShaderRecompiler::IR::ResourceSpecialization specialization;
  uint32_t push_data_cursor = 0;
  // A translation of the program that every thread copies, as the pipeline cache's kept one.
  std::shared_ptr<const ShaderRecompiler::TranslateResult> kept;
  // Single-threaded reference.
  std::vector<u32> spirv;
  std::vector<u32> spirv_plain;
};

bool CopyTranslation(const ShaderRecompiler::TranslateResult &from,
                     ShaderRecompiler::TranslateResult &to) {
  if (!CloneProgram(from.program, to.program)) {
    return false;
  }
  to.decoded_dump = from.decoded_dump;
  to.cfg_dump = from.cfg_dump;
  to.skip_dispatch = from.skip_dispatch;
  return true;
}

// The references specialization leaves to instructions it made dead: CompileProgram's first steps
// on a copy, without its clean-up.
size_t DeadReferencesAfterSpecialization(const Job &job) {
  ShaderRecompiler::TranslateResult copy;
  if (!CopyTranslation(*job.kept, copy)) {
    return 0;
  }
  auto &program = copy.program;
  ShaderRecompiler::IR::ApplyResourceSpecialization(program, job.specialization);
  ShaderRecompiler::IR::RemoveIdentities(program.blocks);
  ShaderRecompiler::IR::EliminateDeadCode(program.blocks);
  return CountDeadPlanReferences(program);
}

struct Outcome {
  bool ok = true;
  std::string message;
};

// One compile of `job`: from a copy of the kept translation or from a fresh one.
Outcome CompileOnce(const Job &job, bool from_copy) {
  ShaderRecompiler::TranslateResult translated;
  if (!from_copy || !CopyTranslation(*job.kept, translated)) {
    translated = ShaderRecompiler::TranslateProgram(job.code, job.options);
  }
  if (CountDeadPlanReferences(translated.program) != 0) {
    return {false, "the translation names instructions it does not have"};
  }
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), job.options,
                                                 job.specialization, job.push_data_cursor);
  if (CountDeadPlanReferences(result.program) != 0) {
    return {false, "the compiled program names instructions it does not have"};
  }
  if (result.spirv != job.spirv || result.spirv_plain != job.spirv_plain) {
    return {false, std::string("SPIR-V differs from the single-threaded compile (") +
                       (from_copy ? "copied" : "fresh") + " translation)"};
  }
  return {};
}

// The reference compile; false when the program is not compiled (skip-dispatch). `strict`: a
// program that names instructions it does not have fails the test; otherwise it is counted in
// `dead_programs`.
bool Finish(Job &job, bool strict, size_t *dead_programs = nullptr) {
  auto translated = ShaderRecompiler::TranslateProgram(job.code, job.options);
  if (translated.skip_dispatch) {
    return false;
  }
  const bool translation_dead = CountDeadPlanReferences(translated.program) != 0;
  Require(job.name.c_str(), "parallel compile translation", !strict || !translation_dead,
          "the translation names instructions it does not have");
  auto kept = std::make_shared<ShaderRecompiler::TranslateResult>();
  if (!CopyTranslation(translated, *kept)) {
    return false;
  }
  job.kept = std::move(kept);
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), job.options,
                                                 job.specialization, job.push_data_cursor);
  const bool compiled_dead = CountDeadPlanReferences(result.program) != 0;
  Require(job.name.c_str(), "parallel compile reference", !strict || !compiled_dead,
          "the compiled program names instructions specialization made dead");
  if (dead_programs != nullptr && (translation_dead || compiled_dead)) {
    (*dead_programs)++;
  }
  job.spirv = std::move(result.spirv);
  job.spirv_plain = std::move(result.spirv_plain);
  return true;
}

// `threads` threads compile every job `rounds` times. Returns the failures.
std::vector<std::string> RunParallel(const std::vector<std::unique_ptr<Job>> &jobs,
                                     uint32_t threads, uint32_t rounds) {
  std::mutex mutex;
  std::vector<std::string> failures;
  std::vector<std::thread> workers;
  const size_t count = jobs.size();
  for (uint32_t thread = 0; thread < threads; thread++) {
    workers.emplace_back([&, thread] {
      // Each thread starts elsewhere in the list, so that different threads compile the same
      // program at once as well as different ones.
      const size_t start = count * thread / threads;
      for (uint32_t round = 0; round < rounds; round++) {
        for (size_t step = 0; step < count; step++) {
          const auto &job = *jobs[(start + step + round) % count];
          const bool from_copy = ((step + thread + round) & 1u) == 0;
          const auto outcome = CompileOnce(job, from_copy);
          if (!outcome.ok) {
            std::scoped_lock lock(mutex);
            if (failures.size() < 16) {
              failures.push_back(job.name + ": " + outcome.message);
            }
          }
        }
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  return failures;
}

// Every compute case that compiles without a device, prepared as CompileCase prepares it.
std::vector<std::unique_ptr<Job>> CaseJobs() {
  std::vector<std::unique_ptr<Job>> jobs;
  for (const auto &test : MakeCases()) {
    if (test.code.empty()) {
      continue;
    }
    auto job = std::make_unique<Job>();
    job->name = test.name;
    job->code = test.code;
    auto user_data = MakeNativeUserData(test.has_user_data ? &test.user_data : nullptr);
    const auto uses_image =
        std::any_of(test.opcodes.begin(), test.opcodes.end(), [](auto op) {
          return op >= ShaderOpcode::IMAGE_GET_RESINFO && op <= ShaderOpcode::IMAGE_GATHER4H;
        });
    if (uses_image && ((user_data[3] >> 28u) & 0xfu) == 0) {
      user_data[3] = static_cast<uint32_t>(Prospero::ImageType::kColor2D) << 28u;
    }
    if (uses_image) {
      user_data[3] = (user_data[3] & ~0xfffu) | test.image_descriptor_swizzle;
    }
    if (!test.has_user_data) {
      user_data[2] = static_cast<u32>(test.initial.size() * sizeof(u32));
    }
    job->user_data.assign(user_data.begin(), user_data.end());
    job->compute = test.compute_info;
    job->compute.host_subgroup_size = 64;
    if (!job->compute.lds_storage) {
      const auto lds = ShaderRecompiler::PlanComputeLds(job->compute.lds_size_dwords,
                                                        g_case_lds_limit_bytes);
      job->compute.lds_size_dwords = lds.dwords;
      job->compute.lds_storage = lds.storage;
    }
    auto &options = job->options;
    options.stage = ShaderType::Compute;
    options.dump_ir = false;
    options.input_info.compute = &job->compute;
    options.user_data = job->user_data;
    if (test.has_compute_info) {
      options.wave_size = test.compute_info.wave_size;
    }
    // The resources the case's own memory gives (CompileCase materializes the same way).
    {
      auto translated = ShaderRecompiler::TranslateProgram(job->code, options);
      if (translated.skip_dispatch) {
        continue;
      }
      const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
      ShaderRecompiler::IR::ResourceSnapshot resources;
      const ShaderRecompiler::IR::SrtRuntime runtime{
          .user_data = options.user_data,
          .shader_base = reinterpret_cast<uint64_t>(job->code.data()),
          .read_memory = ReadTestMemory,
          .userdata = const_cast<std::vector<u32> *>(&test.initial),
      };
      if (!ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources,
                                                      job->specialization)) {
        continue;
      }
    }
    if (Finish(*job, true)) {
      jobs.push_back(std::move(job));
    }
  }
  return jobs;
}

// ctest shader_parallel_compile.
void CheckParallelCompile() {
  const char *name = "ParallelCompile";
  const auto prepare_begin = std::chrono::steady_clock::now();
  const auto jobs = CaseJobs();
  Require(name, "cases", jobs.size() > 100, "too few compute cases compiled");
  size_t dead = 0;
  for (const auto &job : jobs) {
    dead += DeadReferencesAfterSpecialization(*job) != 0 ? 1u : 0u;
  }
  // The zero-stride buffer and FMASK cases: what the clean-up exists for.
  Require(name, "dead references", dead != 0,
          "no case leaves dead plan references after specialization; the check checks nothing");
  const uint32_t threads =
      std::clamp<uint32_t>(std::thread::hardware_concurrency(), 4u, 8u);
  const uint32_t rounds = 3;
  const auto run_begin = std::chrono::steady_clock::now();
  const auto failures = RunParallel(jobs, threads, rounds);
  const auto end = std::chrono::steady_clock::now();
  std::string message;
  for (const auto &failure : failures) {
    message += "\n  " + failure;
  }
  Require(name, "parallel compiles", failures.empty(),
          "compiles on several threads failed:" + message);
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  std::printf("[cpu]     %-32s ok (%zu programs, %zu with dead references after specialization; "
              "%u threads x %u rounds; %.0f ms, reference %.0f ms)\n",
              name, jobs.size(), dead, threads, rounds, ms(run_begin, end),
              ms(prepare_begin, run_begin));
}

template <typename InputInfo>
bool LoadInputInfo(const ShaderJournal::Source &source, InputInfo &info) {
  constexpr auto stored = ShaderStoredInputInfoSize<InputInfo>();
  if (source.input_info.size() != stored) {
    return false;
  }
  std::memcpy(static_cast<void *>(&info), source.input_info.data(), stored);
  info.stage = {};
  ShaderSanitizeStoredInputInfo(info);
  if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
    info.param_alias_mask = 0;
  }
  return true;
}

int JournalCompile(const char *path, uint32_t threads) {
  std::vector<uint8_t> identity;
  if (!ShaderJournal::ReadIdentity(path, identity)) {
    std::printf("%s: not a shader journal\n", path);
    return 2;
  }
  ShaderJournal::Settings settings;
  settings.path = path;
  settings.identity = identity;
  settings.background_writer = false;
  ShaderJournal journal(std::move(settings));
  ShaderRecompiler::IR::SetBindlessImagesEnabled(true); // as on a device with descriptor indexing
  const auto begin = std::chrono::steady_clock::now();
  std::vector<std::unique_ptr<Job>> jobs;
  size_t skipped = 0;
  size_t dead_compiled = 0;
  for (const auto &entry : journal.Entries()) {
    if (entry.source >= journal.Sources().size()) {
      skipped++;
      continue;
    }
    const auto &source = journal.Sources()[entry.source];
    auto job = std::make_unique<Job>();
    job->name = fmt::format("stage {} 0x{:016x} cursor {}", source.stage, source.hash,
                            entry.push_data_cursor);
    job->code = source.code;
    job->back_code = source.back_code;
    job->user_data.assign(source.user_data_count, 0u);
    auto &options = job->options;
    options.stage = static_cast<ShaderType>(source.stage);
    options.shader_hash = source.hash;
    options.wave_size = source.wave_size;
    options.user_data_base = source.user_data_base;
    options.plain_mip_stats_variant = source.plain_mip_stats_variant;
    options.dump_ir = false;
    options.user_data = job->user_data;
    options.back_code = job->back_code;
    bool loaded = false;
    switch (source.kind) {
      case ShaderJournal::Kind::Vertex:
        loaded = LoadInputInfo(source, job->vertex);
        options.input_info.vertex = &job->vertex;
        break;
      case ShaderJournal::Kind::Pixel:
        loaded = LoadInputInfo(source, job->pixel);
        options.input_info.pixel = &job->pixel;
        break;
      case ShaderJournal::Kind::Compute:
        loaded = LoadInputInfo(source, job->compute);
        options.input_info.compute = &job->compute;
        break;
    }
    job->push_data_cursor = entry.push_data_cursor;
    if (job->code.empty() || !loaded ||
        !ShaderRecompiler::IR::DecodeSpecialization(entry.specialization, job->specialization) ||
        !Finish(*job, false, &dead_compiled)) {
      skipped++;
      continue;
    }
    jobs.push_back(std::move(job));
  }
  const auto reference_end = std::chrono::steady_clock::now();
  size_t dead = 0;
  for (const auto &job : jobs) {
    dead += DeadReferencesAfterSpecialization(*job) != 0 ? 1u : 0u;
  }
  // A digest of every reference module, to compare builds.
  uint64_t digest = 0;
  for (const auto &job : jobs) {
    digest = XXH3_64bits_withSeed(job->spirv.data(), job->spirv.size() * sizeof(u32), digest);
    digest = XXH3_64bits_withSeed(job->spirv_plain.data(), job->spirv_plain.size() * sizeof(u32),
                                  digest);
  }
  const auto run_begin = std::chrono::steady_clock::now();
  const auto failures = RunParallel(jobs, threads, 2);
  const auto end = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  std::printf("%s: %zu permutations compiled, %zu skipped; %zu with dead plan references after "
              "specialization, %zu still with them when compiled; SPIR-V digest %016" PRIx64
              "; reference %.0f ms; %u threads x 2 rounds %.0f ms; %zu failures\n",
              path, jobs.size(), skipped, dead, dead_compiled, digest, ms(begin, reference_end),
              threads, ms(run_begin, end), failures.size());
  for (const auto &failure : failures) {
    std::printf("  %s\n", failure.c_str());
  }
  return failures.empty() ? 0 : 1;
}

} // namespace ParallelCompileTests
