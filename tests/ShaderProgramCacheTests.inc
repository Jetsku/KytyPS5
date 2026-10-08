// Persistent program cache tests (KYTY_PROGRAM_CACHE; included by ShaderRecompilerComputeTests.cpp
// after ShaderCodegenTests.inc).
//
// --program-cache-only (CPU): every input of a source key changes the key; every codegen switch,
// CodegenOptions field and host state the emitter reads changes the codegen fingerprint (the file
// identity); the cache file round-trips records, rejects another identity, keeps the intact prefix
// of a truncated or corrupted file and never crashes on one; verification drops records.
// --program-cache-gpu-only: compute dispatches and graphics draws through the renderer (the full
// PipelineCache path) in a cold context, then again in a new context that reloads every program
// from the file instead of translating it: identical results. The codec round trip for every
// compute and fragment case runs in CompileCase and CompileFragmentCase (ProgramCacheRoundTrip).

namespace ProgramCacheTests {

using CodegenTests::ScopedCodegenOptions;
using ShaderRecompiler::CodegenOptions;
using ShaderRecompiler::GetCodegenOptions;

// Sets (or, with nullptr, removes) an environment variable for the scope.
class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : m_name(name) {
    if (const auto *previous = std::getenv(name); previous != nullptr) {
      m_previous = previous;
      m_had = true;
    }
    Set(value);
  }
  ~ScopedEnv() { Set(m_had ? m_previous.c_str() : nullptr); }
  ScopedEnv(const ScopedEnv &) = delete;
  ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
  void Set(const char *value) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
    // An empty value removes the variable.
    _putenv_s(m_name.c_str(), value != nullptr ? value : "");
#else
    if (value != nullptr) {
      setenv(m_name.c_str(), value, 1);
    } else {
      unsetenv(m_name.c_str());
    }
#endif
  }
  std::string m_name;
  std::string m_previous;
  bool m_had = false;
};

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path &path) {
  std::vector<uint8_t> bytes;
  if (std::FILE *file = std::fopen(path.string().c_str(), "rb"); file != nullptr) {
    std::fseek(file, 0, SEEK_END);
    const auto size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    bytes.resize(size > 0 ? static_cast<size_t>(size) : 0);
    if (!bytes.empty() && std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
      bytes.clear();
    }
    std::fclose(file);
  }
  return bytes;
}

void WriteFileBytes(const std::filesystem::path &path, std::span<const uint8_t> bytes) {
  std::FILE *file = std::fopen(path.string().c_str(), "wb");
  Require("ProgramCache", "test file", file != nullptr, "could not create a test file");
  if (!bytes.empty()) {
    std::fwrite(bytes.data(), 1, bytes.size(), file);
  }
  std::fclose(file);
}

void RemoveTestFile(const std::filesystem::path &path) {
  std::error_code error;
  std::filesystem::remove(path, error);
  auto temp = path;
  temp += ".tmp";
  std::filesystem::remove(temp, error);
}

// ---------------------------------------------------------------------------------------------
// Source keys.

void CheckSourceKeys() {
  constexpr const char *name = "ProgramCacheSourceKeys";
  const std::vector<u32> static_state{1, 2, 3};
  const std::vector<u32> code{10, 11, 12, 13};
  const std::vector<u32> back_code{20, 21};
  const ProgramDiskCache::SourceKeyInputs inputs{
      .stage = static_cast<u32>(ShaderType::Pixel),
      .hash = 0x1234567890abcdefull,
      .user_data_count = 16,
      .code_size = 4,
      .static_state = static_state,
      .wave_size = 64,
      .user_data_base = 0,
      .plain_mip_stats_variant = false,
      .code = code,
      .back_code = {},
  };
  ProgramDiskCache::SourceKey base;
  ProgramDiskCache::BuildSourceKey(inputs, base);
  ProgramDiskCache::SourceKey same;
  ProgramDiskCache::BuildSourceKey(inputs, same);
  Require(name, "determinism", same.bytes == base.bytes && same.digest == base.digest,
          "equal inputs gave different keys");

  const auto differs = [&](const char *input, auto &&mutate) {
    auto changed = inputs;
    std::vector<u32> changed_static(static_state.begin(), static_state.end());
    std::vector<u32> changed_code(code.begin(), code.end());
    std::vector<u32> changed_back;
    mutate(changed, changed_static, changed_code, changed_back);
    changed.static_state = changed_static;
    changed.code = changed_code;
    if (!changed_back.empty()) {
      changed.back_code = changed_back;
    }
    ProgramDiskCache::SourceKey key;
    ProgramDiskCache::BuildSourceKey(changed, key);
    Require(name, input, key.bytes != base.bytes && key.digest != base.digest,
            std::string("the key does not depend on ") + input);
  };
  using Inputs = ProgramDiskCache::SourceKeyInputs;
  using Words = std::vector<u32>;
  differs("stage", [](Inputs &in, Words &, Words &, Words &) {
    in.stage = static_cast<u32>(ShaderType::Vertex);
  });
  differs("guest hash", [](Inputs &in, Words &, Words &, Words &) { in.hash ^= 1u; });
  differs("user-data count", [](Inputs &in, Words &, Words &, Words &) { in.user_data_count++; });
  differs("code size", [](Inputs &in, Words &, Words &, Words &) { in.code_size++; });
  differs("a static key word", [](Inputs &, Words &s, Words &, Words &) { s[1] ^= 0x100u; });
  differs("the static key length", [](Inputs &, Words &s, Words &, Words &) { s.push_back(0); });
  differs("wave size", [](Inputs &in, Words &, Words &, Words &) { in.wave_size = 32; });
  differs("user-data base", [](Inputs &in, Words &, Words &, Words &) { in.user_data_base = 8; });
  differs("the plain LOD-stats variant", [](Inputs &in, Words &, Words &, Words &) {
    in.plain_mip_stats_variant = true;
  });
  differs("a code word", [](Inputs &, Words &, Words &c, Words &) { c[3] ^= 0x80000000u; });
  differs("the code length", [](Inputs &, Words &, Words &c, Words &) { c.push_back(0); });
  differs("a merged back half", [&](Inputs &, Words &, Words &, Words &b) { b = back_code; });
  // Moving a word from the code to the back half keeps the concatenation but not the key.
  differs("the code/back-half split", [&](Inputs &, Words &, Words &c, Words &b) {
    b = {c.back()};
    c.pop_back();
  });
  std::printf("[cpu]     %-32s ok\n", name);
}

// ---------------------------------------------------------------------------------------------
// Codegen fingerprint (the file identity).

void CheckFingerprint() {
  constexpr const char *name = "ProgramCacheFingerprint";
  const auto version = ShaderRecompiler::CodegenVersion();
  Require(name, "codegen version",
          version.size() == 64 && std::ranges::all_of(version, [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
          }),
          "the codegen version is not a SHA-256 hex digest");
  const auto base = ShaderRecompiler::CodegenFingerprint();
  Require(name, "determinism", ShaderRecompiler::CodegenFingerprint() == base,
          "the fingerprint changed without a change");

  const auto options_differ = [&](const char *field, auto &&mutate) {
    auto options = GetCodegenOptions();
    mutate(options);
    ScopedCodegenOptions scope(options);
    Require(name, field, ShaderRecompiler::CodegenFingerprint() != base,
            std::string("the fingerprint does not depend on CodegenOptions::") + field);
  };
  options_differ("movrel_range", [](CodegenOptions &o) { o.movrel_range = !o.movrel_range; });
  options_differ("movrel_known_zeros",
                 [](CodegenOptions &o) { o.movrel_known_zeros = !o.movrel_known_zeros; });
  options_differ("movrel_switch", [](CodegenOptions &o) { o.movrel_switch = !o.movrel_switch; });
  options_differ("uniform_lane_reads",
                 [](CodegenOptions &o) { o.uniform_lane_reads = !o.uniform_lane_reads; });
  options_differ("short_f32_helpers",
                 [](CodegenOptions &o) { o.short_f32_helpers = !o.short_f32_helpers; });
  options_differ("fast_float_min_max",
                 [](CodegenOptions &o) { o.fast_float_min_max = !o.fast_float_min_max; });
  options_differ("fast_pkrtz", [](CodegenOptions &o) { o.fast_pkrtz = !o.fast_pkrtz; });
  options_differ("single_f2i_saturation",
                 [](CodegenOptions &o) { o.single_f2i_saturation = !o.single_f2i_saturation; });
  options_differ("lod_stats_gate", [](CodegenOptions &o) { o.lod_stats_gate = !o.lod_stats_gate; });
  options_differ("robust_buffer_loads",
                 [](CodegenOptions &o) { o.robust_buffer_loads = !o.robust_buffer_loads; });
  for (const auto mode : {ShaderRecompiler::MadMode::Exact, ShaderRecompiler::MadMode::Position,
                          ShaderRecompiler::MadMode::Fused}) {
    if (mode != GetCodegenOptions().mad_mode) {
      options_differ("mad_mode", [&](CodegenOptions &o) { o.mad_mode = mode; });
    }
  }
  options_differ("interp_modes", [](CodegenOptions &o) { o.interp_modes = !o.interp_modes; });
  options_differ("sample_offsets", [](CodegenOptions &o) { o.sample_offsets = !o.sample_offsets; });
  options_differ("sample_lod_clamp",
                 [](CodegenOptions &o) { o.sample_lod_clamp = !o.sample_lod_clamp; });
  options_differ("host_ftz_inputs", [](CodegenOptions &o) { o.host_ftz_inputs = !o.host_ftz_inputs; });
  options_differ("exec_selects", [](CodegenOptions &o) { o.exec_selects = !o.exec_selects; });
  options_differ("ps_append_live_election",
                 [](CodegenOptions &o) { o.ps_append_live_election = !o.ps_append_live_election; });
  for (const auto mode : {ShaderRecompiler::PsLiveExec::Off,
                          ShaderRecompiler::PsLiveExec::AppendConsume,
                          ShaderRecompiler::PsLiveExec::All}) {
    if (mode != GetCodegenOptions().ps_live_exec) {
      options_differ("ps_live_exec", [&](CodegenOptions &o) { o.ps_live_exec = mode; });
    }
  }
  options_differ("loop_guard_budget", [](CodegenOptions &o) { o.loop_guard_budget++; });
  options_differ("loop_guard_shaders",
                 [](CodegenOptions &o) { o.loop_guard_shaders.push_back(0x1234u); });
  options_differ("srt_variant_reads",
                 [](CodegenOptions &o) { o.srt_variant_reads = !o.srt_variant_reads; });
  options_differ("mesh_indirect_params",
                 [](CodegenOptions &o) { o.mesh_indirect_params = !o.mesh_indirect_params; });
  options_differ("realtime_clock", [](CodegenOptions &o) { o.realtime_clock = !o.realtime_clock; });
  options_differ("dpp_skip_inactive",
                 [](CodegenOptions &o) { o.dpp_skip_inactive = !o.dpp_skip_inactive; });
  options_differ("lane_reductions", [](CodegenOptions &o) { o.lane_reductions = !o.lane_reductions; });
  options_differ("dispatcher_cap", [](CodegenOptions &o) { o.dispatcher_cap++; });
  options_differ("ir_linear_uses", [](CodegenOptions &o) { o.ir_linear_uses = !o.ir_linear_uses; });
  options_differ("fold_lane_masks", [](CodegenOptions &o) { o.fold_lane_masks = !o.fold_lane_masks; });
  options_differ("ps_helper_atomics_skip",
                 [](CodegenOptions &o) { o.ps_helper_atomics_skip = !o.ps_helper_atomics_skip; });
  options_differ("loop_guard_all", [](CodegenOptions &o) { o.loop_guard_all = !o.loop_guard_all; });
  options_differ("volatile_loads", [](CodegenOptions &o) { o.volatile_loads = !o.volatile_loads; });

  // Host state the emitter reads.
  namespace Spirv = ShaderRecompiler::Spirv;
  const auto float_controls = Spirv::GetHostFloatControls();
  const auto robustness = Spirv::GetHostBufferRobustness();
  const auto image_features = Spirv::GetHostImageFeatures();
  const auto shader_clock = Spirv::GetHostShaderClock();
  const auto host_differs = [&](const char *what, auto &&set) {
    set();
    const bool differs = ShaderRecompiler::CodegenFingerprint() != base;
    Spirv::SetHostFloatControls(float_controls);
    Spirv::SetHostBufferRobustness(robustness);
    Spirv::SetHostImageFeatures(image_features);
    Spirv::SetHostShaderClock(shader_clock);
    Require(name, what, differs, std::string("the fingerprint does not depend on ") + what);
  };
  host_differs("denorm_flush_f32", [&] {
    auto changed = float_controls;
    changed.denorm_flush_f32 = !changed.denorm_flush_f32;
    Spirv::SetHostFloatControls(changed);
  });
  host_differs("denorm_preserve_f16", [&] {
    auto changed = float_controls;
    changed.denorm_preserve_f16 = !changed.denorm_preserve_f16;
    Spirv::SetHostFloatControls(changed);
  });
  host_differs("denorm_preserve_f64", [&] {
    auto changed = float_controls;
    changed.denorm_preserve_f64 = !changed.denorm_preserve_f64;
    Spirv::SetHostFloatControls(changed);
  });
  host_differs("storage_dword_loads_return_zero", [&] {
    Spirv::SetHostBufferRobustness(
        {.storage_dword_loads_return_zero = !robustness.storage_dword_loads_return_zero});
  });
  host_differs("min_lod", [&] {
    auto changed = image_features;
    changed.min_lod = !changed.min_lod;
    Spirv::SetHostImageFeatures(changed);
  });
  host_differs("compute_derivatives", [&] {
    auto changed = image_features;
    changed.compute_derivatives = image_features.compute_derivatives == Spirv::HostComputeDerivatives::Nv
                                      ? Spirv::HostComputeDerivatives::Khr
                                      : Spirv::HostComputeDerivatives::Nv;
    Spirv::SetHostImageFeatures(changed);
  });
  host_differs("shader clock scope", [&] {
    Spirv::SetHostShaderClock({.scope = shader_clock.scope == Spirv::HostClockScope::Device
                                            ? Spirv::HostClockScope::Subgroup
                                            : Spirv::HostClockScope::Device,
                               .shift = shader_clock.shift});
  });
  host_differs("shader clock shift", [&] {
    Spirv::SetHostShaderClock({.scope = shader_clock.scope, .shift = shader_clock.shift == 0 ? 3 : 0});
  });
  Require(name, "host state restored", ShaderRecompiler::CodegenFingerprint() == base,
          "restoring the host state did not restore the fingerprint");

  // Every KYTY_* switch named in the codegen sources, by its raw value.
  const auto switches = ShaderRecompiler::CodegenSwitches();
  const auto listed = [&](std::string_view needle) {
    return std::ranges::any_of(switches, [&](const char *name) { return needle == name; });
  };
  for (const char *required :
       {"KYTY_MOVREL_RANGE", "KYTY_FAST_FMINMAX", "KYTY_FAST_PKRTZ", "KYTY_SINGLE_F2I_SATURATION",
        "KYTY_LOD_STATS_GATE", "KYTY_ROBUST_BUFFER_LOADS", "KYTY_MAD_MODE", "KYTY_INTERP_MODES",
        "KYTY_SAMPLE_OFFSETS", "KYTY_SAMPLE_LOD_CLAMP", "KYTY_HOST_FTZ_INPUTS",
        "KYTY_EXEC_SELECTS", "KYTY_PS_APPEND_LIVE_ELECTION", "KYTY_PS_LIVE_EXEC",
        "KYTY_LOOP_GUARD", "KYTY_LOOP_GUARD_SHADERS", "KYTY_SRT_VARIANT_READS",
        "KYTY_NATIVE_INDIRECT_MESH", "KYTY_LDS_WAITCNT_BARRIER", "KYTY_LOD_STATS_MODE",
        "KYTY_SRT_COMPILED_RECIPES", "KYTY_SRT_DECODED_ROOTS", "KYTY_SRT_ARITHMETIC_TAPES",
        "KYTY_REALTIME_CLOCK", "KYTY_DPP_SKIP_INACTIVE", "KYTY_LANE_REDUCTIONS",
        "KYTY_DISPATCHER_CAP", "KYTY_IR_LINEAR_USES", "KYTY_FOLD_LANE_MASKS"}) {
    Require(name, "codegen switch list", listed(required),
            std::string("the codegen switch list misses ") + required);
  }
  for (const char *excluded : {"KYTY_PROGRAM_CACHE", "KYTY_PROGRAM_CACHE_VERIFY",
                               "KYTY_PROGRAM_CACHE_PATH", "KYTY_PROGRAM_CACHE_SAVE"}) {
    Require(name, "codegen switch list", !listed(excluded),
            std::string("the program cache's own switch is in the fingerprint: ") + excluded);
  }
  for (const char *variable : switches) {
    std::vector<uint8_t> one;
    std::vector<uint8_t> zero;
    {
      ScopedEnv scope(variable, "1");
      one = ShaderRecompiler::CodegenFingerprint();
    }
    {
      ScopedEnv scope(variable, "0");
      zero = ShaderRecompiler::CodegenFingerprint();
    }
    Require(name, "codegen switch", one != base && zero != base && one != zero,
            std::string("the fingerprint does not record ") + variable);
  }
  Require(name, "environment restored", ShaderRecompiler::CodegenFingerprint() == base,
          "restoring the environment did not restore the fingerprint");
  std::printf("[cpu]     %-32s ok (%zu switches)\n", name, switches.size());
}

// ---------------------------------------------------------------------------------------------
// The cache file.

// One translated compute program: its source key and the bytes the cache stores for it.
struct StoredProgram {
  ProgramDiskCache::SourceKey key;
  std::vector<uint8_t> plan;
  std::vector<uint8_t> specialization;
  std::vector<uint8_t> info;
  std::vector<u32> spirv;
};

StoredProgram TranslateForStorage(const TestCase &test) {
  StoredProgram stored;
  // As CompileCase prepares a case without user data.
  auto user_data = MakeNativeUserData(nullptr);
  user_data[2] = static_cast<u32>(test.initial.size() * sizeof(u32));
  ShaderComputeInputInfo compute_info = test.compute_info;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.dump_ir = false;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  auto translated = ShaderRecompiler::TranslateProgram(test.code, options);
  const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
  Require(test.name, "program cache storage",
          ShaderRecompiler::IR::EncodeResourcePlan(plan, stored.plan), "plan encoding failed");
  ShaderRecompiler::IR::ResourceSnapshot resources;
  ShaderRecompiler::IR::ResourceSpecialization specialization;
  std::vector<u32> memory(1024, 0);
  const ShaderRecompiler::IR::SrtRuntime runtime{
      .user_data = options.user_data,
      .shader_base = reinterpret_cast<uint64_t>(test.code.data()),
      .read_memory = ReadTestMemory,
      .userdata = &memory,
  };
  Require(test.name, "program cache storage",
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization),
          "materialization failed");
  ShaderRecompiler::IR::EncodeSpecialization(specialization, stored.specialization);
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  stored.spirv = result.spirv;
  ShaderRecompiler::IR::EncodeCompiledShaderInfo(std::move(result.program).TakeCompiledInfo(),
                                                 stored.info);
  std::vector<u32> static_state;
  BuildStageStaticKey(compute_info, static_state);
  ProgramDiskCache::BuildSourceKey({.stage = static_cast<u32>(ShaderType::Compute),
                                    .hash = XXH3_64bits(test.code.data(), test.code.size() * 4),
                                    .user_data_count = 16,
                                    .code_size = static_cast<u32>(test.code.size()),
                                    .static_state = static_state,
                                    .wave_size = 64,
                                    .code = test.code},
                                   stored.key);
  return stored;
}

bool SameBytes(std::span<const uint8_t> a, std::span<const uint8_t> b) {
  return std::ranges::equal(a, b);
}

std::span<const uint8_t> WordBytes(const std::vector<u32> &words) {
  return {reinterpret_cast<const uint8_t *>(words.data()), words.size() * sizeof(u32)};
}

void CheckCacheFile() {
  constexpr const char *name = "ProgramCacheFile";
  const auto path = std::filesystem::current_path() / "program-cache-file-test.bin";
  RemoveTestFile(path);
  const std::vector<uint8_t> identity_a{'t', 'e', 's', 't', 'A', 1, 2, 3};
  const std::vector<uint8_t> identity_b{'t', 'e', 's', 't', 'B', 1, 2, 3};
  const auto settings = [&](const std::vector<uint8_t> &identity) {
    ProgramDiskCache::Settings result;
    result.path = path;
    result.identity = identity;
    result.periodic_saves = false;
    result.quiet = true;
    return result;
  };
  const auto first = TranslateForStorage(IntegerAddSubMul());
  const auto second = TranslateForStorage(BitwiseOps());
  ProgramDiskCache::SourceKey skipped;
  const std::vector<u32> skipped_code{1, 2, 3};
  ProgramDiskCache::BuildSourceKey({.stage = 1, .hash = 7, .code = skipped_code}, skipped);
  std::vector<uint8_t> other_specialization(first.specialization);
  other_specialization.push_back(0);

  // Write, then reload in a new instance.
  {
    ProgramDiskCache cache(settings(identity_a));
    Require(name, "empty", !cache.FindSource(first.key).has_value(),
            "a missing file produced a record");
    cache.AddSource(first.key, false, first.plan);
    cache.AddSource(first.key, false, first.plan); // duplicate: ignored
    cache.AddSource(second.key, false, second.plan);
    cache.AddSource(skipped, true, {});
    cache.AddPermutation(first.key.digest, 0, first.specialization, first.info, first.spirv, {});
    cache.AddPermutation(first.key.digest, 4, first.specialization, first.info, first.spirv,
                         first.spirv);
    cache.AddPermutation(second.key.digest, 0, second.specialization, second.info, second.spirv,
                         {});
    const auto stats = cache.GetStats();
    Require(name, "adds", stats.added_sources == 3 && stats.added_permutations == 3,
            "duplicate or missing additions");
    Require(name, "flush", cache.Flush(), "the save failed");
    auto temp = path;
    temp += ".tmp";
    Require(name, "atomic write", std::filesystem::exists(path) && !std::filesystem::exists(temp),
            "the save left no file or a temporary file");
  }
  const auto file = ReadFileBytes(path);
  const auto check_complete = [&](ProgramDiskCache &cache) {
    const auto source = cache.FindSource(first.key);
    Require(name, "reload source", source.has_value() && !source->skip_dispatch &&
                                       SameBytes(source->plan, first.plan),
            "the stored plan did not come back");
    const auto skip = cache.FindSource(skipped);
    Require(name, "reload skip", skip.has_value() && skip->skip_dispatch && skip->plan.empty(),
            "the skip-dispatch verdict did not come back");
    const auto permutation = cache.FindPermutation(first.key.digest, 0, first.specialization);
    Require(name, "reload permutation",
            permutation.has_value() && SameBytes(permutation->info, first.info) &&
                SameBytes(permutation->spirv, WordBytes(first.spirv)) &&
                permutation->spirv_plain.empty(),
            "the stored permutation did not come back");
    const auto plain = cache.FindPermutation(first.key.digest, 4, first.specialization);
    Require(name, "reload plain variant",
            plain.has_value() && SameBytes(plain->spirv_plain, WordBytes(first.spirv)),
            "the plain variant did not come back");
    Require(name, "exact permutation key",
            !cache.FindPermutation(first.key.digest, 8, first.specialization).has_value() &&
                !cache.FindPermutation(first.key.digest, 0, other_specialization).has_value() &&
                !cache.FindPermutation(second.key.digest, 4, second.specialization).has_value(),
            "a permutation matched another cursor or specialization");
    auto changed = first.key;
    changed.bytes.back() ^= 1u;
    Require(name, "exact source key", !cache.FindSource(changed).has_value(),
            "a source matched a key with another code word");
  };
  {
    ProgramDiskCache cache(settings(identity_a));
    check_complete(cache);
    const auto stats = cache.GetStats();
    Require(name, "reload", stats.file_found && !stats.header_rejected &&
                                stats.loaded_sources == 3 && stats.loaded_permutations == 3 &&
                                stats.rejected_records == 0,
            "the reloaded file did not hold every record");
  }

  // Another identity (codegen version, switches or device): nothing is used, and the next save
  // replaces the file.
  {
    ProgramDiskCache cache(settings(identity_b));
    const auto stats = cache.GetStats();
    Require(name, "identity", stats.header_rejected && !cache.FindSource(first.key).has_value(),
            "a file with another identity was used");
    cache.AddSource(second.key, false, second.plan);
    Require(name, "identity", cache.Flush(), "the replacing save failed");
  }
  {
    ProgramDiskCache cache(settings(identity_a));
    Require(name, "identity", cache.GetStats().header_rejected,
            "the replaced file still had the old identity");
  }
  {
    ProgramDiskCache cache(settings(identity_b));
    Require(name, "identity",
            cache.FindSource(second.key).has_value() && !cache.FindSource(first.key).has_value(),
            "the replacing save did not hold exactly the new records");
  }

  // Record boundaries of the complete file.
  std::vector<size_t> ends;
  size_t header_end = 0;
  {
    size_t offset = sizeof(ProgramDiskCache::FileMagic) + 8;
    uint64_t identity_size = 0;
    std::memcpy(&identity_size, file.data() + offset, sizeof(identity_size));
    offset += 8 + identity_size + 8;
    header_end = offset;
    while (offset + ProgramDiskCache::RecordHeaderBytes <= file.size()) {
      uint64_t size = 0;
      std::memcpy(&size, file.data() + offset + 8, sizeof(size));
      offset += ProgramDiskCache::RecordHeaderBytes + size;
      ends.push_back(offset);
    }
    Require(name, "layout", ends.size() == 6 && ends.back() == file.size(),
            "unexpected record layout");
  }
  const auto loaded_records = [&](ProgramDiskCache &cache) {
    const auto stats = cache.GetStats();
    return stats.loaded_sources + stats.loaded_permutations;
  };
  // Truncation anywhere: the complete records before the cut are kept, nothing after it, no
  // crash. Every cut inside the header, and around every record boundary; a stride elsewhere.
  std::vector<size_t> cuts;
  for (size_t cut = 0; cut <= header_end + 1; cut++) cuts.push_back(cut);
  for (const auto end : ends) {
    for (size_t delta = 0; delta < 3; delta++) {
      if (end >= delta) cuts.push_back(end - delta);
      if (end + delta <= file.size()) cuts.push_back(end + delta);
    }
  }
  for (size_t cut = header_end; cut < file.size(); cut += 97) cuts.push_back(cut);
  for (const auto cut : cuts) {
    WriteFileBytes(path, std::span(file).first(cut));
    ProgramDiskCache cache(settings(identity_a));
    const auto stats = cache.GetStats();
    const auto complete = static_cast<uint64_t>(std::ranges::count_if(
        ends, [&](size_t end) { return end <= cut; }));
    // A cut exactly between records leaves no partial record behind.
    const bool boundary = cut == header_end || std::ranges::find(ends, cut) != ends.end();
    if (cut < header_end) {
      Require(name, "truncated header", stats.header_rejected && loaded_records(cache) == 0,
              "a file cut inside its header was used");
    } else {
      Require(name, "truncated records",
              !stats.header_rejected && loaded_records(cache) == complete &&
                  stats.rejected_records == (boundary ? 0u : 1u),
              "a file cut at byte " + std::to_string(cut) +
                  " did not keep exactly its complete records");
    }
  }
  // A corrupted byte: in the header the whole file is rejected; in a record, that record and
  // every later one are dropped.
  for (size_t position = 0; position < file.size(); position += position < header_end ? 1 : 13) {
    auto damaged = file;
    damaged[position] ^= 0x5au;
    WriteFileBytes(path, damaged);
    ProgramDiskCache cache(settings(identity_a));
    const auto stats = cache.GetStats();
    if (position < header_end) {
      Require(name, "corrupt header", stats.header_rejected && loaded_records(cache) == 0,
              "a file with a damaged header byte " + std::to_string(position) + " was used");
    } else {
      const auto intact = static_cast<uint64_t>(std::ranges::count_if(
          ends, [&](size_t end) { return end <= position; }));
      Require(name, "corrupt record",
              !stats.header_rejected && loaded_records(cache) == intact &&
                  stats.rejected_records == 1,
              "a damaged byte " + std::to_string(position) +
                  " did not drop exactly its record and the ones after it");
    }
  }
  // Garbage and empty files.
  {
    std::vector<uint8_t> garbage(4096);
    for (size_t i = 0; i < garbage.size(); i++) garbage[i] = static_cast<uint8_t>(i * 131u + 7u);
    WriteFileBytes(path, garbage);
    ProgramDiskCache cache(settings(identity_a));
    Require(name, "garbage", cache.GetStats().header_rejected && loaded_records(cache) == 0,
            "a garbage file was used");
  }
  {
    WriteFileBytes(path, {});
    ProgramDiskCache cache(settings(identity_a));
    Require(name, "empty file", cache.GetStats().header_rejected && loaded_records(cache) == 0,
            "an empty file was used");
  }
  // A damaged file is repaired by the next save (only its intact records remain).
  {
    WriteFileBytes(path, std::span(file).first(ends[3] + 5));
    {
      ProgramDiskCache cache(settings(identity_a));
      Require(name, "repair", cache.Flush(), "the repairing save failed");
    }
    const auto repaired = ReadFileBytes(path);
    Require(name, "repair", repaired.size() == ends[3] &&
                                std::equal(repaired.begin(), repaired.end(), file.begin()),
            "the save did not rewrite exactly the intact records");
  }
  // Verification drops a record: it is not returned again, and the next save leaves it out.
  {
    WriteFileBytes(path, file);
    {
      ProgramDiskCache cache(settings(identity_a));
      const auto permutation = cache.FindPermutation(first.key.digest, 0, first.specialization);
      Require(name, "invalidate", permutation.has_value(), "the permutation was not found");
      cache.Invalidate(permutation->id);
      Require(name, "invalidate",
              !cache.FindPermutation(first.key.digest, 0, first.specialization).has_value(),
              "an invalidated record was returned");
      // A fresh record may take its place.
      cache.AddPermutation(first.key.digest, 0, first.specialization, first.info, first.spirv, {});
      Require(name, "invalidate",
              cache.FindPermutation(first.key.digest, 0, first.specialization).has_value(),
              "a fresh record could not replace an invalidated one");
      Require(name, "invalidate", cache.Flush(), "the save failed");
    }
    ProgramDiskCache cache(settings(identity_a));
    Require(name, "invalidate", loaded_records(cache) == 6, "the replaced record was not saved");
  }
  // Capacity: nothing is added beyond max_file_bytes.
  {
    RemoveTestFile(path);
    auto capped = settings(identity_a);
    // Room for the source record only (the record header, digest, sized key, verdict, sized
    // plan), not for the permutation.
    capped.max_file_bytes = ProgramDiskCache::RecordHeaderBytes + 16 + 4 + first.key.bytes.size() +
                           1 + 4 + first.plan.size() + 16;
    ProgramDiskCache cache(capped);
    cache.AddSource(first.key, false, first.plan);
    cache.AddPermutation(first.key.digest, 0, first.specialization, first.info, first.spirv, {});
    const auto stats = cache.GetStats();
    Require(name, "capacity",
            stats.over_capacity && stats.added_sources == 1 && stats.added_permutations == 0 &&
                !cache.FindPermutation(first.key.digest, 0, first.specialization).has_value(),
            "records were added beyond the file size cap");
  }
  RemoveTestFile(path);
  std::printf("[cpu]     %-32s ok (%zu truncations)\n", name, cuts.size());
}

void RunCpu() {
  CheckSourceKeys();
  CheckFingerprint();
  CheckCacheFile();
}

// ---------------------------------------------------------------------------------------------
// GPU: programs reloaded from the file render what translated ones render.

// Compute shaders storing global id + k (k = 1, 2, 3): distinct sources, one output each.
std::vector<u32> DispatchShader(u32 k) {
  std::vector<u32> code;
  AppendVop3(&code, 0x346u, 1, 4, InlineU32(2), Vgpr(0)); // v1 = (tgid << 2) + tid
  code.push_back(EncodeVop2(0x25u, 2, InlineU32(k), 1));  // v2 = k + v1
  code.push_back(EncodeVop2(0x1au, 1, InlineU32(2), 1));  // v1 = v1 << 2
  code.push_back(EncodeMubuf0(0x1cu));
  code.push_back(EncodeMubuf1(2, 0, 1));
  AppendEnd(&code);
  return code;
}

std::vector<std::vector<u32>> RunDispatches(VulkanHarness &harness, const char *name,
                                            const std::vector<std::vector<u32>> &shaders) {
  constexpr uintptr_t base = 0x0000000208400000ull;
  constexpr uint64_t allocation_size = 0x10000;
  constexpr u32 groups = 4;
  constexpr u32 threads = 4;
  constexpr u32 dwords = groups * threads;
  for (const auto &code : shaders) {
    ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
                      {.type = Prospero::ShaderBinaryType::kCs,
                       .code_size_bytes = static_cast<uint32_t>(code.size() * sizeof(u32))});
  }
  int64_t direct_offset = -1;
  Require(name, "direct allocation",
          Libs::LibKernel::Memory::KernelAllocateDirectMemory(
              0, Libs::LibKernel::Memory::KernelGetDirectMemorySize(), allocation_size,
              allocation_size, 0, &direct_offset) == 0,
          "output allocation failed");
  void *mapped = reinterpret_cast<void *>(base);
  Require(name, "direct mapping",
          Libs::LibKernel::Memory::KernelMapDirectMemory(&mapped, allocation_size, 0x3, 0x10,
                                                         direct_offset, allocation_size) == 0 &&
              mapped == reinterpret_cast<void *>(base),
          "output mapping failed");
  std::memset(mapped, 0xa5, allocation_size);
  std::vector<std::vector<u32>> outputs(shaders.size());
  {
    const auto context_owner = harness.MakeRenderContext();
    auto &context = *context_owner;
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      auto &scheduler = context.GetCommandScheduler();
      auto &cache = context.GetBufferCache();
      auto &registers = processor.GetShCtx();
      context.MapMemory(base, allocation_size);
      for (size_t i = 0; i < shaders.size(); i++) {
        const auto output = base + i * 0x1000u;
        ShaderBufferResource descriptor{};
        descriptor.UpdateAddress48(output);
        descriptor.fields[2] = dwords * sizeof(u32);
        descriptor.fields[3] =
            DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
        registers.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(shaders[i].data()),
                               .num_thread_x = threads, .num_thread_y = 1, .num_thread_z = 1,
                               .wave_size = 64, .user_sgpr = 4, .tgid_x_en = true});
        for (u32 word = 0; word < 4; word++) {
          registers.SetCsUserSgpr(word, descriptor.fields[word], HW::UserSgprType::Unknown);
        }
        processor.DispatchDirect(groups, 1, 1, 0x41u);
        cache.ReadMemory(output, dwords * sizeof(u32));
        outputs[i].resize(dwords);
        Require(name, "output readback",
                LibKernel::Memory::TryReadBacking(output, outputs[i].data(), dwords * sizeof(u32)),
                "the output could not be read back");
      }
      context.UnmapMemory(base, allocation_size);
      scheduler.Finish();
    });
    // KYTY_PROGRAM_CACHE_VERIFY=background: the checks of this context's reloads are done.
    context.GetPipelineCache().WaitProgramChecks();
    LibKernel::Memory::InstallGpuResources(nullptr);
    context.ShutdownGpu();
  } // The pipeline cache goes with the context: its program cache saves the file.
  Require(name, "unmap direct backing",
          Libs::LibKernel::Memory::KernelMunmap(base, allocation_size) == 0,
          "output mapping release failed");
  Require(name, "release direct backing",
          Libs::LibKernel::Memory::KernelReleaseDirectMemory(direct_offset, allocation_size) == 0,
          "output allocation release failed");
  return outputs;
}

void CheckReloadedPrograms(VulkanHarness &harness) {
  constexpr const char *name = "ProgramCacheReload";
  const char *verify_mode = std::getenv("KYTY_PROGRAM_CACHE_VERIFY");
  const bool background = verify_mode != nullptr && std::strcmp(verify_mode, "background") == 0;
  // Verify modes that translate on the command processor (1, exit).
  const bool verify = verify_mode != nullptr && *verify_mode != '\0' &&
                      std::strcmp(verify_mode, "0") != 0 && !background;
  const auto path = std::filesystem::current_path() /
                    (background ? "program-cache-gpu-background-test.bin"
                     : verify   ? "program-cache-gpu-verify-test.bin"
                                : "program-cache-gpu-test.bin");
  RemoveTestFile(path);
  const auto path_text = path.string();
  ScopedEnv enable("KYTY_PROGRAM_CACHE", "1");
  ScopedEnv file("KYTY_PROGRAM_CACHE_PATH", path_text.c_str());
  ScopedEnv saves("KYTY_PROGRAM_CACHE_SAVE", "0");
  const std::vector<std::vector<u32>> shaders{DispatchShader(1), DispatchShader(2),
                                              DispatchShader(3)};

  // Cold: translated, emitted and stored.
  const auto before = PipelineCache::Totals();
  const auto cold = RunDispatches(harness, name, shaders);
  const auto after_cold = PipelineCache::Totals();
  Require(name, "cold run",
          after_cold.permutation_hits == before.permutation_hits &&
              after_cold.translations >= before.translations + shaders.size() &&
              std::filesystem::exists(path),
          "the cold run reloaded programs or saved no file");
  for (size_t i = 0; i < shaders.size(); i++) {
    for (u32 id = 0; id < cold[i].size(); id++) {
      Require(name, "cold output", cold[i][id] == id + static_cast<u32>(i) + 1u,
              "shader " + std::to_string(i) + " word " + std::to_string(id) + " is " +
                  Hex(cold[i][id]));
    }
  }

  // Warm: a new context reloads every program instead of translating it (verify mode translates
  // too, and compares).
  const auto warm = RunDispatches(harness, name, shaders);
  const auto after_warm = PipelineCache::Totals();
  Require(name, "warm run",
          after_warm.source_hits - after_cold.source_hits == shaders.size() &&
              after_warm.permutation_hits - after_cold.permutation_hits == shaders.size() &&
              (verify ? after_warm.translations - after_cold.translations == shaders.size()
                      : after_warm.translations == after_cold.translations),
          "the warm run did not reload every program (or translated without verify mode)");
  Require(name, "reloaded output", warm == cold,
          "a reloaded program wrote other values than the translated one");
  if (verify || background) {
    Require(name, "verify",
            after_warm.verify_checks - after_cold.verify_checks >= 2 * shaders.size() &&
                after_warm.verify_mismatches == before.verify_mismatches,
            "verify mode did not compare every reloaded program, or found a difference");
  }

  // Graphics: guest vertex and pixel programs drawn through GetGraphicsPrograms (the
  // rasterization check asserts its rendered pixels), cold and then reloaded.
  const auto before_draw = PipelineCache::Totals();
  harness.CheckRasterization(false);
  const auto after_draw_cold = PipelineCache::Totals();
  harness.CheckRasterization(false);
  const auto after_draw_warm = PipelineCache::Totals();
  const auto cold_programs = after_draw_cold.programs - before_draw.programs;
  Require(name, "reloaded draws",
          cold_programs != 0 &&
              after_draw_warm.permutation_hits - after_draw_cold.permutation_hits ==
                  after_draw_warm.programs - after_draw_cold.programs &&
              (verify || after_draw_warm.translations == after_draw_cold.translations),
          "the second rasterization check did not reload every graphics program");
  if (verify || background) {
    Require(name, "verify", after_draw_warm.verify_mismatches == before.verify_mismatches,
            "verify mode found a reloaded graphics program different");
  }
  RemoveTestFile(path);
  std::printf("[gpu]     %-32s ok (%" PRIu64 " graphics programs reloaded%s)\n", name,
              after_draw_warm.permutation_hits - after_draw_cold.permutation_hits,
              background ? ", checked in the background" : verify ? ", verified" : "");
}

// KYTY_SHADER_PRECOMPILE (pipeline/shaderPrecompile.h): a run journals the inputs of its
// permutations; the next run (start-up waits for the replay, KYTY_SHADER_PRECOMPILE_WAIT=1)
// finds every program published before its first dispatch or draw, without the persistent program
// cache (every program is translated and emitted by the replay), and renders the same.
void CheckShaderPrecompileReplay(VulkanHarness &harness) {
  constexpr const char *name = "ShaderPrecompileReplay";
  const auto path = std::filesystem::current_path() / "shader-precompile-gpu-test.bin";
  RemoveTestFile(path);
  const auto path_text = path.string();
  ScopedEnv enable("KYTY_SHADER_PRECOMPILE", "1");
  ScopedEnv file("KYTY_SHADER_PRECOMPILE_PATH", path_text.c_str());
  ScopedEnv wait("KYTY_SHADER_PRECOMPILE_WAIT", "1");
  ScopedEnv no_disk("KYTY_PROGRAM_CACHE", "0");
  const std::vector<std::vector<u32>> shaders{DispatchShader(4), DispatchShader(5),
                                              DispatchShader(6)};

  // Cold: compiled by the dispatches and journaled; nothing to replay.
  const auto before = PipelineCache::Totals();
  const auto cold = RunDispatches(harness, name, shaders);
  const auto after_cold = PipelineCache::Totals();
  Require(name, "cold run",
          after_cold.programs - before.programs == shaders.size() &&
              after_cold.replayed == before.replayed && std::filesystem::exists(path),
          "the cold run replayed programs or wrote no journal");

  // Warm: the new context replays the journal at start-up; the dispatches compile nothing.
  const auto warm = RunDispatches(harness, name, shaders);
  const auto after_warm = PipelineCache::Totals();
  Require(name, "warm run",
          after_warm.replayed - after_cold.replayed == shaders.size() &&
              after_warm.programs - after_cold.programs == shaders.size() &&
              after_warm.translations - after_cold.translations == shaders.size(),
          "the replay did not publish exactly the journaled programs before the dispatches");
  Require(name, "replayed output", warm == cold,
          "a replayed program wrote other values than the one the dispatch compiled");

  // Graphics: guest vertex and pixel programs (the stage input infos and render-target export
  // mapping are part of their keys), replayed in a new context.
  RemoveTestFile(path);
  const auto before_draw = PipelineCache::Totals();
  harness.CheckRasterization(false);
  const auto after_draw_cold = PipelineCache::Totals();
  harness.CheckRasterization(false);
  const auto after_draw_warm = PipelineCache::Totals();
  const auto cold_programs = after_draw_cold.programs - before_draw.programs;
  Require(name, "replayed draws",
          cold_programs != 0 &&
              after_draw_warm.replayed - after_draw_cold.replayed == cold_programs &&
              after_draw_warm.programs - after_draw_cold.programs == cold_programs,
          "the second rasterization check did not find every graphics program replayed");
  RemoveTestFile(path);
  std::printf("[gpu]     %-32s ok (%zu dispatch and %" PRIu64 " graphics programs replayed)\n", name,
              shaders.size(), cold_programs);
}

void RunGpu(VulkanHarness &harness) {
  CheckReloadedPrograms(harness);
  CheckShaderPrecompileReplay(harness);
}

// --program-cache-tamper-only, with KYTY_PROGRAM_CACHE_VERIFY=1 or background: a stored
// permutation whose SPIR-V differs from what translation produces (the generator word of its
// header, which leaves the module valid, with the record checksum fixed up) must be caught.
// Synchronous verify uses the fresh module and drops the record; background verify counts the
// difference and drops the record. Either way the next run translates it again, stores the fresh
// record, and a later run reloads it.
void CheckVerifyCatchesDifferences(VulkanHarness &harness) {
  constexpr const char *name = "ProgramCacheVerifyTamper";
  const char *verify_mode = std::getenv("KYTY_PROGRAM_CACHE_VERIFY");
  const bool background = verify_mode != nullptr && std::strcmp(verify_mode, "background") == 0;
  const bool sync = verify_mode != nullptr && std::strcmp(verify_mode, "1") == 0;
  Require(name, "mode", sync || background,
          "run with KYTY_PROGRAM_CACHE_VERIFY=1 or KYTY_PROGRAM_CACHE_VERIFY=background");
  const auto path = std::filesystem::current_path() /
                    (background ? "program-cache-tamper-background-test.bin"
                                : "program-cache-tamper-test.bin");
  RemoveTestFile(path);
  const auto path_text = path.string();
  ScopedEnv enable("KYTY_PROGRAM_CACHE", "1");
  ScopedEnv file("KYTY_PROGRAM_CACHE_PATH", path_text.c_str());
  ScopedEnv saves("KYTY_PROGRAM_CACHE_SAVE", "0");
  const std::vector<std::vector<u32>> shaders{DispatchShader(5)};
  const auto expected = RunDispatches(harness, name, shaders);

  // Change the SPIR-V generator word of the stored permutation and fix up its checksum.
  auto bytes = ReadFileBytes(path);
  size_t offset = sizeof(ProgramDiskCache::FileMagic) + 8;
  uint64_t identity_size = 0;
  std::memcpy(&identity_size, bytes.data() + offset, sizeof(identity_size));
  offset += 8 + identity_size + 8;
  bool tampered = false;
  while (offset + ProgramDiskCache::RecordHeaderBytes <= bytes.size()) {
    uint32_t kind = 0;
    uint64_t size = 0;
    std::memcpy(&kind, bytes.data() + offset + 4, sizeof(kind));
    std::memcpy(&size, bytes.data() + offset + 8, sizeof(size));
    auto *payload = bytes.data() + offset + ProgramDiskCache::RecordHeaderBytes;
    if (kind == ProgramDiskCache::RecordPermutation) {
      size_t at = 16 + 4; // digest, cursor
      for (int sized = 0; sized < 2; sized++) { // specialization, metadata
        uint32_t length = 0;
        std::memcpy(&length, payload + at, sizeof(length));
        at += 4 + length;
      }
      const size_t generator = at + 4 + 2 * sizeof(u32); // word count, then words 0-1
      payload[generator] ^= 0x5au;
      const uint64_t checksum = XXH3_64bits(payload, size);
      std::memcpy(bytes.data() + offset + 16, &checksum, sizeof(checksum));
      tampered = true;
    }
    offset += ProgramDiskCache::RecordHeaderBytes + size;
  }
  Require(name, "tamper", tampered, "no stored permutation found");
  WriteFileBytes(path, bytes);

  const auto before = PipelineCache::Totals();
  const auto verified = RunDispatches(harness, name, shaders);
  const auto after = PipelineCache::Totals();
  Require(name, "difference found",
          after.verify_mismatches - before.verify_mismatches == 1 &&
              after.permutation_hits - before.permutation_hits == (background ? 1u : 0u),
          "verify mode did not report the changed SPIR-V exactly once");
  Require(name, "output", verified == expected,
          "the run that verified the changed record computed other values");
  // The dropped record was replaced by a fresh one (sync) or leaves the next run to translate
  // and store it (background); then it is reloaded without a difference.
  RunDispatches(harness, name, shaders);
  const auto healed = PipelineCache::Totals();
  RunDispatches(harness, name, shaders);
  const auto final_totals = PipelineCache::Totals();
  Require(name, "healed",
          final_totals.permutation_hits - healed.permutation_hits == 1 &&
              final_totals.verify_mismatches == after.verify_mismatches,
          "the replaced record was not reloaded cleanly");
  RemoveTestFile(path);
  std::printf("[gpu]     %-32s ok (%s)\n", name, background ? "background" : "synchronous");
}

// CPU only (--corpus-program-cache <file.bin>): a guest shader dump from
// Profiling/analysis/guest-shader-diagnostic*/original (CS or PS, by file name, with null
// resources as in CompileCorpusShader) through what the persistent cache stores and reloads:
// plan, specialization and metadata round trips, the decoded plan's materialization, and a
// second translation and emission giving the same bytes. Prints one CSV line with the sizes
// and the time of translating, emitting and decoding. Run it once per file.
int CorpusProgramCache(const char *path) {
  using Clock = std::chrono::steady_clock;
  const auto us = [](Clock::time_point a, Clock::time_point b) {
    return static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
  };
  std::vector<u32> code;
  if (std::FILE *file = std::fopen(path, "rb"); file != nullptr) {
    u32 word = 0;
    while (std::fread(&word, sizeof(word), 1, file) == 1) {
      code.push_back(word);
    }
    std::fclose(file);
  }
  if (code.empty()) {
    std::printf("%s,error,open\n", path);
    return 1;
  }
  const std::string name = path;
  ShaderRecompiler::CompileOptions options;
  ShaderComputeInputInfo compute{};
  ShaderPixelInputInfo pixel{};
  if (name.find("_cs_") != std::string::npos) {
    options.stage = ShaderType::Compute;
    compute.threads_num[0] = 64;
    compute.threads_num[1] = 1;
    compute.threads_num[2] = 1;
    compute.thread_ids_num = 3;
    compute.lds_size_dwords = 16384;
    compute.host_subgroup_size = 32;
    compute.wave_size = 64;
    options.input_info.compute = &compute;
    options.wave_size = 64;
  } else if (name.find("_ps_") != std::string::npos) {
    options.stage = ShaderType::Pixel;
    pixel.input_num = 32;
    for (u32 i = 0; i < 32; i++) {
      pixel.interpolator_settings[i] = i;
    }
    pixel.ps_perspective_center_vgpr = 0;
    pixel.ps_system_input_base = 2;
    for (auto &mode : pixel.target_output_mode) {
      mode = 4;
    }
    options.input_info.pixel = &pixel;
    options.plain_mip_stats_variant = true;
  } else {
    std::printf("%s,skip,stage\n", path);
    return 0;
  }
  options.dump_ir = false;
  const auto user_data = MakeNativeUserData(nullptr);
  options.user_data = user_data;
  ShaderRecompiler::Spirv::SetHostBufferRobustness({.storage_dword_loads_return_zero = true});
  ShaderRecompiler::Spirv::SetHostFloatControls(
      {.denorm_flush_f32 = false, .denorm_preserve_f16 = true, .denorm_preserve_f64 = false});
  ShaderRecompiler::Spirv::SetHostImageFeatures({.min_lod = true});
  const auto zero_memory = [](void *, uint64_t, std::span<u32> values) {
    std::fill(values.begin(), values.end(), 0u);
    return true;
  };
  const ShaderRecompiler::IR::SrtRuntime runtime{
      .user_data = options.user_data,
      .shader_base = reinterpret_cast<uint64_t>(code.data()),
      .read_memory = zero_memory,
      .read_specialization_memory = zero_memory,
  };
  using namespace ShaderRecompiler::IR;
  const auto t0 = Clock::now();
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  const auto t1 = Clock::now();
  if (translated.skip_dispatch) {
    std::printf("%s,skip,dispatch\n", path);
    return 0;
  }
  const auto plan = ExtractResourcePlan(translated.program);
  std::vector<uint8_t> plan_bytes;
  const auto t2 = Clock::now();
  Require(path, "plan encoding", EncodeResourcePlan(plan, plan_bytes), "encoding failed");
  const auto t3 = Clock::now();
  ResourcePlan decoded;
  Require(path, "plan decoding", DecodeResourcePlan(plan_bytes, decoded), "decoding failed");
  const auto t4 = Clock::now();
  std::vector<uint8_t> again;
  Require(path, "plan round trip", EncodeResourcePlan(decoded, again) && again == plan_bytes,
          "encoding the decoded plan gave other bytes");
  ResourceSnapshot resources;
  ResourceSpecialization specialization;
  ResourceSnapshot reloaded_resources;
  ResourceSpecialization reloaded_specialization;
  const bool materialized = MaterializeResources(plan, runtime, resources, specialization);
  const bool reloaded =
      MaterializeResources(decoded, runtime, reloaded_resources, reloaded_specialization);
  Require(path, "materialization", materialized == reloaded,
          "the decoded plan materialized where the original did not, or the reverse");
  if (!materialized) {
    // Null memory cannot satisfy this program's resources (as in CompileCorpusShader); the
    // round trips above still ran.
    std::printf("%s,skip,materialize\n", path);
    return 0;
  }
  const char *difference = reloaded_resources.buffers != resources.buffers       ? "buffers"
                           : reloaded_resources.images != resources.images       ? "images"
                           : reloaded_resources.samplers != resources.samplers   ? "samplers"
                           : reloaded_resources.flattened_srt != resources.flattened_srt
                               ? "flattened SRT"
                           : reloaded_resources.user_data != resources.user_data ? "user data"
                           : !(reloaded_resources.uniform_fill == resources.uniform_fill)
                               ? "uniform fill"
                           : !(reloaded_specialization == specialization) ? "specialization"
                                                                          : nullptr;
  Require(path, "materialization", difference == nullptr,
          std::string("the decoded plan materialized other ") +
              (difference != nullptr ? difference : ""));
  std::vector<uint8_t> specialization_bytes;
  EncodeSpecialization(specialization, specialization_bytes);
  ResourceSpecialization specialization_back;
  Require(path, "specialization round trip",
          DecodeSpecialization(specialization_bytes, specialization_back) &&
              specialization_back == specialization,
          "the specialization did not come back");
  const auto t5 = Clock::now();
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  const auto t6 = Clock::now();
  const auto info = std::move(result.program).TakeCompiledInfo();
  std::vector<uint8_t> info_bytes;
  EncodeCompiledShaderInfo(info, info_bytes);
  CompiledShaderInfo info_back;
  const auto t7 = Clock::now();
  const bool info_decoded = DecodeCompiledShaderInfo(info_bytes, info_back);
  const auto t8 = Clock::now();
  Require(path, "metadata round trip", info_decoded && info_back == info,
          "the metadata did not come back");
  // What verify mode relies on: translating and emitting again gives the same bytes.
  auto second = ShaderRecompiler::TranslateProgram(code, options);
  std::vector<uint8_t> second_plan;
  Require(path, "determinism",
          EncodeResourcePlan(ExtractResourcePlan(second.program), second_plan) &&
              second_plan == plan_bytes,
          "a second translation gave another plan");
  auto second_result = ShaderRecompiler::CompileProgram(std::move(second), options, specialization);
  Require(path, "determinism",
          second_result.spirv == result.spirv && second_result.spirv_plain == result.spirv_plain &&
              std::move(second_result.program).TakeCompiledInfo() == info,
          "a second emission gave other SPIR-V or metadata");
  std::printf("%s,ok,%zu,%zu,%zu,%zu,%zu,%lld,%lld,%lld,%lld,%lld\n", path, code.size(),
              plan_bytes.size(), info_bytes.size(), result.spirv.size(), result.spirv_plain.size(),
              us(t0, t1), us(t5, t6), us(t2, t3), us(t3, t4), us(t7, t8));
  return 0;
}

} // namespace ProgramCacheTests
