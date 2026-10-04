#include "graphics/host_gpu/renderer/drawPrep/bindingPlan.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/lodStats.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/pipeline/textureBindingMemo.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace Libs::Graphics::DrawPrep {

namespace {

const char* EnvValue(const char* name) {
	const auto* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? value : nullptr;
}

BindingTotals g_binding_totals;

using ShaderRecompiler::IR::CompiledShaderInfo;
using ShaderRecompiler::IR::DescriptorValue;
using ShaderRecompiler::IR::ResourceSnapshot;

// DecodeNativeDescriptor without its exit: false when the value is too short.
template <typename T>
bool TryDecode(const DescriptorValue& value, T& descriptor) {
	if (value.dword_count < sizeof(T) / sizeof(uint32_t)) {
		return false;
	}
	descriptor = DecodeNativeDescriptor<T>(value);
	return true;
}

// FindBuffers' ranges: decoded and clamped as it does, while ClampRangeSize leaves every size
// unchanged (otherwise the command processor clamps: it logs the clamp or stops the emulator).
bool PlanRanges(const CompiledShaderInfo& program, const ResourceSnapshot& resources,
                StagePlan& stage) {
	const auto count = program.info.buffers.size();
	if (resources.buffers.size() < count) {
		return false;
	}
	stage.ranges.resize(count);
	for (size_t i = 0; i < count; i++) {
		ShaderBufferResource descriptor {};
		if (!TryDecode(resources.buffers[i], descriptor)) {
			return false;
		}
		auto& range              = stage.ranges[i];
		const auto address        = descriptor.Base48();
		const auto requested_size = descriptor.GetSize();
		range.out_of_bounds_mode  = BufferOutOfBoundsCombination(descriptor);
		if (address == 0 || requested_size == 0) {
			range.address = 0;
			range.size    = 0;
			continue;
		}
		if (LibKernel::Memory::ClampRangeSizeQuiet(address, requested_size) != requested_size) {
			return false;
		}
		range.address = address;
		range.size    = requested_size;
	}
	return true;
}

// PrepareBindings' shader data after RebindBuffers' reset and fields (offsets not packed yet).
bool PlanShaderData(const CompiledShaderInfo& program, const ResourceSnapshot& resources,
                    std::vector<uint32_t>& shader_data, bool& mip_stats_active) {
	const auto& layout = program.bindings;
	const auto  dwords = layout.ShaderDataDwords();
	for (const auto reg: layout.user_data_registers) {
		if (reg < program.user_data_base || reg - program.user_data_base >= resources.user_data.size()) {
			return false;
		}
	}
	if (layout.memory_offset_dword > dwords || layout.mip_stats_count > resources.images.size() ||
	    (layout.mip_stats_count != 0 &&
	     layout.MipStatsOffsetDword() + layout.mip_stats_count > dwords)) {
		return false;
	}
	shader_data.clear();
	shader_data.reserve(dwords);
	AppendUserShaderData(program, resources, shader_data);
	shader_data.resize(dwords);
	std::fill(shader_data.begin() + layout.memory_offset_dword, shader_data.end(), 0);
	mip_stats_active = WriteMipStatsFields(program, resources, shader_data);
	return true;
}

// Sampler handles this thread found, per cache instance (never shared; the cache never evicts).
struct RememberedSampler {
	uint64_t                owner = 0;
	std::array<uint32_t, 4> fields {};
	bool                    integer_border = false;
	vk::Sampler             sampler        = nullptr;
};
thread_local std::array<RememberedSampler, 256> t_samplers {};

vk::Sampler FindPlanSampler(SamplerCache& cache, const ShaderSamplerResource& descriptor,
                            bool integer_border) {
	static_assert(sizeof(descriptor.fields) == sizeof(RememberedSampler::fields));
	const auto slot  = (descriptor.fields[0] * 0x9e3779b1u ^ descriptor.fields[1] * 0x85ebca6bu ^
                       descriptor.fields[2] * 0xc2b2ae35u ^ descriptor.fields[3] ^
                       (integer_border ? 0x27d4eb2fu : 0u)) >>
	                  24u;
	auto&      entry = t_samplers[slot % t_samplers.size()];
	const auto owner = cache.InstanceId();
	if (entry.owner == owner && entry.sampler != nullptr && entry.integer_border == integer_border &&
	    std::memcmp(entry.fields.data(), descriptor.fields, sizeof(descriptor.fields)) == 0) {
		return entry.sampler;
	}
	const auto sampler = cache.FindSampler(descriptor, integer_border);
	if (sampler != nullptr) {
		entry.owner = owner;
		std::memcpy(entry.fields.data(), descriptor.fields, sizeof(descriptor.fields));
		entry.integer_border = integer_border;
		entry.sampler        = sampler;
	}
	return sampler;
}

// PrepareBindings' samplers: the handle of every sampler created already (null: the command
// processor resolves it, creating it). False when a descriptor is too short.
bool PlanSamplers(SamplerCache& cache, const CompiledShaderInfo& program,
                  const ResourceSnapshot& resources, StagePlan& stage) {
	const auto count = program.info.samplers.size();
	if (resources.samplers.size() < count) {
		return false;
	}
	stage.samplers.assign(count, vk::Sampler {});
	uint32_t absent = 0;
	for (size_t i = 0; i < count; i++) {
		const auto& value = resources.samplers[i];
		if (value.dword_count < sizeof(ShaderSamplerResource) / sizeof(uint32_t)) {
			return false;
		}
		stage.samplers[i] =
		    FindPlanSampler(cache, NativeSamplerDescriptor(program, static_cast<uint32_t>(i), value),
		                    program.info.samplers[i].integer_border);
		absent += stage.samplers[i] == nullptr ? 1u : 0u;
	}
	if (absent != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingAbstainSamplers, absent);
	}
	return true;
}

// ResolveTexture's memo hashes; with `memo` (P4b-2) also the tags of the entries that hold the
// keys now (TextureBindingMemo::FindHint), valid in `stage.texture_tags_valid`.
bool PlanTextureHashes(const CompiledShaderInfo& program, const ResourceSnapshot& resources,
                       const TextureBindingMemo* memo, StagePlan& stage) {
	const auto count = program.info.images.size();
	if (resources.images.size() < count) {
		return false;
	}
	stage.texture_hashes.resize(count);
	if (memo != nullptr) {
		stage.texture_tags.assign(count, 0);
	}
	uint32_t hints = 0;
	for (size_t i = 0; i < count; i++) {
		ShaderTextureResource descriptor {};
		if (!TryDecode(resources.images[i], descriptor)) {
			return false;
		}
		const auto key          = TextureBindingMemo::MakeKey(program.info.images[i], descriptor.fields);
		stage.texture_hashes[i] = TextureBindingMemo::Hash(key);
		uint64_t tag            = 0;
		if (memo != nullptr && memo->FindHint(key, stage.texture_hashes[i], tag)) {
			stage.texture_tags[i] = tag;
			hints++;
		}
	}
	stage.texture_tags_valid = memo != nullptr;
	if (hints != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingTextureHints, hints);
	}
	return true;
}

// HasShaderBufferWrites where it would not stop the emulator.
bool PlanBufferWrites(const CompiledShaderInfo& program, const ResourceSnapshot& resources,
                      bool& writes) {
	if (resources.buffers.size() != program.info.buffers.size()) {
		return false;
	}
	for (size_t i = 0; i < program.info.buffers.size(); i++) {
		if (program.info.buffers[i].written && resources.buffers[i].dword_count < 4) {
			return false;
		}
	}
	const ShaderStageRuntime runtime {&program, &resources};
	writes = HasShaderBufferWrites(runtime);
	return true;
}

// The attachments PrepareDrawRenderState resolves for the draw, as the pipeline key sees them,
// assuming every target's image is found (the commit compares the resolved ones).
bool PredictTargets(const GraphicContext& graphics, const HW::Context& ctx,
                    const PreparedDraw& prepared, PipelineCache::PipelineTargets& targets) {
	targets = {};
	uint32_t mrt_mask = 0;
	if (prepared.pixel_active) {
		for (const auto& output: prepared.pixel_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				if (output.index >= 32u) {
					return false;
				}
				mrt_mask |= 1u << output.index;
			}
		}
	}
	if (SkipInactivePixelShadersEnabled()) {
		mrt_mask &= DrawColorOutputFilter(ctx); // as PrepareDrawRenderState
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) == 0) {
			continue;
		}
		const auto& rt = ctx.GetRenderTarget(slot);
		if (rt.base.addr == 0 || render_target_mask_slot(ctx.GetRenderTargetMask(), slot) == 0) {
			continue; // no target in this slot
		}
		const auto samples = render_sample_count(rt.attrib.num_fragments);
		if (samples == 0 || rt.attrib.num_samples != rt.attrib.num_fragments) {
			return false; // the resolution stops the emulator
		}
		// The preparation's TargetExportMapping resolved this slot's format already.
		const auto format = TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
		                                                 rt.info.channel_order);
		targets.colors[targets.color_count++] = {.slot           = slot,
		                                         .format         = format.format,
		                                         .samples        = samples,
		                                         .export_mapping = format.export_mapping};
	}
	DepthTargetPrediction depth;
	if (!PredictRenderDepthTarget(graphics, ctx, depth)) {
		return false;
	}
	targets.with_depth               = depth.with_depth;
	targets.depth_format             = depth.format;
	targets.depth_samples            = depth.samples;
	targets.depth_bounds_test_enable = depth.bounds_test_enable;
	targets.depth_min_bounds         = depth.min_bounds;
	targets.depth_max_bounds         = depth.max_bounds;
	return true;
}

uint32_t IndexElementSize(uint32_t index_type_and_size) {
	switch (static_cast<Prospero::IndexType>(index_type_and_size)) {
		case Prospero::IndexType::kIndex16: return 2;
		case Prospero::IndexType::kIndex32: return 4;
		case Prospero::IndexType::kIndex8: return 1;
		default: break;
	}
	return 0;
}

void PlanPipeline(const BindingPlanContext& context, const RegisterSnapshot& registers,
                  const DrawIndexArgs* index_args, const PreparedDraw& prepared,
	              BindingPlan& plan, bool prefetch_only = false) {
	using E = Profiler::FrameEvent;
	// Rare (a pipeline not created yet, the map lock busy): shared totals are fine here.
	const auto abstain = [prefetch_only](E event) {
		if (prefetch_only) return;
		Profiler::CountFrameEvent(event);
		(event == E::DrawPrepBindingAbstainPipelineBusy ? g_binding_totals.pipeline_busy
		                                                : g_binding_totals.pipeline_abstains)
		    .fetch_add(1, std::memory_order_relaxed);
	};
	const auto& ctx   = registers.context;
	const auto& ucfg  = registers.user_config;
	if (!DrawTopology(ucfg, plan.topology)) {
		return abstain(E::DrawPrepBindingAbstainPipeline);
	}
	plan.primitive_restart = false;
	if (index_args != nullptr) {
		const auto element_size = IndexElementSize(index_args->index_type_and_size);
		if (element_size == 0) {
			return abstain(E::DrawPrepBindingAbstainPipeline);
		}
		const bool allow_custom = MeshPrimitiveRestartEnabled() &&
		                          prepared.vertex_info.stage.program->stage == ShaderType::Mesh;
		switch (DecidePrimitiveRestart(ucfg, ctx, element_size, allow_custom)) {
			case RestartDecision::Disabled: break;
			case RestartDecision::Enabled: plan.primitive_restart = true; break;
			case RestartDecision::Scan:
			case RestartDecision::Unsupported: return abstain(E::DrawPrepBindingAbstainPipeline);
		}
	}
	if (!PredictTargets(*context.graphics, ctx, prepared, plan.targets)) {
		return abstain(E::DrawPrepBindingAbstainPipeline);
	}
	auto programs = prepared.programs;
	if (plan.plain_pixel) {
		programs.pixel = PipelineCache::PlainPixelProgram(prepared.pixel_prep);
	}
	if (prefetch_only) {
		context.pipelines->PrefetchGraphicsPipeline(plan.targets, ctx, ucfg, prepared.vertex_info,
		    prepared.pixel_active ? &prepared.pixel_info : nullptr, plan.topology,
		    plan.primitive_restart, programs);
		return;
	}
	const PipelineCache::Pipeline* pipeline   = nullptr;
	uint64_t                       generation = 0;
	switch (context.pipelines->FindGraphicsPipelineForPlan(
	    plan.targets, ctx, ucfg, prepared.vertex_info,
	    prepared.pixel_active ? &prepared.pixel_info : nullptr, plan.topology,
	    plan.primitive_restart, programs, pipeline, generation)) {
		case PipelineCache::PlanLookup::Found:
			plan.pipeline            = pipeline;
			plan.pipeline_generation = generation;
			break;
		case PipelineCache::PlanLookup::Busy:
			return abstain(E::DrawPrepBindingAbstainPipelineBusy);
		case PipelineCache::PlanLookup::Absent:
			if (context.pipelines->PipelinePrefetchEnabled()) {
				context.pipelines->PrefetchGraphicsPipeline(plan.targets, ctx, ucfg, prepared.vertex_info,
				    prepared.pixel_active ? &prepared.pixel_info : nullptr, plan.topology,
				    plan.primitive_restart, programs);
			}
			return abstain(E::DrawPrepBindingAbstainPipeline);
		case PipelineCache::PlanLookup::Unsupported:
			return abstain(E::DrawPrepBindingAbstainPipeline);
	}
}

void PlanStatics(const RegisterSnapshot& registers, const PreparedDraw& prepared,
                 BindingPlan& plan) {
	const auto& vertex_program = *prepared.vertex_info.stage.program;
	bool        vertex_writes  = false;
	bool        pixel_writes   = false;
	if (!PlanBufferWrites(vertex_program, prepared.vertex_prep.resources, vertex_writes) ||
	    (prepared.pixel_active && !PlanBufferWrites(*prepared.pixel_info.stage.program,
	                                                prepared.pixel_prep.resources, pixel_writes))) {
		return;
	}
	plan.shader_write_stages = {};
	if (vertex_writes) {
		plan.shader_write_stages |=
		    ShaderPipelineStages(NativeShaderStage(prepared.vertex_info.logical_stage));
	}
	if (pixel_writes) {
		plan.shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	plan.programs_barrier_safe =
	    ProgramBarrierSafe(vertex_program, plan.plain_pixel) &&
	    (!prepared.pixel_active ||
	     ProgramBarrierSafe(*prepared.pixel_info.stage.program, plan.plain_pixel));
	plan.written_valid = false;
	if (TextureCache::AliasBytesEnabled()) {
		const auto& outputs = vertex_program.info.outputs;
		// Quiet: an unsupported clip-rect rule is left to the command processor, which logs it.
		plan.written_valid = DrawScissorUnionQuiet(
		    registers.context,
		    std::any_of(outputs.begin(), outputs.end(),
		                [](const auto& output) {
			                return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
		                }),
		    plan.written);
	}
	plan.statics_valid = true;
}

// AcquireVertexBuffers' ranges, clamped: valid only when every clamp leaves its range whole.
void PlanVertexRanges(const PreparedDraw& prepared, VertexRangePlan& ranges) {
	const auto& info = prepared.vertex_info;
	if (info.stage.program->stage == ShaderType::Mesh || info.buffers_num < 0 ||
	    info.buffers_num > ShaderVertexInputInfo::RES_MAX || info.resources_num < 0 ||
	    info.resources_num > ShaderVertexInputInfo::RES_MAX) {
		return;
	}
	// VertexBufferDescriptorSize reads the attribute resources of stride-0 buffers.
	for (int i = 0; i < info.buffers_num; i++) {
		const auto& buffer = info.buffers[i];
		if (buffer.attr_num < 0 || buffer.attr_num > ShaderVertexInputBuffer::ATTR_MAX) {
			return;
		}
		for (int j = 0; j < buffer.attr_num; j++) {
			if (buffer.attr_indices[j] < 0 ||
			    buffer.attr_indices[j] >= ShaderVertexInputInfo::RES_MAX) {
				return;
			}
		}
	}
	uint32_t invalid = 0;
	if (!CollectVertexRanges(info, ranges, invalid)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingAbstainRanges);
		return;
	}
	for (uint32_t i = 0; i < ranges.merged_count; i++) {
		auto&      range     = ranges.merged[i];
		const auto requested = range.requested_end - range.base_address;
		if (LibKernel::Memory::ClampRangeSizeQuiet(range.base_address, requested) != requested) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingAbstainRanges);
			return;
		}
		range.acquired_end = range.requested_end;
	}
	uint32_t unassigned = 0;
	ranges.valid        = AssignVertexRanges(info, ranges, unassigned);
}

} // namespace

uint32_t BindingParts() {
	static const uint32_t parts = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_BINDINGS");
		if (value == nullptr || std::strcmp(value, "0") == 0) {
			return 0u;
		}
		uint32_t result = 0;
		if (std::strcmp(value, "1") == 0 || std::strcmp(value, "all") == 0) {
			result = AllBindingParts;
		} else {
			std::string_view list(value);
			while (!list.empty()) {
				const auto comma = list.find(',');
				const auto name  = list.substr(0, comma);
				list = comma == std::string_view::npos ? std::string_view {} : list.substr(comma + 1);
				if (name == "pipeline") {
					result |= static_cast<uint32_t>(BindingPart::Pipeline);
				} else if (name == "buffers") {
					result |= static_cast<uint32_t>(BindingPart::Buffers);
				} else if (name == "userdata") {
					result |= static_cast<uint32_t>(BindingPart::UserData);
				} else if (name == "samplers") {
					result |= static_cast<uint32_t>(BindingPart::Samplers);
				} else if (name == "textures") {
					result |= static_cast<uint32_t>(BindingPart::Textures);
				} else if (name == "statics") {
					result |= static_cast<uint32_t>(BindingPart::Statics);
				} else if (name == "hwcheck") {
					result |= static_cast<uint32_t>(BindingPart::HwCheck);
				} else if (name == "dynamic") {
					result |= static_cast<uint32_t>(BindingPart::Dynamic);
				} else if (name == "texturememo") {
					result |= static_cast<uint32_t>(BindingPart::TextureMemo);
				} else if (!name.empty()) {
					EXIT("KYTY_DRAW_PREP_BINDINGS: unknown part '%.*s' (expected 0, 1 or a list of "
					     "pipeline, buffers, userdata, samplers, textures, statics, hwcheck, "
					     "dynamic, texturememo)\n",
					     static_cast<int>(name.size()), name.data());
				}
			}
		}
		if (result != 0) {
			LOGF("DrawPrep: binding plans on (parts=0x%02x, verify=%d)\n", result,
			     BindingsVerifyMode());
			// Switches the preparing threads read, initialized (and reported) on this thread.
			(void)LodStatsCounter::CountClamped();
			(void)LodStatsCounter::PlainVariant();
			(void)TextureCache::AliasBytesEnabled();
			(void)MeshPrimitiveRestartEnabled();
			(void)PipelineDynamicRasterStateEnabled();
		}
		return result;
	}();
	return parts;
}

int BindingsVerifyMode() {
	static const int mode = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_BINDINGS_VERIFY");
		if (value == nullptr || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

void CountBindingVerifyCheck() {
	g_binding_totals.verify_checks.fetch_add(1, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingVerifyChecks);
}

void ReportBindingMismatch(const char* item, uint32_t detail) {
	g_binding_totals.verify_mismatches.fetch_add(1, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingVerifyMismatches);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("DrawPrepBindingVerify: the plan's %s differs from the serial value (detail %u)\n",
		     item, detail);
	}
	if (BindingsVerifyMode() == 2) {
		EXIT("DrawPrepBindingVerify: a binding plan item differs from the serial value (%s)\n",
		     item);
	}
}

BindingTotals& GetBindingTotals() {
	return g_binding_totals;
}

void PrefetchBindingPipeline(const BindingPlanContext& context, const RegisterSnapshot& registers,
                             const DrawIndexArgs* index_args, const PreparedDraw& prepared) {
	if (!context.pipelines->PipelinePrefetchEnabled() || !prepared.ok) return;
	// A regular pipeline binding plan below already requests the same key.
	if (BindingPartEnabled(BindingParts(), BindingPart::Pipeline)) return;
	BindingPlan plan;
	if (prepared.pixel_active && LodStatsCounter::PlainVariant() == LodStatsCounter::Plain::On &&
	    PipelineCache::PlainPixelProgram(prepared.pixel_prep)) {
		bool active = false;
		static thread_local std::vector<uint32_t> scratch;
		if (!PlanShaderData(*prepared.pixel_info.stage.program, prepared.pixel_prep.resources, scratch, active)) return;
		plan.plain_pixel = !active;
	}
	PlanPipeline(context, registers, index_args, prepared, plan, true);
}

void ComputeBindingPlan(const BindingPlanContext& context, const RegisterSnapshot& registers,
                        const DrawIndexArgs* index_args, const PreparedDraw& prepared,
                        BindingPlan& plan) {
	plan.Reset();
	const auto parts = BindingParts();
	if (parts == 0 || !prepared.ok) {
		return;
	}
	Profiler::ScopedFrameWait wait(Profiler::FrameWait::DrawPrepBindingPlan);
	plan.parts        = parts;
	plan.pixel_active = prepared.pixel_active;
	// Read before the first clamp: unchanged at commit, every clamp below answers the same then.
	plan.vm_generation = LibKernel::Memory::VirtualRangesGeneration();

	const auto& vertex_program = *prepared.vertex_info.stage.program;
	const auto& vertex_data    = prepared.vertex_prep.resources;
	const auto* pixel_program  = prepared.pixel_active ? prepared.pixel_info.stage.program : nullptr;
	const auto& pixel_data     = prepared.pixel_prep.resources;
	if (prepared.pixel_active && pixel_program == nullptr) {
		return;
	}
	const auto plan_stage = [&](const CompiledShaderInfo& program, const ResourceSnapshot& data,
	                            StagePlan& stage) {
		if (BindingPartEnabled(parts, BindingPart::Buffers)) {
			stage.ranges_valid = PlanRanges(program, data, stage);
			if (!stage.ranges_valid) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingAbstainRanges);
			}
		}
		if (BindingPartEnabled(parts, BindingPart::UserData)) {
			stage.shader_data_valid =
			    PlanShaderData(program, data, stage.shader_data, stage.mip_stats_active);
		}
		if (BindingPartEnabled(parts, BindingPart::Samplers)) {
			stage.samplers_valid = PlanSamplers(*context.samplers, program, data, stage);
		}
		// The memo part needs the hashes too (a run finds its entries by them).
		const bool memo = BindingPartEnabled(parts, BindingPart::TextureMemo) &&
		                  context.texture_memo != nullptr && TextureBindingMemo::Enabled();
		if (BindingPartEnabled(parts, BindingPart::Textures) || memo) {
			stage.texture_hashes_valid =
			    PlanTextureHashes(program, data, memo ? context.texture_memo : nullptr, stage);
			stage.texture_tags_valid = stage.texture_tags_valid && stage.texture_hashes_valid;
		}
	};
	plan_stage(vertex_program, vertex_data, plan.vertex);
	if (pixel_program != nullptr) {
		plan_stage(*pixel_program, pixel_data, plan.pixel);
	}

	// KYTY_LOD_STATS_PLAIN_VARIANT=on: ExecutePreparedDraw's choice of the plain pixel program
	// (none of the stage's images has a mip-statistics counter).
	if (pixel_program != nullptr && LodStatsCounter::PlainVariant() == LodStatsCounter::Plain::On &&
	    PipelineCache::PlainPixelProgram(prepared.pixel_prep)) {
		bool mip_stats_active = plan.pixel.mip_stats_active;
		if (!plan.pixel.shader_data_valid) {
			static thread_local std::vector<uint32_t> scratch;
			if (!PlanShaderData(*pixel_program, pixel_data, scratch, mip_stats_active)) {
				return; // no pipeline or statics without the choice
			}
		}
		plan.plain_pixel = !mip_stats_active;
	}
	if (BindingPartEnabled(parts, BindingPart::Statics)) {
		PlanStatics(registers, prepared, plan);
	}
	if (BindingPartEnabled(parts, BindingPart::Buffers)) {
		PlanVertexRanges(prepared, plan.vertex_ranges);
	}
	if (BindingPartEnabled(parts, BindingPart::Pipeline)) {
		PlanPipeline(context, registers, index_args, prepared, plan);
	}
	if (BindingPartEnabled(parts, BindingPart::HwCheck)) {
		plan.hw_checks_quiet = hw_checks_quiet(registers.context, registers.user_config);
	}
	if (BindingPartEnabled(parts, BindingPart::Dynamic)) {
		const auto& outputs = vertex_program.info.outputs;
		(void)PlanDynamicViewports(
		    registers.context, context.graphics->GetPhysicalDeviceProperties().limits,
		    std::any_of(outputs.begin(), outputs.end(),
		                [](const auto& output) {
			                return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
		                }),
		    plan.viewports);
	}
	plan.valid = true;
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingPlans);
}

void CountCommittedPlan(bool used) {
	g_binding_totals.plans.fetch_add(1, std::memory_order_relaxed);
	if (used) {
		g_binding_totals.used.fetch_add(1, std::memory_order_relaxed);
	}
	Profiler::CountFrameEvent(used ? Profiler::FrameEvent::DrawPrepBindingPlansUsed
	                               : Profiler::FrameEvent::DrawPrepBindingPlansDropped);
}

const PipelineCache::Pipeline* PlannedPipeline(const PipelineCache& cache, const BindingPlan& plan,
                                               std::span<const RenderColorInfo> colors,
                                               const RenderDepthInfo& depth,
                                               vk::PrimitiveTopology  topology,
                                               bool primitive_restart_enable, bool plain_pixel) {
	if (plan.pipeline == nullptr) {
		return nullptr;
	}
	if (plan.topology != topology || plan.primitive_restart != primitive_restart_enable ||
	    plan.plain_pixel != plain_pixel ||
	    !PipelineCache::SamePipelineTargets(plan.targets, colors, depth) ||
	    cache.PipelineGeneration() != plan.pipeline_generation) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingFallbackPipeline);
		g_binding_totals.pipeline_fallbacks.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingPipelinesUsed);
	g_binding_totals.pipelines_used.fetch_add(1, std::memory_order_relaxed);
	return plan.pipeline;
}

} // namespace Libs::Graphics::DrawPrep
