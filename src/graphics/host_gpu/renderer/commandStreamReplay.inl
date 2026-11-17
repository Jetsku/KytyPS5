// Included by commandStream.h: Replay<Exec>, the consumer half of the command stream.
//
// An executor provides, with vk::CommandBuffer's argument lists:
//   Begin(const BeginPacket&), Submit(const SubmitPacket&), DrainMarker(uint64_t),
//   EnterSite(const void* site, const void* scope), LeaveSite(),
//   and one method per native command named as in vk::CommandBuffer (beginRendering, draw, ...).
// Every array argument points into the packet (or into ReplayState scratch) and is valid only
// during the call.

namespace Libs::Graphics::CommandStream {

namespace Detail {
// Rebuilds a packet's descriptor writes in state.writes (their infos point into the packet), each
// with dstSet `set`.
inline void RebuildDescriptorWrites(ReplayState& state, const DescriptorWriteRecord* records,
                                    uint32_t write_count, const DescriptorInfo* infos,
                                    uint32_t info_count, vk::DescriptorSet set) {
	state.writes.resize(write_count);
	uint32_t next = 0;
	for (uint32_t i = 0; i < write_count; i++) {
		const auto& record    = records[i];
		auto&       write     = state.writes[i];
		write                 = vk::WriteDescriptorSet {};
		write.dstSet          = set;
		write.dstBinding      = record.binding;
		write.dstArrayElement = record.element;
		write.descriptorCount = record.count;
		write.descriptorType  = record.type;
		EXIT_IF(next + record.count > info_count);
		if (record.is_image != 0) {
			write.pImageInfo = reinterpret_cast<const vk::DescriptorImageInfo*>(infos + next);
		} else {
			write.pBufferInfo = reinterpret_cast<const vk::DescriptorBufferInfo*>(infos + next);
		}
		next += record.count;
	}
	EXIT_IF(next != info_count);
}

// KYTY_EMIT_SPLIT=verify: a verify-only resolved draw against the state the packets before it left
// (ResolvedDrawModel). Every field the record would set must hold the same value there.
inline void CheckResolvedDraw(ReplayState& state, const ResolvedDrawPacket& p, const vk::Buffer* vb,
                              const vk::DeviceSize* vo, const vk::DeviceSize* vs, const DescriptorWriteRecord* writes,
                              const DescriptorInfo* infos, const uint32_t* push, const uint32_t* clip) {
	using P     = ResolvedDrawPacket;
	using M     = ResolvedDrawModel;
	auto& model = state.record_model;
	auto& stats = state.record_stats;
	if ((++stats.verify_records & 1023u) == 0) {
		PublishResolvedDraw(stats, state.record_published);
	}
	if (!model.on) {
		model.on = true; // modelled from the next Begin on
	}
	if (!model.valid) {
		stats.verify_unmodeled++;
		return;
	}
	stats.verify_checks++;
	bool bad[M::FieldCount] = {};
	bad[M::Pipeline]        = model.pipeline != p.pipeline;
	if ((p.flags & P::FlagVertex) != 0) {
		bad[M::Vertex] = model.vb_buffers.size() != p.vertex_count || !model.vb_sizes_set;
		for (uint32_t i = 0; !bad[M::Vertex] && i < p.vertex_count; i++) {
			bad[M::Vertex] = model.vb_buffers[i] != vb[i] || model.vb_offsets[i] != vo[i] || model.vb_sizes[i] != vs[i];
		}
	}
	if ((p.flags & P::FlagIndex) != 0) {
		bad[M::Index] = model.ib_buffer != p.index_buffer || model.ib_offset != p.index_offset ||
		                model.ib_type != p.index_type;
	}
	if ((p.flags & P::FlagPush) != 0 && p.write_count != 0) {
		bad[M::Layout] = model.desc_layout != p.layout;
		uint32_t next  = 0;
		for (uint32_t w = 0; w < p.write_count && !bad[M::Layout]; w++) {
			const auto& record = writes[w];
			for (uint32_t j = 0; j < record.count; j++) {
				const auto* d = model.Find(record.binding, record.element + j);
				if (d == nullptr || d->type != record.type) {
					bad[M::Writes] = true;
				} else if (std::memcmp(&d->info, infos + next + j, sizeof(DescriptorInfo)) != 0) {
					bad[M::Infos] = true;
				}
			}
			next += record.count;
		}
	}
	// Push constants as the expansion leaves them: the block, then the clip planes, then the object id.
	std::array<uint32_t, 64> expect {};
	uint64_t                 mask = 0;
	if ((p.flags & P::FlagPushData) != 0) {
		std::memcpy(expect.data(), push, ResolvedDrawPushDwords * 4u);
		mask |= (uint64_t {1} << ResolvedDrawPushDwords) - 1u;
	}
	for (uint32_t d = 0; d < p.clip_bytes / 4u && p.clip_offset / 4u + d < 64u; d++) {
		expect[p.clip_offset / 4u + d] = clip[d];
		mask |= uint64_t {1} << (p.clip_offset / 4u + d);
	}
	if ((p.flags & P::FlagObjprim) != 0 && p.objprim_offset / 4u < 64u) {
		expect[p.objprim_offset / 4u] = p.object_id;
		mask |= uint64_t {1} << (p.objprim_offset / 4u);
	}
	bad[M::Constants] = (model.constants_valid & mask) != mask;
	for (uint32_t d = 0; !bad[M::Constants] && d < 64u; d++) {
		bad[M::Constants] = ((mask >> d) & 1u) != 0 && model.constants[d] != expect[d];
	}
	const bool     indexed = (p.flags & P::FlagIndexed) != 0;
	const uint32_t args[5] = {p.count, p.instance_count, indexed ? 0u : p.first_vertex,
	                          indexed ? static_cast<uint32_t>(p.vertex_offset) : 0u, p.first_instance};
	bad[M::Draw]           = model.draw_indexed != indexed || std::memcmp(model.draw_args, args, sizeof(args)) != 0;
	bool any               = false;
	for (uint32_t f = 0; f < M::FieldCount; f++) {
		if (bad[f]) {
			stats.mismatches[f]++;
			any = true;
		}
	}
	if (any) {
		PublishResolvedDraw(stats, state.record_published);
	}
	if (any && stats.reported < 16) {
		stats.reported++;
		std::printf("EmitSplit verify: resolved draw differs from the encoded draw:%s%s%s%s%s%s%s%s\n",
		            bad[0] ? " pipeline" : "", bad[1] ? " vertex" : "", bad[2] ? " index" : "", bad[3] ? " layout" : "",
		            bad[4] ? " writes" : "", bad[5] ? " infos" : "", bad[6] ? " constants" : "", bad[7] ? " draw" : "");
		std::fflush(stdout);
	}
}

// KYTY_EMIT_SPLIT: a resolved draw's commands (ResolvedDrawPacket, in its order) into `exec`.
template <typename Exec>
void ExpandResolvedDraw(Exec& exec, ReplayState& state, const ResolvedDrawPacket& p, const vk::Buffer* vb,
                        const vk::DeviceSize* vo, const vk::DeviceSize* vs, const DescriptorWriteRecord* writes,
                        const DescriptorInfo* infos, const uint32_t* push, const uint32_t* clip) {
	using P      = ResolvedDrawPacket;
	auto& shadow = state.record_shadow;
	auto& stats  = state.record_stats;
	if ((++stats.records & 1023u) == 0) {
		PublishResolvedDraw(stats, state.record_published);
	}
	if ((p.flags & P::FlagShadowReset) != 0) {
		shadow.Reset();
		stats.shadow_resets++;
	}
	constexpr auto graphics = vk::PipelineBindPoint::eGraphics;
	if ((p.flags & P::FlagVertex) != 0 && p.vertex_count != 0) {
		exec.bindVertexBuffers2(0, p.vertex_count, vb, vo, vs, nullptr);
	}
	if ((p.flags & P::FlagPushData) != 0) {
		const auto stages = vk::ShaderStageFlags(p.push_stages);
		if ((p.flags & P::FlagConstDedup) != 0 && shadow.SameConstants(p.layout, p.push_stages, push)) {
			stats.constants_avoided++;
		} else {
			exec.pushConstants(p.layout, stages, 0, ResolvedDrawPushDwords * 4u, push);
			stats.constants++;
			if ((p.flags & P::FlagConstDedup) != 0) {
				shadow.KeepConstants(p.layout, p.push_stages, push);
			} else {
				shadow.const_valid = false;
			}
		}
	}
	if ((p.flags & P::FlagPush) != 0 && p.write_count != 0) {
		const bool known_miss = (p.flags & P::FlagKnownMiss) != 0;
		const bool dedup      = (p.flags & P::FlagPushDedup) != 0;
		if (!known_miss && dedup && shadow.SameDescriptors(p.layout, p.write_count, writes, p.info_count, infos)) {
			stats.pushes_avoided++;
		} else {
			RebuildDescriptorWrites(state, writes, p.write_count, infos, p.info_count, nullptr);
			exec.pushDescriptorSetKHR(graphics, p.layout, 0, p.write_count, state.writes.data());
			stats.pushes++;
			if (known_miss || !dedup) {
				shadow.desc_valid = false;
			} else {
				shadow.KeepDescriptors(p.layout, p.write_count, writes, p.info_count, infos);
			}
		}
	}
	if ((p.flags & P::FlagBindless) != 0) {
		exec.bindDescriptorSets(graphics, p.layout, 1, 1, &p.bindless_set, 0, nullptr);
	}
	if ((p.flags & P::FlagIndex) != 0) {
		exec.bindIndexBuffer(p.index_buffer, p.index_offset, static_cast<vk::IndexType>(p.index_type));
	}
	if ((p.flags & P::FlagBind) != 0) {
		exec.bindPipeline(graphics, p.pipeline);
	}
	if (p.clip_bytes != 0) {
		// As the commit thread records them: raw, the block's shadow kept (the planes follow every draw).
		exec.pushConstants(p.layout, vk::ShaderStageFlags(p.clip_stages), p.clip_offset, p.clip_bytes, clip);
	}
	if ((p.flags & P::FlagObjprim) != 0) {
		exec.pushConstants(p.layout, vk::ShaderStageFlags(p.objprim_stages), p.objprim_offset, sizeof(p.object_id),
		                   &p.object_id);
		shadow.const_valid = false;
	}
	if ((p.flags & P::FlagIndexed) != 0) {
		exec.drawIndexed(p.count, p.instance_count, 0, p.vertex_offset, p.first_instance);
	} else {
		exec.draw(p.count, p.instance_count, p.first_vertex, p.first_instance);
	}
}
} // namespace Detail

template <typename Exec>
void Replay(const Header& header, Exec& exec, ReplayState& state, MismatchHandler on_mismatch,
            void* mismatch_context) {
	const SiteBlock*   site   = nullptr;
	const VerifyBlock* verify = nullptr;
	Reader             reader(&header, Detail::PayloadOffset(header, &site, &verify));
	const Op           op = header.op;

	// Every call site tests verify first, so with KYTY_CP_RECORDER_VERIFY off no argument hash is computed.
	// Compares the producer's hash of its original arguments with this side's hash of the
	// arguments passed to the executor, and the sequence number; folds the packet into the
	// command-buffer digest.
	const auto check = [&](uint64_t hash) {
		if (verify == nullptr) {
			return;
		}
		state.checks++;
		if (verify->sequence != state.sequence + 1) {
			state.mismatches++;
			if (on_mismatch != nullptr) {
				on_mismatch(mismatch_context, 1, op, verify->sequence, state.sequence + 1,
				            verify->sequence);
			}
		}
		state.sequence = verify->sequence;
		if (verify->hash != hash) {
			state.mismatches++;
			if (state.mismatches == 1) {
				state.mismatch_op       = op;
				state.mismatch_sequence = verify->sequence;
				state.mismatch_expected = verify->hash;
				state.mismatch_actual   = hash;
			}
			if (on_mismatch != nullptr) {
				on_mismatch(mismatch_context, 0, op, verify->sequence, verify->hash, hash);
			}
		}
		state.cb_digest = FoldDigest(state.cb_digest, op, hash);
		state.cb_packets++;
	};

	if (site != nullptr) {
		exec.EnterSite(site->site, site->scope);
	}
	switch (op) {
		case Op::Wrap:
		case Op::Count: EXIT("CommandStream: unexpected packet %u\n", static_cast<uint32_t>(op));
		case Op::DrainMarker: {
			const auto& p = reader.Get<DrainMarkerPacket>();
			if (verify != nullptr) check(VerifyHash::DrainMarker(p.serial));
			exec.DrainMarker(p.serial);
			break;
		}
		case Op::Begin: {
			const auto& p    = reader.Get<BeginPacket>();
			state.cb_digest  = 0;
			state.cb_packets = 0;
			state.record_shadow.Reset();
			if (state.record_model.on) {
				state.record_model.Begin();
			}
			if (verify != nullptr) check(VerifyHash::Begin(p.command, p.tick));
			exec.Begin(p);
			break;
		}
		case Op::Submit: {
			const auto& p = reader.Get<SubmitPacket>();
			if (verify != nullptr &&
			    (p.cb_digest != state.cb_digest || p.cb_packets != state.cb_packets)) {
				state.mismatches++;
				if (on_mismatch != nullptr) {
					on_mismatch(mismatch_context, 2, op, verify->sequence, p.cb_digest,
					            state.cb_digest);
				}
			}
			if (verify != nullptr) check(VerifyHash::Submit(p));
			exec.Submit(p);
			break;
		}
		case Op::BeginRendering: {
			const auto& p      = reader.Get<BeginRenderingPacket>();
			const auto* colors = reader.GetArray<vk::RenderingAttachmentInfo>(p.colors);
			const vk::RenderingAttachmentInfo* depth =
			    p.has_depth != 0 ? &reader.Get<vk::RenderingAttachmentInfo>() : nullptr;
			const vk::RenderingAttachmentInfo* stencil =
			    p.has_stencil != 0 ? &reader.Get<vk::RenderingAttachmentInfo>() : nullptr;
			vk::RenderingInfo info {};
			info.flags                = p.flags;
			info.renderArea           = p.area;
			info.layerCount           = p.layers;
			info.viewMask             = p.view_mask;
			info.colorAttachmentCount = p.colors;
			info.pColorAttachments    = colors;
			info.pDepthAttachment     = depth;
			info.pStencilAttachment   = stencil;
			if (verify != nullptr) check(VerifyHash::BeginRendering(info));
			exec.beginRendering(info);
			break;
		}
		case Op::EndRendering: {
			if (verify != nullptr) check(VerifyHash::EndRendering());
			exec.endRendering();
			break;
		}
		case Op::PipelineBarrier2: {
			const auto&        p = reader.Get<PipelineBarrier2Packet>();
			vk::DependencyInfo info {};
			info.dependencyFlags          = p.flags;
			info.memoryBarrierCount       = p.memory_count;
			info.pMemoryBarriers          = reader.GetArray<vk::MemoryBarrier2>(p.memory_count);
			info.bufferMemoryBarrierCount = p.buffer_count;
			info.pBufferMemoryBarriers = reader.GetArray<vk::BufferMemoryBarrier2>(p.buffer_count);
			info.imageMemoryBarrierCount = p.image_count;
			info.pImageMemoryBarriers    = reader.GetArray<vk::ImageMemoryBarrier2>(p.image_count);
			if (verify != nullptr) check(VerifyHash::PipelineBarrier2(info));
			exec.pipelineBarrier2(info);
			break;
		}
		case Op::PipelineBarrier: {
			const auto& p       = reader.Get<PipelineBarrierPacket>();
			const auto* memory  = reader.GetArray<vk::MemoryBarrier>(p.memory_count);
			const auto* buffers = reader.GetArray<vk::BufferMemoryBarrier>(p.buffer_count);
			const auto* images  = reader.GetArray<vk::ImageMemoryBarrier>(p.image_count);
			if (verify != nullptr) check(VerifyHash::PipelineBarrier(p.src_stages, p.dst_stages, p.flags, p.memory_count,
			                                  memory, p.buffer_count, buffers, p.image_count,
			                                  images));
			exec.pipelineBarrier(p.src_stages, p.dst_stages, p.flags, p.memory_count, memory,
			                     p.buffer_count, buffers, p.image_count, images);
			break;
		}
		case Op::CopyBuffer: {
			const auto& p       = reader.Get<CopyBufferPacket>();
			const auto* regions = reader.GetArray<vk::BufferCopy>(p.count);
			if (verify != nullptr) check(VerifyHash::CopyBuffer(p.source, p.destination, p.count, regions));
			exec.copyBuffer(p.source, p.destination, p.count, regions);
			break;
		}
		case Op::CopyBufferToImage: {
			const auto& p       = reader.Get<CopyBufferToImagePacket>();
			const auto* regions = reader.GetArray<vk::BufferImageCopy>(p.count);
			if (verify != nullptr) check(
			    VerifyHash::CopyBufferToImage(p.source, p.destination, p.layout, p.count, regions));
			exec.copyBufferToImage(p.source, p.destination, p.layout, p.count, regions);
			break;
		}
		case Op::CopyImageToBuffer: {
			const auto& p       = reader.Get<CopyImageToBufferPacket>();
			const auto* regions = reader.GetArray<vk::BufferImageCopy>(p.count);
			if (verify != nullptr) check(
			    VerifyHash::CopyImageToBuffer(p.source, p.layout, p.destination, p.count, regions));
			exec.copyImageToBuffer(p.source, p.layout, p.destination, p.count, regions);
			break;
		}
		case Op::CopyImage: {
			const auto& p       = reader.Get<CopyImagePacket>();
			const auto* regions = reader.GetArray<vk::ImageCopy>(p.count);
			if (verify != nullptr) check(VerifyHash::CopyImage(p.source, p.source_layout, p.destination,
			                            p.destination_layout, p.count, regions));
			exec.copyImage(p.source, p.source_layout, p.destination, p.destination_layout, p.count,
			               regions);
			break;
		}
		case Op::FillBuffer: {
			const auto& p = reader.Get<FillBufferPacket>();
			if (verify != nullptr) check(VerifyHash::FillBuffer(p.buffer, p.offset, p.size, p.value));
			exec.fillBuffer(p.buffer, p.offset, p.size, p.value);
			break;
		}
		case Op::BindPipeline: {
			const auto& p = reader.Get<BindPipelinePacket>();
			if (verify != nullptr) check(VerifyHash::BindPipeline(p.point, p.pipeline));
			if (state.record_model.valid && p.point == vk::PipelineBindPoint::eGraphics) {
				state.record_model.pipeline = p.pipeline;
			}
			exec.bindPipeline(p.point, p.pipeline);
			break;
		}
		case Op::BindDescriptorSets: {
			const auto& p       = reader.Get<BindDescriptorSetsPacket>();
			const auto* sets    = reader.GetArray<vk::DescriptorSet>(p.set_count);
			const auto* offsets = reader.GetArray<uint32_t>(p.dynamic_count);
			if (verify != nullptr) check(VerifyHash::BindDescriptorSets(p.point, p.layout, p.first_set, p.set_count, sets,
			                                     p.dynamic_count, offsets));
			if (p.point == vk::PipelineBindPoint::eGraphics && p.first_set == 0) {
				state.record_shadow.desc_valid = false;
				if (state.record_model.valid) {
					state.record_model.UnsetDescriptors();
				}
			}
			exec.bindDescriptorSets(p.point, p.layout, p.first_set, p.set_count, sets,
			                        p.dynamic_count, offsets);
			break;
		}
		case Op::PushDescriptorSet: {
			const auto& p       = reader.Get<PushDescriptorSetPacket>();
			const auto* records = reader.GetArray<DescriptorWriteRecord>(p.write_count);
			const auto* infos   = reader.GetArray<DescriptorInfo>(p.info_count);
			Detail::RebuildDescriptorWrites(state, records, p.write_count, infos, p.info_count,
			                                nullptr);
			if (verify != nullptr) check(VerifyHash::PushDescriptorSet(p.point, p.layout, p.set, p.write_count,
			                                    state.writes.data()));
			if (p.point == vk::PipelineBindPoint::eGraphics) {
				state.record_shadow.desc_valid = false;
				if (state.record_model.valid && p.set == 0) {
					state.record_model.Push(p.layout, p.write_count, state.writes.data());
				}
			}
			exec.pushDescriptorSetKHR(p.point, p.layout, p.set, p.write_count, state.writes.data());
			break;
		}
		case Op::UpdateDescriptorSets: {
			const auto& p       = reader.Get<UpdateDescriptorSetsPacket>();
			const auto* records = reader.GetArray<DescriptorWriteRecord>(p.write_count);
			const auto* infos   = reader.GetArray<DescriptorInfo>(p.info_count);
			Detail::RebuildDescriptorWrites(state, records, p.write_count, infos, p.info_count,
			                                p.set);
			if (verify != nullptr) check(VerifyHash::UpdateDescriptorSets(p.set, p.write_count, state.writes.data()));
			exec.updateDescriptorSets(p.set, p.write_count, state.writes.data());
			break;
		}
		case Op::PushConstants: {
			const auto& p    = reader.Get<PushConstantsPacket>();
			const auto* data = reader.GetArray<uint8_t>(p.size);
			if (verify != nullptr) check(VerifyHash::PushConstants(p.layout, p.stages, p.offset, p.size, data));
			state.record_shadow.const_valid = false;
			if (state.record_model.valid) {
				state.record_model.SetConstants(p.offset, p.size, data);
			}
			exec.pushConstants(p.layout, p.stages, p.offset, p.size, data);
			break;
		}
		case Op::BindVertexBuffers2: {
			const auto& p       = reader.Get<BindVertexBuffers2Packet>();
			const auto* buffers = reader.GetArray<vk::Buffer>(p.count);
			const auto* offsets = reader.GetArray<vk::DeviceSize>(p.count);
			const auto* sizes =
			    p.has_sizes != 0 ? reader.GetArray<vk::DeviceSize>(p.count) : nullptr;
			const auto* strides =
			    p.has_strides != 0 ? reader.GetArray<vk::DeviceSize>(p.count) : nullptr;
			if (verify != nullptr) check(
			    VerifyHash::BindVertexBuffers2(p.first, p.count, buffers, offsets, sizes, strides));
			if (state.record_model.valid) {
				auto& m = state.record_model;
				if (p.first == 0 && sizes != nullptr && strides == nullptr) {
					m.vb_buffers.assign(buffers, buffers + p.count);
					m.vb_offsets.assign(offsets, offsets + p.count);
					m.vb_sizes.assign(sizes, sizes + p.count);
					m.vb_sizes_set = true;
				} else {
					m.vb_buffers.clear();
					m.vb_sizes_set = false;
				}
			}
			exec.bindVertexBuffers2(p.first, p.count, buffers, offsets, sizes, strides);
			break;
		}
		case Op::BindIndexBuffer: {
			const auto& p = reader.Get<BindIndexBufferPacket>();
			if (verify != nullptr) check(VerifyHash::BindIndexBuffer(p.buffer, p.offset, p.type));
			if (state.record_model.valid) {
				state.record_model.ib_buffer = p.buffer;
				state.record_model.ib_offset = p.offset;
				state.record_model.ib_type   = static_cast<uint32_t>(p.type);
			}
			exec.bindIndexBuffer(p.buffer, p.offset, p.type);
			break;
		}
		case Op::SetViewportWithCount: {
			const auto& p         = reader.Get<CountPacket>();
			const auto* viewports = reader.GetArray<vk::Viewport>(p.count);
			if (verify != nullptr) check(VerifyHash::Viewports(p.count, viewports));
			exec.setViewportWithCount(p.count, viewports);
			break;
		}
		case Op::SetScissorWithCount: {
			const auto& p        = reader.Get<CountPacket>();
			const auto* scissors = reader.GetArray<vk::Rect2D>(p.count);
			if (verify != nullptr) check(VerifyHash::Scissors(p.count, scissors));
			exec.setScissorWithCount(p.count, scissors);
			break;
		}
		case Op::SetLineWidth: {
			const auto& p = reader.Get<FloatPacket>();
			if (verify != nullptr) check(VerifyHash::Floats(op, &p.value, 1));
			exec.setLineWidth(p.value);
			break;
		}
		case Op::SetBlendConstants: {
			const auto& p = reader.Get<Float4Packet>();
			if (verify != nullptr) check(VerifyHash::Floats(op, p.values, 4));
			exec.setBlendConstants(p.values);
			break;
		}
		case Op::SetDepthTestEnable: {
			const auto& p = reader.Get<UintPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.value));
			exec.setDepthTestEnable(p.value);
			break;
		}
		case Op::SetDepthWriteEnable: {
			const auto& p = reader.Get<UintPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.value));
			exec.setDepthWriteEnable(p.value);
			break;
		}
		case Op::SetDepthCompareOp: {
			const auto& p = reader.Get<UintPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.value));
			exec.setDepthCompareOp(static_cast<vk::CompareOp>(p.value));
			break;
		}
		case Op::SetDepthBiasEnable: {
			const auto& p = reader.Get<UintPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.value));
			exec.setDepthBiasEnable(p.value);
			break;
		}
		case Op::SetDepthBias: {
			const auto& p = reader.Get<Float3Packet>();
			if (verify != nullptr) check(VerifyHash::Floats(op, p.values, 3));
			exec.setDepthBias(p.values[0], p.values[1], p.values[2]);
			break;
		}
		case Op::SetStencilTestEnable: {
			const auto& p = reader.Get<UintPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.value));
			exec.setStencilTestEnable(p.value);
			break;
		}
		case Op::SetStencilOp: {
			const auto& p = reader.Get<StencilOpPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(p.faces),
			                        static_cast<uint64_t>(p.fail), static_cast<uint64_t>(p.pass),
			                        static_cast<uint64_t>(p.depth_fail),
			                        static_cast<uint64_t>(p.compare)));
			exec.setStencilOp(p.faces, p.fail, p.pass, p.depth_fail, p.compare);
			break;
		}
		case Op::SetStencilCompareMask: {
			const auto& p = reader.Get<StencilValuePacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(p.faces), p.value));
			exec.setStencilCompareMask(p.faces, p.value);
			break;
		}
		case Op::SetStencilWriteMask: {
			const auto& p = reader.Get<StencilValuePacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(p.faces), p.value));
			exec.setStencilWriteMask(p.faces, p.value);
			break;
		}
		case Op::SetStencilReference: {
			const auto& p = reader.Get<StencilValuePacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(p.faces), p.value));
			exec.setStencilReference(p.faces, p.value);
			break;
		}
		case Op::SetCullMode: {
			const auto& p = reader.Get<UintPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.value));
			exec.setCullMode(vk::CullModeFlags(p.value));
			break;
		}
		case Op::SetFrontFace: {
			const auto& p = reader.Get<UintPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.value));
			exec.setFrontFace(static_cast<vk::FrontFace>(p.value));
			break;
		}
		case Op::SetDepthBoundsTestEnable: {
			const auto& p = reader.Get<UintPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.value));
			exec.setDepthBoundsTestEnable(p.value);
			break;
		}
		case Op::SetDepthBounds: {
			const auto& p = reader.Get<Float2Packet>();
			if (verify != nullptr) check(VerifyHash::Floats(op, p.values, 2));
			exec.setDepthBounds(p.values[0], p.values[1]);
			break;
		}
		case Op::SetColorWriteEnable: {
			const auto& p       = reader.Get<CountPacket>();
			const auto* enables = reader.GetArray<vk::Bool32>(p.count);
			if (verify != nullptr) check(VerifyHash::ColorWriteEnable(p.count, enables));
			exec.setColorWriteEnableEXT(p.count, enables);
			break;
		}
		case Op::SetAttachmentFeedbackLoopEnable: {
			const auto& p = reader.Get<UintPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.value));
			exec.setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags(p.value));
			break;
		}
		case Op::Draw: {
			const auto& p = reader.Get<DrawPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.vertex_count, p.instance_count, p.first_vertex,
			                        p.first_instance));
			if (state.record_model.valid) {
				auto& m                = state.record_model;
				m.draw_indexed         = false;
				const uint32_t args[5] = {p.vertex_count, p.instance_count, p.first_vertex, 0u, p.first_instance};
				std::memcpy(m.draw_args, args, sizeof(args));
			}
			exec.draw(p.vertex_count, p.instance_count, p.first_vertex, p.first_instance);
			break;
		}
		case Op::DrawIndexed: {
			const auto& p = reader.Get<DrawIndexedPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.index_count, p.instance_count, p.first_index,
			                        static_cast<uint32_t>(p.vertex_offset), p.first_instance));
			if (state.record_model.valid) {
				auto& m        = state.record_model;
				m.draw_indexed = true;
				// A resolved draw's first index is 0: another one cannot match.
				const uint32_t args[5] = {p.index_count, p.instance_count, p.first_index,
				                          static_cast<uint32_t>(p.vertex_offset), p.first_instance};
				std::memcpy(m.draw_args, args, sizeof(args));
			}
			exec.drawIndexed(p.index_count, p.instance_count, p.first_index, p.vertex_offset,
			                 p.first_instance);
			break;
		}
		case Op::DrawMeshTasks: {
			const auto& p = reader.Get<Groups3Packet>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.x, p.y, p.z));
			exec.drawMeshTasksEXT(p.x, p.y, p.z);
			break;
		}
		case Op::DrawMeshTasksIndirect:
		case Op::DrawIndirect:
		case Op::DrawIndexedIndirect: {
			const auto& p           = reader.Get<IndirectPacket>();
			uint64_t    buffer_bits = 0;
			const auto  raw         = static_cast<VkBuffer>(p.buffer);
			std::memcpy(&buffer_bits, &raw, sizeof(buffer_bits));
			if (verify != nullptr) check(VerifyHash::Value(op, buffer_bits, p.offset, p.draw_count, p.stride));
			if (op == Op::DrawMeshTasksIndirect) {
				exec.drawMeshTasksIndirectEXT(p.buffer, p.offset, p.draw_count, p.stride);
			} else if (op == Op::DrawIndirect) {
				exec.drawIndirect(p.buffer, p.offset, p.draw_count, p.stride);
			} else {
				exec.drawIndexedIndirect(p.buffer, p.offset, p.draw_count, p.stride);
			}
			break;
		}
		case Op::DrawMeshTasksIndirectCount:
		case Op::DrawIndirectCount:
		case Op::DrawIndexedIndirectCount: {
			const auto& p           = reader.Get<IndirectCountPacket>();
			uint64_t    buffer_bits = 0;
			uint64_t    count_bits  = 0;
			const auto  raw         = static_cast<VkBuffer>(p.buffer);
			const auto  raw_count   = static_cast<VkBuffer>(p.count_buffer);
			std::memcpy(&buffer_bits, &raw, sizeof(buffer_bits));
			std::memcpy(&count_bits, &raw_count, sizeof(count_bits));
			if (verify != nullptr) check(VerifyHash::Value(op, buffer_bits, p.offset, count_bits, p.count_offset,
			                        p.max_count, p.stride));
			if (op == Op::DrawMeshTasksIndirectCount) {
				exec.drawMeshTasksIndirectCountEXT(p.buffer, p.offset, p.count_buffer,
				                                   p.count_offset, p.max_count, p.stride);
			} else if (op == Op::DrawIndirectCount) {
				exec.drawIndirectCount(p.buffer, p.offset, p.count_buffer, p.count_offset,
				                       p.max_count, p.stride);
			} else {
				exec.drawIndexedIndirectCount(p.buffer, p.offset, p.count_buffer, p.count_offset,
				                              p.max_count, p.stride);
			}
			break;
		}
		case Op::Dispatch: {
			const auto& p = reader.Get<Groups3Packet>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.x, p.y, p.z));
			exec.dispatch(p.x, p.y, p.z);
			break;
		}
		case Op::DispatchBase: {
			const auto& p = reader.Get<DispatchBasePacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.base_x, p.base_y, p.base_z, p.x, p.y, p.z));
			exec.dispatchBase(p.base_x, p.base_y, p.base_z, p.x, p.y, p.z);
			break;
		}
		case Op::DispatchIndirect: {
			const auto& p           = reader.Get<DispatchIndirectPacket>();
			uint64_t    buffer_bits = 0;
			const auto  raw         = static_cast<VkBuffer>(p.buffer);
			std::memcpy(&buffer_bits, &raw, sizeof(buffer_bits));
			if (verify != nullptr) check(VerifyHash::Value(op, buffer_bits, p.offset));
			exec.dispatchIndirect(p.buffer, p.offset);
			break;
		}
		case Op::ResetQueryPool:
		case Op::EndQuery: {
			const auto& p         = reader.Get<QueryRangePacket>();
			uint64_t    pool_bits = 0;
			const auto  raw       = static_cast<VkQueryPool>(p.pool);
			std::memcpy(&pool_bits, &raw, sizeof(pool_bits));
			if (verify != nullptr) check(VerifyHash::Value(op, pool_bits, p.first, p.count));
			if (op == Op::ResetQueryPool) {
				exec.resetQueryPool(p.pool, p.first, p.count);
			} else {
				exec.endQuery(p.pool, p.first);
			}
			break;
		}
		case Op::BeginQuery: {
			const auto& p         = reader.Get<BeginQueryPacket>();
			uint64_t    pool_bits = 0;
			const auto  raw       = static_cast<VkQueryPool>(p.pool);
			std::memcpy(&pool_bits, &raw, sizeof(pool_bits));
			if (verify != nullptr) check(VerifyHash::Value(op, pool_bits, p.query,
			                        static_cast<VkQueryControlFlags>(p.flags)));
			exec.beginQuery(p.pool, p.query, p.flags);
			break;
		}
		case Op::CopyQueryPoolResults: {
			const auto& p         = reader.Get<CopyQueryPoolResultsPacket>();
			uint64_t    pool_bits = 0;
			uint64_t    dest_bits = 0;
			const auto  raw_pool  = static_cast<VkQueryPool>(p.pool);
			const auto  raw_dest  = static_cast<VkBuffer>(p.destination);
			std::memcpy(&pool_bits, &raw_pool, sizeof(pool_bits));
			std::memcpy(&dest_bits, &raw_dest, sizeof(dest_bits));
			if (verify != nullptr) check(VerifyHash::Value(
			    op, pool_bits, p.first, p.count, dest_bits, p.offset,
			    p.stride ^
			        (static_cast<uint64_t>(static_cast<VkQueryResultFlags>(p.flags)) << 40u)));
			exec.copyQueryPoolResults(p.pool, p.first, p.count, p.destination, p.offset, p.stride,
			                          p.flags);
			break;
		}
		case Op::WriteTimestamp2: {
			const auto& p         = reader.Get<WriteTimestampPacket>();
			uint64_t    pool_bits = 0;
			const auto  raw       = static_cast<VkQueryPool>(p.pool);
			std::memcpy(&pool_bits, &raw, sizeof(pool_bits));
			if (verify != nullptr) check(VerifyHash::Value(op, pool_bits, p.query, p.stage));
			const auto stage = vk::PipelineStageFlags2(static_cast<VkPipelineStageFlags2>(p.stage));
			exec.writeTimestamp2(stage, p.pool, p.query);
			break;
		}
		case Op::HostCopy: {
			const auto& p = reader.Get<HostCopyPacket>();
			if (verify != nullptr) check(VerifyHash::Value(op, p.destination, p.source, p.size));
			exec.hostCopy(reinterpret_cast<void*>(static_cast<uintptr_t>(p.destination)), p.source,
			              p.size);
			break;
		}
		case Op::InsertDebugLabel: {
			const auto  size = reader.Get<uint32_t>();
			const auto* name = reader.GetArray<char>(size);
			if (verify != nullptr) check(VerifyHash::DebugLabel(name, size));
			exec.insertDebugLabel(name);
			break;
		}
		case Op::WriteBufferMarker: {
			const auto& p           = reader.Get<WriteBufferMarkerPacket>();
			uint64_t    buffer_bits = 0;
			const auto  raw         = static_cast<VkBuffer>(p.buffer);
			std::memcpy(&buffer_bits, &raw, sizeof(buffer_bits));
			if (verify != nullptr) check(VerifyHash::Value(op, buffer_bits, p.offset, p.stage, p.marker));
			exec.writeBufferMarkerAMD(static_cast<vk::PipelineStageFlagBits>(p.stage), p.buffer, p.offset,
			                          p.marker);
			break;
		}
		case Op::ClearColorImage: {
			const auto&         p      = reader.Get<ClearImagePacket>();
			const auto*         ranges = reader.GetArray<vk::ImageSubresourceRange>(p.count);
			vk::ClearColorValue color {};
			std::memcpy(&color, p.value, sizeof(color));
			if (verify != nullptr) check(VerifyHash::ClearColorImage(p.image, p.layout, color, p.count, ranges));
			exec.clearColorImage(p.image, p.layout, &color, p.count, ranges);
			break;
		}
		case Op::ResolvedDraw: {
			const auto  start  = reader.Offset();
			const auto& p      = reader.Get<ResolvedDrawPacket>();
			const auto* copies = reader.GetArray<HostCopyPacket>(p.copy_count);
			const auto* vb     = reader.GetArray<vk::Buffer>(p.vertex_count);
			const auto* vo     = reader.GetArray<vk::DeviceSize>(p.vertex_count);
			const auto* vs     = reader.GetArray<vk::DeviceSize>(p.vertex_count);
			const auto* writes = reader.GetArray<DescriptorWriteRecord>(p.write_count);
			const auto* infos  = reader.GetArray<DescriptorInfo>(p.info_count);
			const auto* push   = reader.GetArray<uint32_t>(
                (p.flags & ResolvedDrawPacket::FlagPushData) != 0 ? ResolvedDrawPushDwords : 0u);
			const auto* clip = reader.GetArray<uint32_t>(p.clip_bytes / 4u);
			if (verify != nullptr) {
				check(VerifyHash::ResolvedDraw(reinterpret_cast<const uint8_t*>(&header) + start, reader.Offset() - start));
			}
			if ((p.flags & ResolvedDrawPacket::FlagVerifyOnly) != 0) {
				Detail::CheckResolvedDraw(state, p, vb, vo, vs, writes, infos, push, clip);
			} else {
				for (uint32_t i = 0; i < p.copy_count; i++) {
					exec.hostCopy(reinterpret_cast<void*>(static_cast<uintptr_t>(copies[i].destination)), copies[i].source,
					              copies[i].size);
				}
				Detail::ExpandResolvedDraw(exec, state, p, vb, vo, vs, writes, infos, push, clip);
			}
			break;
		}
		case Op::ClearDepthStencilImage: {
			const auto& p      = reader.Get<ClearImagePacket>();
			const auto* ranges = reader.GetArray<vk::ImageSubresourceRange>(p.count);
			const vk::ClearDepthStencilValue value {std::bit_cast<float>(p.value[0]), p.value[1]};
			if (verify != nullptr) check(VerifyHash::ClearDepthStencilImage(p.image, p.layout, value, p.count, ranges));
			exec.clearDepthStencilImage(p.image, p.layout, &value, p.count, ranges);
			break;
		}
	}
	if (site != nullptr) {
		exec.LeaveSite();
	}
}

} // namespace Libs::Graphics::CommandStream
