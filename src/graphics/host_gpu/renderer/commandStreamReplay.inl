// Included by commandStream.h: Replay<Exec>, the consumer half of the command stream.
//
// An executor provides, with vk::CommandBuffer's argument lists:
//   Begin(const BeginPacket&), Submit(const SubmitPacket&), DrainMarker(uint64_t),
//   EnterSite(const void* site, const void* scope), LeaveSite(),
//   and one method per native command named as in vk::CommandBuffer (beginRendering, draw, ...).
// Every array argument points into the packet (or into ReplayState scratch) and is valid only
// during the call.

namespace Libs::Graphics::CommandStream {

template <typename Exec>
void Replay(const Header& header, Exec& exec, ReplayState& state, MismatchHandler on_mismatch,
            void* mismatch_context) {
	const SiteBlock*   site   = nullptr;
	const VerifyBlock* verify = nullptr;
	Reader             reader(&header, Detail::PayloadOffset(header, &site, &verify));
	const Op           op = header.op;

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
			check(VerifyHash::DrainMarker(p.serial));
			exec.DrainMarker(p.serial);
			break;
		}
		case Op::Begin: {
			const auto& p    = reader.Get<BeginPacket>();
			state.cb_digest  = 0;
			state.cb_packets = 0;
			check(VerifyHash::Begin(p.command, p.tick));
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
			check(VerifyHash::Submit(p));
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
			check(VerifyHash::BeginRendering(info));
			exec.beginRendering(info);
			break;
		}
		case Op::EndRendering: {
			check(VerifyHash::EndRendering());
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
			check(VerifyHash::PipelineBarrier2(info));
			exec.pipelineBarrier2(info);
			break;
		}
		case Op::PipelineBarrier: {
			const auto& p       = reader.Get<PipelineBarrierPacket>();
			const auto* memory  = reader.GetArray<vk::MemoryBarrier>(p.memory_count);
			const auto* buffers = reader.GetArray<vk::BufferMemoryBarrier>(p.buffer_count);
			const auto* images  = reader.GetArray<vk::ImageMemoryBarrier>(p.image_count);
			check(VerifyHash::PipelineBarrier(p.src_stages, p.dst_stages, p.flags, p.memory_count,
			                                  memory, p.buffer_count, buffers, p.image_count,
			                                  images));
			exec.pipelineBarrier(p.src_stages, p.dst_stages, p.flags, p.memory_count, memory,
			                     p.buffer_count, buffers, p.image_count, images);
			break;
		}
		case Op::CopyBuffer: {
			const auto& p       = reader.Get<CopyBufferPacket>();
			const auto* regions = reader.GetArray<vk::BufferCopy>(p.count);
			check(VerifyHash::CopyBuffer(p.source, p.destination, p.count, regions));
			exec.copyBuffer(p.source, p.destination, p.count, regions);
			break;
		}
		case Op::CopyBufferToImage: {
			const auto& p       = reader.Get<CopyBufferToImagePacket>();
			const auto* regions = reader.GetArray<vk::BufferImageCopy>(p.count);
			check(
			    VerifyHash::CopyBufferToImage(p.source, p.destination, p.layout, p.count, regions));
			exec.copyBufferToImage(p.source, p.destination, p.layout, p.count, regions);
			break;
		}
		case Op::CopyImageToBuffer: {
			const auto& p       = reader.Get<CopyImageToBufferPacket>();
			const auto* regions = reader.GetArray<vk::BufferImageCopy>(p.count);
			check(
			    VerifyHash::CopyImageToBuffer(p.source, p.layout, p.destination, p.count, regions));
			exec.copyImageToBuffer(p.source, p.layout, p.destination, p.count, regions);
			break;
		}
		case Op::CopyImage: {
			const auto& p       = reader.Get<CopyImagePacket>();
			const auto* regions = reader.GetArray<vk::ImageCopy>(p.count);
			check(VerifyHash::CopyImage(p.source, p.source_layout, p.destination,
			                            p.destination_layout, p.count, regions));
			exec.copyImage(p.source, p.source_layout, p.destination, p.destination_layout, p.count,
			               regions);
			break;
		}
		case Op::FillBuffer: {
			const auto& p = reader.Get<FillBufferPacket>();
			check(VerifyHash::FillBuffer(p.buffer, p.offset, p.size, p.value));
			exec.fillBuffer(p.buffer, p.offset, p.size, p.value);
			break;
		}
		case Op::BindPipeline: {
			const auto& p = reader.Get<BindPipelinePacket>();
			check(VerifyHash::BindPipeline(p.point, p.pipeline));
			exec.bindPipeline(p.point, p.pipeline);
			break;
		}
		case Op::BindDescriptorSets: {
			const auto& p       = reader.Get<BindDescriptorSetsPacket>();
			const auto* sets    = reader.GetArray<vk::DescriptorSet>(p.set_count);
			const auto* offsets = reader.GetArray<uint32_t>(p.dynamic_count);
			check(VerifyHash::BindDescriptorSets(p.point, p.layout, p.first_set, p.set_count, sets,
			                                     p.dynamic_count, offsets));
			exec.bindDescriptorSets(p.point, p.layout, p.first_set, p.set_count, sets,
			                        p.dynamic_count, offsets);
			break;
		}
		case Op::PushDescriptorSet: {
			const auto& p       = reader.Get<PushDescriptorSetPacket>();
			const auto* records = reader.GetArray<DescriptorWriteRecord>(p.write_count);
			const auto* infos   = reader.GetArray<DescriptorInfo>(p.info_count);
			state.writes.resize(p.write_count);
			uint32_t next = 0;
			for (uint32_t i = 0; i < p.write_count; i++) {
				const auto& record    = records[i];
				auto&       write     = state.writes[i];
				write                 = vk::WriteDescriptorSet {};
				write.dstBinding      = record.binding;
				write.dstArrayElement = record.element;
				write.descriptorCount = record.count;
				write.descriptorType  = record.type;
				EXIT_IF(next + record.count > p.info_count);
				if (record.is_image != 0) {
					write.pImageInfo =
					    reinterpret_cast<const vk::DescriptorImageInfo*>(infos + next);
				} else {
					write.pBufferInfo =
					    reinterpret_cast<const vk::DescriptorBufferInfo*>(infos + next);
				}
				next += record.count;
			}
			EXIT_IF(next != p.info_count);
			check(VerifyHash::PushDescriptorSet(p.point, p.layout, p.set, p.write_count,
			                                    state.writes.data()));
			exec.pushDescriptorSetKHR(p.point, p.layout, p.set, p.write_count, state.writes.data());
			break;
		}
		case Op::PushConstants: {
			const auto& p    = reader.Get<PushConstantsPacket>();
			const auto* data = reader.GetArray<uint8_t>(p.size);
			check(VerifyHash::PushConstants(p.layout, p.stages, p.offset, p.size, data));
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
			check(
			    VerifyHash::BindVertexBuffers2(p.first, p.count, buffers, offsets, sizes, strides));
			exec.bindVertexBuffers2(p.first, p.count, buffers, offsets, sizes, strides);
			break;
		}
		case Op::BindIndexBuffer: {
			const auto& p = reader.Get<BindIndexBufferPacket>();
			check(VerifyHash::BindIndexBuffer(p.buffer, p.offset, p.type));
			exec.bindIndexBuffer(p.buffer, p.offset, p.type);
			break;
		}
		case Op::SetViewportWithCount: {
			const auto& p         = reader.Get<CountPacket>();
			const auto* viewports = reader.GetArray<vk::Viewport>(p.count);
			check(VerifyHash::Viewports(p.count, viewports));
			exec.setViewportWithCount(p.count, viewports);
			break;
		}
		case Op::SetScissorWithCount: {
			const auto& p        = reader.Get<CountPacket>();
			const auto* scissors = reader.GetArray<vk::Rect2D>(p.count);
			check(VerifyHash::Scissors(p.count, scissors));
			exec.setScissorWithCount(p.count, scissors);
			break;
		}
		case Op::SetLineWidth: {
			const auto& p = reader.Get<FloatPacket>();
			check(VerifyHash::Floats(op, &p.value, 1));
			exec.setLineWidth(p.value);
			break;
		}
		case Op::SetBlendConstants: {
			const auto& p = reader.Get<Float4Packet>();
			check(VerifyHash::Floats(op, p.values, 4));
			exec.setBlendConstants(p.values);
			break;
		}
		case Op::SetDepthTestEnable: {
			const auto& p = reader.Get<UintPacket>();
			check(VerifyHash::Value(op, p.value));
			exec.setDepthTestEnable(p.value);
			break;
		}
		case Op::SetDepthWriteEnable: {
			const auto& p = reader.Get<UintPacket>();
			check(VerifyHash::Value(op, p.value));
			exec.setDepthWriteEnable(p.value);
			break;
		}
		case Op::SetDepthCompareOp: {
			const auto& p = reader.Get<UintPacket>();
			check(VerifyHash::Value(op, p.value));
			exec.setDepthCompareOp(static_cast<vk::CompareOp>(p.value));
			break;
		}
		case Op::SetDepthBiasEnable: {
			const auto& p = reader.Get<UintPacket>();
			check(VerifyHash::Value(op, p.value));
			exec.setDepthBiasEnable(p.value);
			break;
		}
		case Op::SetDepthBias: {
			const auto& p = reader.Get<Float3Packet>();
			check(VerifyHash::Floats(op, p.values, 3));
			exec.setDepthBias(p.values[0], p.values[1], p.values[2]);
			break;
		}
		case Op::SetStencilTestEnable: {
			const auto& p = reader.Get<UintPacket>();
			check(VerifyHash::Value(op, p.value));
			exec.setStencilTestEnable(p.value);
			break;
		}
		case Op::SetStencilOp: {
			const auto& p = reader.Get<StencilOpPacket>();
			check(VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(p.faces),
			                        static_cast<uint64_t>(p.fail), static_cast<uint64_t>(p.pass),
			                        static_cast<uint64_t>(p.depth_fail),
			                        static_cast<uint64_t>(p.compare)));
			exec.setStencilOp(p.faces, p.fail, p.pass, p.depth_fail, p.compare);
			break;
		}
		case Op::SetStencilCompareMask: {
			const auto& p = reader.Get<StencilValuePacket>();
			check(VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(p.faces), p.value));
			exec.setStencilCompareMask(p.faces, p.value);
			break;
		}
		case Op::SetStencilWriteMask: {
			const auto& p = reader.Get<StencilValuePacket>();
			check(VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(p.faces), p.value));
			exec.setStencilWriteMask(p.faces, p.value);
			break;
		}
		case Op::SetStencilReference: {
			const auto& p = reader.Get<StencilValuePacket>();
			check(VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(p.faces), p.value));
			exec.setStencilReference(p.faces, p.value);
			break;
		}
		case Op::SetCullMode: {
			const auto& p = reader.Get<UintPacket>();
			check(VerifyHash::Value(op, p.value));
			exec.setCullMode(vk::CullModeFlags(p.value));
			break;
		}
		case Op::SetFrontFace: {
			const auto& p = reader.Get<UintPacket>();
			check(VerifyHash::Value(op, p.value));
			exec.setFrontFace(static_cast<vk::FrontFace>(p.value));
			break;
		}
		case Op::SetDepthBoundsTestEnable: {
			const auto& p = reader.Get<UintPacket>();
			check(VerifyHash::Value(op, p.value));
			exec.setDepthBoundsTestEnable(p.value);
			break;
		}
		case Op::SetDepthBounds: {
			const auto& p = reader.Get<Float2Packet>();
			check(VerifyHash::Floats(op, p.values, 2));
			exec.setDepthBounds(p.values[0], p.values[1]);
			break;
		}
		case Op::SetColorWriteEnable: {
			const auto& p       = reader.Get<CountPacket>();
			const auto* enables = reader.GetArray<vk::Bool32>(p.count);
			check(VerifyHash::ColorWriteEnable(p.count, enables));
			exec.setColorWriteEnableEXT(p.count, enables);
			break;
		}
		case Op::SetAttachmentFeedbackLoopEnable: {
			const auto& p = reader.Get<UintPacket>();
			check(VerifyHash::Value(op, p.value));
			exec.setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags(p.value));
			break;
		}
		case Op::Draw: {
			const auto& p = reader.Get<DrawPacket>();
			check(VerifyHash::Value(op, p.vertex_count, p.instance_count, p.first_vertex,
			                        p.first_instance));
			exec.draw(p.vertex_count, p.instance_count, p.first_vertex, p.first_instance);
			break;
		}
		case Op::DrawIndexed: {
			const auto& p = reader.Get<DrawIndexedPacket>();
			check(VerifyHash::Value(op, p.index_count, p.instance_count, p.first_index,
			                        static_cast<uint32_t>(p.vertex_offset), p.first_instance));
			exec.drawIndexed(p.index_count, p.instance_count, p.first_index, p.vertex_offset,
			                 p.first_instance);
			break;
		}
		case Op::DrawMeshTasks: {
			const auto& p = reader.Get<Groups3Packet>();
			check(VerifyHash::Value(op, p.x, p.y, p.z));
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
			check(VerifyHash::Value(op, buffer_bits, p.offset, p.draw_count, p.stride));
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
			check(VerifyHash::Value(op, buffer_bits, p.offset, count_bits, p.count_offset,
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
			check(VerifyHash::Value(op, p.x, p.y, p.z));
			exec.dispatch(p.x, p.y, p.z);
			break;
		}
		case Op::DispatchIndirect: {
			const auto& p           = reader.Get<DispatchIndirectPacket>();
			uint64_t    buffer_bits = 0;
			const auto  raw         = static_cast<VkBuffer>(p.buffer);
			std::memcpy(&buffer_bits, &raw, sizeof(buffer_bits));
			check(VerifyHash::Value(op, buffer_bits, p.offset));
			exec.dispatchIndirect(p.buffer, p.offset);
			break;
		}
		case Op::ResetQueryPool:
		case Op::EndQuery: {
			const auto& p         = reader.Get<QueryRangePacket>();
			uint64_t    pool_bits = 0;
			const auto  raw       = static_cast<VkQueryPool>(p.pool);
			std::memcpy(&pool_bits, &raw, sizeof(pool_bits));
			check(VerifyHash::Value(op, pool_bits, p.first, p.count));
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
			check(VerifyHash::Value(op, pool_bits, p.query,
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
			check(VerifyHash::Value(
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
			check(VerifyHash::Value(op, pool_bits, p.query, p.stage));
			const auto stage = vk::PipelineStageFlags2(static_cast<VkPipelineStageFlags2>(p.stage));
			exec.writeTimestamp2(stage, p.pool, p.query);
			break;
		}
		case Op::BeginConditionalRendering: {
			const auto& p           = reader.Get<ConditionalRenderingPacket>();
			uint64_t    buffer_bits = 0;
			const auto  raw         = static_cast<VkBuffer>(p.buffer);
			std::memcpy(&buffer_bits, &raw, sizeof(buffer_bits));
			check(VerifyHash::Value(op, buffer_bits, p.offset, p.flags));
			vk::ConditionalRenderingBeginInfoEXT info {};
			info.buffer = p.buffer;
			info.offset = p.offset;
			info.flags  = vk::ConditionalRenderingFlagsEXT(
                static_cast<VkConditionalRenderingFlagsEXT>(p.flags));
			exec.beginConditionalRenderingEXT(info);
			break;
		}
		case Op::EndConditionalRendering: {
			check(VerifyHash::Value(op, 0));
			exec.endConditionalRenderingEXT();
			break;
		}
	}
	if (site != nullptr) {
		exec.LeaveSite();
	}
}

} // namespace Libs::Graphics::CommandStream
