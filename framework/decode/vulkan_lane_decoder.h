/*
** Copyright (c) 2026 LunarG, Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
*/

#ifndef GFXRECON_DECODE_VULKAN_LANE_DECODER_H
#define GFXRECON_DECODE_VULKAN_LANE_DECODER_H

#include "decode/decode_allocator.h"
#include "decode/value_decoder.h"
#include "decode/vulkan_lane_tracker.h"
#include "decode/vulkan_replay_consumer_base.h"
#include "format/api_call_id.h"
#include "generated/generated_vulkan_decoder.h"
#include "util/alignment_utils.h"
#include "util/defines.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(decode)

// Replays the calls of a capture on up to three threads. The dispatch thread routes each call to one of two lanes, each
// of which replays its calls in stream order on its own thread:
//
//   record lane: command buffer recording
//   gpu lane:    queue submission and presentation, the waits for the GPU, and the host writes that follow them
//
// The lanes run ahead of each other as far as their dependencies allow: a command buffer is begun only after the reset
// of its pool, and a submission or pool reset follows the recording that came before it. Every other call waits for
// both lanes to be idle and then replays on the dispatch thread, so the object table is only written while the lanes
// are idle.
class VulkanLaneDecoder : public VulkanDecoder
{
  public:
    explicit VulkanLaneDecoder(VulkanReplayConsumerBase* consumer) :
        consumer_(consumer), tracker_(consumer), observer_(consumer, &tracker_), gpu_observer_(consumer, &tracker_)
    {
        // The observer comes first to see the handle ids before the consumer maps them
        gpu_peek_.AddConsumer(&gpu_observer_);
        record_.decoder.AddConsumer(&observer_);
        record_.decoder.AddConsumer(consumer);
        gpu_.decoder.AddConsumer(consumer);
        consumer->SetLaneTracker(&tracker_);
    }

    ~VulkanLaneDecoder() override
    {
        Stop(record_);
        Stop(gpu_);
    }

    void DecodeFunctionCall(format::ApiCallId  call_id,
                            const ApiCallInfo& call_info,
                            const uint8_t*     parameter_buffer,
                            size_t             buffer_size) override
    {
        PrepareHelpers();
        if (UseLanes())
        {
            switch (Classify(call_id))
            {
                case CallClass::kRecord:
                    {
                        uint64_t dep = 0;
                        if (call_id == format::ApiCall_vkBeginCommandBuffer)
                        {
                            dep = BeginDependency(parameter_buffer, buffer_size);
                        }
                        if (Enqueue(record_, call_id, call_info, parameter_buffer, buffer_size, kCall, dep))
                        {
                            return;
                        }
                    }
                    break;
                case CallClass::kGpu:
                    {
                        uint64_t dep = 0;
                        if ((call_id == format::ApiCall_vkQueueSubmit) || (call_id == format::ApiCall_vkQueueSubmit2) ||
                            (call_id == format::ApiCall_vkQueueSubmit2KHR))
                        {
                            dep = record_.enqueued;
                        }
                        if (call_id == format::ApiCall_vkResetCommandPool)
                        {
                            // The reset waits until a command buffer of the pool is begun
                            const format::HandleId pool_id = CommandPoolOfReset(parameter_buffer, buffer_size);
                            if (Enqueue(gpu_, call_id, call_info, parameter_buffer, buffer_size, kParkedReset, 0, nullptr, 0, pool_id))
                            {
                                NoteResetCommandPool(pool_id);
                                flush_needed_ = true;
                                return;
                            }
                        }
                        else if (Enqueue(gpu_, call_id, call_info, parameter_buffer, buffer_size, kCall, dep))
                        {
                            flush_needed_ |= (call_id == format::ApiCall_vkWaitSemaphores) || (call_id == format::ApiCall_vkWaitSemaphoresKHR);
                            return;
                        }
                    }
                    break;
                case CallClass::kOrdered:
                    break;
            }
        }

        BeforeInline();
        VulkanDecoder::DecodeFunctionCall(call_id, call_info, parameter_buffer, buffer_size);
    }

    void DispatchFillMemoryCommand(
        format::ThreadId thread_id, uint64_t memory_id, uint64_t offset, uint64_t size, const uint8_t* data) override
    {
        if (UseLanes())
        {
            PrepareHelpers();
            uint64_t header[3] = { memory_id, offset, size };
            if (Enqueue(gpu_, format::ApiCallId::ApiCall_Unknown, ApiCallInfo{}, reinterpret_cast<uint8_t*>(header), sizeof(header), kFill, 0, data, size))
            {
                return;
            }
        }
        BeforeInline();
        VulkanDecoder::DispatchFillMemoryCommand(thread_id, memory_id, offset, size, data);
    }

    void PreDispatchBarrier() override { BeforeInline(); }

    void WaitIdle() override
    {
        BeforeInline();
        VulkanDecoder::WaitIdle();
    }

  private:
    enum class CallClass
    {
        kRecord,
        kGpu,
        kOrdered
    };

    static CallClass Classify(format::ApiCallId call_id)
    {
        switch (call_id)
        {
        case format::ApiCall_vkCmdBeginConditionalRendering2EXT:
        case format::ApiCall_vkCmdBeginConditionalRenderingEXT:
        case format::ApiCall_vkCmdBeginCustomResolveEXT:
        case format::ApiCall_vkCmdBeginDebugUtilsLabelEXT:
        case format::ApiCall_vkCmdBeginGpaSampleAMD:
        case format::ApiCall_vkCmdBeginGpaSessionAMD:
        case format::ApiCall_vkCmdBeginPerTileExecutionQCOM:
        case format::ApiCall_vkCmdBeginQuery:
        case format::ApiCall_vkCmdBeginQueryIndexedEXT:
        case format::ApiCall_vkCmdBeginRenderPass:
        case format::ApiCall_vkCmdBeginRenderPass2:
        case format::ApiCall_vkCmdBeginRenderPass2KHR:
        case format::ApiCall_vkCmdBeginRendering:
        case format::ApiCall_vkCmdBeginRenderingKHR:
        case format::ApiCall_vkCmdBeginTransformFeedback2EXT:
        case format::ApiCall_vkCmdBeginTransformFeedbackEXT:
        case format::ApiCall_vkCmdBeginVideoCodingKHR:
        case format::ApiCall_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT:
        case format::ApiCall_vkCmdBindDescriptorBufferEmbeddedSamplersEXT:
        case format::ApiCall_vkCmdBindDescriptorBuffersEXT:
        case format::ApiCall_vkCmdBindDescriptorSets:
        case format::ApiCall_vkCmdBindDescriptorSets2:
        case format::ApiCall_vkCmdBindDescriptorSets2KHR:
        case format::ApiCall_vkCmdBindIndexBuffer:
        case format::ApiCall_vkCmdBindIndexBuffer2:
        case format::ApiCall_vkCmdBindIndexBuffer2KHR:
        case format::ApiCall_vkCmdBindIndexBuffer3KHR:
        case format::ApiCall_vkCmdBindInvocationMaskHUAWEI:
        case format::ApiCall_vkCmdBindPipeline:
        case format::ApiCall_vkCmdBindPipelineShaderGroupNV:
        case format::ApiCall_vkCmdBindShadersEXT:
        case format::ApiCall_vkCmdBindShadingRateImageNV:
        case format::ApiCall_vkCmdBindTileMemoryQCOM:
        case format::ApiCall_vkCmdBindTransformFeedbackBuffers2EXT:
        case format::ApiCall_vkCmdBindTransformFeedbackBuffersEXT:
        case format::ApiCall_vkCmdBindVertexBuffers:
        case format::ApiCall_vkCmdBindVertexBuffers2:
        case format::ApiCall_vkCmdBindVertexBuffers2EXT:
        case format::ApiCall_vkCmdBindVertexBuffers3KHR:
        case format::ApiCall_vkCmdBlitImage:
        case format::ApiCall_vkCmdBlitImage2:
        case format::ApiCall_vkCmdBlitImage2KHR:
        case format::ApiCall_vkCmdBuildAccelerationStructureNV:
        case format::ApiCall_vkCmdBuildAccelerationStructuresIndirectKHR:
        case format::ApiCall_vkCmdBuildAccelerationStructuresKHR:
        case format::ApiCall_vkCmdBuildMicromapsEXT:
        case format::ApiCall_vkCmdBuildPartitionedAccelerationStructuresNV:
        case format::ApiCall_vkCmdClearAttachments:
        case format::ApiCall_vkCmdClearColorImage:
        case format::ApiCall_vkCmdClearDepthStencilImage:
        case format::ApiCall_vkCmdControlVideoCodingKHR:
        case format::ApiCall_vkCmdConvertCooperativeVectorMatrixNV:
        case format::ApiCall_vkCmdCopyAccelerationStructureKHR:
        case format::ApiCall_vkCmdCopyAccelerationStructureNV:
        case format::ApiCall_vkCmdCopyAccelerationStructureToMemoryKHR:
        case format::ApiCall_vkCmdCopyBuffer:
        case format::ApiCall_vkCmdCopyBuffer2:
        case format::ApiCall_vkCmdCopyBuffer2KHR:
        case format::ApiCall_vkCmdCopyBufferToImage:
        case format::ApiCall_vkCmdCopyBufferToImage2:
        case format::ApiCall_vkCmdCopyBufferToImage2KHR:
        case format::ApiCall_vkCmdCopyGpaSessionResultsAMD:
        case format::ApiCall_vkCmdCopyImage:
        case format::ApiCall_vkCmdCopyImage2:
        case format::ApiCall_vkCmdCopyImage2KHR:
        case format::ApiCall_vkCmdCopyImageToBuffer:
        case format::ApiCall_vkCmdCopyImageToBuffer2:
        case format::ApiCall_vkCmdCopyImageToBuffer2KHR:
        case format::ApiCall_vkCmdCopyImageToMemoryKHR:
        case format::ApiCall_vkCmdCopyMemoryIndirectKHR:
        case format::ApiCall_vkCmdCopyMemoryKHR:
        case format::ApiCall_vkCmdCopyMemoryToAccelerationStructureKHR:
        case format::ApiCall_vkCmdCopyMemoryToImageIndirectKHR:
        case format::ApiCall_vkCmdCopyMemoryToImageKHR:
        case format::ApiCall_vkCmdCopyMemoryToMicromapEXT:
        case format::ApiCall_vkCmdCopyMicromapEXT:
        case format::ApiCall_vkCmdCopyMicromapToMemoryEXT:
        case format::ApiCall_vkCmdCopyQueryPoolResults:
        case format::ApiCall_vkCmdCopyQueryPoolResultsToMemoryKHR:
        case format::ApiCall_vkCmdCopyTensorARM:
        case format::ApiCall_vkCmdDebugMarkerBeginEXT:
        case format::ApiCall_vkCmdDebugMarkerEndEXT:
        case format::ApiCall_vkCmdDebugMarkerInsertEXT:
        case format::ApiCall_vkCmdDecodeVideoKHR:
        case format::ApiCall_vkCmdDecompressMemoryEXT:
        case format::ApiCall_vkCmdDecompressMemoryIndirectCountEXT:
        case format::ApiCall_vkCmdDispatch:
        case format::ApiCall_vkCmdDispatchBase:
        case format::ApiCall_vkCmdDispatchBaseKHR:
        case format::ApiCall_vkCmdDispatchDataGraphARM:
        case format::ApiCall_vkCmdDispatchIndirect:
        case format::ApiCall_vkCmdDispatchIndirect2KHR:
        case format::ApiCall_vkCmdDispatchTileQCOM:
        case format::ApiCall_vkCmdDraw:
        case format::ApiCall_vkCmdDrawClusterHUAWEI:
        case format::ApiCall_vkCmdDrawClusterIndirectHUAWEI:
        case format::ApiCall_vkCmdDrawIndexed:
        case format::ApiCall_vkCmdDrawIndexedIndirect:
        case format::ApiCall_vkCmdDrawIndexedIndirect2KHR:
        case format::ApiCall_vkCmdDrawIndexedIndirectCount:
        case format::ApiCall_vkCmdDrawIndexedIndirectCount2KHR:
        case format::ApiCall_vkCmdDrawIndexedIndirectCountAMD:
        case format::ApiCall_vkCmdDrawIndexedIndirectCountKHR:
        case format::ApiCall_vkCmdDrawIndirect:
        case format::ApiCall_vkCmdDrawIndirect2KHR:
        case format::ApiCall_vkCmdDrawIndirectByteCount2EXT:
        case format::ApiCall_vkCmdDrawIndirectByteCountEXT:
        case format::ApiCall_vkCmdDrawIndirectCount:
        case format::ApiCall_vkCmdDrawIndirectCount2KHR:
        case format::ApiCall_vkCmdDrawIndirectCountAMD:
        case format::ApiCall_vkCmdDrawIndirectCountKHR:
        case format::ApiCall_vkCmdDrawMeshTasksEXT:
        case format::ApiCall_vkCmdDrawMeshTasksIndirect2EXT:
        case format::ApiCall_vkCmdDrawMeshTasksIndirectCount2EXT:
        case format::ApiCall_vkCmdDrawMeshTasksIndirectCountEXT:
        case format::ApiCall_vkCmdDrawMeshTasksIndirectCountNV:
        case format::ApiCall_vkCmdDrawMeshTasksIndirectEXT:
        case format::ApiCall_vkCmdDrawMeshTasksIndirectNV:
        case format::ApiCall_vkCmdDrawMeshTasksNV:
        case format::ApiCall_vkCmdDrawMultiEXT:
        case format::ApiCall_vkCmdDrawMultiIndexedEXT:
        case format::ApiCall_vkCmdEncodeVideoKHR:
        case format::ApiCall_vkCmdEndConditionalRenderingEXT:
        case format::ApiCall_vkCmdEndDebugUtilsLabelEXT:
        case format::ApiCall_vkCmdEndGpaSampleAMD:
        case format::ApiCall_vkCmdEndGpaSessionAMD:
        case format::ApiCall_vkCmdEndPerTileExecutionQCOM:
        case format::ApiCall_vkCmdEndQuery:
        case format::ApiCall_vkCmdEndQueryIndexedEXT:
        case format::ApiCall_vkCmdEndRenderPass:
        case format::ApiCall_vkCmdEndRenderPass2:
        case format::ApiCall_vkCmdEndRenderPass2KHR:
        case format::ApiCall_vkCmdEndRendering:
        case format::ApiCall_vkCmdEndRendering2EXT:
        case format::ApiCall_vkCmdEndRendering2KHR:
        case format::ApiCall_vkCmdEndRenderingKHR:
        case format::ApiCall_vkCmdEndTransformFeedback2EXT:
        case format::ApiCall_vkCmdEndTransformFeedbackEXT:
        case format::ApiCall_vkCmdEndVideoCodingKHR:
        case format::ApiCall_vkCmdExecuteCommands:
        case format::ApiCall_vkCmdExecuteGeneratedCommandsEXT:
        case format::ApiCall_vkCmdExecuteGeneratedCommandsNV:
        case format::ApiCall_vkCmdFillBuffer:
        case format::ApiCall_vkCmdFillMemoryKHR:
        case format::ApiCall_vkCmdInsertDebugUtilsLabelEXT:
        case format::ApiCall_vkCmdNextSubpass:
        case format::ApiCall_vkCmdNextSubpass2:
        case format::ApiCall_vkCmdNextSubpass2KHR:
        case format::ApiCall_vkCmdOpticalFlowExecuteNV:
        case format::ApiCall_vkCmdPipelineBarrier:
        case format::ApiCall_vkCmdPipelineBarrier2:
        case format::ApiCall_vkCmdPipelineBarrier2KHR:
        case format::ApiCall_vkCmdPreprocessGeneratedCommandsEXT:
        case format::ApiCall_vkCmdPreprocessGeneratedCommandsNV:
        case format::ApiCall_vkCmdProcessCommandsNVX:
        case format::ApiCall_vkCmdPushConstants:
        case format::ApiCall_vkCmdPushConstants2:
        case format::ApiCall_vkCmdPushConstants2KHR:
        case format::ApiCall_vkCmdPushDescriptorSet:
        case format::ApiCall_vkCmdPushDescriptorSet2:
        case format::ApiCall_vkCmdPushDescriptorSet2KHR:
        case format::ApiCall_vkCmdPushDescriptorSetKHR:
        case format::ApiCall_vkCmdPushDescriptorSetWithTemplate:
        case format::ApiCall_vkCmdPushDescriptorSetWithTemplate2:
        case format::ApiCall_vkCmdPushDescriptorSetWithTemplate2KHR:
        case format::ApiCall_vkCmdPushDescriptorSetWithTemplateKHR:
        case format::ApiCall_vkCmdReserveSpaceForCommandsNVX:
        case format::ApiCall_vkCmdResetEvent:
        case format::ApiCall_vkCmdResetEvent2:
        case format::ApiCall_vkCmdResetEvent2KHR:
        case format::ApiCall_vkCmdResetQueryPool:
        case format::ApiCall_vkCmdResolveImage:
        case format::ApiCall_vkCmdResolveImage2:
        case format::ApiCall_vkCmdResolveImage2KHR:
        case format::ApiCall_vkCmdSetAlphaToCoverageEnableEXT:
        case format::ApiCall_vkCmdSetAlphaToOneEnableEXT:
        case format::ApiCall_vkCmdSetAttachmentFeedbackLoopEnableEXT:
        case format::ApiCall_vkCmdSetBlendConstants:
        case format::ApiCall_vkCmdSetCheckpointNV:
        case format::ApiCall_vkCmdSetCoarseSampleOrderNV:
        case format::ApiCall_vkCmdSetColorBlendAdvancedEXT:
        case format::ApiCall_vkCmdSetColorBlendEnableEXT:
        case format::ApiCall_vkCmdSetColorBlendEquationEXT:
        case format::ApiCall_vkCmdSetColorWriteEnableEXT:
        case format::ApiCall_vkCmdSetColorWriteMaskEXT:
        case format::ApiCall_vkCmdSetComputeOccupancyPriorityNV:
        case format::ApiCall_vkCmdSetConservativeRasterizationModeEXT:
        case format::ApiCall_vkCmdSetCoverageModulationModeNV:
        case format::ApiCall_vkCmdSetCoverageModulationTableEnableNV:
        case format::ApiCall_vkCmdSetCoverageModulationTableNV:
        case format::ApiCall_vkCmdSetCoverageReductionModeNV:
        case format::ApiCall_vkCmdSetCoverageToColorEnableNV:
        case format::ApiCall_vkCmdSetCoverageToColorLocationNV:
        case format::ApiCall_vkCmdSetCullMode:
        case format::ApiCall_vkCmdSetCullModeEXT:
        case format::ApiCall_vkCmdSetDepthBias:
        case format::ApiCall_vkCmdSetDepthBias2EXT:
        case format::ApiCall_vkCmdSetDepthBiasEnable:
        case format::ApiCall_vkCmdSetDepthBiasEnableEXT:
        case format::ApiCall_vkCmdSetDepthBounds:
        case format::ApiCall_vkCmdSetDepthBoundsTestEnable:
        case format::ApiCall_vkCmdSetDepthBoundsTestEnableEXT:
        case format::ApiCall_vkCmdSetDepthClampEnableEXT:
        case format::ApiCall_vkCmdSetDepthClampRangeEXT:
        case format::ApiCall_vkCmdSetDepthClipEnableEXT:
        case format::ApiCall_vkCmdSetDepthClipNegativeOneToOneEXT:
        case format::ApiCall_vkCmdSetDepthCompareOp:
        case format::ApiCall_vkCmdSetDepthCompareOpEXT:
        case format::ApiCall_vkCmdSetDepthTestEnable:
        case format::ApiCall_vkCmdSetDepthTestEnableEXT:
        case format::ApiCall_vkCmdSetDepthWriteEnable:
        case format::ApiCall_vkCmdSetDepthWriteEnableEXT:
        case format::ApiCall_vkCmdSetDescriptorBufferOffsets2EXT:
        case format::ApiCall_vkCmdSetDescriptorBufferOffsetsEXT:
        case format::ApiCall_vkCmdSetDeviceMask:
        case format::ApiCall_vkCmdSetDeviceMaskKHR:
        case format::ApiCall_vkCmdSetDiscardRectangleEXT:
        case format::ApiCall_vkCmdSetDiscardRectangleEnableEXT:
        case format::ApiCall_vkCmdSetDiscardRectangleModeEXT:
        case format::ApiCall_vkCmdSetDispatchParametersARM:
        case format::ApiCall_vkCmdSetEvent:
        case format::ApiCall_vkCmdSetEvent2:
        case format::ApiCall_vkCmdSetEvent2KHR:
        case format::ApiCall_vkCmdSetExclusiveScissorEnableNV:
        case format::ApiCall_vkCmdSetExclusiveScissorNV:
        case format::ApiCall_vkCmdSetExtraPrimitiveOverestimationSizeEXT:
        case format::ApiCall_vkCmdSetFragmentShadingRateEnumNV:
        case format::ApiCall_vkCmdSetFragmentShadingRateKHR:
        case format::ApiCall_vkCmdSetFrontFace:
        case format::ApiCall_vkCmdSetFrontFaceEXT:
        case format::ApiCall_vkCmdSetLineRasterizationModeEXT:
        case format::ApiCall_vkCmdSetLineStipple:
        case format::ApiCall_vkCmdSetLineStippleEXT:
        case format::ApiCall_vkCmdSetLineStippleEnableEXT:
        case format::ApiCall_vkCmdSetLineStippleKHR:
        case format::ApiCall_vkCmdSetLineWidth:
        case format::ApiCall_vkCmdSetLogicOpEXT:
        case format::ApiCall_vkCmdSetLogicOpEnableEXT:
        case format::ApiCall_vkCmdSetPatchControlPointsEXT:
        case format::ApiCall_vkCmdSetPerformanceMarkerINTEL:
        case format::ApiCall_vkCmdSetPerformanceOverrideINTEL:
        case format::ApiCall_vkCmdSetPerformanceStreamMarkerINTEL:
        case format::ApiCall_vkCmdSetPolygonModeEXT:
        case format::ApiCall_vkCmdSetPrimitiveRestartEnable:
        case format::ApiCall_vkCmdSetPrimitiveRestartEnableEXT:
        case format::ApiCall_vkCmdSetPrimitiveRestartIndexEXT:
        case format::ApiCall_vkCmdSetPrimitiveTopology:
        case format::ApiCall_vkCmdSetPrimitiveTopologyEXT:
        case format::ApiCall_vkCmdSetProvokingVertexModeEXT:
        case format::ApiCall_vkCmdSetRasterizationSamplesEXT:
        case format::ApiCall_vkCmdSetRasterizationStreamEXT:
        case format::ApiCall_vkCmdSetRasterizerDiscardEnable:
        case format::ApiCall_vkCmdSetRasterizerDiscardEnableEXT:
        case format::ApiCall_vkCmdSetRayTracingPipelineStackSizeKHR:
        case format::ApiCall_vkCmdSetRenderingAttachmentLocations:
        case format::ApiCall_vkCmdSetRenderingAttachmentLocationsKHR:
        case format::ApiCall_vkCmdSetRenderingInputAttachmentIndices:
        case format::ApiCall_vkCmdSetRenderingInputAttachmentIndicesKHR:
        case format::ApiCall_vkCmdSetRepresentativeFragmentTestEnableNV:
        case format::ApiCall_vkCmdSetSampleLocationsEXT:
        case format::ApiCall_vkCmdSetSampleLocationsEnableEXT:
        case format::ApiCall_vkCmdSetSampleMaskEXT:
        case format::ApiCall_vkCmdSetScissor:
        case format::ApiCall_vkCmdSetScissorWithCount:
        case format::ApiCall_vkCmdSetScissorWithCountEXT:
        case format::ApiCall_vkCmdSetShadingRateImageEnableNV:
        case format::ApiCall_vkCmdSetStencilCompareMask:
        case format::ApiCall_vkCmdSetStencilOp:
        case format::ApiCall_vkCmdSetStencilOpEXT:
        case format::ApiCall_vkCmdSetStencilReference:
        case format::ApiCall_vkCmdSetStencilTestEnable:
        case format::ApiCall_vkCmdSetStencilTestEnableEXT:
        case format::ApiCall_vkCmdSetStencilWriteMask:
        case format::ApiCall_vkCmdSetTessellationDomainOriginEXT:
        case format::ApiCall_vkCmdSetVertexInputEXT:
        case format::ApiCall_vkCmdSetViewport:
        case format::ApiCall_vkCmdSetViewportShadingRatePaletteNV:
        case format::ApiCall_vkCmdSetViewportSwizzleNV:
        case format::ApiCall_vkCmdSetViewportWScalingEnableNV:
        case format::ApiCall_vkCmdSetViewportWScalingNV:
        case format::ApiCall_vkCmdSetViewportWithCount:
        case format::ApiCall_vkCmdSetViewportWithCountEXT:
        case format::ApiCall_vkCmdTraceRaysIndirect2KHR:
        case format::ApiCall_vkCmdTraceRaysIndirectKHR:
        case format::ApiCall_vkCmdTraceRaysKHR:
        case format::ApiCall_vkCmdTraceRaysNV:
        case format::ApiCall_vkCmdUpdateBuffer:
        case format::ApiCall_vkCmdUpdateMemoryKHR:
        case format::ApiCall_vkCmdUpdatePipelineIndirectBuffer:
        case format::ApiCall_vkCmdUpdatePipelineIndirectBufferNV:
        case format::ApiCall_vkCmdWaitEvents:
        case format::ApiCall_vkCmdWaitEvents2:
        case format::ApiCall_vkCmdWaitEvents2KHR:
        case format::ApiCall_vkCmdWriteAccelerationStructuresPropertiesKHR:
        case format::ApiCall_vkCmdWriteAccelerationStructuresPropertiesNV:
        case format::ApiCall_vkCmdWriteBufferMarker2AMD:
        case format::ApiCall_vkCmdWriteBufferMarkerAMD:
        case format::ApiCall_vkCmdWriteMarkerToMemoryAMD:
        case format::ApiCall_vkCmdWriteMicromapsPropertiesEXT:
        case format::ApiCall_vkCmdWriteTimestamp:
        case format::ApiCall_vkCmdWriteTimestamp2:
        case format::ApiCall_vkCmdWriteTimestamp2KHR:
        case format::ApiCall_vkBeginCommandBuffer:
        case format::ApiCall_vkEndCommandBuffer:
            return CallClass::kRecord;
        case format::ApiCall_vkWaitSemaphores:
        case format::ApiCall_vkWaitSemaphoresKHR:
        case format::ApiCall_vkResetCommandPool:
        case format::ApiCall_vkUpdateDescriptorSets:
        case format::ApiCall_vkQueueSubmit:
        case format::ApiCall_vkQueueSubmit2:
        case format::ApiCall_vkQueueSubmit2KHR:
        case format::ApiCall_vkQueuePresentKHR:
        case format::ApiCall_vkAcquireNextImageKHR:
        case format::ApiCall_vkAcquireNextImage2KHR:
        case format::ApiCall_vkResetQueryPool:
        case format::ApiCall_vkResetQueryPoolEXT:
        case format::ApiCall_vkGetQueryPoolResults:
        case format::ApiCall_vkWaitForFences:
        case format::ApiCall_vkResetFences:
        case format::ApiCall_vkGetFenceStatus:
        case format::ApiCall_vkMapMemory:
        case format::ApiCall_vkUnmapMemory:
            return CallClass::kGpu;
            default:
                return CallClass::kOrdered;
        }
    }

    // Ring entry: this header, then for function calls the call info and the parameters, padded to 16 bytes.
    struct Entry
    {
        uint32_t size;
        uint32_t call_id;
        uint32_t parameter_size;
        uint32_t kind; // See EntryKind
        uint64_t dep;  // The number of entries the other lane must have completed first
        uint64_t pool; // The command pool of a reset, or of the begin that needs it
    };
    static_assert(sizeof(Entry) == 32);

    enum EntryKind : uint32_t
    {
        kCall,        // A function call
        kFill,        // A memory fill: memory id, offset, size, then the data
        kUnparkReset, // The reset of a command pool that was put off
        kFlush,       // Everything that was put off
        kParkedReset  // The reset of a command pool, to put off
    };

    static constexpr uint32_t kWrapId    = 0xffffffffu;
    static constexpr uint64_t kRingSize  = 8ull << 20;
    static constexpr uint64_t kRingMask  = kRingSize - 1;
    static constexpr uint64_t kSpinCount = 4000;

    struct Lane
    {
        VulkanDecoder              decoder;
        std::unique_ptr<uint8_t[]> ring;
        std::thread                thread;
        Lane*                      other{ nullptr };
        alignas(64) uint64_t enqueued{ 0 }; // Dispatch thread only
        alignas(64) std::atomic<uint64_t> head{ 0 };
        alignas(64) std::atomic<uint64_t> tail{ 0 };
        alignas(64) std::atomic<uint64_t> done{ 0 }; // Entries completed
        alignas(64) std::atomic<bool> sleeping{ false };
        std::atomic<bool> stop{ false };
    };

    static void Relax()
    {
#if defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }

    void PrepareHelpers()
    {
        if (consumer_->LaneDevicesPending())
        {
            DrainAll();
            consumer_->PrepareLaneHelpers();
        }
    }

    // Whether the lanes replay the calls. They stop for good once a call needs the dispatch thread.
    bool UseLanes()
    {
        if (!lanes_off_ && !consumer_->RecordLaneAllowed())
        {
            lanes_off_ = true;
            BeforeInline();
            consumer_->SetLaneTracker(nullptr);
        }
        return !lanes_off_;
    }

    // Everything that the lanes put off is replayed, since a call that is not followed may depend on the GPU
    void FlushGpu()
    {
        if (flush_needed_)
        {
            flush_needed_ = false;
            Enqueue(gpu_, format::ApiCallId::ApiCall_Unknown, ApiCallInfo{}, nullptr, 0, kFlush, record_.enqueued);
            for (auto& pool : pools_)
            {
                if (pool.second.reset_parked)
                {
                    pool.second.reset_parked = false;
                    pool.second.reset_seq    = gpu_.enqueued;
                }
            }
        }
    }

    // Before a call is replayed on the dispatch thread, which has the object tables to itself
    void BeforeInline()
    {
        FlushGpu();
        DrainAll();
        tracker_.NoteInlineCall();
    }

    static format::HandleId CommandPoolOfReset(const uint8_t* parameters, size_t size)
    {
        // vkResetCommandPool(device, commandPool, flags): two handles
        format::HandleId ids[2] = {};
        size_t           read   = ValueDecoder::DecodeHandleIdValue(parameters, size, &ids[0]);
        ValueDecoder::DecodeHandleIdValue(parameters + read, size - read, &ids[1]);
        return ids[1];
    }

    // The dependency of beginning a command buffer: the reset of its pool, which may have been put off
    uint64_t BeginDependency(const uint8_t* parameters, size_t size)
    {
        format::HandleId command_buffer_id = format::kNullHandleId;
        ValueDecoder::DecodeHandleIdValue(parameters, size, &command_buffer_id);
        const format::HandleId pool_id = consumer_->GetCommandBufferPoolId(command_buffer_id);
        PoolState&             pool    = pools_[pool_id];

        // A command buffer that is begun again without a pool reset is reset by the begin, which must follow the
        // completion of its last submission.
        uint64_t& begun_epoch = begun_epoch_[command_buffer_id];
        const bool rebegin    = (begun_epoch == pool.epoch + 1);
        begun_epoch           = pool.epoch + 1;

        uint64_t dep = pool.reset_seq;
        if (pool.reset_parked)
        {
            pool.reset_parked = false;
            Enqueue(gpu_, format::ApiCallId::ApiCall_Unknown, ApiCallInfo{}, nullptr, 0, kUnparkReset, record_.enqueued, nullptr, 0, pool_id);
            dep = pool.reset_seq = gpu_.enqueued;
        }
        else if (rebegin)
        {
            FlushGpu();
            dep = gpu_.enqueued;
        }
        return dep;
    }

    void NoteResetCommandPool(format::HandleId pool_id)
    {
        PoolState& pool   = pools_[pool_id];
        pool.reset_parked = true;
        ++pool.epoch;
    }

    bool Enqueue(Lane&              lane,
                 format::ApiCallId  call_id,
                 const ApiCallInfo& call_info,
                 const uint8_t*     parameters,
                 size_t             size,
                 uint32_t           kind,
                 uint64_t           dep,
                 const uint8_t*     extra      = nullptr,
                 size_t             extra_size = 0,
                 uint64_t           pool       = 0)
    {
        const uint64_t need = util::aligned_value<16>(sizeof(Entry) + sizeof(ApiCallInfo) + size + extra_size);
        if (need > kRingSize / 4)
        {
            return false;
        }

        if (!lane.ring)
        {
            lane.ring.reset(new uint8_t[kRingSize]);
            lane.other  = (&lane == &record_) ? &gpu_ : &record_;
            lane.thread = std::thread([this, &lane]() { Run(lane); });
        }

        uint64_t       head      = lane.head.load(std::memory_order_relaxed);
        const uint64_t pos       = head & kRingMask;
        const uint64_t tail_room = kRingSize - pos;
        const uint64_t total     = (need > tail_room) ? tail_room + need : need;

        for (uint64_t spins = 0; kRingSize - (head - lane.tail.load(std::memory_order_acquire)) < total; ++spins)
        {
            Relax();
            if (spins > kSpinCount)
            {
                std::this_thread::yield();
            }
        }

        if (need > tail_room)
        {
            Entry* wrap   = reinterpret_cast<Entry*>(lane.ring.get() + pos);
            wrap->size    = static_cast<uint32_t>(tail_room);
            wrap->call_id = kWrapId;
            head += tail_room;
        }

        uint8_t* dst        = lane.ring.get() + (head & kRingMask);
        Entry*   entry      = reinterpret_cast<Entry*>(dst);
        entry->size           = static_cast<uint32_t>(need);
        entry->call_id        = static_cast<uint32_t>(call_id);
        entry->parameter_size = static_cast<uint32_t>(size);
        entry->kind           = kind;
        entry->dep            = dep;
        entry->pool           = pool;
        std::memcpy(dst + sizeof(Entry), &call_info, sizeof(ApiCallInfo));
        std::memcpy(dst + sizeof(Entry) + sizeof(ApiCallInfo), parameters, size);
        if (extra_size != 0)
        {
            std::memcpy(dst + sizeof(Entry) + sizeof(ApiCallInfo) + size, extra, extra_size);
        }

        ++lane.enqueued;
        head += need;
        lane.head.store(head, std::memory_order_seq_cst);
        if (lane.sleeping.load(std::memory_order_seq_cst))
        {
            lane.head.notify_one();
        }
        return true;
    }

    void DrainLane(Lane& lane)
    {
        const uint64_t head = lane.head.load(std::memory_order_relaxed);
        for (uint64_t spins = 0; lane.tail.load(std::memory_order_acquire) != head; ++spins)
        {
            Relax();
            if (spins > kSpinCount)
            {
                std::this_thread::yield();
            }
        }
    }

    void DrainAll()
    {
        DrainLane(record_);
        DrainLane(gpu_);
    }

    void Stop(Lane& lane)
    {
        if (lane.thread.joinable())
        {
            DrainLane(lane);
            lane.stop.store(true, std::memory_order_seq_cst);

            // An empty entry wakes the thread, which then sees the stop request
            uint64_t head  = lane.head.load(std::memory_order_relaxed);
            Entry*   entry = reinterpret_cast<Entry*>(lane.ring.get() + (head & kRingMask));
            entry->size    = sizeof(Entry);
            entry->call_id = kWrapId;
            lane.head.store(head + sizeof(Entry), std::memory_order_seq_cst);
            lane.head.notify_one();
            lane.thread.join();
        }
    }

    struct ParkedReset
    {
        format::ApiCallId    call_id{};
        ApiCallInfo          call_info{};
        std::vector<uint8_t> parameters;
    };

    // The GPU lane replays the reset of a command pool once the submissions that use the pool are done
    void RunParkedReset(Lane& lane, format::HandleId pool_id)
    {
        auto it = parked_resets_.find(pool_id);
        if (it == parked_resets_.end())
        {
            return;
        }
        tracker_.BeforePoolReset(pool_id);
        lane.decoder.DecodeFunctionCall(it->second.call_id, it->second.call_info, it->second.parameters.data(), it->second.parameters.size());
        parked_resets_.erase(it);
    }

    void Run(Lane& lane)
    {
        uint64_t tail       = 0;
        uint64_t done       = 0;
        uint64_t other_done = 0; // Last seen progress of the other lane
        for (;;)
        {
            uint64_t head = lane.head.load(std::memory_order_acquire);
            if (head == tail)
            {
                bool ready = false;
                for (uint64_t spins = 0; spins < kSpinCount; ++spins)
                {
                    Relax();
                    if (lane.head.load(std::memory_order_acquire) != tail)
                    {
                        ready = true;
                        break;
                    }
                }
                if (!ready)
                {
                    lane.sleeping.store(true, std::memory_order_seq_cst);
                    if (lane.head.load(std::memory_order_seq_cst) == tail)
                    {
                        if (lane.stop.load(std::memory_order_seq_cst))
                        {
                            lane.sleeping.store(false, std::memory_order_seq_cst);
                            break;
                        }
                        lane.head.wait(tail, std::memory_order_seq_cst);
                    }
                    lane.sleeping.store(false, std::memory_order_seq_cst);
                }
                continue;
            }

            const uint8_t* src   = lane.ring.get() + (tail & kRingMask);
            const Entry*   entry = reinterpret_cast<const Entry*>(src);
            if (entry->call_id != kWrapId)
            {
                for (uint64_t spins = 0; other_done < entry->dep; ++spins)
                {
                    other_done = lane.other->done.load(std::memory_order_acquire);
                    if (other_done >= entry->dep)
                    {
                        break;
                    }
                    Relax();
                    if (spins > kSpinCount)
                    {
                        std::this_thread::yield();
                    }
                }

                DecodeAllocator::Begin();

                const uint8_t* parameters = src + sizeof(Entry) + sizeof(ApiCallInfo);
                ApiCallInfo call_info;
                std::memcpy(&call_info, src + sizeof(Entry), sizeof(ApiCallInfo));
                const auto call_id = static_cast<format::ApiCallId>(entry->call_id);
                switch (entry->kind)
                {
                    case kFill:
                    {
                        const uint64_t* header = reinterpret_cast<const uint64_t*>(parameters);
                        tracker_.BeforeFill(header[0], header[1], header[2]);
                        lane.decoder.DispatchFillMemoryCommand(
                            0, header[0], header[1], header[2], parameters + entry->parameter_size);
                        break;
                    }
                    case kParkedReset:
                    {
                        ParkedReset& parked = parked_resets_[entry->pool];
                        parked.call_id      = call_id;
                        parked.call_info    = call_info;
                        parked.parameters.assign(parameters, parameters + entry->parameter_size);
                        break;
                    }
                    case kUnparkReset:
                        RunParkedReset(lane, entry->pool);
                        break;
                    case kFlush:
                        tracker_.Flush();
                        while (!parked_resets_.empty())
                        {
                            RunParkedReset(lane, parked_resets_.begin()->first);
                        }
                        break;
                    default:
                        if (&lane == &record_)
                        {
                            observer_.NoteCall(call_id, parameters, entry->parameter_size);
                        }
                        else if (VulkanLaneGpuObserver::Follows(call_id))
                        {
                            gpu_peek_.DecodeFunctionCall(call_id, call_info, parameters, entry->parameter_size);
                        }
                        lane.decoder.DecodeFunctionCall(call_id, call_info, parameters, entry->parameter_size);
                        break;
                }
                DecodeAllocator::End();
                ++done;
                lane.done.store(done, std::memory_order_release);
            }
            tail += entry->size;
            lane.tail.store(tail, std::memory_order_release);
        }
        DecodeAllocator::DestroyInstance();
    }

    VulkanReplayConsumerBase* consumer_;
    VulkanLaneTracker         tracker_;
    VulkanLaneRecordObserver  observer_;
    VulkanLaneGpuObserver     gpu_observer_;
    VulkanDecoder             gpu_peek_;
    bool                      lanes_off_{ false };
    bool                      flush_needed_{ false };
    Lane                      record_;
    Lane                      gpu_;
    struct PoolState
    {
        uint64_t reset_seq{ 0 };         // The gpu lane entry that reset the pool last
        uint64_t epoch{ 0 };             // The number of resets
        bool     reset_parked{ false };  // The last reset has not been replayed
    };
    std::map<format::HandleId, PoolState> pools_;
    std::unordered_map<format::HandleId, ParkedReset> parked_resets_; // Gpu lane only
    std::map<format::HandleId, uint64_t>  begun_epoch_; // Per command buffer: the pool epoch of its last begin, plus one
};

GFXRECON_END_NAMESPACE(decode)
GFXRECON_END_NAMESPACE(gfxrecon)

#endif // GFXRECON_DECODE_VULKAN_LANE_DECODER_H
