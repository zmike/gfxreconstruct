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


#ifndef GFXRECON_VULKAN_UNUSED_PIPELINE_MODIFIER_H
#define GFXRECON_VULKAN_UNUSED_PIPELINE_MODIFIER_H

#include "vulkan_modifier_base.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)

/**
 * @brief   Drops the creation of graphics and compute pipelines that nothing ever binds, with the
 *          shader modules only those pipelines used.
 *
 * Replaying a trimmed capture creates every pipeline of the traced application, usually thousands, and
 * each one is a driver compile. Only the pipelines a recorded command buffer binds can affect the
 * replayed frames.
 *
 * The scan pass follows every pipeline from its create call to the places that refer to it: a
 * vkCmdBindPipeline, another pipeline's basePipelineHandle, or the pipeline libraries a kept pipeline
 * links. A create call is dropped when none of the pipelines it makes is referenced. Its vkDestroyPipeline calls
 * go with it. A shader module is dropped when no kept create call names it, together with its
 * vkDestroyShaderModule.
 *
 * A create call that makes both referenced and unreferenced pipelines is kept whole, because the call
 * cannot be shortened without re-encoding its create-infos.
 *
 * The pass does not model every command that names a pipeline handle: ray tracing pipelines are not
 * followed, and a compute pipeline library is never dropped (see PipelineIsLibraryPart in the source).
 * Replay reports a missing handle if a capture needs more.
 */
class VulkanUnusedPipelineModifier : public VulkanModifierBase
{
  public:
    bool CanOptimize() override;
    bool DropsObjects() const override { return true; }

    void Process_vkCreateGraphicsPipelines(const decode::ApiCallInfo&               call_info,
                                           decode::args::CreateGraphicsPipelines& args) override;
    void Process_vkCreateComputePipelines(const decode::ApiCallInfo&              call_info,
                                          decode::args::CreateComputePipelines& args) override;
    void Process_vkCmdBindPipeline(const decode::ApiCallInfo& call_info, decode::args::CmdBindPipeline& args) override;
    void Process_vkCmdBindPipelineShaderGroupNV(const decode::ApiCallInfo&                    call_info,
                                                decode::args::CmdBindPipelineShaderGroupNV& args) override;
    void Process_vkDestroyPipeline(const decode::ApiCallInfo& call_info, decode::args::DestroyPipeline& args) override;
    void Process_vkCreateShaderModule(const decode::ApiCallInfo&         call_info,
                                      decode::args::CreateShaderModule& args) override;
    void Process_vkDestroyShaderModule(const decode::ApiCallInfo&          call_info,
                                       decode::args::DestroyShaderModule& args) override;
    void Process_vkGetShaderModuleIdentifierEXT(const decode::ApiCallInfo&                    call_info,
                                                decode::args::GetShaderModuleIdentifierEXT& args) override;

  private:
    /// One vkCreate*Pipelines call, keyed by its block index.
    struct CreateCall
    {
        std::vector<format::HandleId> pipelines;
        std::vector<format::HandleId> modules;
        std::vector<format::HandleId> libraries;
    };

    template <typename Args>
    void RecordPipelineCreate(const decode::ApiCallInfo& call_info, Args& args);

    static void CollectLibraries(const decode::Decoded_VkGraphicsPipelineCreateInfo& info, CreateCall& call);
    static void CollectLibraries(const decode::Decoded_VkComputePipelineCreateInfo& info, CreateCall& call);
    static void CollectModules(const decode::Decoded_VkGraphicsPipelineCreateInfo& info, CreateCall& call);
    static void CollectModules(const decode::Decoded_VkComputePipelineCreateInfo& info, CreateCall& call);

    std::unordered_map<uint64_t, CreateCall>        create_calls_;
    std::unordered_set<format::HandleId>            referenced_pipelines_;
    std::unordered_set<format::HandleId>            referenced_modules_;

    // Decided by CanOptimize() at the end of the scan pass.
    std::unordered_set<uint64_t>         dropped_create_blocks_;
    std::unordered_set<format::HandleId> dropped_pipelines_;
    std::unordered_set<format::HandleId> dropped_modules_;
};

GFXRECON_END_NAMESPACE(gfxrecon)

#endif // GFXRECON_VULKAN_UNUSED_PIPELINE_MODIFIER_H
