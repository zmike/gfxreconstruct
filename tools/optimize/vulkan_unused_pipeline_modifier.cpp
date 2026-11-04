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

#include "vulkan_unused_pipeline_modifier.h"

#include "decode/vulkan_pnext_node.h"
#include "util/logging.h"

#include <algorithm>
#include <utility>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)

namespace
{

// A graphics pipeline library part is dropped like any other pipeline unless a kept pipeline links it.
bool PipelineIsLibraryPart(const VkGraphicsPipelineCreateInfo* create_info)
{
    GFXRECON_UNREFERENCED_PARAMETER(create_info);
    return false;
}

// A compute pipeline library part is kept, because the pass does not follow how it is linked.
bool PipelineIsLibraryPart(const VkComputePipelineCreateInfo* create_info)
{
    return (create_info->flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) != 0;
}

} // namespace

template <typename Args>
void VulkanUnusedPipelineModifier::RecordPipelineCreate(const decode::ApiCallInfo& call_info, Args& args)
{
    if (IsModificationPass() || args.pPipelines.IsNull() || args.pCreateInfos.IsNull())
    {
        return;
    }

    CreateCall& call = create_calls_[call_info.index];

    const format::HandleId* pipeline_ids = args.pPipelines.GetPointer();
    call.pipelines.assign(pipeline_ids, pipeline_ids + args.pPipelines.GetLength());

    const auto* infos = args.pCreateInfos.GetMetaStructPointer();
    for (size_t i = 0; i < args.pCreateInfos.GetLength(); ++i)
    {
        const auto& info = infos[i];

        if (info.basePipelineHandle != format::kNullHandleId)
        {
            referenced_pipelines_.insert(info.basePipelineHandle);
        }

        if (PipelineIsLibraryPart(info.decoded_value))
        {
            // Keep every pipeline of the call.
            referenced_pipelines_.insert(call.pipelines.begin(), call.pipelines.end());
        }

        CollectLibraries(info, call);
        CollectModules(info, call);
    }
}

void VulkanUnusedPipelineModifier::CollectLibraries(const decode::Decoded_VkGraphicsPipelineCreateInfo& info,
                                                    CreateCall&                                         call)
{
    const auto* libraries = decode::GetPNextMetaStruct<decode::Decoded_VkPipelineLibraryCreateInfoKHR>(info.pNext);
    if (libraries != nullptr && !libraries->pLibraries.IsNull())
    {
        const format::HandleId* ids = libraries->pLibraries.GetPointer();
        call.libraries.insert(call.libraries.end(), ids, ids + libraries->pLibraries.GetLength());
    }
}

void VulkanUnusedPipelineModifier::CollectLibraries(const decode::Decoded_VkComputePipelineCreateInfo& info,
                                                    CreateCall&                                        call)
{
    GFXRECON_UNREFERENCED_PARAMETER(info);
    GFXRECON_UNREFERENCED_PARAMETER(call);
}

void VulkanUnusedPipelineModifier::CollectModules(const decode::Decoded_VkGraphicsPipelineCreateInfo& info,
                                                  CreateCall&                                         call)
{
    if (info.pStages == nullptr)
    {
        return;
    }
    const auto* stages = info.pStages->GetMetaStructPointer();
    for (size_t i = 0; i < info.pStages->GetLength(); ++i)
    {
        if (stages[i].module != format::kNullHandleId)
        {
            call.modules.push_back(stages[i].module);
        }
    }
}

void VulkanUnusedPipelineModifier::CollectModules(const decode::Decoded_VkComputePipelineCreateInfo& info,
                                                  CreateCall&                                        call)
{
    if (info.stage != nullptr && info.stage->module != format::kNullHandleId)
    {
        call.modules.push_back(info.stage->module);
    }
}

void VulkanUnusedPipelineModifier::Process_vkCreateGraphicsPipelines(const decode::ApiCallInfo&               call_info,
                                                                     decode::args::CreateGraphicsPipelines& args)
{
    if (IsModificationPass())
    {
        if (dropped_create_blocks_.contains(call_info.index))
        {
            CurrentBlockEdit().DropBlock();
        }
        return;
    }
    RecordPipelineCreate(call_info, args);
}

void VulkanUnusedPipelineModifier::Process_vkCreateComputePipelines(const decode::ApiCallInfo&              call_info,
                                                                    decode::args::CreateComputePipelines& args)
{
    if (IsModificationPass())
    {
        if (dropped_create_blocks_.contains(call_info.index))
        {
            CurrentBlockEdit().DropBlock();
        }
        return;
    }
    RecordPipelineCreate(call_info, args);
}

void VulkanUnusedPipelineModifier::Process_vkCmdBindPipeline(const decode::ApiCallInfo&      call_info,
                                                             decode::args::CmdBindPipeline& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!IsModificationPass())
    {
        referenced_pipelines_.insert(args.pipeline);
    }
}

void VulkanUnusedPipelineModifier::Process_vkCmdBindPipelineShaderGroupNV(
    const decode::ApiCallInfo& call_info, decode::args::CmdBindPipelineShaderGroupNV& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!IsModificationPass())
    {
        referenced_pipelines_.insert(args.pipeline);
    }
}

void VulkanUnusedPipelineModifier::Process_vkDestroyPipeline(const decode::ApiCallInfo&      call_info,
                                                             decode::args::DestroyPipeline& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (IsModificationPass() && dropped_pipelines_.contains(args.pipeline))
    {
        CurrentBlockEdit().DropBlock();
    }
}

void VulkanUnusedPipelineModifier::Process_vkCreateShaderModule(const decode::ApiCallInfo&         call_info,
                                                                decode::args::CreateShaderModule& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (IsModificationPass() && !args.pShaderModule.IsNull() &&
        dropped_modules_.contains(*args.pShaderModule.GetPointer()))
    {
        CurrentBlockEdit().DropBlock();
    }
}

void VulkanUnusedPipelineModifier::Process_vkDestroyShaderModule(const decode::ApiCallInfo&          call_info,
                                                                 decode::args::DestroyShaderModule& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (IsModificationPass() && dropped_modules_.contains(args.shaderModule))
    {
        CurrentBlockEdit().DropBlock();
    }
}

void VulkanUnusedPipelineModifier::Process_vkGetShaderModuleIdentifierEXT(
    const decode::ApiCallInfo& call_info, decode::args::GetShaderModuleIdentifierEXT& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!IsModificationPass())
    {
        referenced_modules_.insert(args.shaderModule);
    }
}

bool VulkanUnusedPipelineModifier::CanOptimize()
{
    // A kept pipeline keeps the libraries it links. Repeat until no call adds a library.
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (const auto& [index, call] : create_calls_)
        {
            if (std::any_of(call.pipelines.begin(), call.pipelines.end(), [&](format::HandleId pipeline) {
                    return referenced_pipelines_.contains(pipeline);
                }))
            {
                for (format::HandleId library : call.libraries)
                {
                    changed = referenced_pipelines_.insert(library).second || changed;
                }
            }
        }
    }

    // A create call goes only if none of its pipelines is referenced.
    std::unordered_set<format::HandleId> kept_modules;
    for (const auto& [index, call] : create_calls_)
    {
        bool referenced = call.pipelines.empty() ||
                          std::any_of(call.pipelines.begin(), call.pipelines.end(), [&](format::HandleId pipeline) {
                              return referenced_pipelines_.contains(pipeline);
                          });

        if (referenced)
        {
            kept_modules.insert(call.modules.begin(), call.modules.end());
        }
        else
        {
            dropped_create_blocks_.insert(index);
            dropped_pipelines_.insert(call.pipelines.begin(), call.pipelines.end());
        }
    }

    // A module goes if no kept pipeline names it. A module no create call names is left alone.
    for (const auto& [index, call] : create_calls_)
    {
        for (format::HandleId module : call.modules)
        {
            if (!kept_modules.contains(module) && !referenced_modules_.contains(module))
            {
                dropped_modules_.insert(module);
            }
        }
    }

    GFXRECON_LOG_INFO("Unused pipelines: dropping %zu of %zu create call(s), %zu pipeline(s) and %zu shader module(s).",
                      dropped_create_blocks_.size(),
                      create_calls_.size(),
                      dropped_pipelines_.size(),
                      dropped_modules_.size());

    return !dropped_create_blocks_.empty();
}

GFXRECON_END_NAMESPACE(gfxrecon)
