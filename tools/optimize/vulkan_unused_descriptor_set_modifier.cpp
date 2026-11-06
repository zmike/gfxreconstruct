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


#include "vulkan_unused_descriptor_set_modifier.h"

#include "util/logging.h"

#include <algorithm>
#include <utility>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)

void VulkanUnusedDescriptorSetModifier::RecordCall(uint64_t                      block_index,
                                                   std::vector<format::HandleId> written,
                                                   std::vector<format::HandleId> read)
{
    calls_[block_index] = { std::move(written), std::move(read) };
}

void VulkanUnusedDescriptorSetModifier::RecordBind(const format::HandleId* sets, size_t count)
{
    if (sets != nullptr)
    {
        live_sets_.insert(sets, sets + count);
    }
}

void VulkanUnusedDescriptorSetModifier::DropIfDead(uint64_t block_index)
{
    if (dropped_blocks_.contains(block_index))
    {
        CurrentBlockEdit().DropBlock();
    }
}

void VulkanUnusedDescriptorSetModifier::Process_vkAllocateDescriptorSets(const decode::ApiCallInfo&            call_info,
                                                                         decode::args::AllocateDescriptorSets& args)
{
    if (IsModificationPass())
    {
        DropIfDead(call_info.index);
    }
    else if (!args.pDescriptorSets.IsNull())
    {
        const format::HandleId* sets = args.pDescriptorSets.GetPointer();
        RecordCall(call_info.index, { sets, sets + args.pDescriptorSets.GetLength() });
        calls_[call_info.index].allocate = true;
    }
}

void VulkanUnusedDescriptorSetModifier::Process_vkFreeDescriptorSets(const decode::ApiCallInfo&        call_info,
                                                                     decode::args::FreeDescriptorSets& args)
{
    if (IsModificationPass())
    {
        DropIfDead(call_info.index);
    }
    else if (!args.pDescriptorSets.IsNull())
    {
        const format::HandleId* sets = args.pDescriptorSets.GetPointer();
        RecordCall(call_info.index, { sets, sets + args.pDescriptorSets.GetLength() });
    }
}

void VulkanUnusedDescriptorSetModifier::Process_vkUpdateDescriptorSets(const decode::ApiCallInfo&          call_info,
                                                                       decode::args::UpdateDescriptorSets& args)
{
    if (IsModificationPass())
    {
        DropIfDead(call_info.index);
        return;
    }

    std::vector<format::HandleId> written;
    std::vector<format::HandleId> read;

    if (!args.pDescriptorWrites.IsNull())
    {
        const auto* writes = args.pDescriptorWrites.GetMetaStructPointer();
        for (size_t i = 0; i < args.pDescriptorWrites.GetLength(); ++i)
        {
            written.push_back(writes[i].dstSet);
        }
    }
    if (!args.pDescriptorCopies.IsNull())
    {
        const auto* copies = args.pDescriptorCopies.GetMetaStructPointer();
        for (size_t i = 0; i < args.pDescriptorCopies.GetLength(); ++i)
        {
            written.push_back(copies[i].dstSet);
            read.push_back(copies[i].srcSet);
        }
    }
    RecordCall(call_info.index, std::move(written), std::move(read));
}

void VulkanUnusedDescriptorSetModifier::Process_vkUpdateDescriptorSetWithTemplate(
    const decode::ApiCallInfo& call_info, decode::args::UpdateDescriptorSetWithTemplate& args)
{
    if (IsModificationPass())
    {
        DropIfDead(call_info.index);
    }
    else
    {
        RecordCall(call_info.index, { args.descriptorSet });
    }
}

void VulkanUnusedDescriptorSetModifier::Process_vkUpdateDescriptorSetWithTemplateKHR(
    const decode::ApiCallInfo& call_info, decode::args::UpdateDescriptorSetWithTemplateKHR& args)
{
    if (IsModificationPass())
    {
        DropIfDead(call_info.index);
    }
    else
    {
        RecordCall(call_info.index, { args.descriptorSet });
    }
}

void VulkanUnusedDescriptorSetModifier::Process_vkCmdBindDescriptorSets(const decode::ApiCallInfo& call_info,
                                                                        decode::args::CmdBindDescriptorSets& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!IsModificationPass() && !args.pDescriptorSets.IsNull())
    {
        RecordBind(args.pDescriptorSets.GetPointer(), args.pDescriptorSets.GetLength());
    }
}

void VulkanUnusedDescriptorSetModifier::Process_vkCmdBindDescriptorSets2(const decode::ApiCallInfo& call_info,
                                                                         decode::args::CmdBindDescriptorSets2& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!IsModificationPass() && !args.pBindDescriptorSetsInfo.IsNull())
    {
        const auto* info = args.pBindDescriptorSetsInfo.GetMetaStructPointer();
        if (!info->pDescriptorSets.IsNull())
        {
            RecordBind(info->pDescriptorSets.GetPointer(), info->pDescriptorSets.GetLength());
        }
    }
}

void VulkanUnusedDescriptorSetModifier::Process_vkCmdBindDescriptorSets2KHR(
    const decode::ApiCallInfo& call_info, decode::args::CmdBindDescriptorSets2KHR& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!IsModificationPass() && !args.pBindDescriptorSetsInfo.IsNull())
    {
        const auto* info = args.pBindDescriptorSetsInfo.GetMetaStructPointer();
        if (!info->pDescriptorSets.IsNull())
        {
            RecordBind(info->pDescriptorSets.GetPointer(), info->pDescriptorSets.GetLength());
        }
    }
}

bool VulkanUnusedDescriptorSetModifier::CanOptimize()
{
    // A call that names a live set keeps every set it names, and the sets it reads from. Repeat until
    // no call adds a set.
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (const auto& [index, call] : calls_)
        {
            if (!std::any_of(call.written.begin(), call.written.end(), [&](format::HandleId set) {
                    return live_sets_.contains(set);
                }))
            {
                continue;
            }

            for (const auto* sets : { &call.written, &call.read })
            {
                for (format::HandleId set : *sets)
                {
                    changed = live_sets_.insert(set).second || changed;
                }
            }
        }
    }

    for (const auto& [index, call] : calls_)
    {
        bool keeps = call.written.empty() || std::any_of(call.written.begin(), call.written.end(), [&](format::HandleId set) {
                         return live_sets_.contains(set);
                     });
        if (!keeps)
        {
            dropped_blocks_.insert(index);
        }
    }

    GFXRECON_LOG_INFO("Unused descriptor sets: dropping %zu of %zu call(s), %zu set(s) live.",
                      dropped_blocks_.size(),
                      calls_.size(),
                      live_sets_.size());

    return !dropped_blocks_.empty();
}

GFXRECON_END_NAMESPACE(gfxrecon)
