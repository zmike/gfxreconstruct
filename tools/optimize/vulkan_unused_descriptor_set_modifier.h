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


#ifndef GFXRECON_VULKAN_UNUSED_DESCRIPTOR_SET_MODIFIER_H
#define GFXRECON_VULKAN_UNUSED_DESCRIPTOR_SET_MODIFIER_H

#include "vulkan_modifier_base.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)

/**
 * @brief   Drops the allocation, updates, and freeing of descriptor sets that no command buffer binds.
 *
 * A trimmed capture allocates and fills every descriptor set of the traced application, tens of
 * thousands of them, while a frame binds a fraction. A set nothing binds cannot affect the replayed
 * frames, so the vkAllocateDescriptorSets, vkUpdateDescriptorSets, vkUpdateDescriptorSetWithTemplate,
 * and vkFreeDescriptorSets calls that only name such sets are dropped.
 *
 * A set is live when a vkCmdBindDescriptorSets* call names it. A call that names a live set keeps all
 * the sets it names, because the call cannot be shortened without re-encoding it, and the sets such a
 * call copies from stay live as well. Push descriptors name no set and are not affected.
 *
 * Resources only a dropped set referenced are not unreferenced by this pass; running the optimizer on
 * its own output finds them.
 */
class VulkanUnusedDescriptorSetModifier : public VulkanModifierBase
{
  public:
    bool CanOptimize() override;
    bool DropsObjects() const override { return true; }
    void CollectDroppedHandles(std::unordered_set<format::HandleId>& handles) const override
    {
        for (uint64_t index : dropped_blocks_)
        {
            const SetCall& call = calls_.at(index);
            if (call.allocate)
            {
                handles.insert(call.written.begin(), call.written.end());
            }
        }
    }

    void Process_vkAllocateDescriptorSets(const decode::ApiCallInfo&            call_info,
                                          decode::args::AllocateDescriptorSets& args) override;
    void Process_vkFreeDescriptorSets(const decode::ApiCallInfo&        call_info,
                                      decode::args::FreeDescriptorSets& args) override;
    void Process_vkUpdateDescriptorSets(const decode::ApiCallInfo&          call_info,
                                        decode::args::UpdateDescriptorSets& args) override;
    void Process_vkUpdateDescriptorSetWithTemplate(const decode::ApiCallInfo&                     call_info,
                                                   decode::args::UpdateDescriptorSetWithTemplate& args) override;
    void Process_vkUpdateDescriptorSetWithTemplateKHR(const decode::ApiCallInfo&                        call_info,
                                                      decode::args::UpdateDescriptorSetWithTemplateKHR& args) override;
    void Process_vkCmdBindDescriptorSets(const decode::ApiCallInfo&           call_info,
                                         decode::args::CmdBindDescriptorSets& args) override;
    void Process_vkCmdBindDescriptorSets2(const decode::ApiCallInfo&            call_info,
                                          decode::args::CmdBindDescriptorSets2& args) override;
    void Process_vkCmdBindDescriptorSets2KHR(const decode::ApiCallInfo&               call_info,
                                             decode::args::CmdBindDescriptorSets2KHR& args) override;

  private:
    /// The sets one call names, and whether the call writes (and so keeps) or only reads them.
    struct SetCall
    {
        std::vector<format::HandleId> written;
        std::vector<format::HandleId> read;
        bool                          allocate{ false }; ///< An allocate call: its sets exist only through it.
    };

    void RecordCall(uint64_t block_index, std::vector<format::HandleId> written, std::vector<format::HandleId> read = {});
    void RecordBind(const format::HandleId* sets, size_t count);
    void DropIfDead(uint64_t block_index);

    std::unordered_map<uint64_t, SetCall> calls_;
    std::unordered_set<format::HandleId>  live_sets_;

    // Decided by CanOptimize() at the end of the scan pass.
    std::unordered_set<uint64_t> dropped_blocks_;
};

GFXRECON_END_NAMESPACE(gfxrecon)

#endif // GFXRECON_VULKAN_UNUSED_DESCRIPTOR_SET_MODIFIER_H
