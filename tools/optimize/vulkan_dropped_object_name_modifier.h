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


#ifndef GFXRECON_VULKAN_DROPPED_OBJECT_NAME_MODIFIER_H
#define GFXRECON_VULKAN_DROPPED_OBJECT_NAME_MODIFIER_H

#include "vulkan_modifier_base.h"

#include <unordered_set>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)

/**
 * @brief   Drops the debug name and tag calls of objects whose creation another modifier dropped.
 *
 * Naming an object that replay never creates is a call on an invalid handle, which crashes some
 * drivers. This modifier runs only in the modification pass, with the handles the other modifiers
 * report through VulkanModifierBase::CollectDroppedHandles().
 */
class VulkanDroppedObjectNameModifier : public VulkanModifierBase
{
  public:
    explicit VulkanDroppedObjectNameModifier(std::unordered_set<format::HandleId> dropped_handles) :
        dropped_handles_(std::move(dropped_handles))
    {}

    // The optimizer constructs this modifier only when there are handles to drop.
    bool CanOptimize() override { return true; }

    void Process_vkSetDebugUtilsObjectNameEXT(const decode::ApiCallInfo&               call_info,
                                              decode::args::SetDebugUtilsObjectNameEXT& args) override;
    void Process_vkSetDebugUtilsObjectTagEXT(const decode::ApiCallInfo&              call_info,
                                             decode::args::SetDebugUtilsObjectTagEXT& args) override;
    void Process_vkDebugMarkerSetObjectNameEXT(const decode::ApiCallInfo&                 call_info,
                                               decode::args::DebugMarkerSetObjectNameEXT& args) override;
    void Process_vkDebugMarkerSetObjectTagEXT(const decode::ApiCallInfo&                call_info,
                                              decode::args::DebugMarkerSetObjectTagEXT& args) override;

  private:
    void DropIfDropped(uint64_t object);

    std::unordered_set<format::HandleId> dropped_handles_;
};

GFXRECON_END_NAMESPACE(gfxrecon)

#endif // GFXRECON_VULKAN_DROPPED_OBJECT_NAME_MODIFIER_H
