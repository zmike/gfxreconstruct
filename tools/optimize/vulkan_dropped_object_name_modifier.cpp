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


#include "vulkan_dropped_object_name_modifier.h"

GFXRECON_BEGIN_NAMESPACE(gfxrecon)

void VulkanDroppedObjectNameModifier::DropIfDropped(uint64_t object)
{
    if (IsModificationPass() && dropped_handles_.contains(object))
    {
        CurrentBlockEdit().DropBlock();
    }
}

void VulkanDroppedObjectNameModifier::Process_vkSetDebugUtilsObjectNameEXT(
    const decode::ApiCallInfo& call_info, decode::args::SetDebugUtilsObjectNameEXT& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!args.pNameInfo.IsNull())
    {
        DropIfDropped(args.pNameInfo.GetMetaStructPointer()->objectHandle);
    }
}

void VulkanDroppedObjectNameModifier::Process_vkSetDebugUtilsObjectTagEXT(
    const decode::ApiCallInfo& call_info, decode::args::SetDebugUtilsObjectTagEXT& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!args.pTagInfo.IsNull())
    {
        DropIfDropped(args.pTagInfo.GetMetaStructPointer()->objectHandle);
    }
}

void VulkanDroppedObjectNameModifier::Process_vkDebugMarkerSetObjectNameEXT(
    const decode::ApiCallInfo& call_info, decode::args::DebugMarkerSetObjectNameEXT& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!args.pNameInfo.IsNull())
    {
        DropIfDropped(args.pNameInfo.GetMetaStructPointer()->object);
    }
}

void VulkanDroppedObjectNameModifier::Process_vkDebugMarkerSetObjectTagEXT(
    const decode::ApiCallInfo& call_info, decode::args::DebugMarkerSetObjectTagEXT& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);

    if (!args.pTagInfo.IsNull())
    {
        DropIfDropped(args.pTagInfo.GetMetaStructPointer()->object);
    }
}

GFXRECON_END_NAMESPACE(gfxrecon)
