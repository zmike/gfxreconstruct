/*
** Copyright (c) 2026 Valve Corporation
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

#include "decode/common_object_info_table.h"
#include "decode/custom_vulkan_struct_decoders.h"
#include "decode/struct_pointer_decoder.h"
#include "util/defines.h"
#include "vulkan_object_info.h"

#include <cstdint>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(decode)

void HandleDescriptorUpdate(CommonObjectInfoTable&                                    object_info_table,
                            uint32_t                                                  descriptor_write_count,
                            const StructPointerDecoder<Decoded_VkWriteDescriptorSet>* p_descriptor_writes,
                            VulkanDescriptorSetInfo::VulkanDescriptorBindingsInfo*    descriptor_bindings = nullptr,
                            VkShaderStageFlags stage_flags  = VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM,
                            bool               buffers_only = false);

// True for the descriptor types whose tracked state address replacement reads.
bool IsAddressReplacementDescriptorType(VkDescriptorType type);

// Calls fn(binding_info, array_index, element_index) for each of the write's descriptorCount elements, moving on to
// the next binding when an element lies beyond the count of the current one. binding_it is the entry of the write's
// dstBinding in bindings.
template <typename Fn>
void ForEachWrittenDescriptor(VulkanDescriptorSetInfo::VulkanDescriptorBindingsInfo&          bindings,
                              VulkanDescriptorSetInfo::VulkanDescriptorBindingsInfo::iterator binding_it,
                              const VkWriteDescriptorSet&                                     write,
                              VkShaderStageFlags                                              stage_flags,
                              Fn&&                                                            fn)
{
    uint32_t arr_idx = write.dstArrayElement;

    for (uint32_t i = 0; i < write.descriptorCount; ++i)
    {
        if (binding_it->second.count != 0 && arr_idx >= binding_it->second.count)
        {
            // Advance to next non zero count binding
            do
            {
                ++binding_it;
            } while (binding_it != bindings.end() && binding_it->second.count == 0);

            if (binding_it == bindings.end())
            {
                break;
            }
            arr_idx = 0;
        }

        auto& current_binding_info = binding_it->second;

        current_binding_info.desc_type = write.descriptorType;
        if (stage_flags != VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM)
        {
            current_binding_info.stage_flags = stage_flags;
        }

        fn(current_binding_info, arr_idx, i);

        ++arr_idx;
    }
}

// Records the buffer descriptors of a write whose type is one of the buffer types, as HandleDescriptorUpdate does.
// buffer_infos holds the tracked buffer of each of the write's descriptorCount elements.
void HandleBufferDescriptorWrite(VulkanDescriptorSetInfo::VulkanDescriptorBindingsInfo& descriptor_bindings,
                                 const VkWriteDescriptorSet&                            write,
                                 const VulkanBufferInfo* const*                         buffer_infos);

VkShaderStageFlags PipelineBindPointToShaderStageFlags(VkPipelineBindPoint bind_point);

GFXRECON_END_NAMESPACE(decode)
GFXRECON_END_NAMESPACE(gfxrecon)