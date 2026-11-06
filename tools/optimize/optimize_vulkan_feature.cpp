/*
** Copyright (c) 2020-2026 LunarG, Inc.
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

#include "optimize_vulkan_feature.h"

#if defined(GFXRECON_ENABLE_VULKAN)
// The modifier writes its block through gfxrecon_encode, we need to apply the same guard here
#include "vulkan_aliasing_group_modifier.h"
#include "vulkan_dropped_object_name_modifier.h"
#include "vulkan_unused_descriptor_set_modifier.h"
#include "vulkan_unused_pipeline_modifier.h"
#endif
#include "vulkan_file_optimizer.h"
#include "decode/file_processor.h"
#include "generated/generated_vulkan_referenced_block_consumer.h"
#include "generated/generated_vulkan_referenced_resource_consumer.h"
#include "util/api_version_info.h"
#include "util/feature_module_registry.h"
#include "util/logging.h"

#include <cinttypes>
#include <filesystem>
#include <memory>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(optimize)

GFXR_UTIL_REGISTER_FEATURE_CREATOR(OptimizeFeature, OptimizeVulkanFeature)

// Writing the resource aliasing groups block changes where a replayer places aliased resources, so
// every existing caller of this tool gets that change by default and needs a way back.
constexpr char kNoAliasingMetadata[] = "--no-aliasing-metadata";

// Dropping pipelines and descriptor sets changes which objects a replay creates, so each has a way back.
constexpr char kKeepUnusedPipelines[]      = "--keep-unused-pipelines";
constexpr char kKeepUnusedDescriptorSets[] = "--keep-unused-descriptor-sets";

std::vector<util::FeatureOptionDesc> OptimizeVulkanFeature::GetOptionDescs() const
{
    return { { .name          = "",
               .description   = { "Do not detect aliased resources and do not write the resource aliasing",
                                  "groups meta-data block, and remove one the input already has. Replay",
                                  "then places aliased resources one bind at a time, as it does for a",
                                  "capture this tool has not seen." },
               .has_argument  = false,
               .trigger_names = kNoAliasingMetadata },
             { .name          = "",
               .description   = { "Create graphics and compute pipelines that no command buffer binds, and the",
                                  "shader modules only those pipelines use. By default they are not created,",
                                  "which for a trimmed capture removes most of the driver compiles at replay." },
               .has_argument  = false,
               .trigger_names = kKeepUnusedPipelines },
             { .name          = "",
               .description   = { "Allocate and update descriptor sets that no command buffer binds. By default",
                                  "they are not, which for a trimmed capture removes most of the descriptor",
                                  "updates at replay." },
               .has_argument  = false,
               .trigger_names = kKeepUnusedDescriptorSets } };
}

std::string OptimizeVulkanFeature::CompiledHeaderVersionString() const
{
#if defined(GFXRECON_ENABLE_VULKAN)
    return util::GetVulkanHeaderVersionString();
#else
    return "";
#endif
}

void OptimizeVulkanFeature::RegisterDetectionDecoder(decode::FileProcessor& file_processor, uint64_t block_limit)
{
    // Destroy decoder before consumer: decoder holds a raw pointer to consumer.
    detection_decoder_.reset();
    detection_consumer_ = std::make_unique<decode::VulkanDetectionConsumer>(block_limit);
    detection_decoder_  = std::make_unique<decode::VulkanDecoder>();
    detection_decoder_->AddConsumer(detection_consumer_.get());
    file_processor.AddDecoder(detection_decoder_.get());
}

bool OptimizeVulkanFeature::WasDetected() const
{
    return detection_consumer_ != nullptr && detection_consumer_->WasVulkanAPIDetected();
}

bool OptimizeVulkanFeature::ShouldRun(const util::ArgumentParser& args) const
{
    return WasDetected();
}

bool OptimizeVulkanFeature::ScanInput(const std::string&                    input_filename,
                                      const util::ArgumentParser&           args,
                                      std::unordered_set<format::HandleId>& unreferenced_ids,
                                      VulkanFileOptimizer::Modifiers&       modifiers,
                                      bool&                                 dropped_objects)
{
    dropped_objects = false;

    // Modifiers are constructed here, then dropped again if the scan finds nothing for them to do.
    VulkanFileOptimizer::Modifiers candidates;

#if defined(GFXRECON_ENABLE_VULKAN)
    // Runs even with the opt-out, to strip a block an earlier run wrote.
    candidates.push_back(std::make_unique<VulkanAliasingGroupModifier>(!args.IsOptionSet(kNoAliasingMetadata)));
    if (!args.IsOptionSet(kKeepUnusedPipelines))
    {
        candidates.push_back(std::make_unique<VulkanUnusedPipelineModifier>());
    }
    if (!args.IsOptionSet(kKeepUnusedDescriptorSets))
    {
        candidates.push_back(std::make_unique<VulkanUnusedDescriptorSetModifier>());
    }
#endif

    decode::FileProcessor file_processor;
    if (!file_processor.Initialize(input_filename))
    {
        return false;
    }

    decode::VulkanDecoder                    decoder;
    decode::VulkanReferencedResourceConsumer resref_consumer;
    decoder.AddConsumer(&resref_consumer);
    for (auto& candidate : candidates)
    {
        decoder.AddConsumer(candidate.get());
    }
    file_processor.AddDecoder(&decoder);
    file_processor.ProcessAllFrames();

    if (file_processor.GetErrorState() != decode::BlockIOError::kErrorNone)
    {
        GFXRECON_WRITE_CONSOLE("A failure has occurred during file processing");
        return false;
    }

    // Command buffer submissions provide resource usage for frameless captures.
    if (file_processor.GetCurrentFrameNumber() == 0 && !resref_consumer.WasCommandBufferSubmissionSeen())
    {
        GFXRECON_WRITE_CONSOLE("File did not contain any frames or command buffer submissions");
        return false;
    }

    resref_consumer.GetReferencedHandleIds(nullptr, &unreferenced_ids);

    std::unordered_set<format::HandleId> dropped_handles;
    for (auto& candidate : candidates)
    {
        if (candidate->CanOptimize())
        {
            dropped_objects = dropped_objects || candidate->DropsObjects();
            candidate->CollectDroppedHandles(dropped_handles);
            modifiers.push_back(std::move(candidate));
        }
    }

#if defined(GFXRECON_ENABLE_VULKAN)
    // Replay never creates a dropped object, so a debug name or tag that names it is a call on an invalid handle.
    if (!dropped_handles.empty())
    {
        modifiers.push_back(std::make_unique<VulkanDroppedObjectNameModifier>(std::move(dropped_handles)));
    }
#endif
    return true;
}

bool OptimizeVulkanFeature::WriteOptimizedFile(const std::string&                          input_filename,
                                               const std::string&                          output_filename,
                                               const std::unordered_set<format::HandleId>& unreferenced_ids,
                                               VulkanFileOptimizer::Modifiers              modifiers)
{
    uint64_t                     num_blocks = 0;
    std::unordered_set<uint64_t> unreferenced_blocks;

    if (!unreferenced_ids.empty())
    {
        // Collect the block indices that correspond to unreferenced resources.
        decode::FileProcessor file_processor;
        if (!file_processor.Initialize(input_filename))
        {
            return false;
        }

        decode::VulkanDecoder                 decoder;
        decode::VulkanReferencedBlockConsumer block_ref_consumer(unreferenced_ids);
        decoder.AddConsumer(&block_ref_consumer);
        file_processor.AddDecoder(&decoder);
        file_processor.ProcessAllFrames();

        if (file_processor.GetErrorState() != decode::BlockIOError::kErrorNone)
        {
            GFXRECON_WRITE_CONSOLE("A failure has occurred during file processing");
            return false;
        }

        num_blocks          = file_processor.GetCurrentBlockIndex();
        unreferenced_blocks = block_ref_consumer.GetUnreferencedBlocks();
    }

    // Stream the input to the output, dropping unreferenced blocks and applying the modifiers.
    VulkanFileOptimizer file_optimizer(unreferenced_ids, unreferenced_blocks, std::move(modifiers));
    if (!file_optimizer.Initialize(input_filename, output_filename))
    {
        return false;
    }

    file_optimizer.Process();

    if (file_optimizer.GetErrorState() != decode::BlockIOError::kErrorNone)
    {
        GFXRECON_WRITE_CONSOLE("A failure has occurred during file processing");
        return false;
    }

    GFXRECON_WRITE_CONSOLE(
        "Resource filtering complete - Removed %zu / %" PRIu64 " blocks", unreferenced_blocks.size(), num_blocks);
    GFXRECON_WRITE_CONSOLE("\tOriginal file size: %" PRIu64 " bytes", file_optimizer.GetNumBytesRead());
    GFXRECON_WRITE_CONSOLE("\tOptimized file size: %" PRIu64 " bytes", file_optimizer.GetNumBytesWritten());
    return true;
}

bool OptimizeVulkanFeature::Optimize(const std::string&          input_filename,
                                     const std::string&          output_filename,
                                     const util::ArgumentParser& args)
{
    // Dropping a descriptor set unreferences the resources only it named, so one pass can make the next
    // one find more. Each pass reads the previous pass's output, until a pass drops no pipelines or
    // descriptor sets. That last pass also removes the initialization data the earlier drops freed.
    constexpr uint32_t kMaxPasses = 16;

    std::string current = input_filename;

    for (uint32_t pass = 1; pass <= kMaxPasses; ++pass)
    {
        GFXRECON_WRITE_CONSOLE("Scanning Vulkan file %s for optimizations (pass %u).", current.c_str(), pass);

        std::unordered_set<format::HandleId> unreferenced_ids;
        VulkanFileOptimizer::Modifiers       modifiers;
        bool                                 dropped_objects = false;
        if (!ScanInput(current, args, unreferenced_ids, modifiers, dropped_objects))
        {
            if (current != input_filename)
            {
                std::error_code ec;
                std::filesystem::remove(current, ec);
            }
            return false;
        }

        if (unreferenced_ids.empty() && modifiers.empty())
        {
            if (pass == 1)
            {
                GFXRECON_WRITE_CONSOLE("Nothing to optimize. A new file will not be created.");
                return true;
            }
            break;
        }

        if (!unreferenced_ids.empty())
        {
            GFXRECON_WRITE_CONSOLE(
                "Writing optimized file, removing initialization data for %" PRIu64 " unused resources.",
                unreferenced_ids.size());
        }

        const bool        last   = !dropped_objects || (pass == kMaxPasses);
        const std::string target = last ? output_filename : output_filename + ".pass" + std::to_string(pass);
        const bool        ok     = WriteOptimizedFile(current, target, unreferenced_ids, std::move(modifiers));

        std::error_code ec;
        if (current != input_filename)
        {
            std::filesystem::remove(current, ec);
        }
        if (!ok)
        {
            std::filesystem::remove(target, ec);
            return false;
        }

        current = target;
        if (last)
        {
            if (dropped_objects)
            {
                GFXRECON_WRITE_CONSOLE("Stopped after %u passes; the output may still contain unused objects.",
                                       kMaxPasses);
            }
            return true;
        }
    }

    // A later pass found nothing to do, so the previous pass's output is the result.
    std::error_code ec;
    std::filesystem::rename(current, output_filename, ec);
    if (ec)
    {
        GFXRECON_WRITE_CONSOLE("Failed to rename %s to %s: %s",
                               current.c_str(),
                               output_filename.c_str(),
                               ec.message().c_str());
        return false;
    }
    return true;
}

GFXRECON_END_NAMESPACE(optimize)
GFXRECON_END_NAMESPACE(gfxrecon)
