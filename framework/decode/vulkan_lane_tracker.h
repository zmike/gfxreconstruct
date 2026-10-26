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

#ifndef GFXRECON_DECODE_VULKAN_LANE_TRACKER_H
#define GFXRECON_DECODE_VULKAN_LANE_TRACKER_H

#include "decode/value_decoder.h"
#include "decode/vulkan_rebind_allocator.h"
#include "decode/vulkan_replay_consumer_base.h"
#include "format/api_call_id.h"
#include "generated/generated_vulkan_consumer.h"
#include "generated/generated_vulkan_decoder_args.h"
#include "util/defines.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(decode)

// A range of captured memory that a command buffer reads
struct LaneRange
{
    uint64_t memory;
    uint64_t begin;
    uint64_t end;
};

// What the GPU can read when it executes a command buffer, through the buffers that host writes can reach
struct LaneCbReads
{
    std::vector<LaneRange>        ranges;      // Sorted by memory and offset, and merged
    std::vector<format::HandleId> sets;        // Descriptor sets bound, whose contents at submission are read
    std::vector<format::HandleId> query_pools; // Query pools used
    std::vector<VkCommandBuffer>  secondaries; // Command buffers executed
    format::HandleId              pool_id{ format::kNullHandleId };
    bool                          all{ false }; // Reads something that is not tracked
};

// Follows what the submissions in flight read, so that the waits for the GPU in the stream can be put off until a
// host write needs them. Everything here runs on the gpu lane except for the publication of command buffers.
//
// A submission has a number in submission order. The host writes to memory, descriptor sets, and command pools check
// the number of the last submission that used them against the waits that were skipped. Only a single queue is
// followed, and a wait for a timeline semaphore value is skipped only if a submission that signals it is known: its
// signal operation follows every submission before it, so the wait covers all of those.
class VulkanLaneTracker
{
  public:
    explicit VulkanLaneTracker(VulkanReplayConsumerBase* consumer) : consumer_(consumer) {}

    // Dispatch thread: replay of a call that changes the object tables
    void NoteInlineCall() { epoch_.fetch_add(1, std::memory_order_relaxed); }

    // A semaphore is destroyed, and a new one may have its handle. Called with the gpu lane idle.
    void ForgetSemaphore(VkSemaphore semaphore) { semaphores_.erase(semaphore); }
    void ForgetSemaphores() { semaphores_.clear(); }

    // The host signals a semaphore to a value, which the queue submissions do not account for. Waits for values up to
    // it run when they come. Called with the gpu lane idle.
    void NoteHostSignal(VkSemaphore semaphore, uint64_t value)
    {
        SemaphoreSignals& entry = semaphores_[semaphore];
        entry.host_value        = std::max(entry.host_value, value);
    }

    uint32_t Epoch() const { return epoch_.load(std::memory_order_relaxed); }

    // Record lane: a query pool that takes the compacted sizes of acceleration structures, which replay reads back
    void NoteCompactPool(format::HandleId pool_id)
    {
        std::lock_guard<std::mutex> lock(cb_mutex_);
        compact_pools_.insert(pool_id);
        any_compact_pool_.store(true, std::memory_order_release);
    }

    bool IsCompactPool(format::HandleId pool_id)
    {
        // The set is empty unless the capture reads back acceleration structure sizes
        if (!any_compact_pool_.load(std::memory_order_acquire))
        {
            return false;
        }
        std::lock_guard<std::mutex> lock(cb_mutex_);
        return compact_pools_.contains(pool_id);
    }

    // Record lane: a command buffer that is complete
    void PublishCommandBuffer(VkCommandBuffer command_buffer, std::shared_ptr<const LaneCbReads> reads)
    {
        std::lock_guard<std::mutex> lock(cb_mutex_);
        command_buffers_[command_buffer] = std::move(reads);
    }

    // The range a read of a buffer covers. A length of VK_WHOLE_SIZE is up to the end of the buffer. False if the
    // buffer is not followed.
    static bool ResolveBufferRange(const VulkanReplayConsumerBase::LaneBufferBase& base,
                                   VkDeviceSize                                    offset,
                                   VkDeviceSize                                    length,
                                   LaneRange*                                      range)
    {
        if (!base.valid)
        {
            return false;
        }
        offset         = std::min<uint64_t>(offset, base.size);
        const uint64_t end = ((length == VK_WHOLE_SIZE) || (length > base.size - offset)) ? base.size : offset + length;
        *range         = { base.memory, base.offset + offset, base.offset + end };
        return true;
    }

    // ---- gpu lane ----

    void OnSubmit(
        VkQueue queue, VkDevice device, uint32_t queue_family, std::span<const VkSubmitInfo> submits, VkFence fence)
    {
        NoteQueue(device, queue_family);
        for (const VkSubmitInfo& submit : submits)
        {
            const uint64_t*                         values = nullptr;
            const VkTimelineSemaphoreSubmitInfo*    timeline = nullptr;
            for (const VkBaseInStructure* next = reinterpret_cast<const VkBaseInStructure*>(submit.pNext); next != nullptr;
                 next                          = next->pNext)
            {
                if (next->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)
                {
                    timeline = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
                    values   = timeline->pSignalSemaphoreValues;
                }
            }

            const uint64_t n = BeginSubmission(queue);
            for (uint32_t i = 0; i < submit.commandBufferCount; ++i)
            {
                StampCommandBuffer(submit.pCommandBuffers[i], n, 0);
            }
            if ((timeline != nullptr) && (values != nullptr))
            {
                for (uint32_t i = 0; (i < submit.signalSemaphoreCount) && (i < timeline->signalSemaphoreValueCount); ++i)
                {
                    NoteSignal(submit.pSignalSemaphores[i], values[i], n);
                }
            }
        }
        NoteFence(fence);
    }

    void OnSubmit(
        VkQueue queue, VkDevice device, uint32_t queue_family, std::span<const VkSubmitInfo2> submits, VkFence fence)
    {
        NoteQueue(device, queue_family);
        for (const VkSubmitInfo2& submit : submits)
        {
            const uint64_t n = BeginSubmission(queue);
            for (uint32_t i = 0; i < submit.commandBufferInfoCount; ++i)
            {
                StampCommandBuffer(submit.pCommandBufferInfos[i].commandBuffer, n, 0);
            }
            for (uint32_t i = 0; i < submit.signalSemaphoreInfoCount; ++i)
            {
                NoteSignal(submit.pSignalSemaphoreInfos[i].semaphore, submit.pSignalSemaphoreInfos[i].value, n);
            }
        }
        NoteFence(fence);
    }

    // Takes a wait for timeline semaphores to be reached on the host if the submissions it covers are known, to run
    // when something needs them. False if the wait has to run now.
    bool DeferWait(VkDevice device, const VkSemaphoreWaitInfo& info)
    {
        if (disabled_ || ((info.flags & VK_SEMAPHORE_WAIT_ANY_BIT) != 0) || (info.semaphoreCount == 0))
        {
            return false;
        }

        uint64_t cover = 0;
        for (uint32_t i = 0; i < info.semaphoreCount; ++i)
        {
            uint64_t semaphore_cover = 0;
            if (!Cover(info.pSemaphores[i], info.pValues[i], &semaphore_cover))
            {
                return false;
            }
            cover = std::max(cover, semaphore_cover);
        }

        if (cover <= done_)
        {
            return true;
        }

        timeline_seen_    = true;
        PendingWait& wait = pending_.emplace_back();
        wait.device       = device;
        wait.cover        = cover;
        wait.semaphores.assign(info.pSemaphores, info.pSemaphores + info.semaphoreCount);
        wait.values.assign(info.pValues, info.pValues + info.semaphoreCount);
        if (pending_.size() > kMaxPending)
        {
            Complete(pending_.front().cover);
        }
        return true;
    }

    // A host reset of a query pool becomes a command the next submission follows. False if it has to run on the host.
    bool StageQueryReset(VulkanRebindAllocator* allocator, VkQueryPool pool, uint32_t first, uint32_t count)
    {
        if (!CanStage(allocator) || !allocator->StageQueryReset(pool, first, count))
        {
            return false;
        }
        NoteStagedAllocator(allocator);
        query_reset_staged_ = true;
        return true;
    }

    bool TakeQueryResetStaged() { return std::exchange(query_reset_staged_, false); }

    // The host is about to write captured memory. True if the write is to be staged, which puts the copy to the buffers
    // off to the next submission.
    bool BeforeFill(uint64_t memory_id, uint64_t offset, uint64_t size)
    {
        auto it = memories_.find(memory_id);
        if (it == memories_.end())
        {
            // A memory not written before may have been read by anything in flight
            memories_.emplace(memory_id, MemoryState());
            if (!pending_.empty())
            {
                Complete(last_submission_);
            }
            return false;
        }

        MemoryState&   memory  = it->second;
        const uint32_t version = Version();
        if (memory.version != version)
        {
            // Binding resources to the memory is replayed without the lanes, but mapping it is not
            memory.version = version;
            memory.mapped  = consumer_->GetLaneMappedWriteTarget(memory_id, &memory.mapped_offset, &memory.only_buffers);
        }

        const bool     placed = memory.mapped && memory.only_buffers;
        const uint64_t begin  = memory.mapped_offset + offset;

        // Staged writes that the copies have not been submitted for are overwritten by them
        bool staged_overlap = false;
        for (const auto& range : memory.staged)
        {
            if (!placed || ((begin < range.second) && (range.first < begin + size)))
            {
                staged_overlap = true;
                break;
            }
        }

        // Writes that cannot be placed, and memory with images in it, may reach anything
        uint64_t last = 0;
        if (!pending_.empty())
        {
            last = all_stamp_;
            if (!placed)
            {
                last = std::max(last, last_submission_);
            }
            else
            {
                last = std::max(last, MaxStamp(memory.stamps, begin, begin + size));
            }
        }
        if (!staged_overlap && (last <= done_))
        {
            return false;
        }

        if (placed)
        {
            VulkanResourceAllocator::MemoryData allocator_data{};
            VulkanRebindAllocator*              allocator = consumer_->GetLaneFillAllocator(memory_id, &allocator_data);
            uint64_t                            bytes     = 0;
            if (CanStage(allocator) && allocator->CanStageFill(allocator_data, begin, size, &bytes))
            {
                if (bytes != 0)
                {
                    if (memory.staged.empty())
                    {
                        staged_memories_.push_back(memory_id);
                    }
                    if (staged_overlap)
                    {
                        // Buffers that share the bytes are not ordered by their own copies
                        allocator->StageOrder();
                    }
                    memory.staged.emplace_back(begin, begin + size);
                    NoteStagedAllocator(allocator);
                }
                return bytes != 0;
            }
        }

        // The copies are submitted to wait for
        FlushStaged();
        if (staged_overlap)
        {
            last = last_submission_;
        }
        if (last > done_)
        {
            Complete(last);
        }
        return false;
    }

    // A memory was mapped or unmapped
    void NoteMappingChange() { ++mapping_version_; }

    // The host is about to write a descriptor set
    void BeforeSetWrite(format::HandleId set_id)
    {
        if (pending_.empty())
        {
            return;
        }
        // A submission whose reads are not known could have used any set
        const uint64_t stamp = std::max(SetFor(set_id).stamp, all_stamp_);
        if (stamp > done_)
        {
            Complete(stamp);
        }
    }

    // The host is about to reset a command pool
    void BeforePoolReset(format::HandleId pool_id)
    {
        if (pending_.empty())
        {
            return;
        }
        auto           it    = pool_stamps_.find(pool_id);
        const uint64_t stamp = std::max((it != pool_stamps_.end()) ? it->second : 0, all_stamp_);
        if (stamp > done_)
        {
            Complete(stamp);
        }
    }

    // A command buffer is about to be begun again, which resets it
    void BeforeCommandBufferBegin(VkCommandBuffer command_buffer)
    {
        auto it = cb_stamps_.find(command_buffer);
        // A secondary command buffer that an unfollowed primary executed has no stamp of its own
        const uint64_t stamp = std::max((it != cb_stamps_.end()) ? it->second : last_submission_, all_stamp_);
        if (stamp > done_)
        {
            Complete(stamp);
        }
    }

    // The host is about to reset fences, or reset or read a query pool
    void BeforeFenceReset(VkFence fence)
    {
        auto it = fence_stamps_.find(fence);
        if ((it != fence_stamps_.end()) && (it->second > done_))
        {
            Complete(it->second);
        }
    }

    void BeforeQueryPoolUse(format::HandleId query_pool_id)
    {
        auto           it    = query_stamps_.find(query_pool_id);
        const uint64_t stamp = std::max((it != query_stamps_.end()) ? it->second : 0, all_stamp_);
        if (stamp > done_)
        {
            Complete(stamp);
        }
    }

    // Anything else that may depend on the GPU being done
    void Flush()
    {
        FlushStaged();
        if (!pending_.empty())
        {
            Complete(UINT64_MAX);
        }
    }

    void OnBufferDescriptor(format::HandleId set_id,
                            uint32_t         binding,
                            uint32_t         element,
                            format::HandleId buffer_id,
                            VkDeviceSize     offset,
                            VkDeviceSize     range,
                            bool             dynamic)
    {
        Slot slot;
        if (buffer_id != format::kNullHandleId)
        {
            const auto& base = BufferBase(buffer_id);
            // A dynamic offset moves the range up to the end of the buffer
            if (!ResolveBufferRange(base, offset, dynamic ? VK_WHOLE_SIZE : range, &slot.range))
            {
                slot.unknown = true;
            }
            else
            {
                slot.has_range = true;
            }
        }
        SetSlot(set_id, binding, element, slot);
    }

    void OnTexelDescriptor(format::HandleId set_id, uint32_t binding, uint32_t element, format::HandleId view_id)
    {
        Slot slot;
        if (view_id != format::kNullHandleId)
        {
            const VulkanBufferViewInfo* view = consumer_->GetLaneObjectTable().GetVkBufferViewInfo(view_id);
            if (view == nullptr)
            {
                slot.unknown = true;
            }
            else
            {
                const auto& base = BufferBase(view->buffer_id);
                if (!ResolveBufferRange(base, view->offset, view->range, &slot.range))
                {
                    slot.unknown = true;
                }
                else
                {
                    slot.has_range = true;
                }
            }
        }
        SetSlot(set_id, binding, element, slot);
    }

    // A descriptor that reads no buffer
    void OnOtherDescriptor(format::HandleId set_id, uint32_t binding, uint32_t element)
    {
        SetSlot(set_id, binding, element, Slot());
    }

    void OnUnknownSetWrite(format::HandleId set_id)
    {
        BeforeSetWrite(set_id);
        SetFor(set_id).unknown = true;
    }


  private:
    static constexpr uint32_t kGranuleShift = 9;
    static constexpr size_t   kMaxPending   = 32;

    struct Slot
    {
        uint32_t  binding{ 0 };
        uint32_t  element{ 0 };
        bool      has_range{ false };
        bool      unknown{ false };
        LaneRange range{};
    };

    struct SetState
    {
        std::vector<Slot> slots; // Sorted by binding and element
        uint64_t          stamp{ 0 };
        bool              unknown{ false };
    };

    struct PendingWait
    {
        VkDevice                 device{ VK_NULL_HANDLE };
        uint64_t                 cover{ 0 };
        std::vector<VkSemaphore> semaphores;
        std::vector<uint64_t>    values;
    };

    // A memory that the host has written
    struct MemoryState
    {
        std::vector<uint32_t> stamps; // The last submission per granule that read it
        std::vector<std::pair<uint64_t, uint64_t>> staged; // Staged writes not submitted the copies for
        uint32_t              version{ UINT32_MAX };
        uint64_t              mapped_offset{ 0 };
        bool                  mapped{ false };
        bool                  only_buffers{ false };
    };

    struct Signal
    {
        uint64_t value;
        uint64_t submission;
    };

    struct SemaphoreSignals
    {
        std::deque<Signal> signals;
        uint64_t           trimmed_value{ 0 }; // Waits up to this value are done
        bool               trimmed{ false };
        uint64_t           host_value{ 0 }; // Waits up to this value may be met by the host
    };

    uint32_t Version() const { return epoch_.load(std::memory_order_relaxed) + mapping_version_; }

    void CheckEpoch()
    {
        const uint32_t epoch = epoch_.load(std::memory_order_relaxed);
        if (epoch != cached_epoch_)
        {
            cached_epoch_ = epoch;
            buffer_bases_.clear();
            last_buffer_id_ = format::kNullHandleId;
        }
    }

    const VulkanReplayConsumerBase::LaneBufferBase& BufferBase(format::HandleId buffer_id)
    {
        CheckEpoch();
        if (buffer_id != last_buffer_id_)
        {
            auto it = buffer_bases_.find(buffer_id);
            if (it == buffer_bases_.end())
            {
                it = buffer_bases_.emplace(buffer_id, consumer_->GetLaneBufferBase(buffer_id)).first;
            }
            last_buffer_id_   = buffer_id;
            last_buffer_base_ = &it->second;
        }
        return *last_buffer_base_;
    }

    void NoteQueue(VkDevice device, uint32_t queue_family)
    {
        // The copies of staged writes go before the submission, to the queue they were staged for
        FlushStaged();
        if (queue_ == VK_NULL_HANDLE)
        {
            queue_device_ = device;
            queue_family_ = queue_family;
        }
    }

    // Copies and resets can go to the queue the application submits to
    bool CanStage(const VulkanRebindAllocator* allocator) const
    {
        return !disabled_ && timeline_seen_ && (allocator != nullptr) && (allocator->GetDevice() == queue_device_);
    }

    void NoteStagedAllocator(VulkanRebindAllocator* allocator)
    {
        if (std::find(staged_allocators_.begin(), staged_allocators_.end(), allocator) == staged_allocators_.end())
        {
            staged_allocators_.push_back(allocator);
        }
    }

    // Submits the copies of the staged writes. They are submissions that the host waits for as it does for the
    // application's, and that write the ranges.
    void FlushStaged()
    {
        if (staged_allocators_.empty() || (queue_ == VK_NULL_HANDLE))
        {
            return;
        }
        for (VulkanRebindAllocator* allocator : staged_allocators_)
        {
            VkSemaphore semaphore = VK_NULL_HANDLE;
            uint64_t    value     = 0;
            if (!allocator->SubmitStagedFills(queue_, queue_family_, &semaphore, &value))
            {
                continue;
            }
            const uint64_t n    = BeginSubmission(queue_);
            PendingWait&   wait = pending_.emplace_back();
            wait.device         = allocator->GetDevice();
            wait.cover          = n;
            wait.semaphores.push_back(semaphore);
            wait.values.push_back(value);
            for (uint64_t memory_id : staged_memories_)
            {
                for (const auto& range : memories_[memory_id].staged)
                {
                    StampRange({ memory_id, range.first, range.second }, n);
                }
            }
        }
        for (uint64_t memory_id : staged_memories_)
        {
            memories_[memory_id].staged.clear();
        }
        staged_memories_.clear();
        staged_allocators_.clear();
        if (pending_.size() > kMaxPending)
        {
            Complete(pending_.front().cover);
        }
    }

    uint64_t BeginSubmission(VkQueue queue)
    {
        if (queue_ == VK_NULL_HANDLE)
        {
            queue_ = queue;
        }
        else if (queue_ != queue)
        {
            // Completion across queues is not ordered
            disabled_ = true;
            Flush();
        }
        last_submission_ = ++submissions_;
        return last_submission_;
    }

    void NoteFence(VkFence fence)
    {
        if (fence != VK_NULL_HANDLE)
        {
            fence_stamps_[fence] = last_submission_;
        }
    }

    void NoteSignal(VkSemaphore semaphore, uint64_t value, uint64_t submission)
    {
        SemaphoreSignals& entry = semaphores_[semaphore];
        entry.signals.push_back({ value, submission });
        while ((entry.signals.size() > 64) && (entry.signals.front().submission <= done_))
        {
            entry.trimmed_value = entry.signals.front().value;
            entry.trimmed       = true;
            entry.signals.pop_front();
        }
    }

    // The last submission that a wait for the value covers. False if it is not known.
    bool Cover(VkSemaphore semaphore, uint64_t value, uint64_t* cover) const
    {
        // Every timeline semaphore starts at zero
        if (value == 0)
        {
            *cover = 0;
            return true;
        }
        auto it = semaphores_.find(semaphore);
        if (it == semaphores_.end())
        {
            return false;
        }
        const SemaphoreSignals& entry = it->second;
        if (value <= entry.host_value)
        {
            return false;
        }
        if (entry.trimmed && (value <= entry.trimmed_value))
        {
            *cover = 0;
            return true;
        }
        // The wait ends with the first signal that reaches the value
        for (const Signal& signal : entry.signals)
        {
            if (signal.value >= value)
            {
                *cover = signal.submission;
                return true;
            }
        }
        return false;
    }

    // Runs the skipped waits up to the first that covers the submission
    void Complete(uint64_t submission)
    {
        while (!pending_.empty() && (done_ < submission))
        {
            PendingWait& wait = pending_.front();
            if (wait.cover > done_)
            {
                VkSemaphoreWaitInfo info{};
                info.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                info.semaphoreCount = static_cast<uint32_t>(wait.semaphores.size());
                info.pSemaphores    = wait.semaphores.data();
                info.pValues        = wait.values.data();
                consumer_->WaitSemaphoresNow(wait.device, info);
                done_ = wait.cover;
            }
            pending_.pop_front();
        }
    }

    static uint64_t MaxStamp(const std::vector<uint32_t>& stamps, uint64_t begin, uint64_t end)
    {
        if (end <= begin)
        {
            return 0;
        }
        const size_t first = static_cast<size_t>(begin >> kGranuleShift);
        const size_t last  = std::min(static_cast<size_t>((end - 1) >> kGranuleShift) + 1, stamps.size());
        uint32_t     stamp = 0;
        for (size_t i = first; i < last; ++i)
        {
            stamp = std::max(stamp, stamps[i]);
        }
        return stamp;
    }

    void StampRange(const LaneRange& range, uint64_t submission)
    {
        // Only memory that has been written by the host is followed
        auto it = memories_.find(range.memory);
        if ((it == memories_.end()) || (range.end <= range.begin))
        {
            return;
        }
        std::vector<uint32_t>& stamps = it->second.stamps;
        const size_t           first  = static_cast<size_t>(range.begin >> kGranuleShift);
        const size_t           last   = static_cast<size_t>((range.end - 1) >> kGranuleShift) + 1;
        if (stamps.size() < last)
        {
            stamps.resize(last, 0);
        }
        std::fill(stamps.begin() + first, stamps.begin() + last, static_cast<uint32_t>(submission));
    }

    void StampCommandBuffer(VkCommandBuffer command_buffer, uint64_t submission, uint32_t depth)
    {
        cb_stamps_[command_buffer] = submission;
        std::shared_ptr<const LaneCbReads> reads;
        {
            std::lock_guard<std::mutex> lock(cb_mutex_);
            auto                        it = command_buffers_.find(command_buffer);
            if (it != command_buffers_.end())
            {
                reads = it->second;
            }
        }
        if (!reads || (depth > 4))
        {
            all_stamp_ = submission;
            return;
        }

        if (reads->all)
        {
            all_stamp_ = submission;
        }
        for (const LaneRange& range : reads->ranges)
        {
            StampRange(range, submission);
        }
        for (format::HandleId set_id : reads->sets)
        {
            SetState& set = SetFor(set_id);
            set.stamp     = submission;
            if (set.unknown)
            {
                all_stamp_ = submission;
            }
            for (const Slot& slot : set.slots)
            {
                if (slot.unknown)
                {
                    all_stamp_ = submission;
                }
                else if (slot.has_range)
                {
                    StampRange(slot.range, submission);
                }
            }
        }
        pool_stamps_[reads->pool_id] = submission;
        for (format::HandleId query_pool_id : reads->query_pools)
        {
            query_stamps_[query_pool_id] = submission;
        }
        for (VkCommandBuffer secondary : reads->secondaries)
        {
            StampCommandBuffer(secondary, submission, depth + 1);
        }
    }

    SetState& SetFor(format::HandleId set_id)
    {
        if (set_id != last_set_id_)
        {
            last_set_id_ = set_id;
            last_set_    = &sets_[set_id];
        }
        return *last_set_;
    }

    void SetSlot(format::HandleId set_id, uint32_t binding, uint32_t element, Slot slot)
    {
        slot.binding = binding;
        slot.element = element;
        SetState& set = SetFor(set_id);
        auto      it  = std::lower_bound(set.slots.begin(), set.slots.end(), slot, [](const Slot& a, const Slot& b) {
            return (a.binding != b.binding) ? (a.binding < b.binding) : (a.element < b.element);
        });
        const bool found = (it != set.slots.end()) && (it->binding == binding) && (it->element == element);
        if (!slot.has_range && !slot.unknown)
        {
            // A descriptor that reads no buffer has nothing to stamp, so it only clears what was there
            if (found)
            {
                set.slots.erase(it);
            }
        }
        else if (found)
        {
            *it = slot;
        }
        else
        {
            set.slots.insert(it, slot);
        }
    }

    VulkanReplayConsumerBase* consumer_;

    std::atomic<uint32_t> epoch_{ 0 };
    uint32_t              cached_epoch_{ 0 };
    uint32_t              mapping_version_{ 0 };

    std::mutex                                                          cb_mutex_;
    std::unordered_map<VkCommandBuffer, std::shared_ptr<const LaneCbReads>> command_buffers_;
    std::unordered_set<format::HandleId>                                    compact_pools_;
    std::atomic<bool>                                                       any_compact_pool_{ false };

    VkQueue  queue_{ VK_NULL_HANDLE };
    std::unordered_map<VkCommandBuffer, uint64_t> cb_stamps_; // The last submission of each command buffer
    VkDevice                                      queue_device_{ VK_NULL_HANDLE };
    uint32_t                                      queue_family_{ 0 };
    bool                                          timeline_seen_{ false };
    bool                                          query_reset_staged_{ false };
    std::vector<uint64_t>                         staged_memories_;
    std::vector<VulkanRebindAllocator*>           staged_allocators_;
    bool     disabled_{ false };
    uint64_t submissions_{ 0 };
    uint64_t last_submission_{ 0 };
    uint64_t done_{ 0 };      // The last submission that is known to have completed
    uint64_t all_stamp_{ 0 }; // The last submission that read something not followed

    std::deque<PendingWait>                                  pending_;
    std::unordered_map<VkSemaphore, SemaphoreSignals>        semaphores_;
    std::unordered_map<uint64_t, MemoryState>                memories_;
    std::unordered_map<format::HandleId, SetState>           sets_;
    std::unordered_map<format::HandleId, uint64_t>           pool_stamps_;
    std::unordered_map<format::HandleId, uint64_t>           query_stamps_;
    std::unordered_map<VkFence, uint64_t>                    fence_stamps_;
    std::unordered_map<format::HandleId, VulkanReplayConsumerBase::LaneBufferBase> buffer_bases_;
    format::HandleId                                                               last_buffer_id_{ format::kNullHandleId };
    const VulkanReplayConsumerBase::LaneBufferBase*                                last_buffer_base_{ nullptr };
    format::HandleId                                                               last_set_id_{ format::kNullHandleId };
    SetState*                                                                      last_set_{ nullptr };
};

// Follows the buffers that the commands of the record lane read, as a consumer ahead of the replay consumer so that it
// sees the handle ids.
class VulkanLaneRecordObserver : public VulkanConsumer
{
  public:
    VulkanLaneRecordObserver(VulkanReplayConsumerBase* consumer, VulkanLaneTracker* tracker) :
        consumer_(consumer), tracker_(tracker)
    {}

    // Before each call of the lane: the commands that are not followed make the command buffer read everything
    void NoteCall(format::ApiCallId call_id, const uint8_t* parameters, size_t size)
    {
        CheckEpoch();
        switch (call_id)
        {
            // The calls this class follows
            case format::ApiCall_vkBeginCommandBuffer:
            case format::ApiCall_vkEndCommandBuffer:
            case format::ApiCall_vkCmdBindVertexBuffers:
            case format::ApiCall_vkCmdBindVertexBuffers2:
            case format::ApiCall_vkCmdBindIndexBuffer:
            case format::ApiCall_vkCmdBindIndexBuffer2:
            case format::ApiCall_vkCmdBindIndexBuffer2KHR:
            case format::ApiCall_vkCmdBindVertexBuffers2EXT:
            case format::ApiCall_vkCmdBindDescriptorSets:
            case format::ApiCall_vkCmdBindDescriptorSets2:
            case format::ApiCall_vkCmdBindDescriptorSets2KHR:
            case format::ApiCall_vkCmdResetQueryPool:
            case format::ApiCall_vkCmdBeginQueryIndexedEXT:
            case format::ApiCall_vkCmdEndQueryIndexedEXT:
            case format::ApiCall_vkCmdCopyQueryPoolResults:
            case format::ApiCall_vkCmdCopyBuffer:
            case format::ApiCall_vkCmdCopyBufferToImage:
            case format::ApiCall_vkCmdFillBuffer:
            case format::ApiCall_vkCmdUpdateBuffer:
            case format::ApiCall_vkCmdCopyImageToBuffer:
            case format::ApiCall_vkCmdDrawIndirect:
            case format::ApiCall_vkCmdDrawIndexedIndirect:
            case format::ApiCall_vkCmdDispatchIndirect:
            case format::ApiCall_vkCmdExecuteCommands:
            case format::ApiCall_vkCmdBeginQuery:
            case format::ApiCall_vkCmdEndQuery:
            case format::ApiCall_vkCmdWriteTimestamp:
            case format::ApiCall_vkCmdWriteTimestamp2:
            case format::ApiCall_vkCmdWriteTimestamp2KHR:
            case format::ApiCall_vkCmdWriteAccelerationStructuresPropertiesKHR:
                return;
            // The calls that read no buffer
        case format::ApiCall_vkCmdBeginDebugUtilsLabelEXT:
        case format::ApiCall_vkCmdBeginRenderPass:
        case format::ApiCall_vkCmdBeginRenderPass2:
        case format::ApiCall_vkCmdBeginRenderPass2KHR:
        case format::ApiCall_vkCmdBeginRendering:
        case format::ApiCall_vkCmdBeginRenderingKHR:
        case format::ApiCall_vkCmdBindPipeline:
        case format::ApiCall_vkCmdBindShadersEXT:
        case format::ApiCall_vkCmdBlitImage:
        case format::ApiCall_vkCmdBlitImage2:
        case format::ApiCall_vkCmdBlitImage2KHR:
        case format::ApiCall_vkCmdClearAttachments:
        case format::ApiCall_vkCmdClearColorImage:
        case format::ApiCall_vkCmdClearDepthStencilImage:
        case format::ApiCall_vkCmdCopyImage:
        case format::ApiCall_vkCmdCopyImage2:
        case format::ApiCall_vkCmdCopyImage2KHR:
        case format::ApiCall_vkCmdDebugMarkerBeginEXT:
        case format::ApiCall_vkCmdDebugMarkerEndEXT:
        case format::ApiCall_vkCmdDebugMarkerInsertEXT:
        case format::ApiCall_vkCmdDispatch:
        case format::ApiCall_vkCmdDispatchBase:
        case format::ApiCall_vkCmdDispatchBaseKHR:
        case format::ApiCall_vkCmdDraw:
        case format::ApiCall_vkCmdDrawIndexed:
        case format::ApiCall_vkCmdDrawMultiEXT:
        case format::ApiCall_vkCmdDrawMultiIndexedEXT:
        case format::ApiCall_vkCmdEndDebugUtilsLabelEXT:
        case format::ApiCall_vkCmdEndRenderPass:
        case format::ApiCall_vkCmdEndRenderPass2:
        case format::ApiCall_vkCmdEndRenderPass2KHR:
        case format::ApiCall_vkCmdEndRendering:
        case format::ApiCall_vkCmdEndRenderingKHR:
        case format::ApiCall_vkCmdInsertDebugUtilsLabelEXT:
        case format::ApiCall_vkCmdNextSubpass:
        case format::ApiCall_vkCmdNextSubpass2:
        case format::ApiCall_vkCmdNextSubpass2KHR:
        case format::ApiCall_vkCmdPipelineBarrier:
        case format::ApiCall_vkCmdPipelineBarrier2:
        case format::ApiCall_vkCmdPipelineBarrier2KHR:
        case format::ApiCall_vkCmdPushConstants:
        case format::ApiCall_vkCmdResetEvent:
        case format::ApiCall_vkCmdResetEvent2:
        case format::ApiCall_vkCmdResetEvent2KHR:
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
        case format::ApiCall_vkCmdWaitEvents:
        case format::ApiCall_vkCmdWaitEvents2:
        case format::ApiCall_vkCmdWaitEvents2KHR:
        case format::ApiCall_vkCmdPushConstants2:
        case format::ApiCall_vkCmdPushConstants2KHR:
        case format::ApiCall_vkCmdEndRendering2KHR:
        case format::ApiCall_vkCmdEndRendering2EXT:
        case format::ApiCall_vkCmdDrawMeshTasksEXT:
        case format::ApiCall_vkCmdDrawMeshTasksNV:
        case format::ApiCall_vkCmdEndConditionalRenderingEXT:
                return;
            default:
                break;
        }

        format::HandleId command_buffer_id = format::kNullHandleId;
        ValueDecoder::DecodeHandleIdValue(parameters, size, &command_buffer_id);
        GetBuild(command_buffer_id).reads.all = true;
    }

    void Process_vkBeginCommandBuffer(const ApiCallInfo& call_info, args::BeginCommandBuffer& args) override
    {
        CbBuild& build = builds_[args.commandBuffer];
        build        = CbBuild();
        build.reads.pool_id = consumer_->GetCommandBufferPoolId(args.commandBuffer);

        const VulkanCommandBufferInfo* info = consumer_->GetLaneObjectTable().GetVkCommandBufferInfo(args.commandBuffer);
        build.handle                        = (info != nullptr) ? info->handle : VK_NULL_HANDLE;
        build.reads.all                     = (info == nullptr);
        current_id_                         = args.commandBuffer;
        current_build_                      = &build;
    }

    void Process_vkEndCommandBuffer(const ApiCallInfo& call_info, args::EndCommandBuffer& args) override
    {
        CbBuild& build = GetBuild(args.commandBuffer);
        current_id_  = format::kNullHandleId;
        if (build.handle == VK_NULL_HANDLE)
        {
            return;
        }

        auto& ranges = build.reads.ranges;
        std::sort(ranges.begin(), ranges.end(), [](const LaneRange& a, const LaneRange& b) {
            return (a.memory != b.memory) ? (a.memory < b.memory) : (a.begin < b.begin);
        });
        size_t out = 0;
        for (size_t i = 0; i < ranges.size(); ++i)
        {
            if ((out > 0) && (ranges[out - 1].memory == ranges[i].memory) && (ranges[i].begin <= ranges[out - 1].end))
            {
                ranges[out - 1].end = std::max(ranges[out - 1].end, ranges[i].end);
            }
            else
            {
                ranges[out++] = ranges[i];
            }
        }
        ranges.resize(out);
        for (auto* ids : { &build.reads.sets, &build.reads.query_pools })
        {
            std::sort(ids->begin(), ids->end());
            ids->erase(std::unique(ids->begin(), ids->end()), ids->end());
        }

        tracker_->PublishCommandBuffer(build.handle, std::make_shared<const LaneCbReads>(std::move(build.reads)));
        build.reads = LaneCbReads();
    }

    void Process_vkCmdBindVertexBuffers(const ApiCallInfo& call_info, args::CmdBindVertexBuffers& args) override
    {
        BindVertexBuffers(args.commandBuffer, args.bindingCount, args.pBuffers.GetPointer(), args.pOffsets.GetPointer(), nullptr);
    }

    void Process_vkCmdBindVertexBuffers2(const ApiCallInfo& call_info, args::CmdBindVertexBuffers2& args) override
    {
        BindVertexBuffers(args.commandBuffer,
                          args.bindingCount,
                          args.pBuffers.GetPointer(),
                          args.pOffsets.GetPointer(),
                          args.pSizes.GetPointer());
    }

    void Process_vkCmdBindIndexBuffer(const ApiCallInfo& call_info, args::CmdBindIndexBuffer& args) override
    {
        AddRead(args.commandBuffer, args.buffer, args.offset, VK_WHOLE_SIZE);
    }

    void Process_vkCmdBindIndexBuffer2(const ApiCallInfo& call_info, args::CmdBindIndexBuffer2& args) override
    {
        AddRead(args.commandBuffer, args.buffer, args.offset, args.size);
    }

    void Process_vkCmdBindIndexBuffer2KHR(const ApiCallInfo& call_info, args::CmdBindIndexBuffer2KHR& args) override
    {
        AddRead(args.commandBuffer, args.buffer, args.offset, args.size);
    }

    void Process_vkCmdBindVertexBuffers2EXT(const ApiCallInfo& call_info, args::CmdBindVertexBuffers2EXT& args) override
    {
        BindVertexBuffers(args.commandBuffer,
                          args.bindingCount,
                          args.pBuffers.GetPointer(),
                          args.pOffsets.GetPointer(),
                          args.pSizes.GetPointer());
    }

    void BindDescriptorSets(format::HandleId command_buffer_id, const format::HandleId* sets, uint32_t count)
    {
        CbBuild& build = GetBuild(command_buffer_id);
        for (uint32_t i = 0; (sets != nullptr) && (i < count); ++i)
        {
            if (build.reads.sets.empty() || (build.reads.sets.back() != sets[i]))
            {
                build.reads.sets.push_back(sets[i]);
            }
        }
    }

    void BindDescriptorSets2(format::HandleId                                        command_buffer_id,
                             StructPointerDecoder<Decoded_VkBindDescriptorSetsInfo>& info)
    {
        const Decoded_VkBindDescriptorSetsInfo* meta = info.GetMetaStructPointer();
        const VkBindDescriptorSetsInfo*         bind = info.GetPointer();
        if ((meta != nullptr) && (bind != nullptr))
        {
            BindDescriptorSets(command_buffer_id, meta->pDescriptorSets.GetPointer(), bind->descriptorSetCount);
        }
    }

    void Process_vkCmdBindDescriptorSets(const ApiCallInfo& call_info, args::CmdBindDescriptorSets& args) override
    {
        BindDescriptorSets(args.commandBuffer, args.pDescriptorSets.GetPointer(), args.descriptorSetCount);
    }

    void Process_vkCmdBindDescriptorSets2(const ApiCallInfo& call_info, args::CmdBindDescriptorSets2& args) override
    {
        BindDescriptorSets2(args.commandBuffer, args.pBindDescriptorSetsInfo);
    }

    void Process_vkCmdBindDescriptorSets2KHR(const ApiCallInfo&               call_info,
                                             args::CmdBindDescriptorSets2KHR& args) override
    {
        BindDescriptorSets2(args.commandBuffer, args.pBindDescriptorSetsInfo);
    }

    void Process_vkCmdBeginQueryIndexedEXT(const ApiCallInfo& call_info, args::CmdBeginQueryIndexedEXT& args) override
    {
        AddQueryPool(args.commandBuffer, args.queryPool);
    }

    void Process_vkCmdEndQueryIndexedEXT(const ApiCallInfo& call_info, args::CmdEndQueryIndexedEXT& args) override
    {
        AddQueryPool(args.commandBuffer, args.queryPool);
    }

    void Process_vkCmdCopyQueryPoolResults(const ApiCallInfo& call_info, args::CmdCopyQueryPoolResults& args) override
    {
        AddQueryPool(args.commandBuffer, args.queryPool);
        AddRead(args.commandBuffer, args.dstBuffer, args.dstOffset, VK_WHOLE_SIZE);
    }

    void Process_vkCmdCopyBuffer(const ApiCallInfo& call_info, args::CmdCopyBuffer& args) override
    {
        const VkBufferCopy* regions = args.pRegions.GetPointer();
        for (uint32_t i = 0; (regions != nullptr) && (i < args.regionCount); ++i)
        {
            AddRead(args.commandBuffer, args.srcBuffer, regions[i].srcOffset, regions[i].size);
            // A host write must not overtake the GPU write either
            AddRead(args.commandBuffer, args.dstBuffer, regions[i].dstOffset, regions[i].size);
        }
    }

    void Process_vkCmdFillBuffer(const ApiCallInfo& call_info, args::CmdFillBuffer& args) override
    {
        AddRead(args.commandBuffer, args.dstBuffer, args.dstOffset, args.size);
    }

    void Process_vkCmdUpdateBuffer(const ApiCallInfo& call_info, args::CmdUpdateBuffer& args) override
    {
        AddRead(args.commandBuffer, args.dstBuffer, args.dstOffset, args.dataSize);
    }

    void Process_vkCmdCopyImageToBuffer(const ApiCallInfo& call_info, args::CmdCopyImageToBuffer& args) override
    {
        const VkBufferImageCopy* regions = args.pRegions.GetPointer();
        for (uint32_t i = 0; (regions != nullptr) && (i < args.regionCount); ++i)
        {
            AddRead(args.commandBuffer, args.dstBuffer, regions[i].bufferOffset, VK_WHOLE_SIZE);
        }
    }

    void Process_vkCmdCopyBufferToImage(const ApiCallInfo& call_info, args::CmdCopyBufferToImage& args) override
    {
        const VkBufferImageCopy* regions = args.pRegions.GetPointer();
        for (uint32_t i = 0; (regions != nullptr) && (i < args.regionCount); ++i)
        {
            AddRead(args.commandBuffer, args.srcBuffer, regions[i].bufferOffset, VK_WHOLE_SIZE);
        }
    }

    void Process_vkCmdDrawIndirect(const ApiCallInfo& call_info, args::CmdDrawIndirect& args) override
    {
        AddRead(args.commandBuffer, args.buffer, args.offset, VK_WHOLE_SIZE);
    }

    void Process_vkCmdDrawIndexedIndirect(const ApiCallInfo& call_info, args::CmdDrawIndexedIndirect& args) override
    {
        AddRead(args.commandBuffer, args.buffer, args.offset, VK_WHOLE_SIZE);
    }

    void Process_vkCmdDispatchIndirect(const ApiCallInfo& call_info, args::CmdDispatchIndirect& args) override
    {
        AddRead(args.commandBuffer, args.buffer, args.offset, VK_WHOLE_SIZE);
    }

    void Process_vkCmdBeginQuery(const ApiCallInfo& call_info, args::CmdBeginQuery& args) override
    {
        AddQueryPool(args.commandBuffer, args.queryPool);
    }

    void Process_vkCmdEndQuery(const ApiCallInfo& call_info, args::CmdEndQuery& args) override
    {
        AddQueryPool(args.commandBuffer, args.queryPool);
    }

    void Process_vkCmdWriteTimestamp(const ApiCallInfo& call_info, args::CmdWriteTimestamp& args) override
    {
        AddQueryPool(args.commandBuffer, args.queryPool);
    }

    void Process_vkCmdWriteTimestamp2(const ApiCallInfo& call_info, args::CmdWriteTimestamp2& args) override
    {
        AddQueryPool(args.commandBuffer, args.queryPool);
    }

    void Process_vkCmdWriteTimestamp2KHR(const ApiCallInfo& call_info, args::CmdWriteTimestamp2KHR& args) override
    {
        AddQueryPool(args.commandBuffer, args.queryPool);
    }

    void Process_vkCmdWriteAccelerationStructuresPropertiesKHR(
        const ApiCallInfo& call_info, args::CmdWriteAccelerationStructuresPropertiesKHR& args) override
    {
        tracker_->NoteCompactPool(args.queryPool);
        AddQueryPool(args.commandBuffer, args.queryPool);
    }

    void Process_vkCmdResetQueryPool(const ApiCallInfo& call_info, args::CmdResetQueryPool& args) override
    {
        AddQueryPool(args.commandBuffer, args.queryPool);
    }

    void Process_vkCmdExecuteCommands(const ApiCallInfo& call_info, args::CmdExecuteCommands& args) override
    {
        CbBuild&                build = GetBuild(args.commandBuffer);
        const format::HandleId* ids   = args.pCommandBuffers.GetPointer();
        for (uint32_t i = 0; (ids != nullptr) && (i < args.commandBufferCount); ++i)
        {
            const VulkanCommandBufferInfo* info = consumer_->GetLaneObjectTable().GetVkCommandBufferInfo(ids[i]);
            if (info == nullptr)
            {
                build.reads.all = true;
            }
            else
            {
                build.reads.secondaries.push_back(info->handle);
            }
        }
    }

  private:
    struct CbBuild
    {
        LaneCbReads     reads;
        VkCommandBuffer handle{ VK_NULL_HANDLE };
    };

    void CheckEpoch()
    {
        const uint32_t epoch = tracker_->Epoch();
        if (epoch != cached_epoch_)
        {
            cached_epoch_ = epoch;
            buffer_bases_.clear();
            current_id_ = format::kNullHandleId;
        }
    }

    // A command buffer that was not begun through the lane reads everything
    CbBuild& GetBuild(format::HandleId command_buffer_id)
    {
        if (command_buffer_id != current_id_)
        {
            auto it = builds_.find(command_buffer_id);
            if (it == builds_.end())
            {
                it                  = builds_.emplace(command_buffer_id, CbBuild()).first;
                it->second.reads.all = true;
                const VulkanCommandBufferInfo* info = consumer_->GetLaneObjectTable().GetVkCommandBufferInfo(command_buffer_id);
                it->second.handle                   = (info != nullptr) ? info->handle : VK_NULL_HANDLE;
            }
            current_id_    = command_buffer_id;
            current_build_ = &it->second;
        }
        return *current_build_;
    }

    void AddQueryPool(format::HandleId command_buffer_id, format::HandleId query_pool_id)
    {
        auto& pools = GetBuild(command_buffer_id).reads.query_pools;
        if (pools.empty() || (pools.back() != query_pool_id))
        {
            pools.push_back(query_pool_id);
        }
    }

    void AddRead(format::HandleId command_buffer_id, format::HandleId buffer_id, VkDeviceSize offset, VkDeviceSize length)
    {
        CbBuild& build = GetBuild(command_buffer_id);
        if (buffer_id == format::kNullHandleId)
        {
            return;
        }

        auto it = buffer_bases_.find(buffer_id);
        if (it == buffer_bases_.end())
        {
            it = buffer_bases_.emplace(buffer_id, consumer_->GetLaneBufferBase(buffer_id)).first;
        }

        LaneRange range;
        if (!VulkanLaneTracker::ResolveBufferRange(it->second, offset, length, &range))
        {
            build.reads.all = true;
            return;
        }

        // Skip the repeats of the last few
        const size_t count = build.reads.ranges.size();
        for (size_t i = 0; i < std::min<size_t>(count, 8); ++i)
        {
            const LaneRange& other = build.reads.ranges[count - 1 - i];
            if ((other.memory == range.memory) && (other.begin <= range.begin) && (other.end >= range.end))
            {
                return;
            }
        }
        build.reads.ranges.push_back(range);
    }

    void BindVertexBuffers(format::HandleId        command_buffer_id,
                           uint32_t                count,
                           const format::HandleId* buffers,
                           const VkDeviceSize*     offsets,
                           const VkDeviceSize*     sizes)
    {
        for (uint32_t i = 0; (buffers != nullptr) && (i < count); ++i)
        {
            AddRead(command_buffer_id, buffers[i], (offsets != nullptr) ? offsets[i] : 0, (sizes != nullptr) ? sizes[i] : VK_WHOLE_SIZE);
        }
    }

    VulkanReplayConsumerBase* consumer_;
    VulkanLaneTracker*        tracker_;
    uint32_t                  cached_epoch_{ 0 };
    format::HandleId          current_id_{ format::kNullHandleId };
    CbBuild*                  current_build_{ nullptr };

    std::unordered_map<format::HandleId, CbBuild>                                   builds_;
    std::unordered_map<format::HandleId, VulkanReplayConsumerBase::LaneBufferBase> buffer_bases_;
};

// Calls of the gpu lane that wait for the submissions they depend on, as a consumer of a decoder of their own
class VulkanLaneGpuObserver : public VulkanConsumer
{
  public:
    VulkanLaneGpuObserver(VulkanReplayConsumerBase* consumer, VulkanLaneTracker* tracker) :
        consumer_(consumer), tracker_(tracker)
    {}

    // The calls that the dispatch thread replays with the lanes idle
    static bool FollowsInline(format::ApiCallId call_id)
    {
        return (call_id == format::ApiCall_vkSignalSemaphore) || (call_id == format::ApiCall_vkSignalSemaphoreKHR) ||
               (call_id == format::ApiCall_vkQueueBindSparse) || (call_id == format::ApiCall_vkDestroySemaphore) ||
               (call_id == format::ApiCall_vkDestroyDevice);
    }

    static bool Follows(format::ApiCallId call_id)
    {
        switch (call_id)
        {
            case format::ApiCall_vkResetFences:
            case format::ApiCall_vkResetQueryPool:
            case format::ApiCall_vkResetQueryPoolEXT:
            case format::ApiCall_vkGetQueryPoolResults:
            case format::ApiCall_vkMapMemory:
            case format::ApiCall_vkUnmapMemory:
                return true;
            default:
                return false;
        }
    }

    void Process_vkSignalSemaphore(const ApiCallInfo& call_info, args::SignalSemaphore& args) override
    {
        SignalSemaphore(args.pSignalInfo);
    }

    void Process_vkSignalSemaphoreKHR(const ApiCallInfo& call_info, args::SignalSemaphoreKHR& args) override
    {
        SignalSemaphore(args.pSignalInfo);
    }

    void Process_vkQueueBindSparse(const ApiCallInfo& call_info, args::QueueBindSparse& args) override
    {
        const Decoded_VkBindSparseInfo* meta = args.pBindInfo.GetMetaStructPointer();
        const VkBindSparseInfo*         bind = args.pBindInfo.GetPointer();
        for (uint32_t i = 0; (meta != nullptr) && (bind != nullptr) && (i < args.bindInfoCount); ++i)
        {
            const format::HandleId* ids = meta[i].pSignalSemaphores.GetPointer();
            for (uint32_t j = 0; (ids != nullptr) && (j < bind[i].signalSemaphoreCount); ++j)
            {
                const VulkanSemaphoreInfo* info = consumer_->GetLaneObjectTable().GetVkSemaphoreInfo(ids[j]);
                if (info != nullptr)
                {
                    tracker_->NoteHostSignal(info->handle, UINT64_MAX);
                }
            }
        }
    }

    void Process_vkDestroyDevice(const ApiCallInfo& call_info, args::DestroyDevice& args) override
    {
        // The semaphores of the device go, and their handles may come back
        tracker_->ForgetSemaphores();
    }

    void Process_vkDestroySemaphore(const ApiCallInfo& call_info, args::DestroySemaphore& args) override
    {
        const VulkanSemaphoreInfo* info = consumer_->GetLaneObjectTable().GetVkSemaphoreInfo(args.semaphore);
        if (info != nullptr)
        {
            tracker_->ForgetSemaphore(info->handle);
        }
    }

    void Process_vkResetFences(const ApiCallInfo& call_info, args::ResetFences& args) override
    {
        const format::HandleId* ids = args.pFences.GetPointer();
        for (uint32_t i = 0; (ids != nullptr) && (i < args.fenceCount); ++i)
        {
            const VulkanFenceInfo* info = consumer_->GetLaneObjectTable().GetVkFenceInfo(ids[i]);
            if (info != nullptr)
            {
                tracker_->BeforeFenceReset(info->handle);
            }
        }
    }

    void Process_vkResetQueryPool(const ApiCallInfo& call_info, args::ResetQueryPool& args) override
    {
        ResetQueryPool(args.device, args.queryPool, args.firstQuery, args.queryCount);
    }

    void Process_vkResetQueryPoolEXT(const ApiCallInfo& call_info, args::ResetQueryPoolEXT& args) override
    {
        ResetQueryPool(args.device, args.queryPool, args.firstQuery, args.queryCount);
    }

    void Process_vkGetQueryPoolResults(const ApiCallInfo& call_info, args::GetQueryPoolResults& args) override
    {
        // A skipped call does not wait for the GPU
        if (consumer_->IgnoresQueryResults() && !tracker_->IsCompactPool(args.queryPool))
        {
            return;
        }
        tracker_->BeforeQueryPoolUse(args.queryPool);
    }

    void Process_vkMapMemory(const ApiCallInfo& call_info, args::MapMemory& args) override
    {
        tracker_->NoteMappingChange();
    }

    void Process_vkUnmapMemory(const ApiCallInfo& call_info, args::UnmapMemory& args) override
    {
        tracker_->NoteMappingChange();
    }

  private:
    // With the results ignored, the reset goes to a command that the GPU runs once it is done with the pool
    void ResetQueryPool(format::HandleId device_id, format::HandleId pool_id, uint32_t first, uint32_t count)
    {
        if (consumer_->IgnoresQueryResults() && !tracker_->IsCompactPool(pool_id))
        {
            const CommonObjectInfoTable& table       = consumer_->GetLaneObjectTable();
            const VulkanDeviceInfo*      device_info = table.GetVkDeviceInfo(device_id);
            const VulkanQueryPoolInfo*   pool_info   = table.GetVkQueryPoolInfo(pool_id);
            VulkanRebindAllocator*       allocator =
                (device_info != nullptr) ? dynamic_cast<VulkanRebindAllocator*>(device_info->allocator.get()) : nullptr;
            if ((pool_info != nullptr) && tracker_->StageQueryReset(allocator, pool_info->handle, first, count))
            {
                return;
            }
        }
        tracker_->BeforeQueryPoolUse(pool_id);
    }

    void SignalSemaphore(StructPointerDecoder<Decoded_VkSemaphoreSignalInfo>& signal_info)
    {
        const Decoded_VkSemaphoreSignalInfo* meta  = signal_info.GetMetaStructPointer();
        const VkSemaphoreSignalInfo*         value = signal_info.GetPointer();
        const VulkanSemaphoreInfo*           info =
            (meta != nullptr) ? consumer_->GetLaneObjectTable().GetVkSemaphoreInfo(meta->semaphore) : nullptr;
        if ((info != nullptr) && (value != nullptr))
        {
            tracker_->NoteHostSignal(info->handle, value->value);
        }
    }

    VulkanReplayConsumerBase* consumer_;
    VulkanLaneTracker*        tracker_;
};

GFXRECON_END_NAMESPACE(decode)
GFXRECON_END_NAMESPACE(gfxrecon)

#endif // GFXRECON_DECODE_VULKAN_LANE_TRACKER_H
