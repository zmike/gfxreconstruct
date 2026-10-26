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

#ifndef GFXRECON_DECODE_OBJECT_INFO_MAP_H
#define GFXRECON_DECODE_OBJECT_INFO_MAP_H

#include "util/defines.h"
#include "util/logging.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(decode)

// A map from capture handle IDs to object infos. The infos live in an unordered_map, so that pointers to them stay
// valid until they are erased. Lookup() finds them with one probe of a flat table instead of the node walk of the
// unordered_map, which matters because every replayed call looks up its handles. The IDs are neither zero nor all ones.
template <typename Key, typename T>
class ObjectInfoMap : private std::unordered_map<Key, T>
{
    using Map = std::unordered_map<Key, T>;

  public:
    ObjectInfoMap()                                = default;
    ObjectInfoMap(const ObjectInfoMap&)            = delete;
    ObjectInfoMap& operator=(const ObjectInfoMap&) = delete;

    using iterator       = typename Map::iterator;
    using const_iterator = typename Map::const_iterator;

    using Map::begin;
    using Map::empty;
    using Map::end;
    using Map::size;

    T* Lookup(Key id) { return const_cast<T*>(std::as_const(*this).Lookup(id)); }

    const T* Lookup(Key id) const
    {
        if (slots_.empty())
        {
            return nullptr;
        }

        for (size_t i = Home(id);; i = (i + 1) & mask_)
        {
            const Slot& slot = slots_[i];
            if (slot.key == id)
            {
                return slot.info;
            }
            if (slot.key == kEmpty)
            {
                return nullptr;
            }
        }
    }

    iterator find(Key id) { return Map::find(id); }

    const_iterator find(Key id) const { return Map::find(id); }

    std::pair<iterator, bool> emplace(Key id, T&& info)
    {
        GFXRECON_ASSERT(id != kEmpty && id != kTombstone);

        // Keep an empty slot in the table for the lookups to end at
        if ((used_ + 1) * 2 > slots_.size())
        {
            Rebuild();
        }

        auto result = Map::emplace(id, std::move(info));
        if (result.second)
        {
            Insert(id, &result.first->second);
        }
        return result;
    }

    size_t erase(Key id)
    {
        auto it = Map::find(id);
        if (it == Map::end())
        {
            return 0;
        }

        for (size_t i = Home(id);; i = (i + 1) & mask_)
        {
            if (slots_[i].key == id)
            {
                slots_[i] = { kTombstone, nullptr };
                break;
            }
        }
        Map::erase(it);
        return 1;
    }

    void clear()
    {
        Map::clear();
        slots_.clear();
        used_ = 0;
    }

  private:
    static constexpr Key kEmpty     = 0;
    static constexpr Key kTombstone = ~Key(0);

    struct Slot
    {
        Key key;
        T*  info;
    };

    size_t Home(Key id) const { return static_cast<size_t>((id * 0x9E3779B97F4A7C15ull) >> shift_); }

    void Insert(Key id, T* info)
    {
        size_t i = Home(id);
        while ((slots_[i].key != kEmpty) && (slots_[i].key != kTombstone))
        {
            i = (i + 1) & mask_;
        }
        used_ += (slots_[i].key == kEmpty) ? 1 : 0;
        slots_[i] = { id, info };
    }

    // Sizes the table for four times the number of infos, and one more, and drops the tombstones.
    void Rebuild()
    {
        size_t capacity = std::bit_ceil<size_t>(std::max<size_t>(16, (Map::size() + 1) * 4));

        slots_.assign(capacity, { kEmpty, nullptr });
        mask_  = capacity - 1;
        shift_ = 64 - std::countr_zero(capacity);
        used_  = 0;

        for (auto& entry : *static_cast<Map*>(this))
        {
            Insert(entry.first, &entry.second);
        }
    }

    std::vector<Slot> slots_;
    size_t            mask_{ 0 };
    int               shift_{ 0 };
    size_t            used_{ 0 }; ///< Slots holding an info or a tombstone
};

GFXRECON_END_NAMESPACE(decode)
GFXRECON_END_NAMESPACE(gfxrecon)

#endif // GFXRECON_DECODE_OBJECT_INFO_MAP_H
