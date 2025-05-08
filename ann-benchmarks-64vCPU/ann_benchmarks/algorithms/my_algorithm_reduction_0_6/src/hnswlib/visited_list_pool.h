#pragma once
//  ---------------------------------------------------------------------------
//  hnswlib – VisitedList rewrite
//  ---------------------------------------------------------------------------
//  This header contains a modernised, RAII‑safe implementation of the classic
//  VisitedList and VisitedListPool utilities that are used by the HNSW graph
//  search.  It keeps the public interface intentionally minimal while
//  eliminating the major sources of UB and segmentation faults found in the
//  legacy code (manual new[]/delete[], unsigned‑signed mix‑ups, and pool races).
//  ---------------------------------------------------------------------------
//  © 2025 – released under the Apache‑2.0 licence
//  ---------------------------------------------------------------------------

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace hnswlib {

// 16‑bit marker type (fast enough and saves RAM).  If you need >65k concurrent
// queries, simply switch to std::uint32_t.
using marker_t = std::uint16_t;

// ---------------------------------------------------------------------------
//  VisitedList
// ---------------------------------------------------------------------------
//  A tiny container that allows O(1) marking & membership checks without the
//  overhead of std::unordered_set.  A monotonically increasing stamp_ acts as a
//  logical timestamp; when it overflows we lazily clear the whole array.
// ---------------------------------------------------------------------------
class VisitedList {
public:
    explicit VisitedList(std::size_t num_elements)
        : markers_(num_elements, marker_t{0}), stamp_(marker_t{1}) {}

    // non‑copyable, but movable – avoids accidental deep copies on the pool
    VisitedList(const VisitedList&)            = delete;
    VisitedList& operator=(const VisitedList&) = delete;
    VisitedList(VisitedList&&)                 = default;
    VisitedList& operator=(VisitedList&&)      = default;

    /** Begin a fresh query */
    void reset() noexcept {
        ++stamp_;
        if (stamp_ == marker_t{0}) {                 // overflow → full clear
            std::fill(markers_.begin(), markers_.end(), marker_t{0});
            ++stamp_;                                // restart at 1
        }
    }

    /** True ↦ already visited in current query */
    [[nodiscard]] bool visited(std::size_t idx) const noexcept {
        return markers_[idx] == stamp_;
    }

    /** Mark a node as visited */
    void mark(std::size_t idx) noexcept { markers_[idx] = stamp_; }

private:
    std::vector<marker_t> markers_;
    marker_t              stamp_;   // current logical timestamp
};

// ---------------------------------------------------------------------------
//  VisitedListPool (thread‑safe)
// ---------------------------------------------------------------------------
//  A very small object – roughly 2×sizeof(pointer) – so we pool it to amortise
//  allocation cost under heavy concurrency.  std::unique_ptr ensures we never
//  leak a list even on exception paths.
// ---------------------------------------------------------------------------
class VisitedListPool {
public:
    VisitedListPool(std::size_t initial, std::size_t elements_per_list)
        : elem_per_list_(elements_per_list) {
        for (std::size_t i = 0; i < initial; ++i) {
            pool_.emplace_back(std::make_unique<VisitedList>(elem_per_list_));
        }
    }

    /** Obtain a fresh list – O(1) fast‑path, never nullptr */
    [[nodiscard]] std::unique_ptr<VisitedList> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.empty()) {
            return std::make_unique<VisitedList>(elem_per_list_);
        }
        auto ptr = std::move(pool_.front());
        pool_.pop_front();
        ptr->reset();
        return ptr;
    }

    /** Return a list back to the pool; the caller must relinquish ownership */
    void release(std::unique_ptr<VisitedList> vl) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_front(std::move(vl));
    }

private:
    std::deque<std::unique_ptr<VisitedList>> pool_;
    std::mutex                               mutex_;
    std::size_t                              elem_per_list_;
};

} // namespace hnswlib
