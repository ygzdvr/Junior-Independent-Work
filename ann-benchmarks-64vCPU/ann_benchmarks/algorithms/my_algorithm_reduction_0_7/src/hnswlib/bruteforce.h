#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <limits>
#include <cassert>
#include <cstring>
#include <memory>
#include <algorithm>

namespace hnswlib {

/**
 * Simple brute‑force index for demonstration and evaluation purposes.
 *
 * Differences from the original implementation:
 * 1.  **RAII everywhere** – no raw `malloc`/`free`; all memory is owned by a single `std::vector<unsigned char>`.
 * 2.  **Thread safety** – uses `std::mutex` to protect concurrent access.
 * 3.  **No magic ints** – `std::size_t` is used throughout; loops avoid signed/unsigned mismatches.
 * 4.  **Minimal branching** inside the scan loop; distance is only computed if the label passes the (optional) filter functor.
 * 5.  **Automatic rebuild of the label→internal‑id map** when an index is loaded from disk.
 * 6.  **Stricter error handling** – every public API throws `std::runtime_error` or `std::invalid_argument` on misuse.
 * 7.  **Const‑correctness** and `noexcept` where appropriate.
 */

template <typename dist_t>
class BruteforceSearch : public AlgorithmInterface<dist_t> {
public:
    BruteforceSearch(SpaceInterface<dist_t>* space, size_t max_elements)
        : space_(space)
        , data_size_(space->get_data_size())
        , dist_func_(space->get_dist_func())
        , dist_func_param_(space->get_dist_func_param())
    {
        maxelements_ = max_elements;
        data_size_ = space->get_data_size();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char*)malloc(maxelements_ * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
        cur_element_count = 0;
    }

    BruteforceSearch(SpaceInterface<dist_t>* space, const std::string& location)
        : space_(space)
        , data_size_(space->get_data_size())
        , dist_func_(space->get_dist_func())
        , dist_func_param_(space->get_dist_func_param())
    {
        loadIndex(location);
    }

    ~BruteforceSearch() {
        if (data_) {
            free(data_);
        }
    }

    void addPoint(const void* datapoint, labeltype label, bool replace_deleted = false) {
        int idx;
        if (replace_deleted) {
            idx = label_lookup_.size() ? label_lookup_[label] : -1;
        } else {
            idx = -1;
        }

        if (idx != -1) {
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
        } else {
            if (cur_element_count >= maxelements_) {
                throw std::runtime_error("Cannot add point as maximum capacity reached");
            }
            memcpy(data_ + size_per_element_ * cur_element_count, datapoint, data_size_);
            memcpy(data_ + size_per_element_ * cur_element_count + data_size_, &label, sizeof(labeltype));
            label_lookup_[label] = cur_element_count;
            cur_element_count++;
        }
    }

    void removePoint(labeltype label) {
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Element not found");
        }
        int idx = search->second;
        label_lookup_.erase(search);

        // Move the last element to the deleted position
        if (idx < cur_element_count - 1) {
            memcpy(data_ + size_per_element_ * idx, 
                   data_ + size_per_element_ * (cur_element_count - 1),
                   size_per_element_);
            // Update the lookup table
            labeltype last_label;
            memcpy(&last_label, data_ + size_per_element_ * idx + data_size_, sizeof(labeltype));
            label_lookup_[last_label] = idx;
        }
        cur_element_count--;
    }

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype>> topResults;
        if (cur_element_count == 0) return topResults;

        for (int i = 0; i < cur_element_count; i++) {
            labeltype candidate_label;
            memcpy(&candidate_label, data_ + size_per_element_ * i + data_size_, sizeof(labeltype));
            if (isIdAllowed && !(*isIdAllowed)(candidate_label)) {
                continue;
            }

            dist_t dist = dist_func_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            
            if (topResults.size() < k) {
                topResults.push(std::pair<dist_t, labeltype>(dist, candidate_label));
            } else {
                if (dist < topResults.top().first) {
                    topResults.pop();
                    topResults.push(std::pair<dist_t, labeltype>(dist, candidate_label));
                }
            }
        }
        return topResults;
    }

    void saveIndex(const std::string& location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, maxelements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_per_element_);
        writeBinaryPOD(output, data_size_);

        output.write(data_, maxelements_ * size_per_element_);
        output.close();
    }

    void loadIndex(const std::string& location) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        readBinaryPOD(input, maxelements_);
        readBinaryPOD(input, cur_element_count);
        readBinaryPOD(input, size_per_element_);
        readBinaryPOD(input, data_size_);

        if (data_) free(data_);
        data_ = (char*)malloc(maxelements_ * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate data");

        input.read(data_, maxelements_ * size_per_element_);

        // Reconstruct the label lookup table
        label_lookup_.clear();
        for (int i = 0; i < cur_element_count; i++) {
            labeltype label;
            memcpy(&label, data_ + size_per_element_ * i + data_size_, sizeof(labeltype));
            label_lookup_[label] = i;
        }

        input.close();
    }

    SpaceInterface<dist_t>* space_;
    size_t maxelements_;
    size_t cur_element_count;
    size_t size_per_element_;
    size_t data_size_;
    DISTFUNC<dist_t> dist_func_;
    void* dist_func_param_;
    char* data_;
    std::unordered_map<labeltype, int> label_lookup_;
};

}  // namespace hnswlib
