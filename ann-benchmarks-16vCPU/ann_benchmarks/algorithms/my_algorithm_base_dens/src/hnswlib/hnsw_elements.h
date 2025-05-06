#pragma once

#include "hnsw_class_declaration.h"
#include "space_ip.h" // Include space_ip.h for DOT_KERNEL definition
#include <cmath> // Required for acosf
#include <algorithm> // Required for std::max/std::min
#include <vector> // Required for std::vector
#include <queue> // Required for std::priority_queue
#include <unordered_map> // Required for std::unordered_map
#include <numeric> // Required for std::iota (potentially useful later)
#include <cassert> // For assertions

namespace hnswlib {

template<typename dist_t>
void HierarchicalNSW<dist_t>::getNeighborsByProximity(
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    const size_t M) {
    if (top_candidates.size() < M) {
        return;
    }

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
        queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
    }

    while (queue_closest.size()) {
        if (return_list.size() >= M)
            break;
        std::pair<dist_t, tableint> curent_pair = queue_closest.top();
        dist_t dist_to_query = -curent_pair.first;
        queue_closest.pop();
        bool good = true;

        for (std::pair<dist_t, tableint> second_pair : return_list) {
            dist_t curdist =
                    fstdistfunc_(getDataByInternalId(second_pair.second),
                                    getDataByInternalId(curent_pair.second),
                                    dist_func_param_);
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.push_back(curent_pair);
        }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
        top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::getNeighborsByAngularDiversity(
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    const size_t M,
    const void *query_vector) {
    if (top_candidates.size() <= M) {
        return;
    }

    // --- ADDED: Get dimension from dist_func_param_ --- 
    assert(dist_func_param_ != nullptr);
    const size_t dim = *static_cast<const size_t*>(dist_func_param_);
    // --------------------------------------------------

    // 1. Extract all candidates
    std::vector<std::pair<dist_t, tableint>> candidates;
    candidates.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
        candidates.push_back(top_candidates.top());
        top_candidates.pop();
    }
    std::sort(candidates.begin(), candidates.end(), [](const std::pair<dist_t, tableint>& a, const std::pair<dist_t, tableint>& b) {
        return a.first < b.first;
    });

    std::vector<tableint> selected_neighbors;
    selected_neighbors.reserve(M);

    std::unordered_map<tableint, dist_t> original_distances;
    for(const auto& pair : candidates) {
        original_distances[pair.second] = pair.first;
    }

    if (candidates.empty()) {
       // No candidates, nothing to select
    } else {
        // Start with the closest neighbor (based on original distance)
        selected_neighbors.push_back(candidates[0].second);

        std::vector<tableint> remaining_candidate_ids;
        std::vector<const void*> remaining_candidate_vectors; // Store vector pointers for efficiency
        remaining_candidate_ids.reserve(candidates.size() - 1);
        remaining_candidate_vectors.reserve(candidates.size() - 1);
        for (size_t i = 1; i < candidates.size(); ++i) {
            remaining_candidate_ids.push_back(candidates[i].second);
            remaining_candidate_vectors.push_back(getDataByInternalId(candidates[i].second));
        }

        // Greedily select remaining M-1 neighbors
        while (selected_neighbors.size() < M && !remaining_candidate_ids.empty()) {
            tableint best_candidate_id = -1;
            float max_min_angle = -1.0f; // Initialize with impossible angle value (-1)
            int best_candidate_idx_in_remaining = -1;

            for (size_t i = 0; i < remaining_candidate_ids.size(); ++i) {
                tableint current_candidate_id = remaining_candidate_ids[i];
                const float* current_candidate_vector = reinterpret_cast<const float*>(remaining_candidate_vectors[i]);
                float min_angle_to_selected = M_PI + 1.0f; // Store min angle (max similarity) for this candidate, init > PI

                // Calculate similarity/angle with all already selected neighbors
                for (tableint selected_id : selected_neighbors) {
                    // --- MODIFICATION: Use reinterpret_cast for char* -> const float* ---
                    const float* selected_vector = reinterpret_cast<const float*>(getDataByInternalId(selected_id));
                    // --- MODIFICATION END ---

                    // Calculate cosine similarity using the optimized dot product
                    // Assumes vectors are normalized for InnerProductSpace
                    // Use static_cast for void* -> const float* (safe)
                    float cos_sim = hnswlib::detail::DOT_KERNEL(current_candidate_vector, selected_vector, dim); // Use extracted dim

                    // Clamp similarity and calculate angle
                    float clamped_sim = std::max(-1.0f, std::min(1.0f, cos_sim));
                    float angle = acosf(clamped_sim); 

                    if (angle < min_angle_to_selected) {
                        min_angle_to_selected = angle;
                    }
                }

                // If this candidate offers a better *minimum* angle than the best found so far
                if (min_angle_to_selected > max_min_angle) {
                    max_min_angle = min_angle_to_selected;
                    best_candidate_id = current_candidate_id;
                    best_candidate_idx_in_remaining = i;
                }
            }

            // Add the best candidate found in this iteration
            if (best_candidate_id != -1) {
                selected_neighbors.push_back(best_candidate_id);
                // Remove the selected candidate from remaining vectors efficiently
                remaining_candidate_ids.erase(remaining_candidate_ids.begin() + best_candidate_idx_in_remaining);
                remaining_candidate_vectors.erase(remaining_candidate_vectors.begin() + best_candidate_idx_in_remaining);
            } else {
                // This might happen if all remaining candidates are identical to selected ones (angle=0)
                // Or if remaining_candidate_ids was empty initially
                break; 
            }
        }
    }

    // 3. Refill top_candidates with the angularly diverse selected neighbors
    for (tableint selected_id : selected_neighbors) {
         if (original_distances.count(selected_id)) {
             // Use the original distance associated with this neighbor ID
             top_candidates.emplace(original_distances[selected_id], selected_id);
         } else {
             // This should not happen if selected_id came from the initial candidates
             // As a fallback, recalculate distance relative to the query_vector
             // Note: query_vector is the node we are connecting TO, not the candidate itself.
              dist_t fallback_dist = fstdistfunc_(query_vector, getDataByInternalId(selected_id), dist_func_param_);
              top_candidates.emplace(fallback_dist, selected_id);
         }
    }
     // Ensure the size constraint (redundant if M <= original size, but safe)
     while (top_candidates.size() > M) {
         top_candidates.pop();
     }
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::markDelete(labeltype label) {
    // lock all operations with element by label
    std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock <std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
        throw std::runtime_error("Label not found");
    }
    tableint internalId = search->second;
    lock_table.unlock();

    markDeletedInternal(internalId);
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::markDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count);
    if (!isMarkedDeleted(internalId)) {
        unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
        *ll_cur |= DELETE_MARK;
        num_deleted_ += 1;
        if (allow_replace_deleted_) {
            std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
            deleted_elements.insert(internalId);
        }
    } else {
        throw std::runtime_error("The requested to delete element is already deleted");
    }
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::unmarkDelete(labeltype label) {
    // lock all operations with element by label
    std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock <std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
        throw std::runtime_error("Label not found");
    }
    tableint internalId = search->second;
    lock_table.unlock();

    unmarkDeletedInternal(internalId);
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::unmarkDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count);
    if (isMarkedDeleted(internalId)) {
        unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
        *ll_cur &= ~DELETE_MARK;
        num_deleted_ -= 1;
        if (allow_replace_deleted_) {
            std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
            deleted_elements.erase(internalId);
        }
    } else {
        throw std::runtime_error("The requested to undelete element is not deleted");
    }
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::addPoint(const void *data_point, labeltype label, bool replace_deleted) {
    if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
        throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
    }

    // lock all operations with element by label
    std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
    if (!replace_deleted) {
        addPoint(data_point, label, -1);
        return;
    }
    // check if there is vacant place
    tableint internal_id_replaced;
    std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
    bool is_vacant_place = !deleted_elements.empty();
    if (is_vacant_place) {
        internal_id_replaced = *deleted_elements.begin();
        deleted_elements.erase(internal_id_replaced);
    }
    lock_deleted_elements.unlock();

    // if there is no vacant place then add or update point
    // else add point to vacant place
    if (!is_vacant_place) {
        addPoint(data_point, label, -1);
    } else {
        // we assume that there are no concurrent operations on deleted element
        labeltype label_replaced = getExternalLabel(internal_id_replaced);
        setExternalLabel(internal_id_replaced, label);

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        label_lookup_.erase(label_replaced);
        label_lookup_[label] = internal_id_replaced;
        lock_table.unlock();

        unmarkDeletedInternal(internal_id_replaced);
        updatePoint(data_point, internal_id_replaced, 1.0);
    }
}

template<typename dist_t>
template<typename data_t>
std::vector<data_t> HierarchicalNSW<dist_t>::getDataByLabel(labeltype label) const {
    // lock all operations with element by label
    std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
    
    std::unique_lock <std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
        throw std::runtime_error("Label not found");
    }
    tableint internalId = search->second;
    lock_table.unlock();

    char* data_ptrv = getDataByInternalId(internalId);
    size_t dim = *((size_t *) dist_func_param_);
    std::vector<data_t> data;
    data_t* data_ptr = (data_t*) data_ptrv;
    for (size_t i = 0; i < dim; i++) {
        data.push_back(*data_ptr);
        data_ptr += 1;
    }
    return data;
}

}  // namespace hnswlib 