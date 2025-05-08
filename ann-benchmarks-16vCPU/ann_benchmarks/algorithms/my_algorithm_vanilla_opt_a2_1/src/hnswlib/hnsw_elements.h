#pragma once

#include "hnsw_class_declaration.h"
#include <algorithm>
#include <limits>

namespace hnswlib {

template<typename dist_t>
void HierarchicalNSW<dist_t>::getNeighborsByHeuristic2(
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    const size_t M) {
    if (top_candidates.size() < M) {
        return;
    }

    // Store candidates in a vector for easier manipulation
    std::vector<std::pair<dist_t, tableint>> candidates;
    while (top_candidates.size() > 0) {
        candidates.push_back({top_candidates.top().first, top_candidates.top().second});
        top_candidates.pop();
    }
    
    // Sort candidates by distance (ascending)
    std::sort(candidates.begin(), candidates.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // Always include the closest neighbor
    std::vector<std::pair<dist_t, tableint>> selected;
    selected.push_back(candidates[0]);
    
    // Select remaining neighbors with diversity consideration
    while (selected.size() < M && candidates.size() > selected.size()) {
        // Find the candidate that maximizes diversity from already selected points
        size_t best_candidate_idx = selected.size();
        dist_t best_diversity_score = std::numeric_limits<dist_t>::lowest();
        
        for (size_t i = selected.size(); i < candidates.size(); i++) {
            // Calculate the minimum angular diversity from already selected points
            dist_t min_diversity = std::numeric_limits<dist_t>::max();
            
            for (const auto& sel : selected) {
                // Get vectors from internal IDs
                const void* v1 = getDataByInternalId(candidates[i].second);
                const void* v2 = getDataByInternalId(sel.second);
                
                // Calculate angular separation (using dot product)
                // Note: We're approximating angular diversity here
                // Lower dot product means higher angular diversity
                dist_t dot_product = 1.0f - fstdistfunc_(v1, v2, dist_func_param_);
                
                // Normalize dot product to get diversity score (higher is better)
                dist_t diversity = 1.0f - dot_product;
                min_diversity = std::min(min_diversity, diversity);
            }
            
            // Combine distance and diversity:
            // - Weight distance more for closer points
            // - Weight diversity more for further points
            dist_t distance_factor = candidates[i].first / candidates[0].first; // Normalize
            dist_t diversity_weight = 0.2f + 0.8f * distance_factor; // Increase diversity weight with distance
            dist_t combined_score = (1.0f - diversity_weight) * (1.0f / (1.0f + candidates[i].first)) + 
                                    diversity_weight * min_diversity;
            
            if (combined_score > best_diversity_score) {
                best_diversity_score = combined_score;
                best_candidate_idx = i;
            }
        }
        
        // Add the best candidate to selected list and remove it from candidates
        selected.push_back(candidates[best_candidate_idx]);
        candidates.erase(candidates.begin() + best_candidate_idx);
    }
    
    // Apply robust pruning to the selected candidates
    const void* queryPoint = nullptr;  // We don't have a query point in this case
    auto pruned_selected = robustPrune(selected, queryPoint);
    
    // Return the selected candidates through the original priority queue
    for (const auto& sel : pruned_selected) {
        top_candidates.emplace(sel.first, sel.second);
    }
}

template<typename dist_t>
std::vector<std::pair<dist_t, tableint>> HierarchicalNSW<dist_t>::robustPrune(
    const std::vector<std::pair<dist_t, tableint>>& candidates,
    const void* queryPoint) {
    
    if (candidates.size() <= 1) {
        return candidates;  // Nothing to prune if only 0 or 1 candidates
    }
    
    // Create a copy of candidates that we will prune
    std::vector<std::pair<dist_t, tableint>> result = candidates;
    
    // Sort by distance (ascending)
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // We always keep the closest neighbor
    std::vector<std::pair<dist_t, tableint>> pruned_candidates;
    pruned_candidates.push_back(result[0]);
    
    // For each candidate, check if it should be pruned
    for (size_t i = 1; i < result.size(); i++) {
        const tableint candidate_id = result[i].second;
        bool should_prune = false;
        
        // For each already accepted neighbor, check if it covers the current candidate
        for (const auto& accepted : pruned_candidates) {
            const tableint accepted_id = accepted.second;
            
            // Compute distance between the current candidate and accepted neighbor
            dist_t dist_between = fstdistfunc_(
                getDataByInternalId(candidate_id),
                getDataByInternalId(accepted_id),
                dist_func_param_);
            
            // If the distance between the candidate and accepted neighbor is less than
            // alpha * (distance from candidate to query), prune the candidate
            if (dist_between < result[i].first / pruning_alpha_) {
                should_prune = true;
                break;
            }
        }
        
        if (!should_prune) {
            pruned_candidates.push_back(result[i]);
        }
    }
    
    return pruned_candidates;
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