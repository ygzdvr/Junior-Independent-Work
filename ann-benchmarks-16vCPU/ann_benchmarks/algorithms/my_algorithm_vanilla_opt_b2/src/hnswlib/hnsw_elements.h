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
    const void* base_data_ptr = getDataByInternalId(candidates[0].second); // Get base data for diversity calc
    
    // Select remaining neighbors with diversity consideration
    while (selected.size() < M && candidates.size() > selected.size()) {
        // Find the candidate that maximizes diversity from already selected points
        size_t best_candidate_idx = selected.size();
        dist_t best_diversity_score = std::numeric_limits<dist_t>::lowest();
        
        for (size_t i = selected.size(); i < candidates.size(); i++) {
            dist_t min_diversity = std::numeric_limits<dist_t>::max();
            // Get candidate data pointer (or handle compressed? - ASSUMING FULL VEC for now in diversity)
            // For simplicity, diversity calculation still uses full vectors if available
            // A more advanced version might approximate diversity from codes
            const void* v1_full = getDataByInternalId(candidates[i].second);
            
            for (const auto& sel : selected) {
                // Get selected data pointer
                const void* v2_full = getDataByInternalId(sel.second);
                
                // Calculate angular separation using original distance func (assumes full vectors)
                dist_t dot_product = 1.0f - fstdistfunc_(v1_full, v2_full, dist_func_param_);
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
    
    // Apply robust pruning (needs ADC update internal to robustPrune)
    // Pass the *base* vector (closest neighbor) as the reference point for pruning distances
    auto pruned_selected = robustPrune(selected, base_data_ptr);
    
    // Return the pruned candidates
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
    
    std::vector<std::pair<dist_t, tableint>> result = candidates;
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    std::vector<std::pair<dist_t, tableint>> pruned_candidates;
    pruned_candidates.push_back(result[0]); // Always keep the closest
    
    for (size_t i = 1; i < result.size(); i++) {
        const tableint candidate_id = result[i].second;
        const dist_t dist_to_query = result[i].first; // Distance from candidate to queryPoint (already computed)
        bool should_prune = false;
        
        for (const auto& accepted : pruned_candidates) {
            const tableint accepted_id = accepted.second;
            
            // Compute distance between candidate and accepted neighbor
            // Use ADC if compression enabled, otherwise use full vectors
            dist_t dist_between;
            if (use_compression_) {
                // ADC requires the full vector for one side (the accepted node's vector)
                // and the compressed code for the other (the candidate node's code)
                // This assumes robustPrune is called where 'accepted' nodes are already in the graph
                // and 'candidate' nodes might be too.
                 const void* accepted_vec = getDataByInternalId(accepted_id); // Get full vector for accepted
                 dist_between = distanceADC(accepted_vec, candidate_id); // ADC: full vec vs compressed code
            } else {
                 dist_between = fstdistfunc_(
                getDataByInternalId(candidate_id),
                getDataByInternalId(accepted_id),
                dist_func_param_);
            }
            
            // Robust pruning condition
            if (dist_between < dist_to_query / pruning_alpha_) {
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