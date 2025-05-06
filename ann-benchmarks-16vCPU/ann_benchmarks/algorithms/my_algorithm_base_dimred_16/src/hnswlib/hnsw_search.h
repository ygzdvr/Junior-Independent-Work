#pragma once

#include "hnsw_class_declaration.h"

namespace hnswlib {

template<typename dist_t>
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst>
HierarchicalNSW<dist_t>::searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
    auto vl = visited_list_pool_->acquire();
    
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    // For dimensionality reduction, we need to project the query vector if needed
    std::vector<float> projected_query;
    const void* query_to_use = data_point;
    
    if (use_dim_reduction_ && layer >= dim_reduction_threshold_level_) {
        int reduction_index = layer - dim_reduction_threshold_level_;
        if (reduction_index >= 0 && reduction_index < target_dims_.size() && 
            dim_reduction_per_level_[layer] != nullptr) {
            
            size_t target_dim = target_dims_[reduction_index];
            projected_query.resize(target_dim);
            projectVector(data_point, projected_query.data(), layer);
            query_to_use = projected_query.data();
        }
    }

    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
        // Use the appropriate vector data for this level
        char* ep_data = (use_dim_reduction_ && layer >= dim_reduction_threshold_level_) ? 
                         getReducedVectorByInternalId(ep_id, layer) : 
                         getDataByInternalId(ep_id);
                         
        // Use the appropriate distance computation for this level
        dist_t dist = computeDistance(query_to_use, ep_data, layer);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidateSet.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidateSet.emplace(-lowerBound, ep_id);
    }
    vl->mark(ep_id);

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

        int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
        if (layer == 0) {
            data = (int*)get_linklist0(curNodeNum);
        } else {
            data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
        }
        size_t size = getListCount((linklistsizeint*)data);
        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
        _mm_prefetch((char *) (datal), _MM_HINT_T0);
        _mm_prefetch((char *) (datal + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < size; j++) {
            tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
            _mm_prefetch((char *) (datal + j + 1), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
            if (vl->visited(candidate_id)) continue;
            vl->mark(candidate_id);
            
            // Use the appropriate vector data for this level
            char *currObj1 = (use_dim_reduction_ && layer >= dim_reduction_threshold_level_) ? 
                             getReducedVectorByInternalId(candidate_id, layer) : 
                             getDataByInternalId(candidate_id);

            // Use the appropriate distance computation for this level
            dist_t dist1 = computeDistance(query_to_use, currObj1, layer);
            
            if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                if (!isMarkedDeleted(candidate_id))
                    top_candidates.emplace(dist1, candidate_id);

                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
    visited_list_pool_->release(std::move(vl));

    return top_candidates;
}

template<typename dist_t>
template <bool bare_bone_search, bool collect_metrics>
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst>
HierarchicalNSW<dist_t>::searchBaseLayerST(
    tableint ep_id,
    const void *data_point,
    size_t ef,
    BaseFilterFunctor* isIdAllowed,
    BaseSearchStopCondition<dist_t>* stop_condition) const {
    auto vl = visited_list_pool_->acquire();
    
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

    // For base layer (level 0), we always use the full dimension
    // No need to project the query vector here

    dist_t lowerBound;
    if (bare_bone_search || 
        (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
        char* ep_data = getDataByInternalId(ep_id);
        dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
        lowerBound = dist;
        top_candidates.emplace(dist, ep_id);
        if (!bare_bone_search && stop_condition) {
            stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
        }
        candidate_set.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidate_set.emplace(-lowerBound, ep_id);
    }

    vl->mark(ep_id);

    while (!candidate_set.empty()) {
        std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
        dist_t candidate_dist = -current_node_pair.first;

        bool flag_stop_search;
        if (bare_bone_search) {
            flag_stop_search = candidate_dist > lowerBound;
        } else {
            if (stop_condition) {
                flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
            } else {
                flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
            }
        }
        if (flag_stop_search) {
            break;
        }
        candidate_set.pop();

        tableint current_node_id = current_node_pair.second;
        int *data = (int *) get_linklist0(current_node_id);
        size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
        if (collect_metrics) {
            metric_hops++;
            metric_distance_computations+=size;
        }

#ifdef USE_SSE
        _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
        _mm_prefetch((char *) (data + 1 + 64), _MM_HINT_T0);
        _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
        _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
            _mm_prefetch((char *) (data + j + 1), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                            _MM_HINT_T0);  ////////////
#endif
            if (!vl->visited(candidate_id)) {
                vl->mark(candidate_id);

                // For base layer, always use the full dimension vectors
                char *currObj1 = (getDataByInternalId(candidate_id));
                dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                bool flag_consider_candidate;
                if (!bare_bone_search && stop_condition) {
                    flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                } else {
                    flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                }

                if (flag_consider_candidate) {
                    candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                    offsetLevel0_,  ///////////
                                    _MM_HINT_T0);  ////////////////////////
#endif

                    if (bare_bone_search || 
                        (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                        top_candidates.emplace(dist, candidate_id);
                        if (!bare_bone_search && stop_condition) {
                            stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                        }
                    }

                    bool flag_remove_extra = false;
                    if (!bare_bone_search && stop_condition) {
                        flag_remove_extra = stop_condition->should_remove_extra();
                    } else {
                        flag_remove_extra = top_candidates.size() > ef;
                    }
                    while (flag_remove_extra) {
                        tableint id = top_candidates.top().second;
                        top_candidates.pop();
                        if (!bare_bone_search && stop_condition) {
                            stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                    }

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
    }

    visited_list_pool_->release(std::move(vl));
    return top_candidates;
}

template<typename dist_t>
std::priority_queue<std::pair<dist_t, labeltype >>
HierarchicalNSW<dist_t>::searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed) const {
    std::priority_queue<std::pair<dist_t, labeltype >> result;
    if (cur_element_count == 0) return result;

    tableint currObj = enterpoint_node_;
    
    // For tracking reduced vectors per level during the search
    std::vector<float> projected_query_vec;
    std::vector<std::vector<float>> projected_query_cache(maxlevel_ + 1);
    
    // Initial distance calculation with the entry point
    dist_t curdist;
    
    if (use_dim_reduction_ && maxlevel_ >= dim_reduction_threshold_level_) {
        // Project the query for the top level if needed
        int reduction_index = maxlevel_ - dim_reduction_threshold_level_;
        if (reduction_index >= 0 && reduction_index < target_dims_.size() && 
            dim_reduction_per_level_[maxlevel_] != nullptr) {
            
            size_t target_dim = target_dims_[reduction_index];
            projected_query_cache[maxlevel_].resize(target_dim);
            projectVector(query_data, projected_query_cache[maxlevel_].data(), maxlevel_);
            
            // Get the reduced vector for the entry point
            char* ep_data = getReducedVectorByInternalId(enterpoint_node_, maxlevel_);
            
            // Compute distance in the reduced space
            curdist = computeDistance(projected_query_cache[maxlevel_].data(), ep_data, maxlevel_);
        } else {
            // Fallback to full dimension if reduction is not available
            curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
        }
    } else {
        // No dimensionality reduction, use full vectors
        curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
    }

    for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data;

            data = (unsigned int *) get_linklist(currObj, level);
            int size = getListCount(data);
            metric_hops++;
            metric_distance_computations+=size;

            tableint *datal = (tableint *) (data + 1);
            
            // Prepare projected query vector for this level if needed
            const void* query_to_use = query_data;
            
            if (use_dim_reduction_ && level >= dim_reduction_threshold_level_) {
                int reduction_index = level - dim_reduction_threshold_level_;
                if (reduction_index >= 0 && reduction_index < target_dims_.size() && 
                    dim_reduction_per_level_[level] != nullptr) {
                    
                    // Check if we already projected for this level
                    if (projected_query_cache[level].empty()) {
                        size_t target_dim = target_dims_[reduction_index];
                        projected_query_cache[level].resize(target_dim);
                        projectVector(query_data, projected_query_cache[level].data(), level);
                    }
                    
                    query_to_use = projected_query_cache[level].data();
                }
            }
            
            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand < 0 || cand > max_elements_)
                    throw std::runtime_error("cand error");
                
                // Get the appropriate vector for this level and candidate
                char* candidate_vec = (use_dim_reduction_ && level >= dim_reduction_threshold_level_) ? 
                                      getReducedVectorByInternalId(cand, level) : 
                                      getDataByInternalId(cand);
                
                // Compute distance using the appropriate method for this level
                dist_t d = (use_dim_reduction_ && level >= dim_reduction_threshold_level_) ?
                            computeDistance(query_to_use, candidate_vec, level) :
                            fstdistfunc_(query_data, candidate_vec, dist_func_param_);

                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    bool bare_bone_search = !num_deleted_ && !isIdAllowed;
    if (bare_bone_search) {
        top_candidates = searchBaseLayerST<true>(
                currObj, query_data, std::max(ef_, k), isIdAllowed);
    } else {
        top_candidates = searchBaseLayerST<false>(
                currObj, query_data, std::max(ef_, k), isIdAllowed);
    }

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
        std::pair<dist_t, tableint> rez = top_candidates.top();
        result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
        top_candidates.pop();
    }
    return result;
}

template<typename dist_t>
std::vector<std::pair<dist_t, labeltype >>
HierarchicalNSW<dist_t>::searchStopConditionClosest(
    const void *query_data,
    BaseSearchStopCondition<dist_t>& stop_condition,
    BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype >> result;
    if (cur_element_count == 0) return result;

    tableint currObj = enterpoint_node_;
    
    // For tracking reduced vectors per level during the search
    std::vector<std::vector<float>> projected_query_cache(maxlevel_ + 1);
    
    // Initial distance calculation with the entry point
    dist_t curdist;
    
    if (use_dim_reduction_ && maxlevel_ >= dim_reduction_threshold_level_) {
        // Project the query for the top level if needed
        int reduction_index = maxlevel_ - dim_reduction_threshold_level_;
        if (reduction_index >= 0 && reduction_index < target_dims_.size() && 
            dim_reduction_per_level_[maxlevel_] != nullptr) {
            
            size_t target_dim = target_dims_[reduction_index];
            projected_query_cache[maxlevel_].resize(target_dim);
            projectVector(query_data, projected_query_cache[maxlevel_].data(), maxlevel_);
            
            // Get the reduced vector for the entry point
            char* ep_data = getReducedVectorByInternalId(enterpoint_node_, maxlevel_);
            
            // Compute distance in the reduced space
            curdist = computeDistance(projected_query_cache[maxlevel_].data(), ep_data, maxlevel_);
        } else {
            // Fallback to full dimension if reduction is not available
            curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
        }
    } else {
        // No dimensionality reduction, use full vectors
        curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
    }

    for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data;

            data = (unsigned int *) get_linklist(currObj, level);
            int size = getListCount(data);
            metric_hops++;
            metric_distance_computations+=size;

            tableint *datal = (tableint *) (data + 1);
            
            // Prepare projected query vector for this level if needed
            const void* query_to_use = query_data;
            
            if (use_dim_reduction_ && level >= dim_reduction_threshold_level_) {
                int reduction_index = level - dim_reduction_threshold_level_;
                if (reduction_index >= 0 && reduction_index < target_dims_.size() && 
                    dim_reduction_per_level_[level] != nullptr) {
                    
                    // Check if we already projected for this level
                    if (projected_query_cache[level].empty()) {
                        size_t target_dim = target_dims_[reduction_index];
                        projected_query_cache[level].resize(target_dim);
                        projectVector(query_data, projected_query_cache[level].data(), level);
                    }
                    
                    query_to_use = projected_query_cache[level].data();
                }
            }
            
            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand < 0 || cand > max_elements_)
                    throw std::runtime_error("cand error");
                
                // Get the appropriate vector for this level and candidate
                char* candidate_vec = (use_dim_reduction_ && level >= dim_reduction_threshold_level_) ? 
                                      getReducedVectorByInternalId(cand, level) : 
                                      getDataByInternalId(cand);
                
                // Compute distance using the appropriate method for this level
                dist_t d = (use_dim_reduction_ && level >= dim_reduction_threshold_level_) ?
                            computeDistance(query_to_use, candidate_vec, level) :
                            fstdistfunc_(query_data, candidate_vec, dist_func_param_);

                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

    size_t sz = top_candidates.size();
    result.resize(sz);
    while (!top_candidates.empty()) {
        result[--sz] = top_candidates.top();
        top_candidates.pop();
    }

    stop_condition.filter_results(result);

    return result;
}

}  // namespace hnswlib 