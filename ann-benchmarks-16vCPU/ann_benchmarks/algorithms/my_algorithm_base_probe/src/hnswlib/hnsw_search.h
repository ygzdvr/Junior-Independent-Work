#pragma once

#include "hnsw_class_declaration.h"
#include <random> // Need for random number generation
#include <unordered_set> // Need for storing combined results efficiently

namespace hnswlib {

template<typename dist_t>
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst>
HierarchicalNSW<dist_t>::searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
    auto vl = visited_list_pool_->acquire();
    
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
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
            char *currObj1 = (getDataByInternalId(candidate_id));

            dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
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
HierarchicalNSW<dist_t>::searchKnn(const void *query_data, size_t k, size_t num_probes /*= 1*/, BaseFilterFunctor* isIdAllowed /*= nullptr*/) const {
    // Overall result queue (max heap based on distance)
    std::priority_queue<std::pair<dist_t, labeltype >> result;
    // Use a set to keep track of unique results across probes
    std::unordered_set<labeltype> unique_labels; 
    // Combined top candidates from all probes (min heap based on distance, inverted)
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> combined_top_candidates;


    if (cur_element_count == 0) return result;

    // Find the initial entry point using hierarchical descent
    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data;

            data = (unsigned int *) get_linklist(currObj, level);
            int size = getListCount(data);
            // Removed metric updates from here as they will be counted in base layer search

            tableint *datal = (tableint *) (data + 1);
            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand < 0 || cand > max_elements_)
                    throw std::runtime_error("cand error");
                dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    // --- Multi-Probe Logic ---
    std::vector<tableint> entry_points;
    entry_points.push_back(currObj); // Add the initial best entry point

    // Add num_probes - 1 random entry points (if num_probes > 1)
    if (num_probes > 1 && cur_element_count > 1) {
         // Ensure random engine is seeded reasonably, maybe pass seed from constructor?
         // For now, using a simple thread_local generator
         thread_local static std::mt19937 rng(std::random_device{}()); 
         // Uniform distribution over valid internal IDs (0 to cur_element_count - 1)
         std::uniform_int_distribution<tableint> distrib(0, cur_element_count - 1);
         
         std::unordered_set<tableint> used_entry_points;
         used_entry_points.insert(currObj);

         for (size_t i = 1; i < num_probes && entry_points.size() < cur_element_count; ++i) {
             tableint random_node_id;
             do {
                 random_node_id = distrib(rng);
             } while (used_entry_points.count(random_node_id)); // Ensure uniqueness
             
             entry_points.push_back(random_node_id);
             used_entry_points.insert(random_node_id);
         }
    }
    
    // Determine ef for the search: max of ef_ and k
    size_t ef_search = std::max(ef_, k); 
    bool bare_bone_search = !num_deleted_ && !isIdAllowed;

    // Perform search from each entry point
    for (tableint entry_point_id : entry_points) {
         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> probe_top_candidates;
         
         // Use appropriate searchBaseLayerST template specialization
         if (bare_bone_search) {
             probe_top_candidates = searchBaseLayerST<true>(
                     entry_point_id, query_data, ef_search, isIdAllowed);
         } else {
             probe_top_candidates = searchBaseLayerST<false>(
                     entry_point_id, query_data, ef_search, isIdAllowed);
         }

         // Merge results from this probe into the combined candidates
         while (!probe_top_candidates.empty()) {
             combined_top_candidates.push(probe_top_candidates.top());
             // Keep combined results bounded if it grows too large (optional, heuristic)
             // if (combined_top_candidates.size() > ef_search * num_probes) { 
             //    combined_top_candidates.pop(); // remove furthest
             // }
             probe_top_candidates.pop();
         }
    }


    // Extract final top-k results from combined candidates, ensuring uniqueness
    while (combined_top_candidates.size() > 0 && result.size() < k) {
        std::pair<dist_t, tableint> rez = combined_top_candidates.top();
        labeltype external_label = getExternalLabel(rez.second);
        
        // Add to result only if label is not already present
        if (unique_labels.find(external_label) == unique_labels.end()) {
             result.push(std::pair<dist_t, labeltype>(rez.first, external_label));
             unique_labels.insert(external_label);
        }
        combined_top_candidates.pop();
    }

    // The 'result' priority queue is already ordered furthest-first (max-heap).
    // If closer-first is needed, it should be handled by the caller (e.g., searchKnnCloserFirst).
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
    dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

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
            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand < 0 || cand > max_elements_)
                    throw std::runtime_error("cand error");
                dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

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