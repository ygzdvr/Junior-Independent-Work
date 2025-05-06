#pragma once

#include "hnsw_class_declaration.h"

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
template <bool bare_bone_search = false, bool collect_metrics = false>
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst>
HierarchicalNSW<dist_t>::searchBaseLayerST(
    tableint ep_id,
    const void *data_point,
    size_t ef,
    BaseFilterFunctor* isIdAllowed,
    BaseSearchStopCondition<dist_t>* stop_condition) const {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>> candidateSet;

    dist_t lowerBound;
    dist_t dist = use_compression_ ? \
                  distanceADC(data_point, ep_id) : \
                  fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
    if (!isIdAllowed || (*isIdAllowed)(ep_id)) {
      top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
      candidateSet.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
      candidateSet.emplace(-lowerBound, ep_id);
    }

    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> current_node_pair = candidateSet.top();

        if ((-current_node_pair.first) > lowerBound && (!bare_bone_search || top_candidates.size() == ef)) {
            break;
        }
        candidateSet.pop();

        tableint current_node_id = current_node_pair.second;
        linklistsizeint *ll_cur = get_linklist0(current_node_id);
        int size = getListCount(ll_cur);
        //_mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
        //_mm_prefetch((char *)(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_), _MM_HINT_T0);
        tableint *data = (tableint *)(ll_cur + 1);

        for (int i = 0; i < size; i++) {
            tableint candidate_id = data[i];
    //                _mm_prefetch((char *)(visited_array + *(data + i + 1)), _MM_HINT_T0);
    //                _mm_prefetch((char *)(data_level0_memory_ + (*(data + i + 1)) * size_data_per_element_ + offsetData_), _MM_HINT_T0);
            if (visited_array[candidate_id] != visited_array_tag) {
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));
                dist_t dist_cand = use_compression_ ? \
                                   distanceADC(data_point, candidate_id) : \
                                   fstdistfunc_(data_point, currObj1, dist_func_param_);

                bool is_allowed = !isIdAllowed || (*isIdAllowed)(candidate_id);
                bool stop_search = stop_condition && stop_condition->shouldStop(dist_cand, (dist_t)top_candidates.top().first);

                if (stop_search) {
                    // Stop search if stop condition is met
                } else if (top_candidates.size() < ef || lowerBound > dist_cand) {
                   if(is_allowed){
                       candidateSet.emplace(-dist_cand, candidate_id);
                       top_candidates.emplace(dist_cand, candidate_id);
                   }

                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }

                    if (!top_candidates.empty()) {
                        lowerBound = top_candidates.top().first;
                    }
                } else if (is_allowed) {
                    candidateSet.emplace(-dist_cand, candidate_id);
                }
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
}

template<typename dist_t>
std::priority_queue<std::pair<dist_t, labeltype >>
HierarchicalNSW<dist_t>::searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed) const {
    std::priority_queue<std::pair<dist_t, labeltype >> result;
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