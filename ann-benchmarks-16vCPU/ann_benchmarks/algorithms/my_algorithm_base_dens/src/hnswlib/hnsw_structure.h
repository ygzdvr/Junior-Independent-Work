#pragma once

#include "hnsw_class_declaration.h"

namespace hnswlib {

template<typename dist_t>
tableint HierarchicalNSW<dist_t>::mutuallyConnectNewElement(
    const void *data_point,
    tableint cur_c,
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    int level,
    bool isUpdate) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByAngularDiversity(top_candidates, M_, data_point);
    if (top_candidates.size() > M_)
        throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
        selectedNeighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();

    {
        // lock only during the update
        // because during the addition the lock for cur_c is already acquired
        std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
        if (isUpdate) {
            lock.lock();
        }
        linklistsizeint *ll_cur;
        if (level == 0)
            ll_cur = get_linklist0(cur_c);
        else
            ll_cur = get_linklist(cur_c, level);

        if (*ll_cur && !isUpdate) {
            throw std::runtime_error("The newly inserted element should have blank link list");
        }
        setListCount(ll_cur, selectedNeighbors.size());
        tableint *data = (tableint *) (ll_cur + 1);
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            if (data[idx] && !isUpdate)
                throw std::runtime_error("Possible memory corruption");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            data[idx] = selectedNeighbors[idx];
        }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

        linklistsizeint *ll_other;
        if (level == 0)
            ll_other = get_linklist0(selectedNeighbors[idx]);
        else
            ll_other = get_linklist(selectedNeighbors[idx], level);

        size_t sz_link_list_other = getListCount(ll_other);

        if (sz_link_list_other > Mcurmax)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (selectedNeighbors[idx] == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");
        if (level > element_levels_[selectedNeighbors[idx]])
            throw std::runtime_error("Trying to make a link on a non-existent level");

        tableint *data = (tableint *) (ll_other + 1);

        bool is_cur_c_present = false;
        if (isUpdate) {
            for (size_t j = 0; j < sz_link_list_other; j++) {
                if (data[j] == cur_c) {
                    is_cur_c_present = true;
                    break;
                }
            }
        }

        // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
        if (!is_cur_c_present) {
            if (sz_link_list_other < Mcurmax) {
                data[sz_link_list_other] = cur_c;
                setListCount(ll_other, sz_link_list_other + 1);
            } else {
                // finding the "weakest" element to replace it with the new one
                dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                            dist_func_param_);
                // Heuristic:
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(
                            fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                            dist_func_param_), data[j]);
                }

                getNeighborsByAngularDiversity(candidates, Mcurmax, getDataByInternalId(selectedNeighbors[idx]));

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }

                setListCount(ll_other, indx);
            }
        }
    }

    return next_closest_entry_point;
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::resizeIndex(size_t new_max_elements) {
    if (new_max_elements < cur_element_count)
        throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

    visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

    element_levels_.resize(new_max_elements);

    std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

    // Reallocate base layer
    char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
    if (data_level0_memory_new == nullptr)
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
    data_level0_memory_ = data_level0_memory_new;

    // Reallocate all other layers
    char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
    if (linkLists_new == nullptr)
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
    linkLists_ = linkLists_new;

    max_elements_ = new_max_elements;
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
    // update the feature vector associated with existing point with new vector
    memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

    int maxLevelCopy = maxlevel_;
    tableint entryPointCopy = enterpoint_node_;
    // If point to be updated is entry point and graph just contains single element then just return.
    if (entryPointCopy == internalId && cur_element_count == 1)
        return;

    int elemLevel = element_levels_[internalId];
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int layer = 0; layer <= elemLevel; layer++) {
        std::unordered_set<tableint> sCand;
        std::unordered_set<tableint> sNeigh;
        std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
        if (listOneHop.size() == 0)
            continue;

        sCand.insert(internalId);

        for (auto&& elOneHop : listOneHop) {
            sCand.insert(elOneHop);

            if (distribution(update_probability_generator_) > updateNeighborProbability)
                continue;

            sNeigh.insert(elOneHop);

            std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
            for (auto&& elTwoHop : listTwoHop) {
                sCand.insert(elTwoHop);
            }
        }

        for (auto&& neigh : sNeigh) {
            // if (neigh == internalId)
            //     continue;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
            size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
            size_t elementsToKeep = std::min(ef_construction_, size);
            for (auto&& cand : sCand) {
                if (cand == neigh)
                    continue;

                dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                if (candidates.size() < elementsToKeep) {
                    candidates.emplace(distance, cand);
                } else {
                    if (distance < candidates.top().first) {
                        candidates.pop();
                        candidates.emplace(distance, cand);
                    }
                }
            }

            // Retrieve neighbours using heuristic and set connections.
            getNeighborsByAngularDiversity(candidates, layer == 0 ? maxM0_ : maxM_, getDataByInternalId(neigh));

            {
                std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                linklistsizeint *ll_cur;
                ll_cur = get_linklist_at_level(neigh, layer);
                size_t candSize = candidates.size();
                setListCount(ll_cur, candSize);
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < candSize; idx++) {
                    data[idx] = candidates.top().second;
                    candidates.pop();
                }
            }
        }
    }

    repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::repairConnectionsForUpdate(
    const void *dataPoint,
    tableint entryPointInternalId,
    tableint dataPointInternalId,
    int dataPointLevel,
    int maxLevel) {
    tableint currObj = entryPointInternalId;
    if (dataPointLevel < maxLevel) {
        dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
        for (int level = maxLevel; level > dataPointLevel; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;
                std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                data = get_linklist_at_level(currObj, level);
                int size = getListCount(data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
    }

    if (dataPointLevel > maxLevel)
        throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

    for (int level = dataPointLevel; level >= 0; level--) {
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                currObj, dataPoint, level);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
        while (topCandidates.size() > 0) {
            if (topCandidates.top().second != dataPointInternalId)
                filteredTopCandidates.push(topCandidates.top());

            topCandidates.pop();
        }

        // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
        // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
        if (filteredTopCandidates.size() > 0) {
            bool epDeleted = isMarkedDeleted(entryPointInternalId);
            if (epDeleted) {
                filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                if (filteredTopCandidates.size() > ef_construction_)
                    filteredTopCandidates.pop();
            }

            currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
        }
    }
}

template<typename dist_t>
tableint HierarchicalNSW<dist_t>::addPoint(const void *data_point, labeltype label, int level) {
    tableint cur_c = 0;
    {
        // Checking if the element with the same label already exists
        // if so, updating it *instead* of creating a new element.
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search != label_lookup_.end()) {
            tableint existingInternalId = search->second;
            if (allow_replace_deleted_) {
                if (isMarkedDeleted(existingInternalId)) {
                    throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                }
            }
            lock_table.unlock();

            if (isMarkedDeleted(existingInternalId)) {
                unmarkDeletedInternal(existingInternalId);
            }
            updatePoint(data_point, existingInternalId, 1.0);

            return existingInternalId;
        }

        if (cur_element_count >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }

        cur_c = cur_element_count;
        cur_element_count++;
        label_lookup_[label] = cur_c;
    }

    std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = getRandomLevel(mult_);
    if (level > 0)
        curlevel = level;

    element_levels_[cur_c] = curlevel;

    std::unique_lock <std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy)
        templock.unlock();
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

    // Initialisation of the data and label
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    if (curlevel) {
        linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
        if (linkLists_[cur_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
        if (curlevel < maxlevelcopy) {
            dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxlevelcopy; level > curlevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist(currObj, level);
                    int size = getListCount(data);

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        bool epDeleted = isMarkedDeleted(enterpoint_copy);
        for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
            if (level > maxlevelcopy || level < 0)  // possible?
                throw std::runtime_error("Level error");

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                    currObj, data_point, level);
            if (epDeleted) {
                top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();
            }
            currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
        }
    } else {
        // Do nothing for the first element
        enterpoint_node_ = 0;
        maxlevel_ = curlevel;
    }

    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }
    return cur_c;
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::checkIntegrity() {
    int connections_checked = 0;
    std::vector <int > inbound_connections_num(cur_element_count, 0);
    for (int i = 0; i < cur_element_count; i++) {
        for (int l = 0; l <= element_levels_[i]; l++) {
            linklistsizeint *ll_cur = get_linklist_at_level(i, l);
            int size = getListCount(ll_cur);
            tableint *data = (tableint *) (ll_cur + 1);
            std::unordered_set<tableint> s;
            for (int j = 0; j < size; j++) {
                assert(data[j] < cur_element_count);
                assert(data[j] != i);
                inbound_connections_num[data[j]]++;
                s.insert(data[j]);
                connections_checked++;
            }
            assert(s.size() == size);
        }
    }
    if (cur_element_count > 1) {
        int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
        for (int i=0; i < cur_element_count; i++) {
            assert(inbound_connections_num[i] > 0);
            min1 = std::min(inbound_connections_num[i], min1);
            max1 = std::max(inbound_connections_num[i], max1);
        }
        std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
    }
    std::cout << "integrity ok, checked " << connections_checked << " connections\n";
}

}  // namespace hnswlib 