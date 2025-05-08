#pragma once

#include "hnsw_class_declaration.h"

namespace hnswlib {

template<typename dist_t>
void HierarchicalNSW<dist_t>::getNeighborsByHeuristic2(
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
            if (alpha_ * curdist < dist_to_query) {
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