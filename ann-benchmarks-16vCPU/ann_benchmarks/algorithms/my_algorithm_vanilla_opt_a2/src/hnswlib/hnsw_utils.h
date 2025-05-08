#pragma once

#include "hnsw_class_declaration.h"

namespace hnswlib {

template<typename dist_t>
void HierarchicalNSW<dist_t>::setEf(size_t ef) {
    ef_ = ef;
}

template<typename dist_t>
inline std::mutex& HierarchicalNSW<dist_t>::getLabelOpMutex(labeltype label) const {
    // calculate hash
    size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
    return label_op_locks_[lock_id];
}

template<typename dist_t>
inline labeltype HierarchicalNSW<dist_t>::getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
    return return_label;
}

template<typename dist_t>
inline void HierarchicalNSW<dist_t>::setExternalLabel(tableint internal_id, labeltype label) const {
    memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
}

template<typename dist_t>
inline labeltype *HierarchicalNSW<dist_t>::getExternalLabeLp(tableint internal_id) const {
    return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
}

template<typename dist_t>
inline char *HierarchicalNSW<dist_t>::getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
}

template<typename dist_t>
int HierarchicalNSW<dist_t>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int) r;
}

template<typename dist_t>
size_t HierarchicalNSW<dist_t>::getMaxElements() {
    return max_elements_;
}

template<typename dist_t>
size_t HierarchicalNSW<dist_t>::getCurrentElementCount() {
    return cur_element_count;
}

template<typename dist_t>
size_t HierarchicalNSW<dist_t>::getDeletedCount() {
    return num_deleted_;
}

template<typename dist_t>
linklistsizeint *HierarchicalNSW<dist_t>::get_linklist0(tableint internal_id) const {
    return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
}

template<typename dist_t>
linklistsizeint *HierarchicalNSW<dist_t>::get_linklist0(tableint internal_id, char *data_level0_memory_) const {
    return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
}

template<typename dist_t>
linklistsizeint *HierarchicalNSW<dist_t>::get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
}

template<typename dist_t>
linklistsizeint *HierarchicalNSW<dist_t>::get_linklist_at_level(tableint internal_id, int level) const {
    return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
}

template<typename dist_t>
unsigned short int HierarchicalNSW<dist_t>::getListCount(linklistsizeint * ptr) const {
    return *((unsigned short int *)ptr);
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::setListCount(linklistsizeint * ptr, unsigned short int size) const {
    *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
}

template<typename dist_t>
bool HierarchicalNSW<dist_t>::isMarkedDeleted(tableint internalId) const {
    unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
    return *ll_cur & DELETE_MARK;
}

template<typename dist_t>
std::vector<tableint> HierarchicalNSW<dist_t>::getConnectionsWithLock(tableint internalId, int level) {
    std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
    unsigned int *data = get_linklist_at_level(internalId, level);
    int size = getListCount(data);
    std::vector<tableint> result(size);
    tableint *ll = (tableint *) (data + 1);
    memcpy(result.data(), ll, size * sizeof(tableint));
    return result;
}

}  // namespace hnswlib 