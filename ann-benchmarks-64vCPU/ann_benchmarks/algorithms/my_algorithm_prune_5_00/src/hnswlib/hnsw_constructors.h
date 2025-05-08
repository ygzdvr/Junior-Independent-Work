#pragma once

#include "hnsw_class_declaration.h"
#include <iostream>

namespace hnswlib {

template<typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(SpaceInterface<dist_t> *s) {
}

template<typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(
    SpaceInterface<dist_t> *s,
    const std::string &location,
    bool nmslib,
    size_t max_elements,
    bool allow_replace_deleted)
    : allow_replace_deleted_(allow_replace_deleted) {
    loadIndex(location, s, max_elements);
}

template<typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(
    SpaceInterface<dist_t> *s,
    size_t max_elements,
    size_t M,
    size_t ef_construction,
    size_t random_seed,
    bool allow_replace_deleted,
    dist_t alpha)
    : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
        link_list_locks_(max_elements),
        element_levels_(max_elements),
        allow_replace_deleted_(allow_replace_deleted),
        alpha_(alpha) {
    max_elements_ = max_elements;
    num_deleted_ = 0;
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    if ( M <= 10000 ) {
        M_ = M;
    } else {
        HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
        HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
        M_ = 10000;
    }
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory");

    cur_element_count = 0;

    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
    if (linkLists_ == nullptr)
        throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;
}

template<typename dist_t>
HierarchicalNSW<dist_t>::~HierarchicalNSW() {
    clear();
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::clear() {
    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    for (tableint i = 0; i < cur_element_count; i++) {
        if (element_levels_[i] > 0)
            free(linkLists_[i]);
    }
    free(linkLists_);
    linkLists_ = nullptr;
    cur_element_count = 0;
    visited_list_pool_.reset(nullptr);
}

}  // namespace hnswlib 