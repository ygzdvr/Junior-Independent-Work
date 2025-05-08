#pragma once

#include "hnsw_class_declaration.h"
#include <iostream>

namespace hnswlib {

template<typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(SpaceInterface<dist_t> *s) {
    space_ = s;
    fstdistfunc_ = space_->get_dist_func();
    dist_func_param_ = space_->get_dist_func_param();
    cur_element_count = 0;
    num_deleted_ = 0;
}

template<typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(
    SpaceInterface<dist_t> *s,
    const std::string &location,
    bool nmslib,
    size_t max_elements,
    bool allow_replace_deleted)
    : allow_replace_deleted_(allow_replace_deleted) {
    space_ = s;
    fstdistfunc_ = space_->get_dist_func();
    dist_func_param_ = space_->get_dist_func_param();
    cur_element_count = 0;
    num_deleted_ = 0;

    if (!location.empty()) {
        loadIndex(location, s, max_elements);
    }
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
    space_ = s;
    fstdistfunc_ = space_->get_dist_func();
    dist_func_param_ = space_->get_dist_func_param();
    data_size_ = s->get_data_size();
    
    // Dimensionality reduction parameters initialization
    use_dim_reduction_ = false;
    dim_reduction_threshold_level_ = 1;
    full_dim_ = data_size_ / sizeof(float); // Initialize full_dim_
    reduction_type_ = "random"; // Default initialization
    reduced_vectors_memory_ = nullptr;
    reduced_vector_size_ = 0;

    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    data_level0_memory_ = nullptr;
    linkLists_ = nullptr;

    // Initializing level size distributions
    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    mult_ = 1 / log(1.0 * M_);  // Level multiplier
    revSize_ = 1.0 / mult_;

    max_elements_ = max_elements;

    // Calculate sizes based on original HNSWLib approach
    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint); // Links for levels > 0
    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);    // Links for level 0
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype); // Total size for level 0 element
    offsetData_ = size_links_level0_; // Offset for vector data in level 0 block
    label_offset_ = size_links_level0_ + data_size_; // Offset for label in level 0 block
    offsetLevel0_ = 0; // Links start at offset 0 within the level 0 block
    
    // Memory allocation for level 0 elements (data + links + label)
    data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate level 0");
    
    // Allocate memory for storing element levels
    element_levels_ = std::vector<int>(max_elements_);
    std::fill(element_levels_.begin(), element_levels_.end(), 0);
    
    // Memory allocation for links (levels > 0)
    linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
    if (linkLists_ == nullptr)
        throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate link lists");
    
    // Zero initialization for linkLists_
    memset(linkLists_, 0, sizeof(void *) * max_elements_);
    
    // Allocate space for entry point (initially not set)
    enterpoint_node_ = -1;
    
    // Maximum level in the graph
    maxlevel_ = -1;
    
    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));
}

template<typename dist_t>
HierarchicalNSW<dist_t>::~HierarchicalNSW() {
    clear();
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::clear() {
    // Free memory used by the index structure
    if (linkLists_) {
        for (size_t i = 0; i < max_elements_; i++) {
            if (linkLists_[i])
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
    }
    
    if (data_level0_memory_) {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
    }
    
    if (reduced_vectors_memory_) {
        free(reduced_vectors_memory_);
        reduced_vectors_memory_ = nullptr;
    }
    
    if (visited_list_pool_) {
        visited_list_pool_.reset();
    }
    
    element_levels_.clear();
    link_list_locks_.clear();
    label_op_locks_.clear();
    dim_reduction_per_level_.clear();
    
    cur_element_count = 0;
    num_deleted_ = 0;
    max_elements_ = 0;
    enterpoint_node_ = -1;
    maxlevel_ = -1;
}

}  // namespace hnswlib 