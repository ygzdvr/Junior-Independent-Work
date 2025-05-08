#pragma once

#include "hnsw_class_declaration.h"
#include <fstream>

namespace hnswlib {

template<typename dist_t>
size_t HierarchicalNSW<dist_t>::indexFileSize() const {
    size_t size = 0;
    size += sizeof(offsetLevel0_);
    size += sizeof(max_elements_);
    size += sizeof(cur_element_count);
    size += sizeof(size_data_per_element_);
    size += sizeof(label_offset_);
    size += sizeof(offsetData_);
    size += sizeof(maxlevel_);
    size += sizeof(enterpoint_node_);
    size += sizeof(maxM_);

    size += sizeof(maxM0_);
    size += sizeof(M_);
    size += sizeof(mult_);
    size += sizeof(ef_construction_);

    size += cur_element_count * size_data_per_element_;

    for (size_t i = 0; i < cur_element_count; i++) {
        unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
        size += sizeof(linkListSize);
        size += linkListSize;
    }
    return size;
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);

    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);

    output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count; i++) {
        unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
        writeBinaryPOD(output, linkListSize);
        if (linkListSize)
            output.write(linkLists_[i], linkListSize);
    }
    output.close();
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i) {
    std::ifstream input(location, std::ios::binary);

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    clear();
    // get file size:
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);

    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count)
        max_elements = max_elements_;
    max_elements_ = max_elements;
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);

    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    auto pos = input.tellg();

    /// Optional - check if index is ok:
    input.seekg(cur_element_count * size_data_per_element_, input.cur);
    for (size_t i = 0; i < cur_element_count; i++) {
        if (input.tellg() < 0 || input.tellg() >= total_filesize) {
            throw std::runtime_error("Index seems to be corrupted or unsupported");
        }

        unsigned int linkListSize;
        readBinaryPOD(input, linkListSize);
        if (linkListSize != 0) {
            input.seekg(linkListSize, input.cur);
        }
    }

    // throw exception if it either corrupted or old index
    if (input.tellg() != total_filesize)
        throw std::runtime_error("Index seems to be corrupted or unsupported");

    input.clear();
    /// Optional check end

    input.seekg(pos, input.beg);

    data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    std::vector<std::mutex>(max_elements).swap(link_list_locks_);
    std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

    visited_list_pool_.reset(new VisitedListPool(1, max_elements));

    linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
    if (linkLists_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
    element_levels_ = std::vector<int>(max_elements);
    revSize_ = 1.0 / mult_;
    ef_ = 10;
    for (size_t i = 0; i < cur_element_count; i++) {
        label_lookup_[getExternalLabel(i)] = i;
        unsigned int linkListSize;
        readBinaryPOD(input, linkListSize);
        if (linkListSize == 0) {
            element_levels_[i] = 0;
            linkLists_[i] = nullptr;
        } else {
            element_levels_[i] = linkListSize / size_links_per_element_;
            linkLists_[i] = (char *) malloc(linkListSize);
            if (linkLists_[i] == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
            input.read(linkLists_[i], linkListSize);
        }
    }

    for (size_t i = 0; i < cur_element_count; i++) {
        if (isMarkedDeleted(i)) {
            num_deleted_ += 1;
            if (allow_replace_deleted_) deleted_elements.insert(i);
        }
    }

    input.close();

    return;
}

}  // namespace hnswlib 