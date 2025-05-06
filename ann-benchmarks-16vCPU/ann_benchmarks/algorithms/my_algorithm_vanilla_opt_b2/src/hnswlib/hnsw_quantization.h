#pragma once

#include "hnsw_class_declaration.h"
#include <stdexcept>
#include <iostream> // For basic output/logging

namespace hnswlib {

// --- OPQ/PQ Training and Data Access Implementation ---

template<typename dist_t>
void HierarchicalNSW<dist_t>::trainQuantizer(const float* train_data, size_t num_train_points, size_t pq_m, size_t pq_kbits) {
    if (data_size_ == 0) {
        throw std::runtime_error("Data size (dimension) must be known before training quantizer.");
    }
    if (max_elements_ == 0) {
         throw std::runtime_error("Max elements must be set before training quantizer to allocate memory.");
    }
    if (train_data == nullptr || num_train_points == 0) {
        throw std::runtime_error("Training data cannot be null or empty.");
    }

    std::cout << "Training Product Quantizer (m=" << pq_m << ", k*=" << (1 << pq_kbits) << ", " << pq_kbits << " bits)..." << std::endl;

    // 1. Create the quantizer instance
    try {
        pq_ = std::make_unique<ProductQuantizer>(data_size_, pq_m, pq_kbits);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error("Failed to initialize ProductQuantizer: " + std::string(e.what()));
    }
    pq_code_size_ = pq_->code_size;
    std::cout << "  - PQ Code size: " << pq_code_size_ << " bytes" << std::endl;

    // 2. Allocate memory for compressed codes
    if (compressed_codes_memory_) {
        free(compressed_codes_memory_); // Free existing memory if retraining
    }
    compressed_codes_memory_ = (char*)malloc(max_elements_ * pq_code_size_);
    if (compressed_codes_memory_ == nullptr) {
        pq_ = nullptr; // Clean up
        throw std::runtime_error("Failed to allocate memory for compressed codes.");
    }
    memset(compressed_codes_memory_, 0, max_elements_ * pq_code_size_); // Initialize memory

    // 3. Train the quantizer (placeholder call)
    // In a real implementation, this would involve potentially complex
    // OPQ rotation learning and k-means clustering for each subspace.
    try {
         pq_->train(train_data, num_train_points);
         std::cout << "  - Quantizer training finished (Placeholder - OPQ: "
                   << (pq_->opq_trained ? "yes" : "no") << ")." << std::endl;
    } catch (const std::exception& e) {
         free(compressed_codes_memory_); // Clean up allocated memory
         compressed_codes_memory_ = nullptr;
         pq_ = nullptr;
         throw std::runtime_error("Error during quantizer training: " + std::string(e.what()));
    }


    // 4. Set flag to use compression
    use_compression_ = true;
}

// Get a pointer to the compressed code for a given internal ID
template<typename dist_t>
const char* HierarchicalNSW<dist_t>::getCompressedCodePtr(tableint internal_id) const {
    if (!use_compression_) {
        return nullptr; // Or throw error? Returning null seems safer.
    }
    if (internal_id >= max_elements_) {
         throw std::runtime_error("Internal ID out of bounds.");
    }
    return compressed_codes_memory_ + internal_id * pq_code_size_;
}

// Placeholder for ADC distance calculation
template<typename dist_t>
dist_t HierarchicalNSW<dist_t>::distanceADC(const void* query_vector, tableint internal_id) const {
    if (!use_compression_ || !pq_) {
         throw std::runtime_error("DistanceADC called but compression is not enabled or quantizer not trained.");
    }
    const char* code_ptr = getCompressedCodePtr(internal_id);
    // In a real implementation, call the quantizer's ADC method
    return pq_->distanceADC(static_cast<const float*>(query_vector), code_ptr);
}

// Modify getDataByLabel to return decoded data if compressed
template<typename dist_t>
template<typename data_t>
std::vector<data_t> HierarchicalNSW<dist_t>::getDataByLabel(labeltype label) const {
    // Find internal ID (existing logic)
    std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
    std::unique_lock <std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
        throw std::runtime_error("Label not found");
    }
    tableint internalId = search->second;
    lock_table.unlock();
    lock_label.unlock(); // Release locks earlier

    std::vector<data_t> data(data_size_); // Allocate space for the vector

    if (use_compression_) {
        if (!pq_) {
             throw std::runtime_error("Compression enabled but quantizer not available for decoding.");
        }
        const char* code_ptr = getCompressedCodePtr(internalId);
        // Decode the vector (placeholder call)
        // Assuming data_t is float here, might need template specialization if not
        static_assert(std::is_same<data_t, float>::value, "Decoding currently assumes float data type");
        pq_->decode(code_ptr, data.data());
    } else {
        // Retrieve original vector
        char* data_ptrv = getDataByInternalId(internalId);
        memcpy(data.data(), data_ptrv, data_size_ * sizeof(data_t));
    }

    return data;
}


} // namespace hnswlib 