#pragma once

#include "hnsw_class_declaration.h"
#include "hnsw_dim_reduction.h"
#include <memory>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <numeric> // Required for std::iota if used with std::shuffle
#include <random>  // Required for std::shuffle
#include <cmath>   // Required for std::floor, std::log
#include <cstring> // Required for std::memcpy
#include <unordered_map>

namespace hnswlib {

template<typename dist_t>
void HierarchicalNSW<dist_t>::enableDimensionalityReduction(
    size_t full_dim, 
    const std::vector<size_t>& target_dims, 
    int threshold_level, 
    const std::string& reduction_type) {
    
    if (cur_element_count > 0) {
        throw std::runtime_error("Dimensionality reduction must be enabled before adding points to the index");
    }
    
    if (target_dims.empty()) {
        throw std::runtime_error("Target dimensions vector cannot be empty");
    }
    
    if (threshold_level < 0) {
        throw std::runtime_error("Threshold level cannot be negative");
    }
    
    full_dim_ = full_dim;
    target_dims_ = target_dims;
    dim_reduction_threshold_level_ = threshold_level;
    reduction_type_ = reduction_type; // Store the reduction type
    use_dim_reduction_ = true;
    
    // Calculate total size needed for reduced vectors and offsets for each level
    size_t total_reduced_size_per_element = 0;
    // Use max_elements_ to determine potential max level, +1 for safety
    // We resize offsets later if maxlevel_ increases
    reduced_vector_offsets_.assign(max_elements_ + 1, (size_t)-1); // Initialize offsets to invalid marker

    // Determine potential max level based on M and max_elements (approx log_M(max_elements))
    // This is just an initial guess, maxlevel_ is the true authority during build
    int potential_max_level = 0;
    if (max_elements_ > 0 && M_ > 0) {
        potential_max_level = static_cast<int>(floor(log(static_cast<double>(max_elements_)) * revSize_));
    }
    if (reduced_vector_offsets_.size() <= (size_t)potential_max_level) {
        reduced_vector_offsets_.resize(potential_max_level + 1, (size_t)-1);
    }

    // When multiple layers use the same target dimension, the same projection object
    // will be shared across these layers (for consistency). However, each layer still
    // needs its own storage space for the reduced vectors.
    for (int level = 0; level <= potential_max_level; level++) {
        if (level >= dim_reduction_threshold_level_) {
            int reduction_idx = level - dim_reduction_threshold_level_;
            if (reduction_idx < (int)target_dims_.size()) {
                size_t target_dim = target_dims_[reduction_idx];
                if (target_dim < full_dim_) { // Only allocate if actually reduced
                    reduced_vector_offsets_[level] = total_reduced_size_per_element;
                    total_reduced_size_per_element += target_dim * sizeof(float);
                }
            }
        }
        // If not reducing for this level, offset remains -1
    }
    reduced_vector_size_ = total_reduced_size_per_element; // Total size per element in reduced_vectors_memory_
    
    // Initialize dimensionality reduction objects vector (resize later if needed)
    if (dim_reduction_per_level_.size() <= (size_t)potential_max_level) {
        dim_reduction_per_level_.resize(potential_max_level + 1, nullptr);
    }
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::buildDimensionalityReduction() {
    if (!use_dim_reduction_ || cur_element_count == 0) {
        return; // Nothing to do
    }

    // Ensure internal structures are sized correctly for the actual maxlevel_
    if (reduced_vector_offsets_.size() <= (size_t)maxlevel_) {
        size_t old_size = reduced_vector_offsets_.size();
        reduced_vector_offsets_.resize(maxlevel_ + 1, (size_t)-1);
        // Recalculate offsets for new levels if needed (or rely on initial calc)
        // For simplicity, let's assume initial potential_max_level was sufficient or handle error later
    }
    if (dim_reduction_per_level_.size() <= (size_t)maxlevel_) {
         dim_reduction_per_level_.resize(maxlevel_ + 1, nullptr);
    }
    
    // Allocate memory for reduced vectors
    if (reduced_vectors_memory_ != nullptr) {
        free(reduced_vectors_memory_);
    }
    if (reduced_vector_size_ > 0) { // Only allocate if there's something to store
        reduced_vectors_memory_ = (char*)malloc(reduced_vector_size_ * max_elements_);
        if (reduced_vectors_memory_ == nullptr) {
            throw std::runtime_error("Failed to allocate memory for reduced vectors");
        }
    } else {
        reduced_vectors_memory_ = nullptr; // No reduced vectors needed
        return; // Exit if no reduction happens anywhere
    }
    
    // Map to store dimensionality reduction objects by target dimension
    // This allows reusing the same reduction object for multiple levels with the same dimension
    std::unordered_map<size_t, std::shared_ptr<DimensionalityReduction>> dim_reduction_by_target_dim;
    
    // First pass: create dimensionality reduction objects for each unique target dimension
    for (int level = dim_reduction_threshold_level_; level <= maxlevel_; level++) {
        if (level - dim_reduction_threshold_level_ < (int)target_dims_.size()) {
            size_t target_dim = target_dims_[level - dim_reduction_threshold_level_];
            
            if (target_dim >= full_dim_) {
                continue; // Skip levels not actually reducing
            }
            
            // Check if offset is valid for this level (means it should be reduced)
            if ((size_t)level >= reduced_vector_offsets_.size() || reduced_vector_offsets_[level] == (size_t)-1) {
                continue; // Should not happen if logic is correct, but safe check
            }
            
            // Check if we already have a reduction object for this target dimension
            if (dim_reduction_by_target_dim.find(target_dim) == dim_reduction_by_target_dim.end()) {
                // No reduction object exists yet for this dimension, create a new one
                
                // Collect a sample of vectors to train dimensionality reduction
                std::vector<std::vector<float>> training_data;
                size_t n_samples = std::min(size_t(1000), (size_t)cur_element_count);
                std::vector<tableint> sample_ids(cur_element_count);
                std::iota(sample_ids.begin(), sample_ids.end(), 0); // Fill with 0, 1, 2...
                
                // Use std::shuffle with a proper random engine
                std::shuffle(sample_ids.begin(), sample_ids.end(), level_generator_); 

                size_t actual_samples_taken = 0;
                for (tableint id : sample_ids) {
                    if (!isMarkedDeleted(id)) {
                        std::vector<float> vec(full_dim_);
                        float* vec_data = (float*)getDataByInternalId(id);
                        std::copy(vec_data, vec_data + full_dim_, vec.begin());
                        training_data.push_back(vec);
                        actual_samples_taken++;
                        if (actual_samples_taken >= n_samples) break;
                    }
                }
                if (training_data.empty()) continue; // Skip if no valid samples found
                
                // Create dimensionality reduction object using the stored reduction_type_
                std::shared_ptr<DimensionalityReduction> reduction_obj;
                if (reduction_type_ == "pca") {
                    reduction_obj = std::make_shared<PCAReduction>(training_data, target_dim);
                } else { // Default to random projection
                    reduction_obj = std::make_shared<RandomProjection>(full_dim_, target_dim);
                }
                
                // Store the reduction object in our map for reuse
                dim_reduction_by_target_dim[target_dim] = reduction_obj;
            }
        }
    }
    
    // Second pass: assign reduction objects to levels and compute projections
    for (int level = dim_reduction_threshold_level_; level <= maxlevel_; level++) {
        if (level - dim_reduction_threshold_level_ < (int)target_dims_.size()) {
            size_t target_dim = target_dims_[level - dim_reduction_threshold_level_];
            
            if (target_dim >= full_dim_) {
                continue; // Skip levels not actually reducing
            }
            
            // Check if offset is valid for this level
            if ((size_t)level >= reduced_vector_offsets_.size() || reduced_vector_offsets_[level] == (size_t)-1) {
                continue;
            }
            
            // Get the appropriate reduction object from our map
            auto it = dim_reduction_by_target_dim.find(target_dim);
            if (it != dim_reduction_by_target_dim.end()) {
                // Assign the shared reduction object to this level
                dim_reduction_per_level_[level] = it->second;
                
                // Project all existing vectors
                std::vector<float> reduced_vec(target_dim);
                for (tableint i = 0; i < (tableint)cur_element_count; i++) {
                    if (!isMarkedDeleted(i)) {
                        float* vec_data = (float*)getDataByInternalId(i);
                        dim_reduction_per_level_[level]->project(vec_data, reduced_vec.data());
                        
                        // Store the reduced vector
                        char* dst = getReducedVectorByInternalId(i, level); 
                        // Need to check if dst is valid (i.e., points into reduced_vectors_memory_)
                        if (dst != getDataByInternalId(i)) { // Check if reduction applies to this level
                           std::memcpy(dst, reduced_vec.data(), target_dim * sizeof(float));
                        } // Else: No reduction for this level, do nothing.
                    }
                }
            }
        }
    }
}

template<typename dist_t>
inline char* HierarchicalNSW<dist_t>::getReducedVectorByInternalId(tableint internal_id, int level) const {
    if (!use_dim_reduction_ || reduced_vectors_memory_ == nullptr || 
        (size_t)level >= reduced_vector_offsets_.size() || reduced_vector_offsets_[level] == (size_t)-1) {
        // Return original vector if reduction disabled, memory not alloc'd, level out of bounds, 
        // or offset is invalid (meaning no reduction defined/needed for this level)
        return getDataByInternalId(internal_id);
    }
    
    // Get offset for this level within the reduced block
    size_t level_offset = reduced_vector_offsets_[level]; 
    // Get total size per element for all combined reduced levels
    size_t total_reduced_size = reduced_vector_size_;    

    // Calculate pointer into the reduced vectors memory block
    return reduced_vectors_memory_ + internal_id * total_reduced_size + level_offset;
}

template<typename dist_t>
inline size_t HierarchicalNSW<dist_t>::getDimensionAtLevel(int level) const {
    if (!use_dim_reduction_ || level < dim_reduction_threshold_level_) {
        return full_dim_; // Use full dimension for lower levels
    }
    
    int reduction_index = level - dim_reduction_threshold_level_;
    if (reduction_index >= 0 && reduction_index < (int)target_dims_.size()) {
        // Return target dim if defined, otherwise full dim (no reduction for this specific level)
        return (target_dims_[reduction_index] < full_dim_) ? target_dims_[reduction_index] : full_dim_;
    }
    
    return full_dim_; // Default to full dimension if level is above defined target_dims range
}

template<typename dist_t>
void HierarchicalNSW<dist_t>::projectVector(const void* input, void* output, int level) const {
    // Check conditions under which projection should NOT happen
    if (!use_dim_reduction_ || level < dim_reduction_threshold_level_ || 
        (size_t)level >= dim_reduction_per_level_.size() || // Level out of bounds for DR objects
        dim_reduction_per_level_[level] == nullptr) {       // No DR object created for this level (e.g., target_dim >= full_dim)
        
        // Just copy the original vector if no reduction applies to this level
        std::memcpy(output, input, full_dim_ * sizeof(float));
        return;
    }
    
    // Project the vector using the appropriate dimensionality reduction object
    dim_reduction_per_level_[level]->project(input, output);
}

template<typename dist_t>
dist_t HierarchicalNSW<dist_t>::computeDistance(const void* lhs, const void* rhs, int level) const {
    // Check if reduction applies to this level
    if (!use_dim_reduction_ || level < dim_reduction_threshold_level_ || 
        (size_t)level >= dim_reduction_per_level_.size() || 
        dim_reduction_per_level_[level] == nullptr) {       
        
        // Use the original distance function for full dimensions
        return fstdistfunc_(lhs, rhs, dist_func_param_);
    }
    
    // If reduction applies, calculate distance in the reduced space.
    // NOTE: For performance optimization, this implementation currently uses 
    //       Euclidean (L2) distance in the reduced space *regardless* of the 
    //       original space metric (L2, IP, Cosine). Random projections 
    //       approximately preserve L2 and Cosine distances, while PCA is inherently 
    //       L2-based. Using a single fast L2 calculation simplifies the code and 
    //       speeds up high-level graph traversal. The final accurate distance 
    //       calculation still happens at level 0 with the original vectors.
    // TODO: For potentially higher accuracy with angular spaces (at the cost 
    //       of performance), implement inner product calculation here and 
    //       handle normalization if needed based on the original space type.
    
    size_t reduced_dim = dim_reduction_per_level_[level]->getTargetDim();
    const float* lhs_vec = static_cast<const float*>(lhs);
    const float* rhs_vec = static_cast<const float*>(rhs);
    
    // Simple Euclidean distance (squared) in reduced space
    float sum = 0.0f;
    for (size_t i = 0; i < reduced_dim; i++) {
        float diff = lhs_vec[i] - rhs_vec[i];
        sum += diff * diff;
    }
    
    return static_cast<dist_t>(sum);
}

} // namespace hnswlib 