#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <algorithm>
#include <assert.h>

namespace hnswlib {

/**
 * Abstract base class for dimensionality reduction techniques
 */
class DimensionalityReduction {
public:
    virtual ~DimensionalityReduction() = default;
    
    // Project a high-dimensional vector to a lower-dimensional space
    virtual void project(const void* input, void* output) const = 0;
    
    // Get the original dimensionality
    virtual size_t getSourceDim() const = 0;
    
    // Get the reduced dimensionality
    virtual size_t getTargetDim() const = 0;
    
    // Get size in bytes of the target dimension elements
    virtual size_t getTargetElementSize() const = 0;
};

/**
 * Random Projection based dimensionality reduction
 * Simple but effective technique that preserves distances in expectation (Johnson-Lindenstrauss lemma)
 */
class RandomProjection : public DimensionalityReduction {
public:
    RandomProjection(size_t source_dim, size_t target_dim, unsigned int seed = 42) 
        : source_dim_(source_dim), target_dim_(target_dim) {
        
        assert(target_dim_ <= source_dim_);
        
        // Allocate and initialize projection matrix
        projection_matrix_.resize(source_dim_ * target_dim_);
        
        // Initialize random number generator with seed for reproducibility
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(target_dim_)));
        
        // Fill the projection matrix with random values
        for (size_t i = 0; i < source_dim_ * target_dim_; i++) {
            projection_matrix_[i] = dist(rng);
        }
    }
    
    void project(const void* input, void* output) const override {
        const float* input_vector = static_cast<const float*>(input);
        float* output_vector = static_cast<float*>(output);
        
        // Initialize output vector with zeros
        std::fill(output_vector, output_vector + target_dim_, 0.0f);
        
        // Perform matrix multiplication: output = input * projection_matrix
        for (size_t i = 0; i < target_dim_; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < source_dim_; j++) {
                sum += input_vector[j] * projection_matrix_[j * target_dim_ + i];
            }
            output_vector[i] = sum;
        }
    }
    
    size_t getSourceDim() const override {
        return source_dim_;
    }
    
    size_t getTargetDim() const override {
        return target_dim_;
    }
    
    size_t getTargetElementSize() const override {
        return sizeof(float) * target_dim_;
    }
    
private:
    size_t source_dim_;
    size_t target_dim_;
    std::vector<float> projection_matrix_;
};

/**
 * PCA-based dimensionality reduction
 * This implementation requires a training set to compute the principal components
 */
class PCAReduction : public DimensionalityReduction {
public:
    // Constructor with training data
    PCAReduction(const std::vector<std::vector<float>>& training_data, size_t target_dim) 
        : source_dim_(training_data[0].size()), target_dim_(target_dim) {
        
        assert(!training_data.empty());
        assert(target_dim_ <= source_dim_);
        
        // Compute mean vector
        std::vector<float> mean(source_dim_, 0.0f);
        for (const auto& vector : training_data) {
            for (size_t i = 0; i < source_dim_; i++) {
                mean[i] += vector[i];
            }
        }
        for (size_t i = 0; i < source_dim_; i++) {
            mean[i] /= training_data.size();
        }
        
        // Store mean vector for later use
        mean_vector_ = mean;
        
        // Compute covariance matrix (simple implementation, not optimized)
        std::vector<float> covariance(source_dim_ * source_dim_, 0.0f);
        for (const auto& vector : training_data) {
            for (size_t i = 0; i < source_dim_; i++) {
                for (size_t j = 0; j < source_dim_; j++) {
                    covariance[i * source_dim_ + j] += (vector[i] - mean[i]) * (vector[j] - mean[j]);
                }
            }
        }
        for (size_t i = 0; i < source_dim_ * source_dim_; i++) {
            covariance[i] /= (training_data.size() - 1);
        }
        
        // For a real implementation, compute eigenvectors of covariance matrix
        // Here we'll use a placeholder implementation with random vectors for simplicity
        // In production, use a proper linear algebra library to compute eigenvectors
        
        // Placeholder: create random orthogonal vectors
        projection_matrix_.resize(source_dim_ * target_dim_);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < target_dim_; i++) {
            // Generate random vector
            std::vector<float> v(source_dim_);
            for (size_t j = 0; j < source_dim_; j++) {
                v[j] = dist(rng);
            }
            
            // Orthogonalize against previous vectors (Gram-Schmidt)
            for (size_t j = 0; j < i; j++) {
                float dot = 0.0f;
                for (size_t k = 0; k < source_dim_; k++) {
                    dot += v[k] * projection_matrix_[k * target_dim_ + j];
                }
                for (size_t k = 0; k < source_dim_; k++) {
                    v[k] -= dot * projection_matrix_[k * target_dim_ + j];
                }
            }
            
            // Normalize
            float norm = 0.0f;
            for (size_t j = 0; j < source_dim_; j++) {
                norm += v[j] * v[j];
            }
            norm = std::sqrt(norm);
            for (size_t j = 0; j < source_dim_; j++) {
                v[j] /= norm;
            }
            
            // Store the eigenvector
            for (size_t j = 0; j < source_dim_; j++) {
                projection_matrix_[j * target_dim_ + i] = v[j];
            }
        }
    }
    
    // Static factory method to create from existing data
    static std::shared_ptr<PCAReduction> createFromData(const float* data, size_t n_vectors, size_t source_dim, size_t target_dim) {
        std::vector<std::vector<float>> training_data;
        for (size_t i = 0; i < n_vectors; i++) {
            std::vector<float> vector(source_dim);
            for (size_t j = 0; j < source_dim; j++) {
                vector[j] = data[i * source_dim + j];
            }
            training_data.push_back(vector);
        }
        return std::make_shared<PCAReduction>(training_data, target_dim);
    }
    
    void project(const void* input, void* output) const override {
        const float* input_vector = static_cast<const float*>(input);
        float* output_vector = static_cast<float*>(output);
        
        // Initialize output vector with zeros
        std::fill(output_vector, output_vector + target_dim_, 0.0f);
        
        // Center the input by subtracting the mean
        std::vector<float> centered(source_dim_);
        for (size_t i = 0; i < source_dim_; i++) {
            centered[i] = input_vector[i] - mean_vector_[i];
        }
        
        // Project onto principal components: output = centered_input * projection_matrix
        for (size_t i = 0; i < target_dim_; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < source_dim_; j++) {
                sum += centered[j] * projection_matrix_[j * target_dim_ + i];
            }
            output_vector[i] = sum;
        }
    }
    
    size_t getSourceDim() const override {
        return source_dim_;
    }
    
    size_t getTargetDim() const override {
        return target_dim_;
    }
    
    size_t getTargetElementSize() const override {
        return sizeof(float) * target_dim_;
    }
    
private:
    size_t source_dim_;
    size_t target_dim_;
    std::vector<float> mean_vector_;
    std::vector<float> projection_matrix_; // Eigenvectors as columns
};

} // namespace hnswlib 