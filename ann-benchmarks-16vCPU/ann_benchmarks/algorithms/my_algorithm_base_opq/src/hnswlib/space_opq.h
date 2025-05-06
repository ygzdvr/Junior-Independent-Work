#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <numeric>
#include <cmath>
#include "hnswlib.h"
// #include <faiss/impl/ProductQuantizer.h> // We'll need this eventually

// Reuse optimized L2 for sub-vectors if possible (requires careful integration)
#include "space_l2.h"

#if defined(_MSC_VER)
#   define OPQ_HSW_FORCE_INLINE __forceinline
#else
#   define OPQ_HSW_FORCE_INLINE inline __attribute__((always_inline))
#endif

namespace hnswlib {

// Forward declaration needed if not including full Faiss header yet
// namespace faiss { struct ProductQuantizer; }

namespace detail {

// Placeholder for optimized L2 square distance for sub-vectors
// Ideally reuse or adapt functions from space_l2.h
OPQ_HSW_FORCE_INLINE
float l2_sqr_sub(const float* a, const float* b, std::size_t dim) {
    return l2_sqr_scalar(a, b, dim); // Fallback to scalar for now
}

struct OPQDistanceState {
    const float* pq_centroids = nullptr;
    size_t pq_m = 0;
    size_t pq_k = 256; // Typically 256 for uint8_t codes
    size_t dsub = 0;
    size_t dim = 0; // Original dimension
};

// Asymmetric Distance Computation (Query: float, DB: uint8_t code)
OPQ_HSW_FORCE_INLINE
float distance_opq_adc(const float* query_vec, const uint8_t* db_code, const OPQDistanceState* state) {
    float total_dist = 0.0f;
    if (!state || !state->pq_centroids || state->pq_m == 0 || state->dsub == 0) {
        // Handle error or return max distance
        return std::numeric_limits<float>::max();
    }

    for (size_t m = 0; m < state->pq_m; ++m) {
        const float* sub_query = query_vec + m * state->dsub;
        uint8_t code_m = db_code[m];

        // Calculate centroid address
        const float* centroid = state->pq_centroids + (m * state->pq_k + code_m) * state->dsub;

        // Accumulate L2 squared distance for the sub-vector
        total_dist += l2_sqr_sub(sub_query, centroid, state->dsub);
    }
    return total_dist;
}

// Wrapper function matching the SpaceInterface signature
float distance_opq_wrapper(const void* query_vec_void, const void* db_code_void, const void* state_void) {
    const float* query_vec = static_cast<const float*>(query_vec_void);
    const uint8_t* db_code = static_cast<const uint8_t*>(db_code_void);
    const OPQDistanceState* state = static_cast<const OPQDistanceState*>(state_void);

    // Need to handle the query transformation (rotation R) if not done prior to search
    // Assuming query_vec is already R * original_query
    
    return distance_opq_adc(query_vec, db_code, state);
}

// ------------------ OPQ Space wrapper class ----------------------

// Note: We are calculating distances between a float query and uint8_t codes,
// but the *result* of the distance calculation is float.
struct OPQSpace final : public SpaceInterface<float> {
    using dist_t = float;
    using FnPtr = dist_t (*)(const void*, const void*, const void*);

    explicit OPQSpace(std::size_t dim, std::size_t pq_m)
        : dim_(dim), pq_m_(pq_m) {
        if (dim % pq_m != 0) {
            throw std::runtime_error("Dimension must be divisible by pq_m for OPQSpace.");
        }
        dsub_ = dim / pq_m;
        data_size_ = pq_m_ * sizeof(uint8_t); // Store pq_m bytes per vector

        // Initialize distance state
        dist_state_.pq_m = pq_m_;
        dist_state_.dsub = dsub_;
        dist_state_.dim = dim_;
        dist_state_.pq_k = 256; // Assume uint8 codes -> k=256
        dist_state_.pq_centroids = nullptr; // Must be set later
    }

    // Method to set the PQ centroids (obtained from Faiss training in Python)
    void set_pq_centroids(const float* centroids) {
        if (!centroids) {
             throw std::runtime_error("PQ centroids pointer cannot be null.");
        }
        dist_state_.pq_centroids = centroids;
        std::cout << "OPQSpace: PQ centroids set. m=" << pq_m_ 
                  << ", dsub=" << dsub_ << ", k=" << dist_state_.pq_k << std::endl;
    }

    [[nodiscard]] FnPtr get_dist_func() override {
        if (!dist_state_.pq_centroids) {
             throw std::runtime_error("PQ centroids must be set before getting distance function.");
        }
        return &distance_opq_wrapper;
    }

    // The parameter passed to the distance function will be the state struct
    [[nodiscard]] void* get_dist_func_param() override {
         if (!dist_state_.pq_centroids) {
             throw std::runtime_error("PQ centroids must be set before getting distance function param.");
        }
        return &dist_state_;
    }
    
    // Data size is the size of the compressed code
    [[nodiscard]] std::size_t get_data_size() override { return data_size_; }

private:
    std::size_t dim_;       // Original dimension
    std::size_t pq_m_;      // Number of subquantizers/bytes
    std::size_t dsub_;      // Subvector dimension
    std::size_t data_size_; // Size of stored code (pq_m bytes)

    OPQDistanceState dist_state_; // Holds data needed by the static distance function
};

} // namespace detail

// Expose OPQSpace in the hnswlib namespace
using OPQSpace = detail::OPQSpace;

} // namespace hnswlib 