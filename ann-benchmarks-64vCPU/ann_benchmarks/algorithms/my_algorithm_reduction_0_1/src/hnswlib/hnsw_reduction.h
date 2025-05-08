#pragma once

#include "hnsw_class_declaration.h"
#include <random>
#include <cmath>

namespace hnswlib {

// Helper function for squared L2 distance calculation
template <typename dist_t>
inline dist_t l2Square(const float* a, const float* b, size_t dim) {
    dist_t result = 0;
    for (size_t i = 0; i < dim; i++) {
        dist_t diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

// Initialize projection matrices for Johnson-Lindenstrauss dimension reduction
template <typename dist_t>
void HierarchicalNSW<dist_t>::initProjections(size_t max_level,
                                          float target_eps /* =0.10f */,
                                          size_t min_dim   /* =32    */) {
    level0_dim_ = data_size_ / sizeof(float);  // store the original d

    proj_dim_.assign(max_level + 1, level0_dim_); // default = no compression
    proj_mat_.resize(max_level + 1);              // empty for uncompressed
    proj_data_.resize(max_level + 1);

    std::mt19937 gen(42);
    std::normal_distribution<float> g(0.0f, 1.0f);

    for (size_t l = 1; l <= max_level; ++l) {
        /* rough layer size   n_l ≈ N / 2^l */
        size_t n_l = std::max<size_t>(1, max_elements_ >> l);

        /* theoretical JL dimension */
        double jl_dim = 8.0 / (target_eps * target_eps) * std::log(static_cast<double>(n_l));
        size_t k_l    = static_cast<size_t>(std::ceil(jl_dim));

        /* hard caps:
         *   • never go below min_dim (accuracy safeguard)
         *   • never go above original d (makes no sense)               */
        if (k_l >= level0_dim_ || k_l < min_dim) {
            proj_dim_[l] = level0_dim_;   // mark as "no projection"
            continue;
        }

        proj_dim_[l] = k_l;               // we *do* project this layer

        /* sample R^(l)  — rows are normal(0,1/k) */
        std::vector<float> R(k_l * level0_dim_);
        float scale = 1.0f / std::sqrt(static_cast<float>(k_l));
        for (float& c : R) c = g(gen) * scale;
        proj_mat_[l].swap(R);
    }
}

// Store projected point coordinates across layers
template <typename dist_t>
void HierarchicalNSW<dist_t>::storePointAcrossLayers(tableint id,
                                                 const float* raw) {
    for (size_t l = 1; l < proj_dim_.size(); ++l) {
        size_t k = proj_dim_[l];
        if (k == level0_dim_) continue;           // this layer uncompressed

        size_t offset = id * k;
        if (proj_data_[l].size() < (id + 1) * k)
            proj_data_[l].resize((id + 1) * k);

        const float* R = proj_mat_[l].data();
        float* dest    = proj_data_[l].data() + offset;

        for (size_t r = 0; r < k; ++r) {
            const float* row = R + r * level0_dim_;
            float sum = 0.f;
            for (size_t c = 0; c < level0_dim_; ++c)
                sum += row[c] * raw[c];
            dest[r] = sum;
        }
    }
}

// Calculate distance between points based on layer
template <typename dist_t>
inline dist_t HierarchicalNSW<dist_t>::layerDist(tableint a,
                                             tableint b,
                                             size_t level) const {
    if (level == 0 || proj_dim_[level] == level0_dim_) {
        // full-dimension distance
        const float* pa = reinterpret_cast<const float*>(getDataByInternalId(a));
        const float* pb = reinterpret_cast<const float*>(getDataByInternalId(b));
        return l2Square<dist_t>(pa, pb, level0_dim_);
    }

    /* compressed distance */
    size_t k = proj_dim_[level];
    const float* pa = proj_data_[level].data() + a * k;
    const float* pb = proj_data_[level].data() + b * k;
    return l2Square<dist_t>(pa, pb, k);
}

} // namespace hnswlib 