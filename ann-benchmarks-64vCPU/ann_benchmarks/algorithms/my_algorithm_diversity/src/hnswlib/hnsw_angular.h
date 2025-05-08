#pragma once

#include "hnsw_class_declaration.h"
#include <algorithm>
#include <cmath>

namespace hnswlib {

template <typename dist_t>
inline float HierarchicalNSW<dist_t>::cosFromV(tableint v,
                                               tableint a,
                                               tableint b) const {
    const float* va = reinterpret_cast<const float*>(getDataByInternalId(a));
    const float* vb = reinterpret_cast<const float*>(getDataByInternalId(b));
    const float* vv = reinterpret_cast<const float*>(getDataByInternalId(v));
    size_t dim = data_size_ / sizeof(float);

    float dot_ab = 0.f, dot_va = 0.f, dot_vb = 0.f;
    float na = 0.f, nb = 0.f, nv = 0.f;

    for (size_t i = 0; i < dim; ++i) {
        float pa = va[i], pb = vb[i], pv = vv[i];
        dot_ab += pa * pb;
        dot_va += pv * pa;
        dot_vb += pv * pb;
        na += pa * pa;
        nb += pb * pb;
        nv += pv * pv;
    }

    float num = dot_ab - dot_va - dot_vb + nv;
    float norm_a = na + nv - 2.f * dot_va;          // |a-v|^2
    float norm_b = nb + nv - 2.f * dot_vb;          // |b-v|^2
    float denom = std::sqrt(norm_a * norm_b);

    return (denom == 0.f) ? 1.f : num / denom;      // safe fallback
}


template <typename dist_t>
void HierarchicalNSW<dist_t>::selectAngularDiverseNeighbors(
        tableint v,
        std::vector<std::pair<dist_t,tableint>>& candidates,
        size_t M,
        float eps_rad,
        std::vector<tableint>& pool)
{
    pool.clear();
    if (candidates.empty()) return;

    // 1. add the closest candidate (distance field is already computed)
    auto best = std::min_element(candidates.begin(), candidates.end(),
                               [](auto& lhs, auto& rhs){ return lhs.first < rhs.first; });
    pool.push_back(best->second);

    // Use class member variable if eps_rad is not explicitly provided (or is 0)
    const float cos_eps = std::cos(eps_rad > 0.0f ? eps_rad : eps_rad_);

    // 2. make at most M-1 extra passes
    while (pool.size() < M) {
        float worst_cos = 1.f;          // 1 = 0°   (we search for the smallest cos → largest angle)
        int   worst_idx = -1;

        // search the candidate that is farthest (in angle) from its closest chosen neighbour
        for (auto& pr : candidates) {
            tableint c = pr.second;

            // skip if already chosen
            if (std::find(pool.begin(), pool.end(), c) != pool.end())
                continue;

            float best_cos = 1.f;
            for (tableint w : pool) {
                best_cos = std::min(best_cos, cosFromV(v, c, w));
                if (best_cos < cos_eps) break;  // no need to check further, already good enough
            }
            if (best_cos > cos_eps && best_cos < worst_cos) {
                worst_cos = best_cos;
                worst_idx = static_cast<int>(c);
            }
        }

        // stop if every remaining candidate is within eps_rad
        if (worst_idx == -1) break;

        pool.push_back(static_cast<tableint>(worst_idx));
    }

    // 3. overwrite the vector 'candidates' so the caller can link them
    candidates.clear();
    for (tableint id : pool) candidates.emplace_back(0.f /*unused*/, id);
}

} // namespace hnswlib 