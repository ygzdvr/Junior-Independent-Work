#pragma once

#include "hnsw_types.h"
#include "visited_list_pool.h"
#include <atomic>
#include <random>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <memory>
#include <string>

namespace hnswlib {

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    // Helper struct for comparing distances
    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
                                std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };
    dist_t pruning_alpha_{1.2f};  // Alpha parameter for robust pruning (DiskANN-inspired)

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    // Constructors and Destructor
    HierarchicalNSW(SpaceInterface<dist_t> *s);
    HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements = 0, bool allow_replace_deleted = false);
    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100, bool allow_replace_deleted = false);
    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M, size_t ef_construction, size_t random_seed, bool allow_replace_deleted, dist_t alpha);
    ~HierarchicalNSW();
    void clear();

    // Parameter settings
    void setEf(size_t ef);
    void setPruningAlpha(dist_t alpha) { pruning_alpha_ = alpha; }
    dist_t getPruningAlpha() const { return pruning_alpha_; }

    // Low-level data access
    std::mutex& getLabelOpMutex(labeltype label) const;
    labeltype getExternalLabel(tableint internal_id) const;
    void setExternalLabel(tableint internal_id, labeltype label) const;
    labeltype *getExternalLabeLp(tableint internal_id) const;
    char *getDataByInternalId(tableint internal_id) const;
    int getRandomLevel(double reverse_size);
    size_t getMaxElements();
    size_t getCurrentElementCount();
    size_t getDeletedCount();
    linklistsizeint *get_linklist0(tableint internal_id) const;
    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const;
    linklistsizeint *get_linklist(tableint internal_id, int level) const;
    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const;
    unsigned short int getListCount(linklistsizeint *ptr) const;
    void setListCount(linklistsizeint *ptr, unsigned short int size) const;
    bool isMarkedDeleted(tableint internalId) const;
    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level);

    // Element Operations
    void markDelete(labeltype label);
    void markDeletedInternal(tableint internalId);
    void unmarkDelete(labeltype label);
    void unmarkDeletedInternal(tableint internalId);
    void addPoint(const void *data_point, labeltype label, bool replace_deleted);
    
    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const;

    // Structure Operations
    tableint addPoint(const void *data_point, labeltype label, int level = -1);
    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability);
    void checkIntegrity();
    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate);
    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M);
    std::vector<std::pair<dist_t, tableint>> robustPrune(
        const std::vector<std::pair<dist_t, tableint>>& candidates,
        const void* queryPoint);
    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel);
    void resizeIndex(size_t new_max_elements);

    // Search Methods
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer);
    
    template <bool bare_bone_search = false, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const;
    
    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;
    
    std::vector<std::pair<dist_t, labeltype>>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const;

    // Index Persistence
    size_t indexFileSize() const;
    void saveIndex(const std::string &location);
    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0);

    // Robust Pruning parameter (DiskANN)
    float pruning_alpha_ = 1.0f; // Default value, can be overridden

    // **********************************
    // * Internal methods declarations
    // **********************************
};

}  // namespace hnswlib 