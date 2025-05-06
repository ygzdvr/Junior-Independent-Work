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
#include <cmath> // For std::ceil

namespace hnswlib {

// Forward declaration for the quantizer
struct ProductQuantizer;

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
    std::atomic<int> maxlevel_{-1}; // Use atomic for thread-safe max level tracking

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global; // Used for critical sections like initial entry point setting
    std::vector<std::mutex> link_list_locks_;

    std::atomic<tableint> enterpoint_node_{-1}; // Use atomic for thread-safe entry point tracking

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    // --- OPQ Compression Members ---
    bool use_compression_{false};
    std::unique_ptr<ProductQuantizer> pq_{nullptr}; // Pointer to the Product Quantizer model
    char *compressed_codes_memory_{nullptr}; // Memory to store compressed codes
    size_t pq_code_size_{0}; // Size of one compressed code in bytes
    // --- End OPQ Compression Members ---

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
    const char *getCompressedCodePtr(tableint internal_id) const; // Get pointer to compressed code
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
    std::vector<data_t> getDataByLabel(labeltype label) const; // Returns decoded vector if compressed

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

    // --- OPQ Compression Methods ---
    void trainQuantizer(const float* train_data, size_t num_train_points, size_t pq_m, size_t pq_kbits);
    // Placeholder for ADC distance function
    dist_t distanceADC(const void* query_vector, tableint internal_id) const;
    // --- End OPQ Compression Methods ---

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
};

// Basic Product Quantizer structure (placeholder)
struct ProductQuantizer {
    size_t d;           // Original dimension
    size_t m;           // Number of subquantizers
    size_t kbits;       // Bits per subquantizer centroid index
    size_t code_size;   // Total size of the code in bytes
    size_t dsub;        // Dimension of each subspace (d / m)
    size_t ksub;        // Number of centroids per subquantizer (2^kbits)
    
    std::vector<float> centroids; // Codebooks stored contiguously [m][ksub][dsub]
    // OPQ specific: Orthogonal rotation matrix R (d x d)
    std::vector<float> rotation_matrix;
    bool opq_trained = false; // Flag if OPQ rotation is trained

    ProductQuantizer(size_t d_, size_t m_, size_t kbits_) : 
        d(d_), m(m_), kbits(kbits_), ksub(1 << kbits_), dsub(d / m) {
        if (d % m != 0) {
            throw std::runtime_error("Original dimension d must be divisible by m");
        }
        code_size = (size_t)std::ceil((double)m * kbits / 8.0); // Calculate bytes needed
        centroids.resize(m * ksub * dsub);
        rotation_matrix.resize(d * d); // Initialize rotation matrix storage
        // Initialize rotation matrix to identity initially
        for(size_t i=0; i<d; ++i) {
            for(size_t j=0; j<d; ++j) {
                 rotation_matrix[i * d + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    // Placeholder: Encode a vector into a code
    void encode(const float* vec, char* code) const {
        // TODO: Implement OPQ rotation + PQ encoding
        // 1. Apply rotation: rotated_vec = R * vec
        // 2. For each subspace i = 0 to m-1:
        //    a. Find nearest centroid index in codebook i for subspace i of rotated_vec
        //    b. Store the index (packed into bytes in 'code')
        memset(code, 0, code_size); // Zero out for simplicity now
    }

    // Placeholder: Decode a code into an approximate vector
    void decode(const char* code, float* vec) const {
        // TODO: Implement PQ decoding + inverse OPQ rotation
        // 1. Unpack indices from 'code'
        // 2. For each subspace i = 0 to m-1:
        //    a. Retrieve centroid vector for index i
        //    b. Place centroid into the corresponding subspace of 'reconstructed_rotated_vec'
        // 3. Apply inverse rotation: vec = R^T * reconstructed_rotated_vec
        memset(vec, 0, sizeof(float) * d); // Zero out for simplicity now
    }

    // Placeholder: Compute distance between query vector and code (ADC)
    float distanceADC(const float* query_vec, const char* code) const {
        // TODO: Implement Asymmetric Distance Calculation for OPQ
        // 1. Apply rotation to query: rotated_query = R * query_vec
        // 2. Initialize distance = 0
        // 3. Unpack indices idx[0]...idx[m-1] from 'code'
        // 4. For each subspace i = 0 to m-1:
        //    a. Get centroid C = centroids[i][idx[i]]
        //    b. Get corresponding subspace S_q of rotated_query
        //    c. Add squared Euclidean distance ||S_q - C||^2 to total distance
        // 5. Return total distance (or sqrt for L2)
        return std::numeric_limits<float>::max(); // Placeholder 
    }
    
    // Placeholder: Train OPQ rotation and PQ codebooks
    void train(const float* train_data, size_t num_train_points) {
        // TODO: Implement OPQ training (e.g., using iterative optimization)
        //       and k-means for each PQ subspace on the rotated data.
        // Set opq_trained = true after training rotation.
        opq_trained = false; // Not actually trained here
    }
};

}  // namespace hnswlib 