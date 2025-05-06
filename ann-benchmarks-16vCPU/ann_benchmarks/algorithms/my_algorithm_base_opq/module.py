import hnswlib
import numpy as np
import faiss

from ..base.module import BaseANN


class MyAlgorithmBaseOPQ(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        if self.metric != "l2":
            # Faiss OPQ only supports L2 (inner product transformation exists but is more complex)
            # Hnswlib PQ also typically uses L2.
            raise ValueError("OPQ currently only supports metric 'euclidean' (l2)")

        self.method_param = method_param
        self.efConstruction = method_param["efConstruction"]
        self.M = method_param["M"]
        self.opq_m = method_param.get("opq_m") # Get OPQ bytes per code (M in Faiss PQ)
        self.opq_train_size = method_param.get("opq_train_size", 0) # Get OPQ training size

        self.index_inited = False
        self.opq_index = None # Faiss OPQ index for preprocessing/encoding
        self.p = None # hnswlib index instance
        self.name = f"hnswlib(M={self.M},efC={self.efConstruction})"
        if self.opq_m:
            self.name += f"-OPQ{self.opq_m}"

    def fit(self, X):
        if self.index_inited:
            print("Index already fitted")
            return

        dim = X.shape[1]
        num_elements = X.shape[0]

        if self.opq_m:
            print(f"Fitting with OPQ (opq_m={self.opq_m})")
            if X.dtype != np.float32:
                X = X.astype(np.float32)

            # 1. Train OPQ
            train_size = min(self.opq_train_size, num_elements) if self.opq_train_size > 0 else num_elements
            if train_size < self.opq_m * 256: # Faiss requires enough training data
                 print(f"Warning: OPQ training size {train_size} might be too small for opq_m={self.opq_m}. Adjust opq_train_size.")
                 train_size = max(train_size, self.opq_m * 256) # Attempt to increase if possible
                 train_size = min(train_size, num_elements)

            print(f"Training OPQ on {train_size} vectors...")
            X_train = X[np.random.choice(num_elements, train_size, replace=False)]

            # Use IndexOPQ for training the rotation R and the PQ codebooks
            self.opq_index = faiss.IndexOPQ(dim, self.opq_m)
            self.opq_index.verbose = True
            self.opq_index.train(X_train)
            print("OPQ training complete.")

            # 2. Encode data using the trained OPQ index
            print("Encoding data with OPQ...")
            codes = self.opq_index.sa_encode(X)
            print(f"Encoding complete. Code shape: {codes.shape}, dtype: {codes.dtype}")

            # 3. Initialize HNSW Index with OPQ parameter
            print(f"Initializing hnswlib Index with opq_m={self.opq_m}")
            self.p = hnswlib.Index(space=self.metric, dim=dim, opq_m=self.opq_m) # Pass opq_m

            # 4. Set PQ Centroids in C++ index
            print("Extracting PQ centroids from Faiss...")
            if hasattr(self.opq_index, 'pq'):
                pq = self.opq_index.pq
            elif hasattr(self.opq_index, 'chain') and len(self.opq_index.chain) > 0 and hasattr(self.opq_index.chain[-1], 'pq'): # Handle chains e.g. OPQ -> PQ
                pq = self.opq_index.chain[-1].pq
            else:
                raise RuntimeError("Could not extract ProductQuantizer from Faiss IndexOPQ")
            
            centroids = faiss.vector_to_array(pq.centroids)
            expected_centroids = pq.M * pq.ksub * pq.dsub
            print(f"Centroids shape: {centroids.shape}, Expected size: {expected_centroids}")
            assert centroids.size == expected_centroids, "Centroid array size mismatch"
            assert centroids.dtype == np.float32, "Centroids must be float32"
            
            print("Setting PQ centroids in hnswlib index...")
            self.p.set_pq_centroids(centroids) # Call the bound C++ method
            print("PQ centroids set.")

            # 5. Initialize HNSW index structure
            self.p.init_index(
                max_elements=num_elements,
                ef_construction=self.efConstruction,
                M=self.M
            )

            # 6. Add encoded items using the correct function
            print("Adding encoded items to HNSW index...")
            data_labels = np.arange(num_elements)
            self.p.add_items_codes(codes, data_labels) # Use add_items_codes
            print("Adding items complete.")

        else:
            # Original HNSW implementation (no OPQ)
            print("Fitting with standard HNSW (no OPQ)")
            self.p = hnswlib.Index(space=self.metric, dim=dim) # No opq_m passed
            self.p.init_index(
                max_elements=num_elements, ef_construction=self.efConstruction, M=self.M
            )
            data_labels = np.arange(num_elements)
            self.p.add_items(np.asarray(X).astype('float32'), data_labels)

        self.p.set_num_threads(1)
        self.index_inited = True

    def set_query_arguments(self, ef):
        if not self.p:
            raise RuntimeError("Index not fitted yet")
        self.p.set_ef(ef)
        # Update name (already done in init)
        self.name = f"hnswlib(M={self.M},efC={self.efConstruction})"
        if self.opq_m:
            self.name += f"-OPQ{self.opq_m}"
        self.name += f"-ef{ef}"

    def query(self, v, n):
        if not self.p:
            raise RuntimeError("Index not fitted yet")
        
        query_vector = np.expand_dims(v, axis=0).astype('float32')

        if self.opq_index:
            # === IMPORTANT: Apply OPQ rotation to the query vector ===
            # The C++ ADC expects the query vector to be rotated (R * v)
            # sa_encode applies the rotation and returns codes, we need the rotated vector.
            # We can get this via the `apply_preprocess` method of the VT (VectorTransform) chain.
            # The OPQ matrix (rotation R) is typically the first transform in IndexOPQ.
            if hasattr(self.opq_index, 'chain') and len(self.opq_index.chain) > 0: 
                # Assuming the first element in chain is the LinearTransform (rotation R)
                opq_transform = self.opq_index.chain[0]
                query_vector = opq_transform.apply(query_vector)
                # print(f"Rotated query vector shape: {query_vector.shape}")
            else:
                 print("Warning: Could not find OPQ rotation matrix in Faiss index chain. Query may be inaccurate.")
                 # Proceed with un-rotated query - likely won't work correctly with ADC

            # Pass the (potentially rotated) query vector to knn_query
            labels, distances = self.p.knn_query(query_vector, k=n)
            return labels[0]
        else:
            # Original HNSW query
            labels, distances = self.p.knn_query(query_vector, k=n)
            return labels[0]

    def freeIndex(self):
        del self.p
        del self.opq_index
        self.p = None
        self.opq_index = None
        self.index_inited = False
