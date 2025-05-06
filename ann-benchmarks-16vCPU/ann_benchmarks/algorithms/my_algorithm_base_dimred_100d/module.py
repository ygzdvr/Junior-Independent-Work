import hnswlib
import numpy as np

from ..base.module import BaseANN


class MyAlgorithmBaseDimRed(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        
        # Initialize the index with standard HNSW parameters
        self.p.init_index(
            max_elements=len(X),
            M=self.method_param["M"],
            ef_construction=self.method_param["efConstruction"],
            alpha=self.method_param["alpha"]
        )
        
        # Enable dimensionality reduction if specified in parameters
        if "use_dim_reduction" in self.method_param and self.method_param["use_dim_reduction"]:
            # Get dimensionality reduction parameters
            reduction_type = self.method_param.get("reduction_type", "random")
            threshold_level = self.method_param.get("reduction_threshold", 1)
            
            # Get reduction dimensions for each level above threshold
            target_dims = self.method_param.get("target_dims", [])
            
            # If target_dims is a float, it's a fraction of original dimensions
            if isinstance(target_dims, float):
                orig_dim = len(X[0])
                # Calculate actual dimensions - starting from ~50% and reducing by half at each level
                dims = []
                current_dim = int(orig_dim * target_dims)
                while current_dim > 10:  # Keep at least 10 dimensions
                    dims.append(current_dim)
                    current_dim = current_dim // 2
                target_dims = dims
            
            # Enable dimensionality reduction in the index
            if target_dims:
                self.p.enable_dim_reduction(
                    full_dim=len(X[0]),
                    target_dims=target_dims,
                    threshold_level=threshold_level,
                    reduction_type=reduction_type
                )
        
        # Add items to the index
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        
        # Build dimensionality reduction after adding all items
        if "use_dim_reduction" in self.method_param and self.method_param["use_dim_reduction"]:
            self.p.build_dim_reduction()
            
        if hasattr(self.p, 'get_num_layers'):
            num_layers = self.p.get_num_layers()
            print(f"HNSW index number of layers: {num_layers}")
        else:
             print("Warning: Custom get_num_layers method not found in hnswlib.")

        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        self.name = "hnswlib (%s, 'efQuery': %s)" % (self.method_param, ef)

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def freeIndex(self):
        del self.p
