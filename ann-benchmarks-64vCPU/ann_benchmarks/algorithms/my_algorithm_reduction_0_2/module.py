import hnswlib
import numpy as np

from ..base.module import BaseANN


class MyAlgorithmReduction(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        # Extract JL projection parameters if present
        self.use_jl = self.method_param.get("useJL", False)
        self.jl_target_eps = self.method_param.get("jlTargetEps", 0.10)
        self.jl_min_dim = self.method_param.get("jlMinDim", 32)
        # print(self.method_param,save_index,query_param)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(
            max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"]
        )
        
        # Initialize JL projections if enabled
        if self.use_jl:
            # Get the maximum level from the index object (using max_level=M for simplicity)
            max_level = self.method_param["M"]
            self.p.init_projections(max_level, self.jl_target_eps, self.jl_min_dim)
            
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        jl_info = ""
        if self.use_jl:
            jl_info = f", 'useJL': True, 'jlTargetEps': {self.jl_target_eps}, 'jlMinDim': {self.jl_min_dim}"
        self.name = f"hnswlib reduction ({self.method_param}, 'efQuery': {ef}{jl_info})"

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def freeIndex(self):
        del self.p
