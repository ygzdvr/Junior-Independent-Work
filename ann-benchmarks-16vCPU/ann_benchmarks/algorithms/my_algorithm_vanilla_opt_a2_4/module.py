import hnswlib
import numpy as np

from ..base.module import BaseANN


class MyAlgorithmVanilla(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        
        # Initialize with alpha parameter for robust pruning if specified
        if "pruningAlpha" in self.method_param:
            alpha = float(self.method_param["pruningAlpha"])
            # Store alpha for later use, e.g., during add_items if the C++ code supports it
            self.pruning_alpha = alpha
            self.p.init_index(
                max_elements=len(X),
                ef_construction=self.method_param["efConstruction"],
                M=self.method_param["M"]
            )
            print(f"Using pruning alpha: {alpha}")
        else:
            self.pruning_alpha = None # Or a default value if appropriate
            # Also initialize index if pruningAlpha is not specified
            self.p.init_index(
                max_elements=len(X),
                ef_construction=self.method_param["efConstruction"],
                M=self.method_param["M"]
            )
        
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        pruning_info = ""
        if "pruningAlpha" in self.method_param:
            pruning_info = f", 'pruningAlpha': {self.method_param['pruningAlpha']}"
        self.name = f"hnswlib ({self.method_param}{pruning_info}, 'efQuery': {ef})"

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def freeIndex(self):
        del self.p
