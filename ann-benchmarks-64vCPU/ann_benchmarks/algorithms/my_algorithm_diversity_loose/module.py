import hnswlib
import numpy as np

from ..base.module import BaseANN


class MyAlgorithmDiversity(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        self.eps_rad = method_param.get("eps_rad", 0.1)  # Default to 0.1 if not provided
        # print(self.method_param,save_index,query_param)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        
        # Initialize index with all parameters including eps_rad
        self.p.init_index(
            max_elements=len(X),
            M=self.method_param["M"],
            ef_construction=self.method_param["efConstruction"],
            random_seed=100,
            allow_replace_deleted=False,
            eps_rad=self.eps_rad
        )
        
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
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
