import hnswlib
import numpy as np

from ..base.module import BaseANN


class MyAlgorithmBaseProbe(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        self.num_probes = self.method_param.get("numProbes", 1)
        self.ef_query = None
        # print(self.method_param,save_index,query_param)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(
            max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"]
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        self.ef_query = ef
        name_params = f"M: {self.method_param['M']}, efConstruction: {self.method_param['efConstruction']}, efQuery: {ef}"
        if self.num_probes > 1:
            name_params += f", numProbes: {self.num_probes}"
        self.name = f"myalgo_base_probe({name_params})"

    def query(self, v, n):
        if self.ef_query is None:
            raise RuntimeError("set_query_arguments must be called before query")
        labels, distances = self.p.knn_query(np.expand_dims(v, axis=0), k=n, num_probes=self.num_probes)
        return labels[0]

    def freeIndex(self):
        del self.p
