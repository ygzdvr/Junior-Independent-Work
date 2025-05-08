import hnswlib
import numpy as np

from ..base.module import BaseANN

TRAIN_SET_SIZE = 100000 # Number of points to use for PQ/OPQ training

class MyAlgorithmVanilla(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        self.pq_m = method_param.get("pq_m")
        self.pq_kbits = method_param.get("pq_kbits", 8) # Default 8 bits
        self.name = f"my_algo_b2({method_param})"

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        
        # Initialize index structure (without compression yet)
        pruning_alpha = float(self.method_param.get("pruningAlpha", 1.2))
            self.p.init_index(
                max_elements=len(X), 
                ef_construction=self.method_param["efConstruction"], 
                M=self.method_param["M"],
            pruning_alpha=pruning_alpha
            )

        # --- Train PQ/OPQ if pq_m is specified ---
        if self.pq_m:
            print(f"Training PQ/OPQ (m={self.pq_m}, kbits={self.pq_kbits}) on {TRAIN_SET_SIZE} points...")
            # Select a subset for training
            train_indices = np.random.choice(len(X), min(TRAIN_SET_SIZE, len(X)), replace=False)
            X_train = X[train_indices].astype('float32') # Ensure float32 for training
            
            try:
                self.p.train_pq(X_train, self.pq_m, self.pq_kbits)
                print("PQ/OPQ training complete. Compression enabled.")
            except Exception as e:
                print(f"Error during PQ training: {e}. Proceeding without compression.")
                self.pq_m = None # Disable compression if training fails
        else:
            print("PQ/OPQ compression not specified (pq_m not set). Proceeding without compression.")
        # --- End Training --- 
        
        # Add all items (will be compressed if training was successful)
        print(f"Adding {len(X)} items...")
        data_labels = np.arange(len(X))
        self.p.add_items(X.astype('float32'), data_labels, num_threads=-1) # Use multiple threads
        print("Item addition complete.")
        # self.p.set_num_threads(1) # Set back to 1 for querying? Depends on benchmark setup

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        # Update name based on actual parameters used (including compression)
        pruning_info = f", 'pruningAlpha': {self.p.pruning_alpha}"
        pq_info = ""
        if self.p.is_compressed:
            pq_info = f", 'pq_m': {self.p.pq_m}, 'pq_kbits': {self.p.pq_kbits}"
        base_params = f"M: {self.method_param['M']}, efConstruction: {self.method_param['efConstruction']}"
        self.name = f"hnswlib ({base_params}{pruning_info}{pq_info}, 'efQuery': {ef})"

    def query(self, v, n):
        return self.p.knn_query(np.expand_dims(v, axis=0).astype('float32'), k=n)[0][0]

    def freeIndex(self):
        del self.p
