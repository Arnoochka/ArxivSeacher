from sklearn.base import BaseEstimator
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

class MMR(BaseEstimator):
    def __init__(self, lambda_param: float = 0.8):
        self.lambda_param = lambda_param
    
    def predict(self, doc_embeddings: NDArray, query_embedding: NDArray, top_k: int) -> Tuple[NDArray, NDArray]:
        n_docs = doc_embeddings.shape[0]
        selected = []
        candidate_idx = list(range(n_docs))
        
        sim_query = self._cosine_metric(doc_embeddings, query_embedding)
        first_idx = sim_query.argmax()
        selected.append(first_idx)
        candidate_idx.remove(first_idx)
        score = np.zeros(shape=(top_k), dtype=np.float32)
        score[0] = sim_query[first_idx]
        
        for k in range(1, top_k):
            sim_selected = np.max(self._cosine_metric(doc_embeddings[candidate_idx], doc_embeddings[selected]), axis=1)
            mmr_score = self.lambda_param * sim_query[candidate_idx] - (1 - self.lambda_param) * sim_selected
            idx_in_candidates = np.argmax(mmr_score)
            idx = candidate_idx[idx_in_candidates]
            score[k] = mmr_score[idx_in_candidates]
            selected.append(idx)
            candidate_idx.remove(idx)
        
        return np.array(selected), score
    
    def _cosine_metric(sel, a: NDArray, b: NDArray) -> NDArray:
        return a @ b.T