import numpy as np
from sklearn.base import BaseEstimator
from numpy.typing import NDArray
from typing import Tuple

class PageRank(BaseEstimator):
    def __init__(self,
                 k_neighbors: int = 5,
                 alpha: float = 0.85,
                 max_iter: int = 100,
                 tol: float = 1e-6):
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def predict(self, embeddings: NDArray, top_k: int = 100) -> Tuple[NDArray, NDArray]:
        n = embeddings.shape[0]
        sim_matrix = self._cosine_metric(embeddings, embeddings)
        A = np.zeros((n, n))
        for i in range(n):
            sims = np.argsort(sim_matrix[i])[::-1][1:self.k_neighbors+1]
            for j in sims:
                if sim_matrix[i, j] > 0:
                    A[i, j] = sim_matrix[i, j]
                    
        M = A.T / (A.sum(axis=1) + 1e-8)[:, None]
        score = np.ones(n) / n
        for _ in range(self.max_iter):
            score_new = self.alpha * M @ score + (1 - self.alpha) / n
            if np.linalg.norm(score_new - score, 1) < self.tol:
                break
            score = score_new
        
        return np.argsort(score)[::-1][:top_k], score
    
    def _cosine_metric(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b.T
