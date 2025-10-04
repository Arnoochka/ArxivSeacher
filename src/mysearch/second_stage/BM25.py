from sklearn.base import BaseEstimator
from typing import List, Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from numpy.typing import NDArray

class BM25(BaseEstimator):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs_: int = None
        self.docs_len_: NDArray = None
        self.avgdl_: float = None
        self.terms_: Dict[str, int] = {}
        self.q_matrix_: csr_matrix = None
        self.idf_: NDArray = None

    def fit(self, docs_tokenized: List[List[str]]) -> "BM25":
        self.n_docs_ = len(docs_tokenized)
        self.docs_len_ = np.array([len(doc) for doc in docs_tokenized], dtype=np.int32)
        self.avgdl_ = float(self.docs_len_.sum() / self.n_docs_)

        row_idxs, col_idxs, data = [], [], []
        terms = {}
        global_idx = 0

        for doc_idx, doc in enumerate(docs_tokenized):
            doc_freq = {}
            for token in doc:
                doc_freq[token] = doc_freq.get(token, 0) + 1
            for token, freq in doc_freq.items():
                if token not in terms:
                    terms[token] = global_idx
                    global_idx += 1
                col_idxs.append(terms[token])
                row_idxs.append(doc_idx)
                data.append(freq)

        self.q_matrix_ = csr_matrix((data, (col_idxs, row_idxs)),
                                    shape=(global_idx, self.n_docs_),
                                    dtype=np.int64)
        self.terms_ = terms

        df = np.diff(self.q_matrix_.indptr)
        self.idf_ = np.log((self.n_docs_ - df + 0.5) / (df + 0.5) + 1)

        return self
    
    def predict(self, query_tokens: List[str], top_k: int = 500) -> Tuple[NDArray, NDArray]:
        scores = np.zeros(self.n_docs_, dtype=np.float32)

        for token in query_tokens:
            if token not in self.terms_:
                continue
            term_idx = self.terms_[token]
            idf = self.idf_[term_idx]
            row = self.q_matrix_.getrow(term_idx)
            docs = row.indices
            freqs = row.data.astype(np.float32)

            denom = freqs + self.k1 * (
                1 - self.b + self.b * self.docs_len_[docs] / self.avgdl_
            )
            scores[docs] += idf * (freqs * (self.k1 + 1)) / (denom + 1e-10)
        
        positive_indices = np.where(scores > 0)[0]
        indices = positive_indices[np.argsort(scores[positive_indices])[::-1]][:top_k]

        return indices, scores