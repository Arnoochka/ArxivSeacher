from sklearn.base import TransformerMixin, BaseEstimator
from typing import List, Tuple
from collections import Counter
import numpy as np
from mysearch.utils import Parallelizer
from scipy.sparse import csr_matrix


class TFIDFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 vocab_size: int):
        self.vocab_size = vocab_size
        self.idf_ = None
        
    def fit(self, docs: List[List[int]]) -> "TFIDFTransformer":
        n_docs = len(docs)
        df = np.zeros(self.vocab_size, dtype=np.int32)
        for doc in docs:
            unique_tokens = np.unique(np.array(doc, dtype=np.int32))
            df[unique_tokens] += 1
        self.idf_ = np.log(n_docs / (1.0 + df))
        return self
    

    def transform(self, docs: List[List[int]]) -> csr_matrix:
        indexes_values = Parallelizer.processing_list(docs, self._compute_tfidf)
        indexes, values = zip(*indexes_values)
        indexes = list(indexes)
        values = list(values)
        row_ind = Parallelizer.concat([[doc_idx] * len(idxs) for doc_idx, idxs in enumerate(indexes)])
        col_ind = Parallelizer.concat(indexes)
        data = Parallelizer.concat(values)
        tfidf_matrix = csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(len(docs), self.vocab_size),
            dtype=np.float64
        )
        return tfidf_matrix
    
    def _compute_tfidf(self, doc: List[int]) -> Tuple[List, List]:
        idxs, counts = np.unique(np.array(doc, dtype=np.int32), return_counts=True)
        tf = counts / max(len(doc), 1)
        idf = self.idf_[idxs]
        tfidf = tf * idf
        return list(idxs), list(tfidf)
    
class TFIDFVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features: int):
        self.max_features = max_features
        self.transformer_ = None
        
    def fit(self, tokenized_docs: List[List[str]]) -> "TFIDFVectorizer":
        all_tokens = Parallelizer.concat(
            Parallelizer.processing_list(
                tokenized_docs,
                lambda doc: [token for token in doc]
                )
            )
        vocab_freq = Counter(all_tokens)
        top_tokens = [word for word, _ in vocab_freq.most_common(self.max_features)]
        self.vocabulary_ = {word: idx for idx, word in enumerate(top_tokens)}
        encoded_docs = Parallelizer.processing_list(tokenized_docs, self._encode)
        self.transformer_ = TFIDFTransformer(vocab_size=len(self.vocabulary_))
        self.transformer_.fit(encoded_docs)
        return self
    
    def transform(self, tokenized_docs: List[List[str]]) -> csr_matrix:
        encoded_docs = Parallelizer.processing_list(tokenized_docs, self._encode)
        return self.transformer_.transform(encoded_docs)
    
    def _encode(self, doc: List[str]) -> List[int]:
        return [self.vocabulary_[token] for token in doc if token in self.vocabulary_]