from sklearn.base import BaseEstimator
import numpy as np
from typing import Callable, Dict, List
from sentence_transformers import SentenceTransformer
from mysearch.utils import Parallelizer
from .BM25 import BM25
from .MMR import MMR
import pandas as pd
from .PageRank import PageRank


class SemanticFilter(BaseEstimator):
    def __init__(self,
                 preprocessor: Callable,
                 bm25_kwargs: Dict = {},
                 mmr_kwargs: Dict = {},
                 pagerank_kwargs: Dict = {}):
        self.preprocessor = preprocessor
        self.bm25 = BM25(**bm25_kwargs)
        self.mmr = MMR(**mmr_kwargs)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.pagerank = PageRank(**pagerank_kwargs)
        self.texts_ = None
        
    def fit(self, articles: pd.DataFrame) -> "SemanticFilter":
        tokenized_titles, tokenized_abstracts = self._preprocess(articles)
        tokenized_texts = [title + abstract for title, abstract in zip(tokenized_titles, tokenized_abstracts)]
        self.bm25.fit(tokenized_texts)
        self.texts_ = Parallelizer.processing_list(tokenized_texts, lambda tokens: ' '.join(tokens))
        return self
    
    def predict(self,
                query_tokens: List[str],
                bm_top_k: int = 1000,
                mmr_top_k: int = 300,
                pr_top_k: int = 100):

        # BM25
        bm25_indices, bm25_score = self.bm25.predict(query_tokens, top_k=bm_top_k)
        indices = bm25_indices
        bm25_score = bm25_score[indices]
        # Encoder
        embeddings = self.embedder.encode([' '.join(query_tokens)] + [self.texts_[idx] for idx in indices])
        query_embedding = embeddings[0] / np.linalg.norm(embeddings[0])
        doc_embeddings = embeddings[1:] / np.linalg.norm(embeddings[1:], axis=1, keepdims=True)
        # MMR
        mmr_indices, mmr_score = self.mmr.predict(doc_embeddings, query_embedding, top_k=mmr_top_k)
        indices = indices[mmr_indices]
        doc_embeddings = doc_embeddings[mmr_indices]
        # PageRank
        pr_indices, pr_score = self.pagerank.predict(doc_embeddings, top_k=pr_top_k)
        indices = indices[pr_indices]

        return (indices,
                bm25_score[mmr_indices][pr_indices],
                mmr_score[pr_indices],
                pr_score[pr_indices])



        
    def _preprocess(self, articles: pd.DataFrame) -> tuple[List[str], List[str]]:
        titles = [title for title in articles['title']]
        abstracts = [abstract for abstract in articles['abstract']]
        tokenized_titles = Parallelizer.processing_list(titles, self.preprocessor)
        tokenized_abstracts= Parallelizer.processing_list(abstracts, self.preprocessor)
        return tokenized_titles, tokenized_abstracts