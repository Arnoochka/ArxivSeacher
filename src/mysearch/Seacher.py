from sklearn.base import BaseEstimator
from typing import Tuple, Union
from numpy.typing import NDArray
from mysearch.first_stage import ArticleCollector
from mysearch.second_stage import SemanticFilter
from mysearch.utils import Preprocessor
import pandas as pd

class Seacher(BaseEstimator):
    def __init__(self,
                 collector_path: str,
                 semantic_filter: SemanticFilter = SemanticFilter,
                 preprocessor: Preprocessor = Preprocessor
                 ):
        self.collector = ArticleCollector.load(collector_path)
        self.semantic_filter = semantic_filter(preprocessor)
        self.preprocessor = preprocessor
        super().__init__()
        
    def predict(self, question: str,
                num_articles: int = 1000,
                num_articles_per_request: int = 100,
                upper_k: int = 3,
                down_k: int = 3,
                bm_top_k: int = 300,
                mmr_top_k: int = 100,
                pr_top_k: int = 25,
                return_score: bool = False,
                ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, NDArray, NDArray, NDArray]]:
        question = Preprocessor(question)
        articles = self.collector(question,
                                  num_articles,
                                  num_articles_per_request,
                                  upper_k,
                                  down_k)
        
        filter = self.semantic_filter.fit(articles)
        indices, bm25_score, mmr_score, pr_score = filter.predict(question,
                                                                  bm_top_k,
                                                                  mmr_top_k,
                                                                  pr_top_k)
        selected_articles = articles.iloc[indices]
        if return_score:
            return (selected_articles,
                    bm25_score,
                    mmr_score,
                    pr_score)
        else: 
            return selected_articles
        