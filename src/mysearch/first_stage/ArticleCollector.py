import numpy as np
from sklearn.base import BaseEstimator, clone
from typing import List, Dict, Callable
from .LogisticRegression import LogisticRegression
from .TFIDF import TFIDFVectorizer
from mysearch.utils import Parallelizer
import pandas as pd
import xml.etree.ElementTree as ET
import asyncio
import aiohttp
import joblib

class CategoryEstimator(BaseEstimator):
    def __init__(self,
                 categories: Dict[str, List[str]],
                 preprocessor: Callable,
                 vocab_size: int = 10000,
                 log_kwargs: Dict = {}):
        super().__init__()
        self.vocab_size = vocab_size
        self.preprocessor = preprocessor
        self.categories_ = categories
        self.vectorizer_ = TFIDFVectorizer(self.vocab_size)
        self.upper_classifier_ = LogisticRegression(**log_kwargs)
        self.down_classifiers_: List[LogisticRegression] = None
        
        self._build_category_mappings()

    def _build_category_mappings(self):
        self.upper_name_idx = {name: idx for idx, name in enumerate(self.categories_.keys())}
        self.upper_idx_name = {idx: name for name, idx in self.upper_name_idx.items()}
        self.down_name_idx = {}
        self.down_idx_name = {}

        for upper_name, down_list in self.categories_.items():
            upper_idx = self.upper_name_idx[upper_name]
            self.down_name_idx[upper_name] = {
                down_name: idx for idx, down_name in enumerate(down_list)
            }
            self.down_idx_name[upper_idx] = {
                idx: down_name for idx, down_name in enumerate(down_list)
            }
    def get_upper_idx(self, upper_name: str) -> int:
        return self.upper_name_idx[upper_name]
    def get_upper_name(self, upper_idx: int) -> str:
        return self.upper_idx_name[upper_idx]
    def get_down_idx(self, upper_name: str, down_name: str) -> int:
        return self.down_name_idx[upper_name][down_name]
    def get_down_name(self, upper_idx: int, down_idx: int) -> str:
        return self.down_idx_name[upper_idx][down_idx]
    
    def fit(self, docs: List[str], labels: List[str]) -> "ArticleCollector":
        tokenized_docs = Parallelizer.processing_list(docs, self.preprocessor)
        upper_names = list(self.categories_.keys())
        n_categories = len(upper_names)
        self.down_classifiers_ = [clone(self.upper_classifier_) for _ in range(n_categories)]
        
        upper_labels = np.array([self.get_upper_idx(label.split(".")[0]) for label in labels])
        X_vectorized = self.vectorizer_.fit_transform(tokenized_docs)
        self.upper_classifier_.fit(X_vectorized, upper_labels)
        
        for upper_idx, upper_name in enumerate(upper_names):
            indices = []
            down_labels = []
            for idx, label in enumerate(labels):
                label_upper_name, label_down_name = label.split(".")
                if label_upper_name == upper_name:
                    indices.append(idx)
                    down_labels.append(self.get_down_idx(label_upper_name, label_down_name))
            if not indices:
                continue
            X_down = X_vectorized[indices]
            self.down_classifiers_[upper_idx].fit(X_down, down_labels)

        return self
    
    def predict(self, preprocessed_question: List[int], upper_k: int, down_k: int) -> List[str]:
        X_vectorized = self.vectorizer_.transform([preprocessed_question])
        upper_preds = self.upper_classifier_.predict(X_vectorized, k=upper_k)[0]
        results = []
        text_vector = X_vectorized[0]
        for upper_idx in upper_preds:
            clf = self.down_classifiers_[upper_idx]
            down_idxs = clf.predict(text_vector, k=down_k)
            for sample_idxs in down_idxs:
                for down_idx in sample_idxs:
                    upper_name = self.get_upper_name(upper_idx)
                    down_name = self.get_down_name(upper_idx, down_idx)
                    results.append(f"{upper_name}.{down_name}")

        return results
    
    def save(self, filepath: str):
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "CategoryEstimator":
        return joblib.load(filepath)

class ArticleCollector(CategoryEstimator):
    def __init__(self,
                 categories: Dict[str, List[str]],
                 preprocessor: Callable,
                 vocab_size: int = 10000,
                 log_kwargs: Dict = {}):
        super().__init__(categories, preprocessor, vocab_size, log_kwargs)
        
    def __call__(self,
                 question: str,
                 num_articles: int,
                 num_articles_per_request: int,
                 upper_k: int = 3,
                 dowm_k: int = 3
                 ) -> pd.DataFrame:
        categories = self.predict(question, upper_k, dowm_k)
        articles = self.search_arxiv_articles(categories,
                                              num_articles=num_articles,
                                              max_results_per_request=num_articles_per_request)
        return articles
    @staticmethod
    async def _fetch_category(session,
                              arxiv_category: str,
                              num_articles: int,
                              max_results_per_request: int) -> List[dict]:
        base_url = "http://export.arxiv.org/api/query"
        all_articles = []
        start = 0

        while start < num_articles:
            params = {
                'search_query': f'cat:{arxiv_category}',
                'start': start,
                'max_results': max_results_per_request,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            try:
                async with session.get(base_url, params=params) as response:
                    response.raise_for_status()
                    content = await response.text()

                    root = ET.fromstring(content)

                    ns = {
                        'atom': 'http://www.w3.org/2005/Atom',
                        'arxiv': 'http://arxiv.org/schemas/atom'
                    }

                    entries = root.findall('atom:entry', ns)
                    if not entries:
                        break

                    for entry in entries:
                        article_id = entry.find('atom:id', ns).text.split("/")[-1].split("v")[0]
                        title = entry.find('atom:title', ns).text.strip()
                        abstract = entry.find('atom:summary', ns).text.strip()

                        authors = [
                            author.find('atom:name', ns).text
                            for author in entry.findall('atom:author', ns)
                        ]
                        submitter = authors[0] if authors else None

                        doi_elem = entry.find('arxiv:doi', ns)
                        doi = doi_elem.text if doi_elem is not None else None

                        license_elem = entry.find('arxiv:license', ns)
                        license_url = license_elem.attrib['href'] if license_elem is not None else None

                        categories = [cat.attrib['term'] for cat in entry.findall('atom:category', ns)]
                        date = entry.find('atom:published', ns).text

                        pdf_url = None
                        for link in entry.findall('atom:link', ns):
                            if link.attrib.get('type') == 'application/pdf':
                                pdf_url = link.attrib['href']
                                break

                        all_articles.append({
                            'id': article_id,
                            'submitter': submitter,
                            'authors': authors,
                            'title': title,
                            'doi': doi,
                            'categories': categories,
                            'license': license_url,
                            'abstract': abstract,
                            'date': date,
                            'pdf_url': pdf_url
                        })

                    start += len(entries)

            except Exception as e:
                print(f"{arxiv_category}: {e}")
                break

        return all_articles

    @staticmethod
    async def _fetch_all_categories_async(arxiv_categoriers: List[str],
                                          num_articles: int,
                                          max_results_per_request: int
                                          ) -> pd.DataFrame:
        async with aiohttp.ClientSession() as session:
            tasks = [
                ArticleCollector._fetch_category(session, cat, num_articles, max_results_per_request)
                for cat in arxiv_categoriers
            ]
            results = await asyncio.gather(*tasks)

        all_articles = []
        for result in results:
            all_articles.extend(result)

        return pd.DataFrame(all_articles)

    @staticmethod
    def search_arxiv_articles(arxiv_categoriers: List[str],
                              num_articles: int = 1000,
                              max_results_per_request: int = 100
                              ) -> pd.DataFrame:
        return asyncio.run(
            ArticleCollector._fetch_all_categories_async(arxiv_categoriers,
                                                       num_articles,
                                                       max_results_per_request)
        )
    

