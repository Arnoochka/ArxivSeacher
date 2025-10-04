from joblib import Parallel, delayed
from typing import List, Callable, Iterable
import numpy as np
import re
import spacy

class Parallelizer:
    @staticmethod
    def processing_list(inputs: Iterable, func: Callable) -> List:
        return Parallel(n_jobs=-1)(delayed(func)(input) for input in inputs)
    
    @staticmethod
    def concat(lists: List, start = []) -> List:
        def concat(local_list: List) -> List:
            return sum(local_list, start)
        n = int(np.sqrt(len(lists)))   
        while n > 10:
            local_lists = [lists[k:k+n].copy() for k in range(0, len(lists), n)]
            lists = Parallelizer.processing_list(local_lists, concat)
            n = int(np.sqrt(len(lists)))  
        return sum(lists, [])


class Preprocessor:
    LATEX_CMD = re.compile(r'\\[a-zA-Z]+')
    LATEX_INLINE = re.compile(r'\$.*?\$')
    LATEX_ENV = re.compile(r'\\begin\{.*?\}.*?\\end\{.*?\}', re.DOTALL)
    PUNCT_NUM = re.compile(r'[^a-zA-Z\s]')  
    NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    
    def __new__(cls, text: str) -> List[str]:
        text = text.lower()
        text = cls.LATEX_ENV.sub(" ", text)
        text = cls.LATEX_INLINE.sub(" ", text)
        text = cls.LATEX_CMD.sub(" ", text)
        text = cls.PUNCT_NUM.sub(" ", text)
        doc = cls.NLP(text)

        tokens = []
        for token in doc:
            if token.is_stop or token.is_space or len(token.text) < 3:
                continue
            lemma = token.lemma_.strip()
            if lemma:
                tokens.append(lemma)

        return tokens
