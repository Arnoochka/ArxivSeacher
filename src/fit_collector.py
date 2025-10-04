import json
import pandas as pd
from mysearch.first_stage import ArticleCollector
from mysearch.utils import Preprocessor
from time import time

if __name__ == "__main__":
    df = []
    idx = 0
    max_idx = 3 * 10**5
    with open("metadata-arxiv.json", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                df.append(json.loads(line))
                idx += 1
            if not idx < max_idx:
                break
            
    df = pd.DataFrame(df)       
    df = pd.DataFrame(df)      
    texts = []
    labels = []
    categories = {}

    for text, category in zip(df['abstract'], df['categories']):
        texts.append(text)
        cat = category.split(" ")[0]
        if len(cat.split(".")) == 1:
            cat = f"{cat}.{cat}"
        upper_cat, down_cat = cat.split(".")
        if upper_cat not in categories.keys():
            categories[upper_cat] = [down_cat]
        elif down_cat not in categories[upper_cat]:
            categories[upper_cat].append(down_cat)
        labels.append(f"{upper_cat}.{down_cat}")
    print("fit collector")
    start = time()
    collector = ArticleCollector(categories, Preprocessor)
    collector.fit(texts, labels)
    collector.save(f"save_models/collector{max_idx}.pkl")
    end = time()
    print(f"time: {end - start}")