from mysearch import Seacher
from time import time

if __name__ == "__main__":
    start = time()
    seacher = Seacher("save_models/collector300000.pkl")
    articles = seacher.predict("quantum mechanic")
    end = time()
    answer = [f"{title}: {url}" for title, url in zip(articles['title'],articles['pdf_url'])]
    print(f"time: {end-start:.4f} (s)")
    for i, result in enumerate(answer):
        print(f"{i+1}. {result}")
