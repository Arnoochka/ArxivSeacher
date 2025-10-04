from mysearch import Seacher
from time import time

# if __name__ == "__main__":
#     collector = ArticleCollector.load("save_models/collector300000.pkl")
#     print("начало эксперимента")
#     start = time()
#     question = Preprocessor("quantum mechanic")
#     articles = collector(question, 1000, 100, 3, 3)
#     abstracts = [abstract for abstract in articles['abstract']]
#     filter = SemanticFilter(Preprocessor).fit(articles)
#     indices, bm25_score, mmr_score, pr_score = filter.predict(question,
#                                                               bm_top_k=300,
#                                                               mmr_top_k=100,
#                                                               pr_top_k=25)
#     selected_articles = articles.iloc[indices]
#     answer = [f"{title}: {url}" for title, url in zip(selected_articles['title'],selected_articles['pdf_url'])]
#     end = time()
#     print(f"time: {end-start}")
#     for i, result in enumerate(answer):
#         print(f"{i+1}. {result}")
#     print(f"time: {end-start}")

if __name__ == "__main__":
    start = time()
    seacher = Seacher("save_models/collector300000.pkl")
    articles = seacher.predict("quantum mechanic")
    end = time()
    answer = [f"{title}: {url}" for title, url in zip(articles['title'],articles['pdf_url'])]
    print(f"time: {end-start:.4f} (s)")
    for i, result in enumerate(answer):
        print(f"{i+1}. {result}")
