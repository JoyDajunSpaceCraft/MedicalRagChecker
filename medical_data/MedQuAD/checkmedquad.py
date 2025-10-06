# Dense only
from medquad_retriever import MedQuADRetriever
ret = MedQuADRetriever(corpus_dir="medquad_corpus",
                       model_name="pritamdeka/S-PubMedBert-MS-MARCO",
                       use_hybrid=False)
hits = ret.search("What are the symptoms of keratoderma with woolly hair?", top_k=5)
for h in hits:
    print(round(h["score"],4), h["source"], h["doc_id"])

# Hybrid (BM25 + dense)  —— 需要先: pip install rank-bm25
ret_h = MedQuADRetriever("medquad_corpus", "pritamdeka/S-PubMedBert-MS-MARCO",
                         use_hybrid=True, alpha_dense=0.65)
hits = ret_h.search("treatment for chronic migraine", top_k=8)
print(hits)