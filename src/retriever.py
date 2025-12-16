import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

class RAGRetriever:
    def __init__(self, chunks_file):
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        self.corpus = [chunk["text"] for chunk in self.chunks]
        
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.chunk_embeddings = self.embedder.encode(self.corpus, convert_to_tensor=True)

    def search_bm25(self, query, top_k=10):
        scores = self.bm25.get_scores(query.lower().split())
        top_n = np.argsort(scores)[::-1][:top_k]
        return [{"id": self.chunks[i]["id"], "text": self.chunks[i]["text"], "score": float(scores[i])} for i in top_n]

    def search_semantic(self, query, top_k=10):
        query_vec = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_vec, self.chunk_embeddings, top_k=top_k)[0]
        return [{"id": self.chunks[h['corpus_id']]["id"], "text": self.chunks[h['corpus_id']]["text"], "score": h['score']} for h in hits]

class RAGSystem:
    def __init__(self, retriever):
        self.retriever = retriever
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query, strategy="hybrid", top_k=3):
        candidates = []
        
        initial_k = top_k * 5 

        if strategy == "bm25":
            candidates = self.retriever.search_bm25(query, top_k=initial_k)
            
        elif strategy == "semantic":
            candidates = self.retriever.search_semantic(query, top_k=initial_k)
            
        else: 
            bm25_res = self.retriever.search_bm25(query, top_k=initial_k)
            sem_res = self.retriever.search_semantic(query, top_k=initial_k)
            
            unique_docs = {d['id']: d for d in bm25_res + sem_res}
            candidates = list(unique_docs.values())
        
        if not candidates:
            return []
            
        pairs = [[query, doc['text']] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        for i, doc in enumerate(candidates):
            doc['rerank_score'] = float(scores[i])
            
        final_results = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        return final_results[:top_k]