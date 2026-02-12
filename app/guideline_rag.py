
import json
import logging
import pickle
import os
from pathlib import Path
from typing import List, Dict, Optional

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class GuidelineRAG:
    """Retrieval Augmented Generation engine for ACG Guidelines."""
    
    def __init__(self, data_path: str = "data/acg_guidelines.json", model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger("medrec.rag")
        self.data_path = Path(data_path)
        self.cache_path = self.data_path.with_suffix(".pkl")
        self.guidelines = []
        self.embeddings = None
        self.model = None
        self.vectorizer = None # For TF-IDF fallback
        self.tfidf_matrix = None
        
        self._load_data()
        self._initialize_model(model_name)

    def _load_data(self):
        if not self.data_path.exists():
            self.logger.warning(f"Guideline data not found at {self.data_path}")
            return
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.guidelines = json.load(f)
            
        self.logger.info(f"Loaded {len(self.guidelines)} guidelines.")

    def _initialize_model(self, model_name: str):
        # Check for cached embeddings first
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    # Check if cache matches current data
                    if len(data.get('guidelines', [])) == len(self.guidelines):
                        self.embeddings = data['embeddings']
                        self.logger.info("Loaded embeddings from cache.")
                        # We still need the model for query encoding
                        if HAS_TRANSFORMERS:
                             self.model = SentenceTransformer(model_name)
                        return
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")

        # Compute new embeddings
        if HAS_TRANSFORMERS:
            try:
                self.logger.info(f"Loading SentenceTransformer: {model_name}")
                self.model = SentenceTransformer(model_name)
                texts = [g['content'] for g in self.guidelines]
                self.embeddings = self.model.encode(texts, convert_to_tensor=True)
                
                # Save cache
                with open(self.cache_path, 'wb') as f:
                    pickle.dump({'guidelines': self.guidelines, 'embeddings': self.embeddings}, f)
                    
                return
            except Exception as e:
                self.logger.error(f"SentenceTransformer initialization failed: {e}")
        
        # Fallback to TF-IDF
        if HAS_SKLEARN:
            self.logger.info("Falling back to TF-IDF for RAG.")
            self.vectorizer = TfidfVectorizer(stop_words='english')
            texts = [g['content'] for g in self.guidelines]
            if texts:
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.logger.error("No suitable RAG backend available (needs sentence-transformers or scikit-learn).")

    def retrieve(self, query: str, k: int = 2) -> List[Dict]:
        """Retrieve top-k relevant guidelines for a query."""
        if not self.guidelines:
            return []
            
        if not query.strip():
            return []

        hits = []

        # Strategy 1: Dense Retrieval (Sentence Transformers)
        if self.model is not None and self.embeddings is not None:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_embedding, self.embeddings)[0]
            # Get top k
            top_results = list(enumerate(scores.tolist()))
            top_results.sort(key=lambda x: x[1], reverse=True)
            
            for idx, score in top_results[:k]:
                if score > 0.3: # Minimum threshold
                    hits.append(self.guidelines[idx])
            
            return hits

        # Strategy 2: Sparse Retrieval (TF-IDF)
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = scores.argsort()[::-1][:k]
            
            for idx in top_indices:
                if scores[idx] > 0.1:
                    hits.append(self.guidelines[idx])
            return hits
            
        return []

    def format_for_prompt(self, guidelines: List[Dict]) -> str:
        """Format retrieved guidelines for insertion into LLM prompt."""
        if not guidelines:
            return ""
            
        formatted = "### Relevant Guidelines (Use if applicable):\n"
        for i, g in enumerate(guidelines):
            formatted += f"{i+1}. **{g['topic']}**: {g['content']}\n"
        return formatted
