import os
import re
from rank_bm25 import BM25Okapi
from typing import List, Tuple

class SimpleRetriever:
    def __init__(self, docs_path: str = "docs/"):
        self.docs_path = docs_path
        self.chunks = []
        self.bm25 = None
        self.load_documents()
    
    def load_documents(self):
        """Load and chunk all markdown documents."""
        for filename in os.listdir(self.docs_path):
            if filename.endswith('.md'):
                with open(os.path.join(self.docs_path, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    chunks = self._chunk_document(content, filename)
                    self.chunks.extend(chunks)
        
        # Build BM25 index
        tokenized_chunks = [self._tokenize(chunk['content']) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def _chunk_document(self, content: str, filename: str) -> List[dict]:
        """Split document into paragraph-level chunks."""
        # Split by paragraphs (blank lines)
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append({
                    'id': f"{filename.replace('.md', '')}::chunk{i}",
                    'content': para.strip(),
                    'source': filename,
                    'score': 0.0
                })
        return chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return re.findall(r'\w+', text.lower())
    
    def retrieve(self, query: str, k: int = 3) -> List[dict]:
        """Retrieve top-k relevant chunks with scores."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Combine chunks with scores
        scored_chunks = []
        for chunk, score in zip(self.chunks, scores):
            chunk_copy = chunk.copy()
            chunk_copy['score'] = float(score)
            scored_chunks.append(chunk_copy)
        
        # Sort by score and return top-k
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        return scored_chunks[:k]