from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class DocumentProcessor:
    """Splits documents into smaller chunks"""
    def __init__(self, chunk_size=200, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    def process(self, text: str) -> list:
        chunks = self.splitter.split_text(text)
        print(f"Document split into {len(chunks)} chunks")
        return chunks

class Embedder:
    """Converts text into vectors"""
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def embed(self, texts: list) -> np.ndarray:
        return self.model.encode(texts)
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query])[0]

class Retriever:
    """Finds relevant chunks for a query"""
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index = None
        self.chunks = []
    def build_index(self, chunks: list):
        self.chunks = chunks
        embeddings = self.embedder.embed(chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        print(f"Index built with {self.index.ntotal} chunks!")
    def retrieve(self, query: str, top_k: int = 2) -> list:
        query_embedding = self.embedder.embed_query(query)
        query_embedding = np.array([query_embedding]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0]]
