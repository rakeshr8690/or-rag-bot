"""
Vector database handler for storing and retrieving embeddings.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
import numpy as np
from pathlib import Path

from ..config.settings import VECTOR_DB_PATH, COLLECTION_NAME, TOP_K_RESULTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Handle vector storage and retrieval using ChromaDB."""
    
    def __init__(self, db_path: Path = VECTOR_DB_PATH, 
                 collection_name: str = COLLECTION_NAME):
        """
        Initialize vector store.
        
        Args:
            db_path: Path to store database
            collection_name: Name of the collection
        """
        logger.info(f"Initializing ChromaDB at {db_path}")
        
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Collection '{collection_name}' ready")
        
    def add_documents(self, chunks: List[Dict], batch_size: int = 100):
        """
        Add document chunks to vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'chunk_id', 'embedding'
            batch_size: Batch size for adding documents
        """
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [chunk['chunk_id'] for chunk in batch]
            texts = [chunk['text'] for chunk in batch]
            embeddings = [chunk['embedding'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Added batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")
            
        logger.info(f"Successfully added {len(chunks)} chunks")
        
    def query(self, query_embedding: List[float], 
              top_k: int = TOP_K_RESULTS,
              filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Query the vector store.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Query results with documents and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0]
        }
        
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        return {
            'total_documents': count,
            'collection_name': self.collection.name
        }
        
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(name=self.collection.name)
        logger.info(f"Deleted collection: {self.collection.name}")
        
    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        collection_name = self.collection.name
        self.delete_collection()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Reset collection: {collection_name}")


class FAISSVectorStore:
    """Alternative vector store using FAISS (for faster retrieval at scale)."""
    
    def __init__(self, dimension: int = 768):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
            
        logger.info(f"Initializing FAISS index with dimension {dimension}")
        
        self.faiss = faiss
        self.index = self.faiss.IndexFlatIP(dimension)
        self.dimension = dimension
        self.documents = []
        self.metadatas = []
        self.ids = []
        
    def add_documents(self, chunks: List[Dict]):
        """
        Add documents to FAISS index.
        
        Args:
            chunks: List of chunk dictionaries
        """
        logger.info(f"Adding {len(chunks)} chunks to FAISS index...")
        
        embeddings = np.array([chunk['embedding'] for chunk in chunks], dtype='float32')
        
        self.faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        
        self.documents.extend([chunk['text'] for chunk in chunks])
        self.metadatas.extend([chunk['metadata'] for chunk in chunks])
        self.ids.extend([chunk['chunk_id'] for chunk in chunks])
        
        logger.info(f"FAISS index now contains {self.index.ntotal} vectors")
        
    def query(self, query_embedding: List[float], top_k: int = TOP_K_RESULTS) -> Dict:
        """
        Query the FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Query results
        """
        import faiss
        
        query_vec = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        results = {
            'ids': [self.ids[idx] for idx in indices[0]],
            'documents': [self.documents[idx] for idx in indices[0]],
            'distances': distances[0].tolist(),
            'metadatas': [self.metadatas[idx] for idx in indices[0]]
        }
        
        return results
        
    def save_index(self, path: str):
        """Save FAISS index to disk."""
        import faiss
        import pickle
        
        faiss.write_index(self.index, f"{path}/faiss.index")
        
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'ids': self.ids
            }, f)
            
        logger.info(f"FAISS index saved to {path}")
        
    def load_index(self, path: str):
        """Load FAISS index from disk."""
        import faiss
        import pickle
        
        self.index = faiss.read_index(f"{path}/faiss.index")
        
        with open(f"{path}/metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadatas = data['metadatas']
            self.ids = data['ids']
            
        logger.info(f"FAISS index loaded from {path}")