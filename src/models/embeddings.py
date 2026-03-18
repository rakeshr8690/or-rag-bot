"""
Embedding model handler for converting text to vectors.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import logging
from tqdm import tqdm

from ..config.settings import EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingHandler:
    """Handle text embeddings using sentence transformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of sentence-transformers model
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
        
    def embed_batch(self, texts: List[str], batch_size: int = 32, 
                    show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Embedding {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
        
    def embed_chunks(self, chunks: List[dict], batch_size: int = 32) -> List[dict]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            batch_size: Batch size for processing
            
        Returns:
            List of chunks with embeddings added
        """
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
            
        return chunks
        
    def compute_similarity(self, query_embedding: np.ndarray, 
                          doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding matrix
            
        Returns:
            Similarity scores
        """
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
        
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model._modules['0'].auto_model.config._name_or_path,
            'embedding_dimension': self.embedding_dim,
            'max_sequence_length': self.model.max_seq_length
        }