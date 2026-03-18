"""
Main script for building the OR RAG system.
Run this to download data, process it, and create the vector database.
"""

import logging
from pathlib import Path

from src.data.data_loader import ORDataLoader
from src.data.data_processor import ORDataProcessor
from src.data.document_structurer import DocumentStructurer
from src.models.embeddings import EmbeddingHandler
from src.models.vector_store import VectorStore
from src.utils.chunking import SemanticChunker
from src.config.settings import DATA_RAW_PATH, DATA_PROCESSED_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline for building the RAG system."""
    
    logger.info("="*60)
    logger.info("Starting OR RAG Bot Setup Pipeline")
    logger.info("="*60)
    
    logger.info("\n[Step 1/6] Downloading datasets...")
    loader = ORDataLoader()
    
    try:
        datasets = loader.download_all_datasets()
        logger.info(f"Downloaded {len(datasets)} datasets")
    except Exception as e:
        logger.warning(f"Some datasets failed to download: {e}")
        datasets = {}
        for file in DATA_RAW_PATH.glob("*.parquet"):
            name = file.stem
            datasets[name] = loader.load_local_dataset(file.name)
            
    if not datasets:
        logger.error("No datasets available. Please download manually.")
        return
        
    logger.info("\n[Step 2/6] Processing datasets...")
    processor = ORDataProcessor()
    
    processed_datasets = {}
    for name, df in datasets.items():
        logger.info(f"Processing {name}...")
        processed_df = processor.process_dataset(df)
        processed_df = processor.deduplicate(processed_df)
        processed_datasets[name] = processed_df
        
        output_path = DATA_PROCESSED_PATH / f"{name}_processed.parquet"
        processed_df.to_parquet(output_path)
        logger.info(f"Saved to {output_path}")
        
    logger.info("\n[Step 3/6] Structuring documents...")
    structurer = DocumentStructurer()
    
    all_documents = []
    for name, df in processed_datasets.items():
        logger.info(f"Structuring {name}...")
        docs = structurer.structure_dataset(df)
        all_documents.extend(docs)
        
    logger.info(f"Total structured documents: {len(all_documents)}")
    
    import json
    structured_path = DATA_PROCESSED_PATH / "structured_documents.json"
    with open(structured_path, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved structured documents to {structured_path}")
    
    logger.info("\n[Step 4/6] Chunking documents...")
    chunker = SemanticChunker()
    
    chunks = chunker.chunk_documents(all_documents, method='section')
    logger.info(f"Created {len(chunks)} chunks")
    
    logger.info("\n[Step 5/6] Generating embeddings...")
    embedding_handler = EmbeddingHandler()
    
    chunks_with_embeddings = embedding_handler.embed_chunks(chunks)
    logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    logger.info("\n[Step 6/6] Storing in vector database...")
    vector_store = VectorStore()
    
    try:
        stats = vector_store.get_collection_stats()
        if stats['total_documents'] > 0:
            logger.info("Resetting existing collection...")
            vector_store.reset_collection()
    except:
        pass
        
    vector_store.add_documents(chunks_with_embeddings)
    
    stats = vector_store.get_collection_stats()
    logger.info("\n" + "="*60)
    logger.info("Setup Complete!")
    logger.info("="*60)
    logger.info(f"Total documents in vector store: {stats['total_documents']}")
    logger.info(f"Collection name: {stats['collection_name']}")
    logger.info("\nYou can now run the Flask app with: python src/app.py")
    logger.info("="*60)


if __name__ == "__main__":
    main()