"""
Text chunking utilities for semantic document splitting.
"""

from typing import List, Dict
import re
import logging

from ..config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunk documents while preserving semantic meaning."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_by_section(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk document by semantic sections.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries
        """
        section_headers = [
            'PROBLEM TITLE:', 'BUSINESS CONTEXT:', 'DECISION VARIABLES:',
            'OBJECTIVE:', 'CONSTRAINTS:', 'NUMERICAL DATA:',
            'MATHEMATICAL FORMULATION:', 'SOLUTION APPROACH:',
            'OPTIMAL SOLUTION:', 'SENSITIVITY ANALYSIS NOTES:',
            'RELATED PROBLEMS:', 'KEYWORDS:'
        ]
        
        sections = []
        current_section = {'header': 'HEADER', 'content': ''}
        
        for line in text.split('\n'):
            line = line.strip()
            
            is_header = any(line.startswith(header) for header in section_headers)
            
            if is_header and current_section['content']:
                sections.append(current_section)
                current_section = {'header': line, 'content': ''}
            elif is_header:
                current_section['header'] = line
            else:
                current_section['content'] += line + '\n'
                
        if current_section['content']:
            sections.append(current_section)
            
        chunks = []
        problem_id = metadata.get('problem_id', 'UNKNOWN') if metadata else 'UNKNOWN'
        
        for idx, section in enumerate(sections):
            if section['header'] == 'HEADER':
                continue
                
            chunk_text = f"{section['header']}\n{section['content']}"
            
            chunk_with_context = f"[Problem ID: {problem_id}]\n{chunk_text}"
            
            chunk_dict = {
                'text': chunk_with_context,
                'chunk_id': f"{problem_id}_chunk_{idx}",
                'section': section['header'],
                'metadata': metadata if metadata else {}
            }
            
            chunks.append(chunk_dict)
            
        return chunks
        
    def chunk_by_size(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk document by character count with overlap.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        problem_id = metadata.get('problem_id', 'UNKNOWN') if metadata else 'UNKNOWN'
        
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                    
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_dict = {
                    'text': f"[Problem ID: {problem_id}]\n{chunk_text}",
                    'chunk_id': f"{problem_id}_chunk_{chunk_num}",
                    'metadata': metadata if metadata else {}
                }
                chunks.append(chunk_dict)
                chunk_num += 1
                
            start = end - self.overlap
            
        return chunks
        
    def chunk_documents(self, documents: List[Dict], method: str = 'section') -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            method: 'section' or 'size'
            
        Returns:
            List of all chunks
        """
        logger.info(f"Chunking {len(documents)} documents using '{method}' method...")
        
        all_chunks = []
        
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            if method == 'section':
                chunks = self.chunk_by_section(text, metadata)
            elif method == 'size':
                chunks = self.chunk_by_size(text, metadata)
            else:
                raise ValueError(f"Unknown chunking method: {method}")
                
            all_chunks.extend(chunks)
            
        logger.info(f"Created {len(all_chunks)} chunks")
        return all_chunks
        
    def optimize_chunk_size(self, documents: List[Dict]) -> int:
        """
        Suggest optimal chunk size based on document characteristics.
        
        Args:
            documents: List of documents
            
        Returns:
            Recommended chunk size
        """
        lengths = [len(doc.get('text', '')) for doc in documents]
        avg_length = sum(lengths) / len(lengths) if lengths else 1000
        
        optimal_size = int(avg_length / 4)
        
        optimal_size = max(500, min(2000, optimal_size))
        
        logger.info(f"Recommended chunk size: {optimal_size}")
        return optimal_size