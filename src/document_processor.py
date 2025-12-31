"""
Document Processing Module
Handles PDF/text loading, chunking, and topic extraction.
"""

from typing import List, Dict, Optional
from pathlib import Path
import hashlib
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from config import settings


class DocumentProcessor:
    """
    Processes documents for the PrepBot.
    Handles loading, chunking, and metadata extraction.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path.
        
        Args:
            file_path: Path to the document (PDF or TXT)
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine loader based on file extension
        if path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() in ['.txt', '.md']:
            loader = TextLoader(str(path))
        else:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                "Supported types: .pdf, .txt, .md"
            )
        
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata['source_file'] = path.name
            doc.metadata['file_path'] = str(path)
        
        return documents
    
    def load_text(self, text: str, source_name: str = "direct_input") -> List[Document]:
        """
        Load text directly as a Document.
        
        Args:
            text: The text content
            source_name: Name to identify this text source
            
        Returns:
            List containing a single Document object
        """
        doc = Document(
            page_content=text,
            metadata={
                'source_file': source_name,
                'source_type': 'direct_input'
            }
        )
        return [doc]
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
    
    def process_document(
        self,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        source_name: str = "document"
    ) -> Dict:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to file (if loading from file)
            text: Text content (if loading directly)
            source_name: Name for the document source
            
        Returns:
            Dictionary containing:
                - chunks: List of chunked documents
                - metadata: Document metadata
                - doc_id: Unique document identifier
        """
        # Load document
        if file_path:
            documents = self.load_document(file_path)
            source_name = Path(file_path).name
        elif text:
            documents = self.load_text(text, source_name)
        else:
            raise ValueError("Either file_path or text must be provided")
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Generate document ID (hash of content)
        content = "\n".join([doc.page_content for doc in documents])
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Collect metadata
        total_length = sum(len(doc.page_content) for doc in documents)
        metadata = {
            'doc_id': doc_id,
            'source_name': source_name,
            'total_chunks': len(chunks),
            'total_length': total_length,
            'avg_chunk_size': total_length // len(chunks) if chunks else 0,
        }
        
        # Add doc_id to each chunk
        for chunk in chunks:
            chunk.metadata['doc_id'] = doc_id
        
        return {
            'chunks': chunks,
            'metadata': metadata,
            'doc_id': doc_id
        }