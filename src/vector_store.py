"""
Vector Store Module
Manages ChromaDB for document storage and retrieval using AWS Bedrock embeddings.
"""

from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from bedrock_models import get_embeddings


class VectorStore:
    """
    Manages vector storage and retrieval using ChromaDB and Bedrock embeddings.
    Supports multiple document collections and similarity search.
    """
    
    def __init__(self, collection_name: str = "default"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.embeddings = get_embeddings()
        self.persist_directory = str(settings.vector_db_dir / collection_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"✓ Vector store initialized: {collection_name}")
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to the vector store in batches.
        
        Args:
            documents: List of Document objects to add (chunks)
            batch_size: Number of documents to process at once
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        print(f"📊 Storing {len(documents)} text chunks in vector database...")
        
        # Add documents in batches to avoid memory issues
        all_ids = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = self.vectorstore.add_documents(batch)
            all_ids.extend(ids)
            print(f"  ✓ Embedded and stored {min(i + batch_size, len(documents))}/{len(documents)} chunks")
        
        print(f"✓ Successfully stored all {len(all_ids)} chunks")
        return all_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of most similar Documents
        """
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of tuples (Document, similarity_score)
        """
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        return results
    
    def search_by_document_id(
        self,
        doc_id: str,
        k: int = 10
    ) -> List[Document]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            doc_id: Document ID to search for
            k: Maximum number of chunks to return
            
        Returns:
            List of document chunks
        """
        results = self.vectorstore.similarity_search(
            query="",  # Empty query with filter
            k=k,
            filter={"doc_id": doc_id}
        )
        return results
    
    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict] = None
    ):
        """
        Get a LangChain retriever for use in chains.
        
        Args:
            search_type: Type of search ("similarity", "mmr", etc.)
            search_kwargs: Additional search parameters
            
        Returns:
            LangChain retriever object
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✓ Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Warning: Could not delete collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "document_count": 0,
                "error": str(e)
            }
    
    def list_documents(self) -> List[str]:
        """
        List all unique document IDs in the collection.
        
        Returns:
            List of document IDs
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            # Get all documents with their metadata
            results = collection.get(include=["metadatas"])
            
            # Extract unique doc_ids
            doc_ids = set()
            for metadata in results.get("metadatas", []):
                if "doc_id" in metadata:
                    doc_ids.add(metadata["doc_id"])
            
            return sorted(list(doc_ids))
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []


class VectorStoreManager:
    """
    Manages multiple vector store collections.
    Useful for organizing documents by topic or session.
    """
    
    def __init__(self):
        """Initialize the vector store manager."""
        self.stores: Dict[str, VectorStore] = {}
    
    def get_or_create_store(self, collection_name: str) -> VectorStore:
        """
        Get an existing store or create a new one.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            VectorStore instance
        """
        if collection_name not in self.stores:
            self.stores[collection_name] = VectorStore(collection_name)
        
        return self.stores[collection_name]
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        vector_db_path = settings.vector_db_dir
        if not vector_db_path.exists():
            return []
        
        # List subdirectories (each is a collection)
        collections = [
            d.name for d in vector_db_path.iterdir() 
            if d.is_dir()
        ]
        return collections
    
    def delete_collection(self, collection_name: str):
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
        """
        if collection_name in self.stores:
            self.stores[collection_name].delete_collection()
            del self.stores[collection_name]
        else:
            # Create temporary store to delete
            store = VectorStore(collection_name)
            store.delete_collection()


# Global manager instance
vector_store_manager = VectorStoreManager()


def get_vector_store(collection_name: str = "default") -> VectorStore:
    """
    Get or create a vector store.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        VectorStore instance
    """
    return vector_store_manager.get_or_create_store(collection_name)