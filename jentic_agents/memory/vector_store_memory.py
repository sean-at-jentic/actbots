"""
Memory backend using ChromaDB for semantic search.
"""
import chromadb
from typing import Any, List, Optional
from .base_memory import BaseMemory


class VectorStoreMemory(BaseMemory):
    """
    Memory system using ChromaDB for vector search.

    This allows for semantic retrieval of memories based on text similarity.
    """
    def __init__(self, collection_name: str = "agent_memory", persist_directory: str = None):
        """
        Initialize the vector store memory.

        Args:
            collection_name: Name of the ChromaDB collection to use.
            persist_directory: Directory to persist the database. If None, uses in-memory storage.
        """
        self._collection_name = collection_name
        
        # Create client - persistent or in-memory
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        
        # Create or get collection
        try:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        except chromadb.errors.UniqueConstraintError:
            # Collection already exists
            self._collection = self._client.get_collection(name=collection_name)

    def store(self, key: str, value: str) -> None:
        """
        Store a value and its vector embedding.

        Args:
            key: Unique identifier for the stored value.
            value: Text data to store.
        """
        # ChromaDB automatically generates embeddings and handles duplicates
        self._collection.upsert(
            documents=[value],
            ids=[key]
        )

    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by its exact key.

        Args:
            key: Unique identifier for the value.
            
        Returns:
            Stored value, or None if key not found.
        """
        try:
            result = self._collection.get(ids=[key])
            return result['documents'][0] if result['documents'] else None
        except Exception:
            return None

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search for memories semantically similar to the query.

        Args:
            query: The text to search for.
            top_k: The number of results to return.

        Returns:
            A list of dictionaries, each containing the key, value and similarity score.
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        search_results = []
        if results['ids'] and results['ids'][0]:  # Check if we have results
            for i, (key, value, distance) in enumerate(zip(
                results['ids'][0], 
                results['documents'][0], 
                results['distances'][0]
            )):
                search_results.append({
                    "key": key,
                    "value": value,
                    "score": 1 - distance  # Convert distance to similarity score (0-1)
                })
        
        return search_results

    def delete(self, key: str) -> bool:
        """
        Delete a stored value by its key.

        Args:
            key: Unique identifier for the value to delete.
            
        Returns:
            True if value was deleted, False if key not found.
        """
        try:
            # Check if key exists first
            result = self._collection.get(ids=[key])
            if not result['documents']:
                return False
            
            # Delete the document
            self._collection.delete(ids=[key])
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all stored values."""
        # Delete the collection and recreate it
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def keys(self) -> list[str]:
        """Get all stored keys."""
        try:
            result = self._collection.get()
            return result['ids'] if result['ids'] else []
        except Exception:
            return []

    def close(self):
        """Close the database connection (no-op for ChromaDB)."""
        pass  # ChromaDB handles cleanup automatically
        
    def __len__(self) -> int:
        """Return number of stored items."""
        try:
            return self._collection.count()
        except Exception:
            return 0
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in storage."""
        try:
            result = self._collection.get(ids=[key])
            return bool(result['documents'])
        except Exception:
            return False 