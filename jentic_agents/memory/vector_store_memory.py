"""
Memory backend using ChromaDB for semantic search.
"""
import chromadb
from typing import Any, List, Optional
from .base_memory import BaseMemory
from ..utils.logger import get_logger
from ..utils.block_timer import Timer
import os

logger = get_logger(__name__)


class VectorStoreMemory(BaseMemory):
    """
    Memory system using ChromaDB for vector search.

    This allows for semantic retrieval of memories based on text similarity.
    """
    def __init__(self, collection_name: str = "agent_memory"):
        """
        Initialize the vector store memory.

        Args:
            collection_name: Name of the ChromaDB collection to use.
        """
        self._collection_name = collection_name
        
        # Try to get persist directory from environment
        # YOU MUST SET THE CHROMA_DB_PATH ENVIRONMENT VARIABLE TO USE PERSISTENT STORAGE - OTHERWISE IT WILL USE IN-MEMORY STORAGE
        persist_directory = None
        try:
            persist_directory = os.environ.get("CHROMA_DB_PATH")
            logger.info(f"CHROMA_DB_PATH: {persist_directory}")
        except Exception as e:
            logger.warning(f"Failed to fetch VECTOR_STORE_PERSIST_DIR from environment: {e}")
            logger.warning("Using in-memory storage.")
            persist_directory = None
        
        logger.info(f"Initializing VectorStoreMemory with collection: {collection_name}")
        
        # Create client - persistent or in-memory
        if persist_directory:
            logger.info(f"Using persistent storage at: {persist_directory}")
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            logger.info("Using in-memory storage.")
            self._client = chromadb.Client()
        
        # Create or get collection
        try:
            with Timer(f"Create collection '{collection_name}'"):
                self._collection = self._client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
            logger.info(f"Created new ChromaDB collection: {collection_name}")
        except chromadb.errors.UniqueConstraintError:
            # Collection already exists
            with Timer(f"Get collection '{collection_name}'"):
                self._collection = self._client.get_collection(name=collection_name)
            logger.info(f"Using existing ChromaDB collection: {collection_name}")

    def store(self, key: str, value: str) -> None:
        """
        Store a value and its vector embedding.

        Args:
            key: Unique identifier for the stored value.
            value: Text data to store.
        """
        logger.debug(f"Storing key: {key}")
        with Timer(f"Store key '{key}'"):
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
        logger.debug(f"Retrieving key: {key}")
        try:
            with Timer(f"Retrieve key '{key}'"):
                result = self._collection.get(ids=[key])
            
            doc = result['documents'][0] if result['documents'] else None
            if doc:
                logger.debug(f"Key '{key}' found.")
            else:
                logger.debug(f"Key '{key}' not found.")
            return doc
        except Exception as e:
            logger.error(f"Error retrieving key '{key}': {e}")
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
        logger.info(f"Searching for '{query}' with top_k={top_k}")
        with Timer(f"Search for '{query}'"):
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
        
        logger.info(f"Found {len(search_results)} results for query: '{query}'")
        return search_results

    def delete(self, key: str) -> bool:
        """
        Delete a stored value by its key.

        Args:
            key: Unique identifier for the value to delete.
            
        Returns:
            True if value was deleted, False if key not found.
        """
        logger.debug(f"Attempting to delete key: {key}")
        try:
            # Check if key exists first
            with Timer(f"Check existence for delete on key '{key}'"):
                result = self._collection.get(ids=[key])

            if not result['documents']:
                logger.debug(f"Key '{key}' not found for deletion.")
                return False
            
            # Delete the document
            with Timer(f"Delete key '{key}'"):
                self._collection.delete(ids=[key])
            logger.info(f"Deleted key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting key '{key}': {e}")
            return False

    def clear(self) -> None:
        """Clear all stored values."""
        logger.warning(f"Clearing all data from collection: {self._collection_name}")
        with Timer(f"Clear collection '{self._collection_name}'"):
            # Delete the collection and recreate it
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        logger.info(f"Collection '{self._collection_name}' cleared and recreated.")

    def keys(self) -> list[str]:
        """Get all stored keys."""
        logger.debug("Retrieving all keys.")
        try:
            with Timer("Get all keys"):
                result = self._collection.get()
            keys = result['ids'] if result['ids'] else []
            logger.debug(f"Retrieved {len(keys)} keys.")
            return keys
        except Exception as e:
            logger.error(f"Error retrieving all keys: {e}")
            return []

    def close(self):
        """Close the database connection (no-op for ChromaDB)."""
        logger.info("ChromaDB client does not require explicit closing.")
        pass  # ChromaDB handles cleanup automatically
        
    def __len__(self) -> int:
        """Return number of stored items."""
        logger.debug("Getting item count.")
        try:
            with Timer("Count items"):
                count = self._collection.count()
            logger.debug(f"Collection has {count} items.")
            return count
        except Exception as e:
            logger.error(f"Error getting item count: {e}")
            return 0
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in storage."""
        logger.debug(f"Checking for existence of key: {key}")
        try:
            with Timer(f"Check existence of key '{key}'"):
                result = self._collection.get(ids=[key])
            exists = bool(result['documents'])
            logger.debug(f"Key '{key}' exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking for key '{key}': {e}")
            return False 