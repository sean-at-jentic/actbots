"""
Simple dictionary-based memory implementation for development and testing.
"""
from typing import Any, Optional
from .base_memory import BaseMemory
from ..utils.logger import get_logger
from ..utils.block_timer import Timer

logger = get_logger(__name__)


class ScratchPadMemory(BaseMemory):
    """
    Simple in-memory storage using a dictionary.
    
    This is suitable for development, testing, and single-session use cases.
    Data is lost when the process terminates.
    """
    
    def __init__(self):
        """Initialize empty scratch pad memory."""
        logger.info("Initializing ScratchPadMemory")
        self._storage: dict[str, Any] = {}
        logger.debug("ScratchPadMemory initialized with empty storage")
    
    def store(self, key: str, value: Any) -> None:
        """
        Store a value under the given key.
        
        Args:
            key: Unique identifier for the stored value
            value: Data to store
        """
        logger.debug(f"Storing key: {key}")
        with Timer(f"Store key '{key}'"):
            self._storage[key] = value
        logger.debug(f"Successfully stored key: {key}")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Args:
            key: Unique identifier for the value
            
        Returns:
            Stored value, or None if key not found
        """
        logger.debug(f"Retrieving key: {key}")
        with Timer(f"Retrieve key '{key}'"):
            value = self._storage.get(key)
        
        if value is not None:
            logger.debug(f"Key '{key}' found")
        else:
            logger.debug(f"Key '{key}' not found")
        return value
    
    def delete(self, key: str) -> bool:
        """
        Delete a stored value.
        
        Args:
            key: Unique identifier for the value to delete
            
        Returns:
            True if value was deleted, False if key not found
        """
        logger.debug(f"Attempting to delete key: {key}")
        with Timer(f"Delete key '{key}'"):
            if key in self._storage:
                del self._storage[key]
                logger.info(f"Deleted key: {key}")
                return True
            else:
                logger.debug(f"Key '{key}' not found for deletion")
                return False
    
    def clear(self) -> None:
        """
        Clear all stored values.
        """
        logger.warning("Clearing all stored values from ScratchPadMemory")
        with Timer("Clear all storage"):
            self._storage.clear()
        logger.info("ScratchPadMemory cleared")
    
    def keys(self) -> list[str]:
        """
        Get all stored keys.
        
        Returns:
            List of all keys in memory
        """
        logger.debug("Retrieving all keys")
        with Timer("Get all keys"):
            keys = list(self._storage.keys())
        logger.debug(f"Retrieved {len(keys)} keys")
        return keys
    
    def __len__(self) -> int:
        """Return number of stored items."""
        logger.debug("Getting item count")
        with Timer("Count items"):
            count = len(self._storage)
        logger.debug(f"ScratchPadMemory has {count} items")
        return count
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in storage."""
        logger.debug(f"Checking for existence of key: {key}")
        with Timer(f"Check existence of key '{key}'"):
            exists = key in self._storage
        logger.debug(f"Key '{key}' exists: {exists}")
        return exists
