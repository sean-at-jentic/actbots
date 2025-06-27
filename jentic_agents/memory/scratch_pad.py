"""
Simple dictionary-based memory implementation for development and testing.
"""
from typing import Any, Optional
from .base_memory import BaseMemory


class ScratchPadMemory(BaseMemory):
    """
    Simple in-memory storage using a dictionary.
    
    This is suitable for development, testing, and single-session use cases.
    Data is lost when the process terminates.
    """
    
    def __init__(self):
        """Initialize empty scratch pad memory."""
        self._storage: dict[str, Any] = {}
    
    def store(self, key: str, value: Any) -> None:
        """
        Store a value under the given key.
        
        Args:
            key: Unique identifier for the stored value
            value: Data to store
        """
        self._storage[key] = value
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Args:
            key: Unique identifier for the value
            
        Returns:
            Stored value, or None if key not found
        """
        return self._storage.get(key)
    
    def delete(self, key: str) -> bool:
        """
        Delete a stored value.
        
        Args:
            key: Unique identifier for the value to delete
            
        Returns:
            True if value was deleted, False if key not found
        """
        if key in self._storage:
            del self._storage[key]
            return True
        return False
    
    def clear(self) -> None:
        """
        Clear all stored values.
        """
        self._storage.clear()
    
    def keys(self) -> list[str]:
        """
        Get all stored keys.
        
        Returns:
            List of all keys in memory
        """
        return list(self._storage.keys())
    
    def __len__(self) -> int:
        """Return number of stored items."""
        return len(self._storage)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in storage."""
        return key in self._storage
