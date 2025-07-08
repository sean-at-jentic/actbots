"""
Abstract base class for memory backends that store and retrieve information.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMemory(ABC):
    """
    Abstract base class for memory implementations.
    """

    @abstractmethod
    def store(self, key: str, value: Any) -> None:
        """
        Store a value under the given key.

        Args:
            key: Unique identifier for the stored value
            value: Data to store (will be serialized as needed)
        """
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.

        Args:
            key: Unique identifier for the value

        Returns:
            Stored value, or None if key not found
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a stored value.

        Args:
            key: Unique identifier for the value to delete

        Returns:
            True if value was deleted, False if key not found
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all stored values.
        """
        pass

    @abstractmethod
    def keys(self) -> list[str]:
        """
        Get all stored keys.

        Returns:
            List of all keys in memory
        """
        pass

    @abstractmethod
    def resolve_placeholders(self, obj: Any) -> Any:
        """
        Recursively substitute memory placeholders inside an object.
        """
        pass

    @abstractmethod
    def validate_placeholders(self, args: Dict[str, Any], required_fields: list) -> tuple[Optional[str], Optional[str]]:
        """
        Validates placeholders in arguments, returning an error and correction prompt if needed.
        """
        pass
