#!/usr/bin/env python3
"""
Test cases for memory implementations.
"""
import sys
import os
import pytest

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.memory.vector_store_memory import VectorStoreMemory


class TestScratchPadMemory:
    """Test cases for ScratchPadMemory class."""
    
    def setup_method(self):
        """Set up a fresh memory instance for each test."""
        self.memory = ScratchPadMemory()
    
    def test_initialization(self):
        """Test that memory initializes correctly."""
        assert len(self.memory) == 0
        assert list(self.memory.keys()) == []
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving values."""
        # Store different types of values
        self.memory.store("string_key", "hello world")
        self.memory.store("int_key", 42)
        self.memory.store("list_key", [1, 2, 3])
        self.memory.store("dict_key", {"name": "test", "value": 123})
        
        # Retrieve values
        assert self.memory.retrieve("string_key") == "hello world"
        assert self.memory.retrieve("int_key") == 42
        assert self.memory.retrieve("list_key") == [1, 2, 3]
        assert self.memory.retrieve("dict_key") == {"name": "test", "value": 123}
        
        # Test retrieving non-existent key
        assert self.memory.retrieve("non_existent") is None
    
    def test_delete(self):
        """Test deleting values."""
        # Store some values
        self.memory.store("key1", "value1")
        self.memory.store("key2", "value2")
        
        # Delete existing key
        assert self.memory.delete("key1") is True
        assert self.memory.retrieve("key1") is None
        assert self.memory.retrieve("key2") == "value2"
        
        # Try to delete non-existent key
        assert self.memory.delete("non_existent") is False
        
        # Try to delete already deleted key
        assert self.memory.delete("key1") is False
    
    def test_clear(self):
        """Test clearing all values."""
        # Store some values
        self.memory.store("key1", "value1")
        self.memory.store("key2", "value2")
        self.memory.store("key3", "value3")
        
        assert len(self.memory) == 3
        
        # Clear all values
        self.memory.clear()
        
        assert len(self.memory) == 0
        assert list(self.memory.keys()) == []
        assert self.memory.retrieve("key1") is None
    
    def test_keys(self):
        """Test getting all keys."""
        # Empty memory
        assert self.memory.keys() == []
        
        # Store some values
        self.memory.store("alpha", 1)
        self.memory.store("beta", 2)
        self.memory.store("gamma", 3)
        
        keys = self.memory.keys()
        assert len(keys) == 3
        assert set(keys) == {"alpha", "beta", "gamma"}
    
    def test_len(self):
        """Test length functionality."""
        assert len(self.memory) == 0
        
        self.memory.store("key1", "value1")
        assert len(self.memory) == 1
        
        self.memory.store("key2", "value2")
        assert len(self.memory) == 2
        
        self.memory.delete("key1")
        assert len(self.memory) == 1
        
        self.memory.clear()
        assert len(self.memory) == 0
    
    def test_contains(self):
        """Test membership testing with 'in' operator."""
        assert "key1" not in self.memory
        
        self.memory.store("key1", "value1")
        assert "key1" in self.memory
        assert "key2" not in self.memory
        
        self.memory.store("key2", "value2")
        assert "key1" in self.memory
        assert "key2" in self.memory
        
        self.memory.delete("key1")
        assert "key1" not in self.memory
        assert "key2" in self.memory
    
    def test_overwrite_value(self):
        """Test overwriting existing values."""
        self.memory.store("key1", "original_value")
        assert self.memory.retrieve("key1") == "original_value"
        
        self.memory.store("key1", "new_value")
        assert self.memory.retrieve("key1") == "new_value"
        assert len(self.memory) == 1  # Should still be one item
    
    def test_empty_and_none_values(self):
        """Test storing empty strings and None values."""
        self.memory.store("empty_string", "")
        self.memory.store("none_value", None)
        self.memory.store("zero", 0)
        self.memory.store("empty_list", [])
        self.memory.store("empty_dict", {})
        
        assert self.memory.retrieve("empty_string") == ""
        assert self.memory.retrieve("none_value") is None
        assert self.memory.retrieve("zero") == 0
        assert self.memory.retrieve("empty_list") == []
        assert self.memory.retrieve("empty_dict") == {}
        
        # All keys should exist
        assert "empty_string" in self.memory
        assert "none_value" in self.memory
        assert "zero" in self.memory
        assert "empty_list" in self.memory
        assert "empty_dict" in self.memory


class TestVectorStoreMemory:
    """Test cases for VectorStoreMemory class."""
    
    def setup_method(self):
        """Set up a fresh memory instance for each test."""
        self.memory = VectorStoreMemory(collection_name="test_collection")
    
    def teardown_method(self):
        """Clean up after each test."""
        self.memory.clear()
        self.memory.close()
    
    def test_initialization(self):
        """Test that memory initializes correctly."""
        assert len(self.memory) == 0
        assert list(self.memory.keys()) == []
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving values."""
        # Store some text data
        self.memory.store("fact1", "The capital of France is Paris")
        self.memory.store("fact2", "Python is a programming language")
        self.memory.store("fact3", "The Pacific Ocean is the largest ocean")
        
        # Retrieve by exact key
        assert self.memory.retrieve("fact1") == "The capital of France is Paris"
        assert self.memory.retrieve("fact2") == "Python is a programming language"
        assert self.memory.retrieve("fact3") == "The Pacific Ocean is the largest ocean"
        
        # Test retrieving non-existent key
        assert self.memory.retrieve("non_existent") is None
    
    def test_search(self):
        """Test semantic search functionality."""
        # Store some related content
        self.memory.store("math1", "The square root of 16 is 4")
        self.memory.store("math2", "Calculus is a branch of mathematics")
        self.memory.store("prog1", "Python is used for data science")
        self.memory.store("prog2", "JavaScript runs in web browsers")
        self.memory.store("geo1", "Mount Everest is the tallest mountain")
        
        # Search for mathematics-related content
        math_results = self.memory.search("mathematics and numbers", top_k=3)
        assert len(math_results) > 0
        
        # Check that results have the expected structure
        for result in math_results:
            assert "key" in result
            assert "value" in result
            assert "score" in result
            assert 0 <= result["score"] <= 1
        
        # Search for programming content
        prog_results = self.memory.search("programming languages", top_k=2)
        assert len(prog_results) > 0
    
    def test_delete(self):
        """Test deleting values."""
        # Store some values
        self.memory.store("temp1", "Temporary data 1")
        self.memory.store("temp2", "Temporary data 2")
        self.memory.store("keep", "Keep this data")
        
        assert len(self.memory) == 3
        
        # Delete existing key
        assert self.memory.delete("temp1") is True
        assert self.memory.retrieve("temp1") is None
        assert self.memory.retrieve("temp2") == "Temporary data 2"
        assert len(self.memory) == 2
        
        # Try to delete non-existent key
        assert self.memory.delete("non_existent") is False
        
        # Try to delete already deleted key
        assert self.memory.delete("temp1") is False
    
    def test_clear(self):
        """Test clearing all values."""
        # Store some values
        self.memory.store("item1", "First item")
        self.memory.store("item2", "Second item")
        self.memory.store("item3", "Third item")
        
        assert len(self.memory) == 3
        
        # Clear all values
        self.memory.clear()
        
        assert len(self.memory) == 0
        assert list(self.memory.keys()) == []
        assert self.memory.retrieve("item1") is None
    
    def test_keys(self):
        """Test getting all keys."""
        # Empty memory
        assert self.memory.keys() == []
        
        # Store some values
        self.memory.store("alpha", "Alpha value")
        self.memory.store("beta", "Beta value")
        self.memory.store("gamma", "Gamma value")
        
        keys = self.memory.keys()
        assert len(keys) == 3
        assert set(keys) == {"alpha", "beta", "gamma"}
    
    def test_len(self):
        """Test length functionality."""
        assert len(self.memory) == 0
        
        self.memory.store("item1", "Value 1")
        assert len(self.memory) == 1
        
        self.memory.store("item2", "Value 2")
        assert len(self.memory) == 2
        
        self.memory.delete("item1")
        assert len(self.memory) == 1
        
        self.memory.clear()
        assert len(self.memory) == 0
    
    def test_contains(self):
        """Test membership testing with 'in' operator."""
        assert "key1" not in self.memory
        
        self.memory.store("key1", "Some value")
        assert "key1" in self.memory
        assert "key2" not in self.memory
        
        self.memory.store("key2", "Another value")
        assert "key1" in self.memory
        assert "key2" in self.memory
        
        self.memory.delete("key1")
        assert "key1" not in self.memory
        assert "key2" in self.memory
    
    def test_overwrite_value(self):
        """Test overwriting existing values."""
        self.memory.store("doc1", "Original document content")
        assert self.memory.retrieve("doc1") == "Original document content"
        
        self.memory.store("doc1", "Updated document content")
        assert self.memory.retrieve("doc1") == "Updated document content"
        assert len(self.memory) == 1  # Should still be one item


def test_scratch_pad_memory_integration():
    """Integration test for ScratchPadMemory."""
    print("🧠 Testing ScratchPadMemory Integration")
    print("=" * 50)
    
    memory = ScratchPadMemory()
    
    # Test basic operations
    print("Testing basic store/retrieve operations...")
    memory.store("user_name", "Alice")
    memory.store("user_age", 30)
    memory.store("preferences", {"theme": "dark", "language": "en"})
    
    assert memory.retrieve("user_name") == "Alice"
    assert memory.retrieve("user_age") == 30
    assert memory.retrieve("preferences") == {"theme": "dark", "language": "en"}
    
    print(f"Memory contains {len(memory)} items")
    print(f"Keys: {memory.keys()}")
    
    # Test deletion
    print("Testing deletion...")
    assert memory.delete("user_age") is True
    assert memory.retrieve("user_age") is None
    assert len(memory) == 2
    
    # Test membership
    print("Testing membership...")
    assert "user_name" in memory
    assert "user_age" not in memory
    
    print("✅ ScratchPadMemory integration test completed successfully!")


def test_vector_store_memory_integration():
    """Integration test for VectorStoreMemory."""
    print("🚀 Testing VectorStoreMemory Integration")
    print("=" * 50)
    
    # Initialize memory
    print("Initializing VectorStoreMemory...")
    memory = VectorStoreMemory(collection_name="integration_test")
    
    # Test basic storage and retrieval
    print("\n📝 Testing basic store/retrieve operations:")
    
    test_data = {
        "math_fact": "The square root of 16 is 4",
        "python_tip": "You can use list comprehensions to create lists efficiently",
        "weather": "It's sunny and warm today with blue skies",
        "cooking": "To make pasta, boil water with salt and cook for 8-10 minutes",
        "history": "The Roman Empire fell in 476 AD"
    }
    
    # Store the data
    for key, value in test_data.items():
        memory.store(key, value)
        print(f"  ✓ Stored: {key}")
    
    # Test retrieval by exact key
    print(f"\n🔍 Testing exact key retrieval:")
    retrieved = memory.retrieve("math_fact")
    print(f"  Retrieved 'math_fact': {retrieved}")
    assert retrieved == "The square root of 16 is 4"
    
    # Test keys() method
    print(f"\n📋 All stored keys: {memory.keys()}")
    print(f"Memory contains {len(memory)} items")
    assert len(memory) == 5
    
    # Test semantic search
    print(f"\n🔍 Testing semantic search:")
    
    search_queries = [
        "mathematics and numbers",
        "programming in Python", 
        "outdoor conditions",
        "food preparation",
        "ancient civilizations"
    ]
    
    for query in search_queries:
        print(f"\n  Query: '{query}'")
        results = memory.search(query, top_k=2)
        assert len(results) > 0
        for i, result in enumerate(results, 1):
            print(f"    {i}. Key: {result['key']}")
            print(f"       Value: {result['value']}")
            print(f"       Score: {result['score']:.3f}")
    
    # Test deletion
    print(f"\n🗑️  Testing deletion:")
    deleted = memory.delete("weather")
    print(f"  Deleted 'weather': {deleted}")
    assert deleted is True
    keys_after_delete = memory.keys()
    print(f"  Keys after deletion: {keys_after_delete}")
    assert "weather" not in keys_after_delete
    assert len(memory) == 4
    
    # Test clear
    print(f"\n🧹 Testing clear:")
    memory.clear()
    keys_after_clear = memory.keys()
    print(f"  Keys after clear: {keys_after_clear}")
    print(f"  Memory contains {len(memory)} items")
    assert len(memory) == 0
    assert keys_after_clear == []
    
    memory.close()
    print("\n✅ VectorStoreMemory integration test completed successfully!")


if __name__ == "__main__":
    test_scratch_pad_memory_integration()
    test_vector_store_memory_integration()
