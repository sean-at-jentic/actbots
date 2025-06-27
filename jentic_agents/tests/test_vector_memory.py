#!/usr/bin/env python3
"""
Test script for VectorStoreMemory implementation.
"""
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from jentic_agents.memory.vector_store_memory import VectorStoreMemory


def test_vector_store_memory():
    """Test the VectorStoreMemory implementation."""
    print("🚀 Testing VectorStoreMemory")
    print("=" * 50)
    
    # Initialize memory
    print("Initializing VectorStoreMemory...")
    memory = VectorStoreMemory()
    
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
    
    # Test keys() method
    print(f"\n📋 All stored keys: {memory.keys()}")
    print(f"Memory contains {len(memory)} items")
    
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
        for i, result in enumerate(results, 1):
            print(f"    {i}. Key: {result['key']}")
            print(f"       Value: {result['value']}")
            print(f"       Score: {result['score']:.3f}")
    
    # Test deletion
    print(f"\n🗑️  Testing deletion:")
    deleted = memory.delete("weather")
    print(f"  Deleted 'weather': {deleted}")
    print(f"  Keys after deletion: {memory.keys()}")
    
    # Test clear
    print(f"\n🧹 Testing clear:")
    memory.clear()
    print(f"  Keys after clear: {memory.keys()}")
    print(f"  Memory contains {len(memory)} items")
    
    memory.close()
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_vector_store_memory() 