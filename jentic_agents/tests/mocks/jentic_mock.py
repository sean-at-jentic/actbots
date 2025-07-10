"""
A stateful, realistic mock of the Jentic service for robust testing.
"""
import time
from typing import Any, Dict, List, Optional
import asyncio

# Mock Tool Database
# A simple in-memory list of dictionaries representing our tool catalog.
# Each tool has an 'id', 'name', 'description', 'inputs' (matching the real
# Jentic schema), and a lambda for simulating execution.

MOCK_TOOL_DB: List[Dict[str, Any]] = [
    {
        "id": "calculator_01",
        "name": "Simple Calculator",
        "description": "Performs basic arithmetic operations.",
        "inputs": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "description": "One of: add, subtract, multiply, divide"},
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"}
            },
            "required": ["operation", "a", "b"]
        },
        "execute": lambda params: {
            "add": params['a'] + params['b'],
            "subtract": params['a'] - params['b'],
            "multiply": params['a'] * params['b'],
            "divide": params['a'] / params['b']
        }.get(params.get('operation'), "Invalid operation")
    },
    {
        "id": "weather_reporter_01",
        "name": "Weather Reporter",
        "description": "Gets the current weather for a given city.",
        "inputs": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city to get the weather for."}
            },
            "required": ["city"]
        },
        "execute": lambda params: f"The weather in {params.get('city', 'Neverland')} is sunny."
    },
    {
        "id": "ping_01",
        "name": "Ping Tool",
        "description": "A simple tool that returns 'pong'.",
        "inputs": {"type": "object", "properties": {}},
        "execute": lambda params: "pong"
    }
]


class MockRequest:
    """A simple mock for the Jentic SDK's request objects."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class JenticMock:
    """
    A mock Jentic service that mimics the real SDK's behavior with a local,
    in-memory tool database.
    """
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._tool_db = MOCK_TOOL_DB
        self._cache: Dict[str, Dict] = {}

    async def search_api_capabilities(self, request) -> Dict[str, Dict]:
        """Simulates searching for tools, matching the real Jentic structure."""
        query = request.capability_description.lower()
        
        hits = [
            tool for tool in self._tool_db
            if query in tool['name'].lower() or query in tool['description'].lower()
        ]
        
        operations = []
        for hit in hits:
            operations.append({
                "operation_uuid": hit['id'],
                "summary": hit['name'],
                "description": hit['description'],
                "api_name": "mock_api",
                "path": f"/mock/{hit['id']}",
                "method": "POST"
            })
        
        return {"operations": operations, "workflows": []}

    async def load_execution_info(self, operation_uuids: List[str], workflow_uuids: List[str] = [], **kwargs) -> Dict:
        """Simulates loading a tool's detailed definition."""
        tool_id = operation_uuids[0]
        tool = next((t for t in self._tool_db if t['id'] == tool_id), None)
        if not tool:
            return {"operations": {}, "workflows": {}}
        
        # Return a structure similar to the real SDK
        return {
            "operations": {
                tool['id']: {
                    "operation_uuid": tool['id'],
                    "summary": tool['name'],
                    "inputs": tool['inputs'],
                    "api_name": "mock_api",
                }
            },
            "workflows": {}
        }

    async def execute_operation(self, tool_id: str, params: Dict) -> Dict:
        """Simulates executing a tool, matching the real Jentic structure."""
        tool = next((t for t in self._tool_db if t['id'] == tool_id), None)
        if not tool:
            return {"success": False, "output": {"error": "Tool not found"}}
            
        try:
            result_val = tool['execute'](params)
            return {
                "success": True,
                "output": {"result": result_val}
            }
        except Exception as e:
            return {"success": False, "output": {"error": str(e)}}

    # Sync wrappers for easier use in non-async tests if needed
    def search(self, query: str) -> List[Dict]:
        """Convenience sync wrapper for search."""
        request = MockRequest(capability_description=query)
        return asyncio.run(self.search_api_capabilities(request))
    
    def load(self, tool_id: str) -> Dict:
        """Convenience sync wrapper for load."""
        return asyncio.run(self.load_execution_info(operation_uuids=[tool_id]))

    def execute(self, tool_id: str, params: Dict) -> Dict:
        """Convenience sync wrapper for execute."""
        return asyncio.run(self.execute_operation(tool_id, params)) 