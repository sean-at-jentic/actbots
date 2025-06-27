"""
Thin wrapper around jentic-sdk for centralized auth, retries, and logging.
"""
import logging
import os
from typing import Any, Dict, List, Optional

#comment
logger = logging.getLogger(__name__)


class JenticClient:
    """
    Centralized adapter over jentic-sdk that exposes search, load, and execute.
    This client is designed to work directly with live Jentic services and
    requires the Jentic SDK to be installed.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Jentic client.

        Args:
            api_key: Jentic API key. If None, reads from JENTIC_API_KEY environment variable.
        
        Raises:
            ImportError: If the 'jentic' SDK is not installed.
        """
        self.api_key = api_key or os.getenv('JENTIC_API_KEY')
        self._tool_metadata_cache: Dict[str, Dict[str, Any]] = {}

        try:
            from jentic import execute, search_apis, load_execution_info
            self._jentic_execute = execute
            self._jentic_search = search_apis
            self._jentic_load = load_execution_info
            logger.info("JenticClient initialized with REAL Jentic services.")
        except ImportError as e:
            logger.error("The 'jentic' SDK is required to use JenticClient. Please install it.")
            raise ImportError("The 'jentic' package is not installed. Please run 'pip install jentic'.") from e

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for workflows and operations matching a query. Caches metadata for later use.
        """
        logger.info(f"Searching for tools matching query: '{query}' (top_k={top_k})")
        results = self._jentic_search(capability_description=query, max_results=top_k)
        return self._format_and_cache_search_results(results.get('matches', {}), top_k)

    def _format_and_cache_search_results(self, matches: Dict[str, List[Dict]], top_k: int) -> List[Dict[str, Any]]:
        """Formats search results and caches tool metadata."""
        formatted_results = []
        
        for tool_type in ['workflows', 'operations']:
            for tool in matches.get(tool_type, []):
                tool_id = tool.get('workflow_id') or tool.get('operation_uuid')
                if not tool_id:
                    continue

                formatted_tool = {
                    "id": tool_id,
                    "name": tool.get('summary', 'Unnamed Tool'),
                    "description": tool.get('description') or f"{tool.get('method')} {tool.get('path')}",
                    "type": "workflow" if tool_type == 'workflows' else "operation",
                    "api_name": tool.get('api_name', 'unknown'),
                    "parameters": {}  # Loaded on demand by load()
                }
                formatted_results.append(formatted_tool)
                self._tool_metadata_cache[tool_id] = {
                    "type": formatted_tool["type"],
                    "api_name": formatted_tool["api_name"]
                }
        
        return formatted_results[:top_k]

    def load(self, tool_id: str) -> Dict[str, Any]:
        """
        Load the detailed definition for a specific tool by its ID.
        Uses cached metadata to determine if it's a workflow or operation.
        """
        logger.info(f"Loading tool definition for ID: {tool_id}")

        tool_meta = self._tool_metadata_cache.get(tool_id)
        if not tool_meta:
            raise ValueError(f"Tool '{tool_id}' not found in cache. Must be discovered via search() first.")

        load_kwargs = {
            "api_name": tool_meta['api_name'],
            "workflow_uuids": [tool_id] if tool_meta['type'] == 'workflow' else [],
            "operation_uuids": [tool_id] if tool_meta['type'] == 'operation' else []
        }
        results = self._jentic_load(**load_kwargs)
        return self._format_load_results(tool_id, results)


    def _format_load_results(self, tool_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Formats loaded tool definition into a consistent structure."""
        if 'workflows' in results and results['workflows']:
            workflow_key = list(results['workflows'].keys())[0]
            workflow = results['workflows'][workflow_key]
            return {
                "id": tool_id,
                "name": workflow['summary'],
                "description": workflow['description'],
                "type": "workflow",
                "parameters": workflow.get('inputs', {}).get('properties', {}),
                "executable": True,
            }
        elif 'operations' in results and results['operations']:
            operation = results['operations'][tool_id]
            return {
                "id": tool_id,
                "name": operation['summary'],
                "description": f"{operation['method']} {operation['path']}",
                "type": "operation",
                "parameters": operation.get('inputs', {}).get('properties', {}),
                "required": operation.get('inputs', {}).get('required', []),
                "executable": True,
            }
        raise ValueError(f"Could not format load result for tool_id: {tool_id}")

    def execute(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters. Uses cached metadata to determine execution type.
        """
        logger.info(f"Executing tool '{tool_id}' with params: {params}")
        
        tool_meta = self._tool_metadata_cache.get(tool_id)
        if not tool_meta:
            raise ValueError(f"Tool '{tool_id}' not found in cache. Must be discovered via search() first.")
        
        try:
            result = self._jentic_execute(
                execution_type=tool_meta['type'],
                uuid=tool_id,
                inputs=params
            )
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Jentic execution failed for tool '{tool_id}': {e}")
            # Re-raise the exception to be handled by the agent's error handling logic
            raise e
