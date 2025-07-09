"""
Thin wrapper around jentic-sdk for centralized auth, retries, and logging.
"""

import logging
import os
from typing import Any, Dict, List, Optional

# Standard module logger
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
        self.api_key = api_key or os.getenv("JENTIC_API_KEY")
        self._tool_metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Lazily-import the official Jentic SDK.  We do this here (instead of at
        # module import-time) so unit-tests can monkey-patch the import path.
        try:
            from jentic import Jentic  # type: ignore

            self._sdk_client = Jentic(api_key=self.api_key)  # async SDK instance

            # Models are only needed inside methods, so we import them lazily to
            # avoid bloating start-up time.
            import importlib

            self._sdk_models = importlib.import_module("jentic.models")

            logger.info("JenticClient initialized with Jentic services.")

        except ImportError as exc:
            logger.error(
                "The 'jentic' SDK could not be imported – ensure it is installed and available in the current environment."
            )
            raise ImportError(
                "The 'jentic' package is not installed or is an incompatible version. "
                "Please run 'pip install -U jentic'."
            ) from exc

        # Internal helper to synchronously call the async SDK methods.
        import asyncio

        def _sync(coro):
            """Run *coro* in a new event loop and return the result.

            If an event-loop is already running (e.g. inside a Jupyter notebook),
            fall back to `asyncio.get_event_loop().run_until_complete`.
            """
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Avoid deadlocks / unexpected behaviour in environments that
                # already have a running event loop (e.g. Jupyter). For now we
                # fail fast so the caller can decide how to integrate the async
                # SDK properly.
                raise RuntimeError(
                    "JenticClient synchronous wrapper called from within an "
                    "active asyncio event-loop. Use the async SDK directly or "
                    "await the underlying coroutine instead."
                )

            return asyncio.run(coro)

        self._sync = _sync

    def _is_async_context(self) -> bool:
        """Check if we're running in an async context."""
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            is_async = loop.is_running()
            logger.debug(f"_is_async_context check: {is_async}, loop: {loop}")
            return is_async
        except RuntimeError as e:
            logger.debug(f"_is_async_context check: False, RuntimeError: {e}")
            return False

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Async version of search for use in async contexts.
        """
        logger.info(f"Searching for tools: '{query}' (top {top_k})")

        # Build request model for the SDK.
        RequestModel = getattr(self._sdk_models, "ApiCapabilitySearchRequest")
        search_request = RequestModel(capability_description=query, max_results=top_k)

        # Call the async SDK directly.
        results = await self._sdk_client.search_api_capabilities(search_request)

        # Pydantic model ➔ dict
        if hasattr(results, "model_dump"):
            results_dict = results.model_dump(exclude_none=False)
        else:
            # Fallback for non-Pydantic objects.
            results_dict = dict(results)

        return self._format_and_cache_search_results(results_dict, top_k)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for workflows and operations matching a query. Caches metadata for later use.
        """
        # Check if we're in an async context and use async method if so
        if self._is_async_context():
            logger.info(f"Detected async context, using ThreadPoolExecutor for search: {query}")
            import asyncio
            import concurrent.futures
            
            try:
                # Create a new thread to run the async operation
                def run_async():
                    logger.info(f"Running search_async in new thread for: {query}")
                    # Create a new event loop in this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.search_async(query, top_k))
                        logger.info(f"ThreadPoolExecutor search completed successfully for: {query}")
                        return result
                    except Exception as e:
                        logger.error(f"ThreadPoolExecutor search failed for {query}: {e}")
                        raise
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    result = future.result()
                    logger.info(f"ThreadPoolExecutor returned result for {query}: {len(result)} items")
                    return result
            except Exception as e:
                logger.error(f"ThreadPoolExecutor approach failed for {query}: {e}, falling back to sync")
                # Don't fall back to sync - re-raise the exception
                raise RuntimeError(f"Async context detected but ThreadPoolExecutor failed: {e}") from e
        
        logger.info(f"Searching for tools: '{query}' (top {top_k})")

        # Build request model for the SDK.
        RequestModel = getattr(self._sdk_models, "ApiCapabilitySearchRequest")

        search_request = RequestModel(capability_description=query, max_results=top_k)

        # Call the async SDK synchronously.
        results = self._sync(self._sdk_client.search_api_capabilities(search_request))

        # Pydantic model ➔ dict
        if hasattr(results, "model_dump"):
            results_dict = results.model_dump(exclude_none=False)
        else:
            # Fallback for non-Pydantic objects.
            results_dict = dict(results)

        return self._format_and_cache_search_results(results_dict, top_k)

    def _format_and_cache_search_results(
        self, payload: Dict[str, Any], top_k: int
    ) -> List[Dict[str, Any]]:
        """Formats search results and caches tool metadata."""
        formatted_results = []

        # API returns e.g. {"workflows": [...], "operations": [...]}  – iterate over both.
        for tool_type in ("workflows", "operations"):
            for tool in payload.get(tool_type, []):
                tool_id = tool.get("workflow_id") or tool.get("operation_uuid")
                if not tool_id:
                    continue

                formatted_tool = {
                    "id": tool_id,
                    "name": tool.get("summary", "Unnamed Tool"),
                    "description": tool.get("description")
                    or f"{tool.get('method')} {tool.get('path')}",
                    "type": "workflow" if tool_type == "workflows" else "operation",
                    "api_name": tool.get("api_name", "unknown"),
                    "parameters": {},  # Loaded on demand by load()
                }
                formatted_results.append(formatted_tool)
                self._tool_metadata_cache[tool_id] = {
                    "type": formatted_tool["type"],
                    "api_name": formatted_tool["api_name"],
                }

        return formatted_results[:top_k]

    async def load_async(self, tool_id: str) -> Dict[str, Any]:
        """
        Async version of load for use in async contexts.
        """
        logger.info(f"Loading tool definition for: {tool_id}")

        tool_meta = self._tool_metadata_cache.get(tool_id)
        if not tool_meta:
            raise ValueError(f"Tool '{tool_id}' not found in cache. Must be discovered via search() first.")

        # Prepare and execute load request via SDK
        results = await self._sdk_client.load_execution_info(
            workflow_uuids=[tool_id] if tool_meta["type"] == "workflow" else [],
            operation_uuids=[tool_id] if tool_meta["type"] == "operation" else [],
            api_name=tool_meta["api_name"],
        )

        # Convert Pydantic → dict for downstream processing.
        if hasattr(results, "model_dump"):
            results = results.model_dump(exclude_none=False)

        return self._format_load_results(tool_id, results)

    def load(self, tool_id: str) -> Dict[str, Any]:
        """
        Load the detailed definition for a specific tool by its ID.
        Uses cached metadata to determine if it's a workflow or operation.
        """
        # Check if we're in an async context and use async method if so
        if self._is_async_context():
            logger.info(f"Detected async context, using ThreadPoolExecutor for load: {tool_id}")
            import concurrent.futures
            
            try:
                # Create a new thread to run the async operation
                def run_async():
                    logger.info(f"Running load_async in new thread for: {tool_id}")
                    # Create a new event loop in this thread
                    import asyncio
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.load_async(tool_id))
                        logger.info(f"ThreadPoolExecutor load completed successfully for: {tool_id}")
                        return result
                    except Exception as e:
                        logger.error(f"ThreadPoolExecutor load failed for {tool_id}: {e}")
                        raise
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    result = future.result()
                    logger.info(f"ThreadPoolExecutor returned load result for {tool_id}")
                    return result
            except Exception as e:
                logger.error(f"ThreadPoolExecutor approach failed for load {tool_id}: {e}")
                # Don't fall back to sync - re-raise the exception
                raise RuntimeError(f"Async context detected but ThreadPoolExecutor failed: {e}") from e
        
        logger.info(f"Loading tool definition for: {tool_id}")

        tool_meta = self._tool_metadata_cache.get(tool_id)
        if not tool_meta:
            raise ValueError(
                f"Tool '{tool_id}' not found in cache. Must be discovered via search() first."
            )

        # Prepare and execute load request via SDK
        load_coro = self._sdk_client.load_execution_info(
            workflow_uuids=[tool_id] if tool_meta["type"] == "workflow" else [],
            operation_uuids=[tool_id] if tool_meta["type"] == "operation" else [],
            api_name=tool_meta["api_name"],
        )

        results = self._sync(load_coro)

        # Convert Pydantic → dict for downstream processing.
        if hasattr(results, "model_dump"):
            results = results.model_dump(exclude_none=False)

        return self._format_load_results(tool_id, results)

    def _format_load_results(
        self, tool_id: str, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formats loaded tool definition into a consistent structure."""

        # Check for workflows by iterating through them and matching the UUID
        if "workflows" in results and results["workflows"]:
            for workflow_key, workflow_data in results["workflows"].items():
                if workflow_data.get("workflow_uuid") == tool_id:
                    return {
                        "id": tool_id,
                        "name": workflow_data.get("summary", "Unnamed Workflow"),
                        "description": workflow_data.get("description", ""),
                        "type": "workflow",
                        "parameters": workflow_data.get("inputs", {}).get(
                            "properties", {}
                        ),
                        "executable": True,
                    }

        # Check for operations, assuming they are keyed by ID
        if "operations" in results and results["operations"]:
            if tool_id in results["operations"]:
                operation = results["operations"][tool_id]
                return {
                    "id": tool_id,
                    "name": operation.get("summary", "Unnamed Operation"),
                    "description": f"{operation.get('method')} {operation.get('path')}",
                    "type": "operation",
                    "parameters": operation.get("inputs", {}).get("properties", {}),
                    "required": operation.get("inputs", {}).get("required", []),
                    "executable": True,
                }

        logger.error(
            f"Failed to find tool '{tool_id}' in load results payload. Payload received: {results}"
        )
        raise ValueError(
            f"Could not format load result for tool_id: {tool_id}. "
            "The tool was not found in the payload returned by the Jentic API."
        )

    async def execute_async(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of execute for use in async contexts.
        """
        logger.info(f"Executing tool: {tool_id}")
        
        tool_meta = self._tool_metadata_cache.get(tool_id)
        if not tool_meta:
            raise ValueError(f"Tool '{tool_id}' not found in cache. Must be discovered via search() first.")
        
        try:
            if tool_meta["type"] == "workflow":
                result = await self._sdk_client.execute_workflow(tool_id, params)
            else:
                result = await self._sdk_client.execute_operation(tool_id, params)

            return {"status": "success", "result": result}

        except Exception as exc:
            logger.error("Jentic execution failed for tool '%s': %s", tool_id, exc)
            raise

    def execute(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters. Uses cached metadata to determine execution type.
        """
        # Check if we're in an async context and use async method if so
        if self._is_async_context():
            logger.info(f"Detected async context, using ThreadPoolExecutor for execute: {tool_id}")
            import concurrent.futures
            
            try:
                # Create a new thread to run the async operation
                def run_async():
                    logger.info(f"Running execute_async in new thread for: {tool_id}")
                    # Create a new event loop in this thread
                    import asyncio
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.execute_async(tool_id, params))
                        logger.info(f"ThreadPoolExecutor execute completed successfully for: {tool_id}")
                        return result
                    except Exception as e:
                        logger.error(f"ThreadPoolExecutor execute failed for {tool_id}: {e}")
                        raise
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    result = future.result()
                    logger.info(f"ThreadPoolExecutor returned execute result for {tool_id}")
                    return result
            except Exception as e:
                logger.error(f"ThreadPoolExecutor approach failed for execute {tool_id}: {e}")
                # Don't fall back to sync - re-raise the exception
                raise RuntimeError(f"Async context detected but ThreadPoolExecutor failed: {e}") from e
        
        logger.info(f"Executing tool: {tool_id}")

        tool_meta = self._tool_metadata_cache.get(tool_id)
        if not tool_meta:
            raise ValueError(
                f"Tool '{tool_id}' not found in cache. Must be discovered via search() first."
            )

        try:
            if tool_meta["type"] == "workflow":
                exec_coro = self._sdk_client.execute_workflow(tool_id, params)
            else:
                exec_coro = self._sdk_client.execute_operation(tool_id, params)

            result = self._sync(exec_coro)

            return {"status": "success", "result": result}

        except Exception as exc:
            logger.error("Jentic execution failed for tool '%s': %s", tool_id, exc)
            raise
