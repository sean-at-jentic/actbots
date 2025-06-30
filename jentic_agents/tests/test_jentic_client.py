"""
Unit tests for JenticClient platform wrapper.
"""
import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

from ..platform.jentic_client import JenticClient


class TestJenticClient:
    """Test cases for JenticClient"""
    
    @pytest.fixture
    def mock_jentic_sdk(self):
        """Mock the Jentic SDK for testing"""
        # Create mock objects
        mock_jentic_module = Mock()
        mock_sdk_client = Mock()
        mock_sdk_models = Mock()
        
        # Set up the mock Jentic class to return our mock client
        mock_jentic_module.Jentic = Mock(return_value=mock_sdk_client)
        
        # Mock search results
        mock_search_response = Mock()
        mock_search_response.model_dump.return_value = {
            'workflows': [
                {
                    'workflow_id': 'echo_tool_001',
                    'summary': 'Echo Tool',
                    'description': 'Echoes input message',
                    'method': 'POST',
                    'path': '/echo',
                    'api_name': 'test_api'
                },
                {
                    'workflow_id': 'ping_tool_001',
                    'summary': 'Ping Tool', 
                    'description': 'Simple ping tool',
                    'method': 'GET',
                    'path': '/ping',
                    'api_name': 'test_api'
                },
                {
                    'workflow_id': 'third_tool_001',
                    'summary': 'Third Tool',
                    'description': 'Another test tool',
                    'method': 'POST', 
                    'path': '/third',
                    'api_name': 'test_api'
                }
            ],
            'operations': []
        }
        
        # Mock load results - return different data based on tool_id
        def mock_load_response_func(*args, **kwargs):
            mock_response = Mock()
            # Check which tool is being loaded by examining the workflow_uuids parameter
            workflow_uuids = kwargs.get('workflow_uuids', [])
            if workflow_uuids and 'ping_tool_001' in workflow_uuids:
                # Return only ping tool data
                mock_response.model_dump.return_value = {
                    'workflows': {
                        'ping_tool_001': {
                            'summary': 'Ping Tool',
                            'description': 'Simple ping tool', 
                            'inputs': {
                                'properties': {}
                            }
                        }
                    },
                    'operations': {}
                }
            else:
                # Default to echo tool data
                mock_response.model_dump.return_value = {
                    'workflows': {
                        'echo_tool_001': {
                            'summary': 'Echo Tool',
                            'description': 'Echoes input message',
                            'inputs': {
                                'properties': {
                                    'message': {'type': 'string'}
                                }
                            }
                        }
                    },
                    'operations': {}
                }
            return mock_response
        
        # Configure SDK client methods to return async coroutines
        async def mock_search_async(*args, **kwargs):
            return mock_search_response
        
        async def mock_load_async(*args, **kwargs):
            return mock_load_response_func(*args, **kwargs)
        
        async def mock_execute_workflow_async(*args, **kwargs):
            # Return different results based on the workflow being executed
            # args[0] should be the workflow_id
            if len(args) > 0 and 'ping' in str(args[0]):
                return "pong"
            return "Echo: Hello, World!"
        
        async def mock_execute_operation_async(*args, **kwargs):
            return "pong"
        
        mock_sdk_client.search_api_capabilities = Mock(side_effect=mock_search_async)
        mock_sdk_client.load_execution_info = Mock(side_effect=mock_load_async)
        mock_sdk_client.execute_workflow = Mock(side_effect=mock_execute_workflow_async)
        mock_sdk_client.execute_operation = Mock(side_effect=mock_execute_operation_async)
        
        # Mock the request model
        mock_request_model = Mock()
        mock_sdk_models.ApiCapabilitySearchRequest = mock_request_model
        
        # Patch the imports in the JenticClient module
        with patch.dict('sys.modules', {'jentic': mock_jentic_module}):
            with patch('importlib.import_module', return_value=mock_sdk_models):
                yield mock_sdk_client
    
    def test_init(self):
        """Test client initialization"""
        client = JenticClient(api_key="test_key")
        assert client.api_key == "test_key"
    
    def test_init_no_key(self):
        """Test client initialization without API key"""
        client = JenticClient()
        assert client.api_key is None
    
    def test_search(self, mock_jentic_sdk):
        """Test search functionality returns expected mock data"""
        client = JenticClient()
        results = client.search("test query", top_k=3)
        
        assert isinstance(results, list)
        assert len(results) == 3  # Mock returns 3 tools
        assert all(isinstance(result, dict) for result in results)
        assert all("id" in result and "name" in result for result in results)
    
    def test_load_echo_tool(self, mock_jentic_sdk):
        """Test loading echo tool definition"""
        client = JenticClient()
        # First search to populate cache
        client.search("echo", top_k=5)
        tool = client.load("echo_tool_001")
        
        assert tool["id"] == "echo_tool_001"
        assert tool["name"] == "Echo Tool"
        assert "parameters" in tool
        assert tool["executable"] is True
    
    def test_load_ping_tool(self, mock_jentic_sdk):
        """Test loading ping tool definition"""
        client = JenticClient()
        # First search to populate cache
        client.search("ping", top_k=5)
        tool = client.load("ping_tool_001")
        
        assert tool["id"] == "ping_tool_001"
        assert tool["name"] == "Ping Tool"
        assert tool["parameters"] == {}
        assert tool["executable"] is True
    
    def test_load_unknown_tool(self):
        """Test loading unknown tool raises ValueError"""
        client = JenticClient()
        
        with pytest.raises(ValueError, match=r"Tool 'unknown_tool' not found in cache\. Must be discovered via search\(\) first\."):
            client.load("unknown_tool")
    
    def test_execute_echo_tool(self, mock_jentic_sdk):
        """Test executing echo tool"""
        client = JenticClient()
        # First search to populate cache
        client.search("echo", top_k=5)
        result = client.execute("echo_tool_001", {"message": "Hello, World!"})
        
        assert result["status"] == "success"
        assert result["result"] == "Echo: Hello, World!"
    
    def test_execute_ping_tool(self, mock_jentic_sdk):
        """Test executing ping tool"""
        client = JenticClient()
        # First search to populate cache
        client.search("ping", top_k=5)
        result = client.execute("ping_tool_001", {})
        
        assert result["status"] == "success"
        assert result["result"] == "pong"
    
    def test_execute_unknown_tool(self):
        """Test executing unknown tool raises ValueError"""
        client = JenticClient()
        
        with pytest.raises(ValueError, match=r"Tool 'unknown_tool' not found in cache\. Must be discovered via search\(\) first\."):
            client.execute("unknown_tool", {})
