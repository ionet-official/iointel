"""
Smart Tool Discovery System

This module provides lazy loading of tools with credential checking.
It builds on existing patterns in the codebase for environment variables and API keys.
"""

import os
import importlib
from typing import Dict, List, Optional, Type, Any
from pathlib import Path
from dotenv import load_dotenv
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.utilities.helpers import make_logger
from iointel.src.utilities.constants import _get_env_var

logger = make_logger(__name__)


def get_tool_env_var(tool_name: str, suffix: str, default=None):
    """
    Get environment variable for a tool using existing patterns.
    
    Tries multiple prefixes: TOOL_NAME_SUFFIX, TOOL_SUFFIX, SUFFIX
    
    Args:
        tool_name: Name of the tool (e.g., 'openweather')
        suffix: Suffix to look for (e.g., 'API_KEY')
        default: Default value if not found
    
    Returns:
        Environment variable value or default
    """
    # Try tool-specific first: OPENWEATHER_API_KEY
    tool_specific = f"{tool_name.upper()}_{suffix}"
    if value := os.getenv(tool_specific):
        return value
    
    # Try generic patterns: OPENAI_API_KEY, IO_API_KEY
    return _get_env_var(suffix, default)


class ToolDiscovery:
    """
    Discovers and lazily loads agno tools with credential checking.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize tool discovery.
        
        Args:
            env_file: Path to environment file for credential checking
        """
        self.env_file = env_file
        self._discovered_tools: Dict[str, Type] = {}
        self._loaded_instances: Dict[str, Any] = {}
        
        if env_file:
            load_dotenv(env_file)
    
    def discover_agno_tools(self) -> Dict[str, Type]:
        """
        Discover all available agno tools by scanning the agno directory.
        
        Returns:
            Dict mapping tool names to their classes
        """
        if self._discovered_tools:
            return self._discovered_tools
        
        agno_dir = Path(__file__).parent / "agno"
        
        for py_file in agno_dir.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name in ["common.py", "__init__.py"]:
                continue
                
            module_name = py_file.stem
            
            try:
                # Import the module
                module = importlib.import_module(f"iointel.src.agent_methods.tools.agno.{module_name}")
                
                # Find classes that inherit from the agno base
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name.lower() == module_name.lower() and
                        hasattr(attr, '_get_tool')):
                        
                        self._discovered_tools[module_name] = attr
                        logger.debug(f"Discovered agno tool: {module_name} -> {attr}")
                        
            except Exception as e:
                logger.warning(f"Failed to discover tool {module_name}: {e}")
        
        return self._discovered_tools
    
    def get_tool_credentials(self, tool_name: str) -> Dict[str, Any]:
        """
        Get credentials for a specific tool using existing patterns.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dict of credential parameters for the tool
        """
        creds = {}
        
        # Define credential patterns for each tool
        credential_patterns = {
            "openweather": {"api_key": "API_KEY"},
            "serpapi": {"api_key": "API_KEY"},
            "tavily": {"api_key": "API_KEY"},
            "twillio": {"api_key": "API_KEY"},
            "gmail": {"credentials_path": "CREDENTIALS_PATH"},
            "github": {"token": "TOKEN"},
            "slack": {"bot_token": "BOT_TOKEN"},
            "discord": {"bot_token": "BOT_TOKEN"},
            "openai": {"api_key": "API_KEY"},
            "confluence": {"api_key": "API_KEY", "base_url": "BASE_URL"},
            # Add more patterns as needed
        }
        
        patterns = credential_patterns.get(tool_name, {})
        
        for param_name, env_suffix in patterns.items():
            value = get_tool_env_var(tool_name, env_suffix)
            if value:
                creds[param_name] = value
        
        return creds
    
    def can_load_tool(self, tool_name: str) -> bool:
        """
        Check if a tool can be loaded (has required credentials or doesn't need them).
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            bool: True if tool can be loaded
        """
        discovered = self.discover_agno_tools()
        
        if tool_name not in discovered:
            return False
        
        tool_class = discovered[tool_name]
        
        # Check if tool has credential requirements by inspecting its fields
        import inspect
        try:
            sig = inspect.signature(tool_class.__init__)
            params = sig.parameters
            
            # Tools with no special credential params can always be loaded
            simple_tools = ["shell", "file", "csv", "arxiv", "calculator"]
            if tool_name in simple_tools:
                return True
            
            # Check if tool has api_key or similar credential fields
            if 'api_key' in params or hasattr(tool_class, 'api_key'):
                creds = self.get_tool_credentials(tool_name)
                has_api_key = 'api_key' in creds
                logger.debug(f"Tool '{tool_name}' credential check: has_api_key={has_api_key}")
                return has_api_key
            
            # If no special credential requirements, assume it can be loaded
            return True
            
        except Exception as e:
            logger.warning(f"Error checking tool '{tool_name}' credentials: {e}")
            return False
    
    def load_tool(self, tool_name: str) -> Any:
        """
        Lazily load and instantiate a tool.
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            Tool instance
            
        Raises:
            ValueError: If tool not found or credentials missing
        """
        if tool_name in self._loaded_instances:
            return self._loaded_instances[tool_name]
        
        discovered = self.discover_agno_tools()
        
        if tool_name not in discovered:
            available = list(discovered.keys())
            raise ValueError(f"Tool '{tool_name}' not found. Available: {available}")
        
        if not self.can_load_tool(tool_name):
            raise ValueError(f"Tool '{tool_name}' missing required credentials")
        
        # Get credentials for the tool
        creds = self.get_tool_credentials(tool_name)
        
        # Instantiate the tool
        tool_class = discovered[tool_name]
        try:
            instance = tool_class(**creds)
            self._loaded_instances[tool_name] = instance
            logger.info(f"Loaded tool: {tool_name} with creds: {list(creds.keys())}")
            return instance
        except Exception as e:
            raise ValueError(f"Failed to instantiate tool '{tool_name}': {e}")
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tools that can be loaded.
        
        Returns:
            List of tool names that can be instantiated
        """
        discovered = self.discover_agno_tools()
        available = []
        
        for tool_name in discovered.keys():
            if self.can_load_tool(tool_name):
                available.append(tool_name)
            else:
                logger.debug(f"Tool '{tool_name}' skipped due to missing credentials")
        
        return available
    
    def load_available_tools(self) -> List[str]:
        """
        Load all available tools and return list of tool names in registry.
        
        Returns:
            List of tool names that were loaded into the registry
        """
        available = self.get_available_tools()
        
        for tool_name in available:
            try:
                self.load_tool(tool_name)
            except Exception as e:
                logger.warning(f"Failed to load tool '{tool_name}': {e}")
        
        return list(TOOLS_REGISTRY.keys())


# Global instance for easy access
_discovery = ToolDiscovery()


def load_tools_from_env(env_file: Optional[str] = None) -> List[str]:
    """
    Load all available tools from environment and return list of tool names.
    This is the main function that replaces the old load_tools_from_env.
    
    Args:
        env_file: Path to environment file for credential checking
        
    Returns:
        List of tool names that were loaded into the registry
    """
    if env_file:
        discovery = ToolDiscovery(env_file)
    else:
        discovery = _discovery
    
    return discovery.load_available_tools()


def get_available_tools(env_file: Optional[str] = None) -> List[str]:
    """
    Get list of available tools without loading them.
    
    Args:
        env_file: Path to environment file for credential checking
        
    Returns:
        List of available tool names
    """
    if env_file:
        discovery = ToolDiscovery(env_file)
    else:
        discovery = _discovery
    
    return discovery.get_available_tools()


def load_specific_tools(tool_names: List[str], env_file: Optional[str] = None) -> List[str]:
    """
    Load specific tools by name.
    
    Args:
        tool_names: List of tool names to load
        env_file: Path to environment file for credential checking
        
    Returns:
        List of tool names that were successfully loaded
    """
    if env_file:
        discovery = ToolDiscovery(env_file)
    else:
        discovery = _discovery
    
    loaded = []
    for tool_name in tool_names:
        try:
            discovery.load_tool(tool_name)
            loaded.append(tool_name)
        except Exception as e:
            logger.warning(f"Failed to load tool '{tool_name}': {e}")
    
    return loaded