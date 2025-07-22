"""
Smart Tool Loader for IOIntel

This module provides intelligent tool loading that:
1. Checks for required API keys in environment variables
2. Only initializes tools with valid credentials
3. Logs missing/invalid configurations
4. Returns only properly registered tools
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from ...utilities.registries import TOOLS_REGISTRY

# Configure logging
logger = logging.getLogger(__name__)

# Map tools to their required environment variables
TOOL_ENV_REQUIREMENTS = {
    # Core tools with API requirements
    "coinmarketcap": ["COINMARKETCAP_API_KEY"],
    "firecrawl": ["FIRECRAWL_API_KEY"],
    "searxng": ["SEARXNG_URL"],  # URL for self-hosted instance
    "wolfram": ["WOLFRAM_API_KEY"],
    "retrieval_engine": ["RETRIEVAL_ENGINE_URL"],  # Base URL for retrieval engine
    
    # Agno tools with API requirements
    "agentql": ["AGENTQL_API_KEY"],
    "airflow": ["AIRFLOW_API_URL", "AIRFLOW_API_KEY"],
    "arxiv": [],  # No API key needed
    "aws_lambda": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    "calculator": [],  # No API key needed
    "cartesia": ["CARTESIA_API_KEY"],
    "conditional_gate": [],  # No API key needed
    "user_input": [],  # No API key needed
    "clickup": ["CLICKUP_API_KEY"],
    "confluence": ["CONFLUENCE_API_KEY", "CONFLUENCE_BASE_URL"],
    "crawl4ai": [],  # No API key needed for basic usage
    "csv": [],  # No API key needed
    "dalle": ["OPENAI_API_KEY"],
    "file": [],  # No API key needed
    "google_calendar": ["GOOGLE_CALENDAR_CREDENTIALS_PATH"],
    "shell": [],  # No API key needed
    "yfinance": [],  # No API key needed
}

# Map tool names to their initialization functions
TOOL_INITIALIZERS = {
    "coinmarketcap": lambda: _init_coinmarketcap(),
    "context_tree": lambda: _init_context_tree(),
    "duckduckgo": lambda: _init_duckduckgo(),
    "firecrawl": lambda: _init_firecrawl(),
    "retrieval_engine": lambda: _init_retrieval_engine(),
    "searxng": lambda: _init_searxng(),
    "solscan": lambda: _init_solscan(),
    "wolfram": lambda: _init_wolfram(),
    "agentql": lambda: _init_agentql(),
    "calculator": lambda: _init_calculator(),
    "conditional_gate": lambda: _init_conditional_gate(),
    "user_input": lambda: _init_user_input(),
    "yfinance": lambda: _init_yfinance(),
    "file": lambda: _init_file_toolkit(),
    "shell": lambda: _init_shell_tool(),
    "csv": lambda: _init_csv_toolkit(),
    "arxiv": lambda: _init_arxiv_toolkit(),
    "crawl4ai": lambda: _init_crawl4ai_toolkit(),
}


def check_env_requirements(tool_name: str, requirements: List[str]) -> bool:
    """Check if all required environment variables are present for a tool."""
    missing = []
    for req in requirements:
        if not os.getenv(req):
            missing.append(req)
    
    if missing:
        logger.warning(
            f"Tool '{tool_name}' missing required environment variables: {', '.join(missing)}. "
            f"This tool will not be available."
        )
        return False
    return True


def _init_coinmarketcap():
    """Initialize CoinMarketCap tools."""
    # The functions are already registered via @register_tool


def _init_context_tree():
    """Initialize Context Tree agent."""
    # The tree instance already has @register_tool methods


def _init_duckduckgo():
    """Initialize DuckDuckGo search."""
    # Functions are registered via @register_tool


def _init_firecrawl():
    """Initialize Firecrawl crawler."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    
    # Create instance - auto-registers its @register_tool methods


def _init_retrieval_engine():
    """Initialize Retrieval Engine."""
    base_url = os.getenv("RETRIEVAL_ENGINE_URL")
    api_key = os.getenv("RETRIEVAL_ENGINE_API_KEY")
    
    # RetrievalEngine has tools already registered via @register_tool on its methods
    # We just need to instantiate it to make those tools available


def _init_searxng():
    """Initialize SearxNG client."""
    try:
        from .searxng import SearxngClient
        base_url = os.getenv("SEARXNG_URL", "http://localhost:8888")
        
        # Create instance and use its registered tools
        client = SearxngClient(base_url=base_url)
        
        # Return the actual registered tool names
        return ["searxng.search", "searxng.get_urls"]
    except Exception as e:
        logger.warning(f"SearxNG not available: {e}")
        return []


def _init_solscan():
    """Initialize Solscan tools."""
    # Solscan module has no @register_tool decorated functions yet
    return []


def _init_wolfram():
    """Initialize Wolfram Alpha."""
    from .wolfram import Wolfram
    api_key = os.getenv("WOLFRAM_API_KEY")
    wolfram = Wolfram(api_key=api_key)
    # Auto-registers its @register_tool methods


def _init_agentql():
    """Initialize AgentQL."""
    try:
        from .agno.agentql import AgentQL
        api_key = os.getenv("AGENTQL_API_KEY")
        
        agentql = AgentQL(api_key=api_key)
        # AgentQL should register its own tools
        return []  # Return empty, tools are auto-registered
    except ImportError as e:
        logger.warning(f"AgentQL not available: {e}")
        return []


def _init_calculator():
    """Initialize Calculator."""
    try:
        from .agno.calculator import Calculator
        
        calc = Calculator()
        # Return the actual registered tool names
        return ["calculator_add", "calculator_subtract", "calculator_multiply", "calculator_divide", 
                "calculator_exponentiate", "calculator_square_root", "calculator_factorial", 
                "calculator_is_prime"]
    except ImportError as e:
        logger.warning(f"Calculator not available: {e}")
        return []


def _init_conditional_gate():
    """Initialize Conditional Gate tools."""
    try:
        # Import the module to register the tools
        from . import conditional_gate
        # Tools are auto-registered via @register_tool decorators
        return ["conditional_gate", "threshold_gate"]
    except ImportError as e:
        logger.warning(f"Conditional Gate not available: {e}")
        return []


def _init_user_input():
    """Initialize User Input tools."""
    try:
        # Import the module to register the tools
        from . import user_input
        # Tools are auto-registered via @register_tool decorators
        return ["user_input", "prompt_tool"]
    except ImportError as e:
        logger.warning(f"User Input not available: {e}")
        return []


def _init_yfinance():
    """Initialize YFinance tools."""
    try:
        from .agno.yfinance import YFinance
        
        yf = YFinance()
        # YFinance should register its own tools
        return []  # Return empty, tools are auto-registered
    except ImportError as e:
        logger.warning(f"YFinance not available: {e}")
        return []


def _init_file_toolkit():
    """Initialize File toolkit."""
    try:
        from .agno.file import File
        
        toolkit = File()
        # File should register its own tools
        return []  # Return empty, tools are auto-registered
    except ImportError as e:
        logger.warning(f"File toolkit not available: {e}")
        return []


def _init_shell_tool():
    """Initialize Shell tool."""
    try:
        from .agno.shell import Shell
        
        shell = Shell()
        # Shell should register its own tools
        return []  # Return empty, tools are auto-registered
    except ImportError as e:
        logger.warning(f"Shell tool not available: {e}")
        return []


def _init_csv_toolkit():
    """Initialize CSV toolkit."""
    try:
        from .agno.csv import Csv
        
        toolkit = Csv()
        # Csv should register its own tools
        return []  # Return empty, tools are auto-registered
    except ImportError as e:
        logger.warning(f"CSV toolkit not available: {e}")
        return []


def _init_arxiv_toolkit():
    """Initialize Arxiv toolkit."""
    try:
        from .agno.arxiv import Arxiv
        
        toolkit = Arxiv()
        # Arxiv should register its own tools
        return []  # Return empty, tools are auto-registered
    except ImportError as e:
        logger.warning(f"Arxiv toolkit not available: {e}")
        return []


def _init_crawl4ai_toolkit():
    """Initialize Crawl4AI toolkit."""
    try:
        from .agno.crawl4ai import Crawl4ai
        
        toolkit = Crawl4ai()
        # Crawl4ai should register its own tools
        return []  # Return empty, tools are auto-registered
    except ImportError as e:
        logger.warning(f"Crawl4AI toolkit not available: {e}")
        return []


def load_tools_from_env(env_file: str = "creds.env") -> List[str]:
    """
    Load tools based on available credentials in the environment.
    
    Args:
        env_file: Path to the environment file containing credentials
        
    Returns:
        List of tool names that can be passed to an Agent
    """
    # Load environment variables
    if env_file and os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"Loaded environment from {env_file}")
    else:
        logger.warning(f"Environment file {env_file} not found, using existing environment")
    
    # Check each tool's requirements and initialize if satisfied
    for tool_name, initializer in TOOL_INITIALIZERS.items():
        requirements = TOOL_ENV_REQUIREMENTS.get(tool_name, [])
        
        if check_env_requirements(tool_name, requirements):
            try:
                initializer()  # Just instantiate, tools auto-register themselves
                logger.info(f"Successfully initialized {tool_name}")
            except Exception as e:
                logger.error(f"Failed to initialize {tool_name}: {str(e)}")
        else:
            logger.info(f"Skipping {tool_name} due to missing requirements")
    
    # Instance Factory: Create instances and register bound methods
    from ...utilities.registries import TOOLS_REGISTRY
    from ..data_models.datamodels import Tool
    import inspect
    
    logger.info("🏭 Starting instance factory for bound method tools...")
    
    # Create instances for classes that need them
    tool_instances = {}
    
    # Context Tree instance
    try:
        from ..tools.context_tree import ContextTree
        context_tree_instance = ContextTree()
        tool_instances['context_tree'] = context_tree_instance
        logger.info("✅ Created ContextTree instance")
    except Exception as e:
        logger.warning(f"Failed to create ContextTree instance: {e}")
    
    # Firecrawl instance
    try:
        from ..tools.firecrawl import Crawler
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        if firecrawl_api_key:
            crawler_instance = Crawler(api_key=firecrawl_api_key)
            tool_instances['crawler'] = crawler_instance
            logger.info("✅ Created Crawler instance")
    except Exception as e:
        logger.warning(f"Failed to create Crawler instance: {e}")
    
    # Wolfram instance
    try:
        from ..tools.wolfram import Wolfram
        wolfram_api_key = os.getenv("WOLFRAM_API_KEY")
        if wolfram_api_key:
            wolfram_instance = Wolfram(api_key=wolfram_api_key)
            tool_instances['wolfram'] = wolfram_instance
            logger.info("✅ Created Wolfram instance")
    except Exception as e:
        logger.warning(f"Failed to create Wolfram instance: {e}")
    
    # SearxNG instance
    try:
        from ..tools.searxng import SearxngClient
        searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8888")
        searxng_instance = SearxngClient(base_url=searxng_url)
        tool_instances['searxng'] = searxng_instance
        logger.info("✅ Created SearxngClient instance")
    except Exception as e:
        logger.warning(f"Failed to create SearxngClient instance: {e}")
    
    # Replace unbound method registrations with bound method registrations
    unbound_tools_to_replace = []
    
    for tool_name, tool in TOOLS_REGISTRY.items():
        try:
            if hasattr(tool, 'fn') and tool.fn:
                sig = inspect.signature(tool.fn)
                params = list(sig.parameters.keys())
                
                # Check if this is an unbound method (has 'self' parameter)
                if params and params[0] == 'self':
                    unbound_tools_to_replace.append((tool_name, tool))
                    logger.debug(f"🔍 Found unbound method tool: {tool_name}")
        except Exception as e:
            logger.debug(f"Error inspecting tool {tool_name}: {e}")
    
    logger.info(f"🔧 Found {len(unbound_tools_to_replace)} unbound method tools to fix")
    
    # Map unbound tools to their bound instances
    tool_class_mapping = {
        # ContextTree methods
        'read_context_tree': ('context_tree', 'read'),
        'create_context_tree': ('context_tree', 'create'),
        'append_context_tree': ('context_tree', 'append'),
        'update_context_tree': ('context_tree', 'update'),
        'delete_context_tree': ('context_tree', 'delete'),
        'summary_context_tree': ('context_tree', 'summary'),
        'load_context_tree': ('context_tree', 'load_tree'),
        'save_context_tree': ('context_tree', 'save_tree'),
        
        # Crawler methods (Firecrawl)
        'Crawler-scrape_url': ('crawler', 'scrape_url'),
        'Crawler-async_scrape_url': ('crawler', 'async_scrape_url'),
        'Crawler-crawl_url': ('crawler', 'crawl_url'),
        
        # Wolfram methods
        'Wolfram-query': ('wolfram', 'query'),
        
        # SearxNG methods
        'searxng.search': ('searxng', 'search'),
        'searxng.get_urls': ('searxng', 'get_urls'),
    }
    
    # Replace unbound registrations with bound ones
    replaced_count = 0
    for tool_name, unbound_tool in unbound_tools_to_replace:
        if tool_name in tool_class_mapping:
            instance_key, method_name = tool_class_mapping[tool_name]
            
            if instance_key in tool_instances:
                instance = tool_instances[instance_key]
                
                # Get the bound method
                if hasattr(instance, method_name):
                    bound_method = getattr(instance, method_name)
                    
                    # Create new Tool from bound method
                    try:
                        bound_tool = Tool.from_function(bound_method, name=tool_name)
                        
                        # Replace in registry
                        TOOLS_REGISTRY[tool_name] = bound_tool
                        logger.info(f"✅ Replaced {tool_name} with bound method from {instance_key}.{method_name}")
                        replaced_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to create bound tool for {tool_name}: {e}")
                else:
                    logger.warning(f"Method {method_name} not found on {instance_key} instance")
            else:
                logger.debug(f"Instance {instance_key} not available for {tool_name}")
        else:
            logger.debug(f"No mapping found for unbound tool: {tool_name}")
    
    logger.info(f"🎉 Instance factory complete: replaced {replaced_count} unbound tools with bound methods")
    
    # Now get all available tools (should include both function-based and bound method tools)
    available_tools = []
    
    for tool_name, tool in TOOLS_REGISTRY.items():
        try:
            if hasattr(tool, 'fn') and tool.fn:
                available_tools.append(tool_name)
        except Exception:
            available_tools.append(tool_name)
    
    logger.info(f"Total available tools: {len(available_tools)}")
    logger.debug(f"Available tools: {available_tools}")
    
    return available_tools


def get_registered_tools() -> Dict[str, Any]:
    """Get all currently registered tools from the global registry."""
    return TOOLS_REGISTRY.copy()


def get_tool_by_name(name: str) -> Optional[Any]:
    """Get a specific tool by name from the registry."""
    return TOOLS_REGISTRY.get(name)