from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel

class ToolResult(BaseModel):
    """Result from a tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BaseTool(ABC):
    """Base class for all tools in the RL environment"""
    
    name: str
    description: str
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def validate_params(self, **kwargs) -> bool:
        """Validate the parameters for tool execution"""
        pass
    
    def get_description(self) -> str:
        """Get tool description for the agent"""
        return self.description
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema for the agent"""
        return {} 