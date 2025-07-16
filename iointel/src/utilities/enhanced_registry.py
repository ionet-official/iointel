"""
Enhanced tool registry with advanced features for iointel.

This module provides a comprehensive tool registry system with categorization,
search capabilities, performance tracking, and analytics.
"""

import json
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from ..agent_methods.data_models.datamodels import Tool


class ToolCategory(str, Enum):
    """Tool categories for better organization."""
    SEARCH = "search"
    COMMUNICATION = "communication"
    DATA = "data"
    AI = "ai"
    FILE = "file"
    WEB = "web"
    ANALYSIS = "analysis"
    UTILITY = "utility"
    SOCIAL = "social"
    PRODUCTIVITY = "productivity"
    CLOUD = "cloud"
    MEDIA = "media"
    FINANCE = "finance"
    RESEARCH = "research"


@dataclass
class ToolMetadata:
    """Enhanced metadata for tools."""
    category: ToolCategory
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    dependencies: Set[str] = field(default_factory=set)
    cost_tier: str = "free"  # free, low, medium, high
    rate_limit: Optional[int] = None
    requires_auth: bool = False
    documentation_url: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "tags": list(self.tags),
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "dependencies": list(self.dependencies),
            "cost_tier": self.cost_tier,
            "rate_limit": self.rate_limit,
            "requires_auth": self.requires_auth,
            "documentation_url": self.documentation_url,
            "examples": self.examples,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolMetadata":
        """Create from dictionary."""
        return cls(
            category=ToolCategory(data["category"]),
            tags=set(data.get("tags", [])),
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            dependencies=set(data.get("dependencies", [])),
            cost_tier=data.get("cost_tier", "free"),
            rate_limit=data.get("rate_limit"),
            requires_auth=data.get("requires_auth", False),
            documentation_url=data.get("documentation_url"),
            examples=data.get("examples", []),
        )


@dataclass
class ToolPerformanceMetrics:
    """Performance metrics for tools."""
    usage_count: int = 0
    last_used: Optional[datetime] = None
    success_rate: float = 1.0
    average_response_time: float = 0.0
    total_execution_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "total_execution_time": self.total_execution_time,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPerformanceMetrics":
        """Create from dictionary."""
        return cls(
            usage_count=data.get("usage_count", 0),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            success_rate=data.get("success_rate", 1.0),
            average_response_time=data.get("average_response_time", 0.0),
            total_execution_time=data.get("total_execution_time", 0.0),
            error_count=data.get("error_count", 0),
            last_error=data.get("last_error"),
        )


class EnhancedTool(Tool):
    """Enhanced tool with richer metadata and performance tracking."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Any,
        fn: Any,
        body: Optional[str] = None,
        fn_metadata: Optional[Any] = None,
        fn_self: Optional[Any] = None,
        metadata: Optional[ToolMetadata] = None,
        performance: Optional[ToolPerformanceMetrics] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
            body=body,
            fn_metadata=fn_metadata,
            fn_self=fn_self,
        )
        self.metadata = metadata or ToolMetadata(category=ToolCategory.UTILITY)
        self.performance = performance or ToolPerformanceMetrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "body": self.body,
            "metadata": self.metadata.to_dict(),
            "performance": self.performance.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedTool":
        """Create from dictionary (for deserialization)."""
        # Note: This is a simplified version - actual implementation would need
        # to handle function reconstruction properly
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data["parameters"],
            fn=lambda: None,  # Placeholder
            body=data.get("body"),
            metadata=ToolMetadata.from_dict(data.get("metadata", {})),
            performance=ToolPerformanceMetrics.from_dict(data.get("performance", {})),
        )


class EnhancedToolRegistry:
    """
    Advanced tool registry with categorization, search, and analytics.
    """
    
    def __init__(self, persistence_path: Optional[Path] = None):
        self.tools: Dict[str, EnhancedTool] = {}
        self.categories: Dict[ToolCategory, Set[str]] = defaultdict(set)
        self.tags: Dict[str, Set[str]] = defaultdict(set)
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.persistence_path = persistence_path or Path.home() / ".iointel" / "tool_registry.json"
        self._ensure_persistence_dir()
        self.load_from_disk()
    
    def _ensure_persistence_dir(self):
        """Ensure persistence directory exists."""
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
    
    def register_tool(self, tool: EnhancedTool) -> None:
        """Register a tool with enhanced metadata."""
        self.tools[tool.name] = tool
        self.categories[tool.metadata.category].add(tool.name)
        
        for tag in tool.metadata.tags:
            self.tags[tag].add(tool.name)
        
        for dep in tool.metadata.dependencies:
            self.dependencies[dep].add(tool.name)
        
        self.save_to_disk()
    
    def get_tool(self, name: str) -> Optional[EnhancedTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def search_tools(
        self,
        query: str = "",
        category: Optional[ToolCategory] = None,
        tags: Optional[Set[str]] = None,
        has_dependencies: Optional[bool] = None,
        requires_auth: Optional[bool] = None,
        cost_tier: Optional[str] = None,
        min_success_rate: Optional[float] = None,
        max_response_time: Optional[float] = None,
    ) -> List[EnhancedTool]:
        """Advanced tool search with multiple filters."""
        results = []
        
        for tool in self.tools.values():
            # Text search in name and description
            if query and query.lower() not in tool.name.lower() and \
               query.lower() not in (tool.description or "").lower():
                continue
            
            # Category filter
            if category and tool.metadata.category != category:
                continue
            
            # Tags filter
            if tags and not tags.intersection(tool.metadata.tags):
                continue
            
            # Dependencies filter
            if has_dependencies is not None:
                if has_dependencies and not tool.metadata.dependencies:
                    continue
                if not has_dependencies and tool.metadata.dependencies:
                    continue
            
            # Auth filter
            if requires_auth is not None and tool.metadata.requires_auth != requires_auth:
                continue
            
            # Cost tier filter
            if cost_tier and tool.metadata.cost_tier != cost_tier:
                continue
            
            # Performance filters
            if min_success_rate is not None and tool.performance.success_rate < min_success_rate:
                continue
            
            if max_response_time is not None and tool.performance.average_response_time > max_response_time:
                continue
            
            results.append(tool)
        
        return results
    
    def get_tools_by_category(self, category: ToolCategory) -> List[EnhancedTool]:
        """Get all tools in a specific category."""
        return [self.tools[name] for name in self.categories[category] if name in self.tools]
    
    def get_tools_by_tag(self, tag: str) -> List[EnhancedTool]:
        """Get all tools with a specific tag."""
        return [self.tools[name] for name in self.tags[tag] if name in self.tools]
    
    def get_tool_dependencies(self, tool_name: str) -> Set[str]:
        """Get dependencies for a specific tool."""
        if tool_name in self.tools:
            return self.tools[tool_name].metadata.dependencies
        return set()
    
    def get_dependent_tools(self, tool_name: str) -> Set[str]:
        """Get tools that depend on a specific tool."""
        return self.dependencies.get(tool_name, set())
    
    def get_popular_tools(self, limit: int = 10) -> List[EnhancedTool]:
        """Get most popular tools by usage count."""
        return sorted(
            self.tools.values(),
            key=lambda t: t.performance.usage_count,
            reverse=True
        )[:limit]
    
    def get_best_performing_tools(self, limit: int = 10) -> List[EnhancedTool]:
        """Get best performing tools by success rate and response time."""
        return sorted(
            self.tools.values(),
            key=lambda t: (t.performance.success_rate, -t.performance.average_response_time),
            reverse=True
        )[:limit]
    
    def get_tool_analytics(self) -> Dict[str, Any]:
        """Get comprehensive registry analytics."""
        total_tools = len(self.tools)
        if total_tools == 0:
            return {"total_tools": 0}
        
        # Category distribution
        category_stats = {}
        for category, tools in self.categories.items():
            category_stats[category.value] = len(tools)
        
        # Tag distribution
        tag_stats = {}
        for tag, tools in self.tags.items():
            tag_stats[tag] = len(tools)
        
        # Performance stats
        success_rates = [t.performance.success_rate for t in self.tools.values()]
        response_times = [t.performance.average_response_time for t in self.tools.values()]
        usage_counts = [t.performance.usage_count for t in self.tools.values()]
        
        return {
            "total_tools": total_tools,
            "categories": category_stats,
            "most_used_tags": dict(sorted(tag_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            "tools_requiring_auth": sum(1 for t in self.tools.values() if t.metadata.requires_auth),
            "performance": {
                "average_success_rate": sum(success_rates) / len(success_rates),
                "average_response_time": sum(response_times) / len(response_times),
                "total_usage": sum(usage_counts),
                "most_used_tool": max(self.tools.values(), key=lambda t: t.performance.usage_count).name,
                "best_performing_tool": max(self.tools.values(), key=lambda t: t.performance.success_rate).name,
            },
            "dependencies": {
                "tools_with_dependencies": sum(1 for t in self.tools.values() if t.metadata.dependencies),
                "most_depended_on": max(self.dependencies.items(), key=lambda x: len(x[1]))[0] if self.dependencies else None,
            },
        }
    
    def validate_dependencies(self) -> List[str]:
        """Validate that all tool dependencies are satisfied."""
        errors = []
        for tool_name, tool in self.tools.items():
            for dep in tool.metadata.dependencies:
                if dep not in self.tools:
                    errors.append(f"Tool '{tool_name}' depends on missing tool '{dep}'")
        return errors
    
    def update_tool_performance(
        self,
        tool_name: str,
        execution_time: float,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Update tool performance metrics."""
        if tool_name not in self.tools:
            return
        
        tool = self.tools[tool_name]
        metrics = tool.performance
        
        # Update usage count
        metrics.usage_count += 1
        metrics.last_used = datetime.now()
        
        # Update success rate
        total_calls = metrics.usage_count
        old_success_rate = metrics.success_rate
        new_success_rate = (old_success_rate * (total_calls - 1) + (1 if success else 0)) / total_calls
        metrics.success_rate = new_success_rate
        
        # Update response time
        metrics.total_execution_time += execution_time
        metrics.average_response_time = metrics.total_execution_time / total_calls
        
        # Update error info
        if not success:
            metrics.error_count += 1
            metrics.last_error = error
        
        # Update metadata timestamp
        tool.metadata.updated_at = datetime.now()
        
        self.save_to_disk()
    
    def save_to_disk(self) -> None:
        """Save registry to disk for persistence."""
        data = {
            "tools": {name: tool.to_dict() for name, tool in self.tools.items()},
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0",
                "total_tools": len(self.tools),
            }
        }
        
        try:
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save registry to disk: {e}")
    
    def load_from_disk(self) -> None:
        """Load registry from disk."""
        if not self.persistence_path.exists():
            return
        
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
            
            for name, tool_data in data.get("tools", {}).items():
                # For now, we'll skip loading tools from disk as it requires
                # function reconstruction. In a real implementation, this would
                # need proper serialization/deserialization of functions.
                pass
                
        except Exception as e:
            print(f"Warning: Could not load registry from disk: {e}")
    
    def migrate_from_legacy_registry(self) -> None:
        """Migrate tools from legacy TOOLS_REGISTRY."""
        try:
            from .registries import TOOLS_REGISTRY
            
            for name, tool in TOOLS_REGISTRY.items():
                # Infer category and metadata
                category = self._infer_tool_category(name, tool.description)
                tags = self._infer_tool_tags(name, tool.description)
                
                # Create enhanced tool
                enhanced_tool = EnhancedTool(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                    fn=tool.fn,
                    body=tool.body,
                    fn_metadata=tool.fn_metadata,
                    fn_self=tool.fn_self,
                    metadata=ToolMetadata(
                        category=category,
                        tags=tags,
                        version="1.0.0",
                        created_at=datetime.now(),
                    ),
                )
                
                self.register_tool(enhanced_tool)
                
        except ImportError:
            print("Warning: Could not import legacy TOOLS_REGISTRY")
    
    def _infer_tool_category(self, name: str, description: str) -> ToolCategory:
        """Infer tool category from name and description."""
        name_lower = name.lower()
        desc_lower = (description or "").lower()
        
        # Category inference rules
        if any(term in name_lower for term in ["search", "google", "duck", "tavily", "exa"]):
            return ToolCategory.SEARCH
        elif any(term in name_lower for term in ["email", "slack", "discord", "telegram", "whatsapp"]):
            return ToolCategory.COMMUNICATION
        elif any(term in name_lower for term in ["csv", "pandas", "sql", "data", "sheets"]):
            return ToolCategory.DATA
        elif any(term in name_lower for term in ["openai", "gpt", "claude", "ai", "cartesia", "elevenlabs"]):
            return ToolCategory.AI
        elif any(term in name_lower for term in ["file", "filesystem", "read", "write"]):
            return ToolCategory.FILE
        elif any(term in name_lower for term in ["web", "crawl", "browser", "scrape", "spider"]):
            return ToolCategory.WEB
        elif any(term in name_lower for term in ["reddit", "twitter", "social", "hackernews"]):
            return ToolCategory.SOCIAL
        elif any(term in name_lower for term in ["jira", "linear", "trello", "todoist", "productivity"]):
            return ToolCategory.PRODUCTIVITY
        elif any(term in name_lower for term in ["aws", "google", "cloud", "lambda", "bigquery"]):
            return ToolCategory.CLOUD
        elif any(term in name_lower for term in ["youtube", "video", "media", "giphy"]):
            return ToolCategory.MEDIA
        elif any(term in name_lower for term in ["finance", "stock", "yfinance"]):
            return ToolCategory.FINANCE
        elif any(term in name_lower for term in ["arxiv", "pubmed", "research", "wikipedia"]):
            return ToolCategory.RESEARCH
        else:
            return ToolCategory.UTILITY
    
    def _infer_tool_tags(self, name: str, description: str) -> Set[str]:
        """Infer tool tags from name and description."""
        tags = set()
        name_lower = name.lower()
        desc_lower = (description or "").lower()
        
        # Common tags
        tag_keywords = {
            "api": ["api", "rest", "endpoint"],
            "async": ["async", "asynchronous"],
            "auth": ["auth", "authentication", "login"],
            "cloud": ["cloud", "aws", "azure", "gcp"],
            "database": ["database", "db", "sql", "nosql"],
            "file": ["file", "filesystem", "storage"],
            "network": ["network", "http", "https", "request"],
            "realtime": ["realtime", "live", "stream"],
            "security": ["security", "encrypt", "secure"],
            "text": ["text", "string", "parse"],
            "web": ["web", "browser", "html", "css"],
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in name_lower or keyword in desc_lower for keyword in keywords):
                tags.add(tag)
        
        return tags


# Global enhanced registry instance
enhanced_registry = EnhancedToolRegistry()


def get_enhanced_registry() -> EnhancedToolRegistry:
    """Get the global enhanced registry instance."""
    return enhanced_registry