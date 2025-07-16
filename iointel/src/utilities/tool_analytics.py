"""
Tool performance tracking and analytics for iointel.

This module provides decorators and utilities for tracking tool performance,
usage patterns, and generating analytics reports.
"""

import time
import asyncio
from functools import wraps
from typing import Callable, Any, Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from .enhanced_registry import enhanced_registry


def track_tool_performance(tool_name: str):
    """
    Decorator to track tool usage and performance metrics.
    
    Args:
        tool_name: Name of the tool to track
    
    Returns:
        Decorated function with performance tracking
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_msg = None
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_msg = str(e)
                raise e
            finally:
                execution_time = time.time() - start_time
                enhanced_registry.update_tool_performance(
                    tool_name, execution_time, success, error_msg
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_msg = None
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_msg = str(e)
                raise e
            finally:
                execution_time = time.time() - start_time
                enhanced_registry.update_tool_performance(
                    tool_name, execution_time, success, error_msg
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class ToolAnalytics:
    """
    Comprehensive tool analytics and reporting system.
    """
    
    def __init__(self, registry=None):
        self.registry = registry or enhanced_registry
        self.analytics_path = Path.home() / ".iointel" / "analytics"
        self.analytics_path.mkdir(parents=True, exist_ok=True)
    
    def generate_usage_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate a comprehensive usage report.
        
        Args:
            days: Number of days to include in the report
        
        Returns:
            Dictionary containing usage statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        total_tools = len(self.registry.tools)
        used_tools = 0
        total_usage = 0
        total_execution_time = 0
        
        tool_stats = []
        
        for tool_name, tool in self.registry.tools.items():
            metrics = tool.performance
            
            # Filter by date (simplified - in practice would need proper date filtering)
            if metrics.last_used and metrics.last_used > cutoff_date:
                used_tools += 1
            
            total_usage += metrics.usage_count
            total_execution_time += metrics.total_execution_time
            
            tool_stats.append({
                "name": tool_name,
                "category": tool.metadata.category.value,
                "usage_count": metrics.usage_count,
                "success_rate": metrics.success_rate,
                "average_response_time": metrics.average_response_time,
                "total_execution_time": metrics.total_execution_time,
                "error_count": metrics.error_count,
                "last_used": metrics.last_used.isoformat() if metrics.last_used else None,
            })
        
        # Sort by usage count
        tool_stats.sort(key=lambda x: x["usage_count"], reverse=True)
        
        return {
            "report_date": datetime.now().isoformat(),
            "period_days": days,
            "summary": {
                "total_tools": total_tools,
                "used_tools": used_tools,
                "usage_rate": used_tools / total_tools if total_tools > 0 else 0,
                "total_usage": total_usage,
                "total_execution_time": total_execution_time,
                "average_execution_time": total_execution_time / total_usage if total_usage > 0 else 0,
            },
            "top_tools": tool_stats[:10],
            "category_breakdown": self._get_category_breakdown(),
            "performance_insights": self._get_performance_insights(),
        }
    
    def _get_category_breakdown(self) -> Dict[str, Any]:
        """Get breakdown of tool usage by category."""
        category_stats = {}
        
        for category, tool_names in self.registry.categories.items():
            if not tool_names:
                continue
            
            tools = [self.registry.tools[name] for name in tool_names if name in self.registry.tools]
            
            total_usage = sum(tool.performance.usage_count for tool in tools)
            total_execution_time = sum(tool.performance.total_execution_time for tool in tools)
            average_success_rate = sum(tool.performance.success_rate for tool in tools) / len(tools)
            
            category_stats[category.value] = {
                "tool_count": len(tools),
                "total_usage": total_usage,
                "total_execution_time": total_execution_time,
                "average_success_rate": average_success_rate,
                "usage_per_tool": total_usage / len(tools) if tools else 0,
            }
        
        return category_stats
    
    def _get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        insights = {
            "slow_tools": [],
            "unreliable_tools": [],
            "underutilized_tools": [],
            "recommendations": [],
        }
        
        for tool_name, tool in self.registry.tools.items():
            metrics = tool.performance
            
            # Identify slow tools (> 5 seconds average)
            if metrics.average_response_time > 5.0:
                insights["slow_tools"].append({
                    "name": tool_name,
                    "response_time": metrics.average_response_time,
                })
            
            # Identify unreliable tools (< 90% success rate)
            if metrics.success_rate < 0.9 and metrics.usage_count > 5:
                insights["unreliable_tools"].append({
                    "name": tool_name,
                    "success_rate": metrics.success_rate,
                    "error_count": metrics.error_count,
                })
            
            # Identify underutilized tools (no usage in last 30 days)
            if metrics.usage_count == 0 or (
                metrics.last_used and 
                metrics.last_used < datetime.now() - timedelta(days=30)
            ):
                insights["underutilized_tools"].append({
                    "name": tool_name,
                    "category": tool.metadata.category.value,
                    "last_used": metrics.last_used.isoformat() if metrics.last_used else None,
                })
        
        # Generate recommendations
        if insights["slow_tools"]:
            insights["recommendations"].append(
                "Consider optimizing slow tools or implementing caching for better performance."
            )
        
        if insights["unreliable_tools"]:
            insights["recommendations"].append(
                "Review and improve error handling for unreliable tools."
            )
        
        if len(insights["underutilized_tools"]) > len(self.registry.tools) * 0.3:
            insights["recommendations"].append(
                "Consider removing or refactoring underutilized tools to reduce registry bloat."
            )
        
        return insights
    
    def export_analytics_report(self, filename: Optional[str] = None) -> Path:
        """
        Export analytics report to JSON file.
        
        Args:
            filename: Optional filename, defaults to timestamped report
        
        Returns:
            Path to the exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tool_analytics_{timestamp}.json"
        
        report_path = self.analytics_path / filename
        report = self.generate_usage_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
    
    def get_tool_recommendations(self, context: str = "general") -> List[Dict[str, Any]]:
        """
        Get tool recommendations based on context and usage patterns.
        
        Args:
            context: Context for recommendations (e.g., "data_analysis", "web_scraping")
        
        Returns:
            List of recommended tools with explanations
        """
        recommendations = []
        
        # Get tools by category based on context
        context_categories = {
            "data_analysis": ["DATA", "ANALYSIS", "UTILITY"],
            "web_scraping": ["WEB", "SEARCH", "UTILITY"],
            "communication": ["COMMUNICATION", "SOCIAL"],
            "research": ["RESEARCH", "SEARCH", "WEB"],
            "productivity": ["PRODUCTIVITY", "UTILITY"],
            "ai_tasks": ["AI", "UTILITY"],
        }
        
        relevant_categories = context_categories.get(context, ["UTILITY"])
        
        for category_name in relevant_categories:
            try:
                from .enhanced_registry import ToolCategory
                category = ToolCategory(category_name.lower())
                tools = self.registry.get_tools_by_category(category)
                
                # Sort by performance and usage
                tools.sort(key=lambda t: (
                    t.performance.success_rate,
                    -t.performance.average_response_time,
                    t.performance.usage_count
                ), reverse=True)
                
                for tool in tools[:3]:  # Top 3 per category
                    recommendations.append({
                        "name": tool.name,
                        "category": category_name,
                        "description": tool.description,
                        "success_rate": tool.performance.success_rate,
                        "average_response_time": tool.performance.average_response_time,
                        "usage_count": tool.performance.usage_count,
                        "reason": f"Top performing tool in {category_name} category",
                    })
                    
            except ValueError:
                continue
        
        return recommendations[:10]  # Return top 10 overall
    
    def track_agent_tool_usage(self, agent_name: str, tool_name: str, success: bool):
        """
        Track tool usage per agent for collaborative analytics.
        
        Args:
            agent_name: Name of the agent using the tool
            tool_name: Name of the tool being used
            success: Whether the tool usage was successful
        """
        usage_log_path = self.analytics_path / "agent_tool_usage.json"
        
        # Load existing log
        if usage_log_path.exists():
            with open(usage_log_path, 'r') as f:
                usage_log = json.load(f)
        else:
            usage_log = {}
        
        # Update log
        if agent_name not in usage_log:
            usage_log[agent_name] = {}
        
        if tool_name not in usage_log[agent_name]:
            usage_log[agent_name][tool_name] = {
                "total_uses": 0,
                "successful_uses": 0,
                "last_used": None,
            }
        
        usage_log[agent_name][tool_name]["total_uses"] += 1
        if success:
            usage_log[agent_name][tool_name]["successful_uses"] += 1
        usage_log[agent_name][tool_name]["last_used"] = datetime.now().isoformat()
        
        # Save log
        with open(usage_log_path, 'w') as f:
            json.dump(usage_log, f, indent=2)


# Global analytics instance
analytics = ToolAnalytics()


def get_analytics() -> ToolAnalytics:
    """Get the global analytics instance."""
    return analytics