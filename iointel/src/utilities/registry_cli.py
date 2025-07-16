#!/usr/bin/env python3
"""
Command-line interface for managing the enhanced tool registry.
"""

import argparse
import sys
import json
from typing import Dict, Any, List
from .enhanced_registry import enhanced_registry, ToolCategory
from .tool_analytics import analytics


def format_tool_list(tools: List[Any]) -> str:
    """Format tool list for display."""
    if not tools:
        return "No tools found."
    
    output = []
    for tool in tools:
        category = tool.metadata.category.value
        usage = tool.performance.usage_count
        success_rate = tool.performance.success_rate * 100
        
        output.append(
            f"  {tool.name:<25} {category:<15} "
            f"Usage: {usage:<5} Success: {success_rate:.1f}%"
        )
    
    return "\n".join(output)


def format_tool_info(tool: Any) -> str:
    """Format detailed tool information."""
    if not tool:
        return "Tool not found."
    
    info = []
    info.append(f"Name: {tool.name}")
    info.append(f"Description: {tool.description or 'No description'}")
    info.append(f"Category: {tool.metadata.category.value}")
    info.append(f"Tags: {', '.join(tool.metadata.tags) if tool.metadata.tags else 'None'}")
    info.append(f"Version: {tool.metadata.version}")
    info.append(f"Author: {tool.metadata.author or 'Unknown'}")
    info.append(f"Created: {tool.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    info.append(f"Updated: {tool.metadata.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    info.append(f"Cost Tier: {tool.metadata.cost_tier}")
    info.append(f"Requires Auth: {tool.metadata.requires_auth}")
    info.append(f"Dependencies: {', '.join(tool.metadata.dependencies) if tool.metadata.dependencies else 'None'}")
    
    # Performance metrics
    info.append("\nPerformance Metrics:")
    info.append(f"  Usage Count: {tool.performance.usage_count}")
    info.append(f"  Success Rate: {tool.performance.success_rate:.2%}")
    info.append(f"  Average Response Time: {tool.performance.average_response_time:.2f}s")
    info.append(f"  Total Execution Time: {tool.performance.total_execution_time:.2f}s")
    info.append(f"  Error Count: {tool.performance.error_count}")
    
    if tool.performance.last_used:
        info.append(f"  Last Used: {tool.performance.last_used.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        info.append("  Last Used: Never")
    
    if tool.performance.last_error:
        info.append(f"  Last Error: {tool.performance.last_error}")
    
    return "\n".join(info)


def format_analytics_summary(analytics_data: Dict[str, Any]) -> str:
    """Format analytics summary for display."""
    summary = analytics_data.get("summary", {})
    
    output = []
    output.append("Registry Analytics Summary")
    output.append("=" * 30)
    output.append(f"Total Tools: {summary.get('total_tools', 0)}")
    output.append(f"Used Tools: {summary.get('used_tools', 0)}")
    output.append(f"Usage Rate: {summary.get('usage_rate', 0):.1%}")
    output.append(f"Total Usage: {summary.get('total_usage', 0)}")
    output.append(f"Total Execution Time: {summary.get('total_execution_time', 0):.2f}s")
    output.append(f"Average Execution Time: {summary.get('average_execution_time', 0):.2f}s")
    
    # Category breakdown
    category_breakdown = analytics_data.get("category_breakdown", {})
    if category_breakdown:
        output.append("\nCategory Breakdown:")
        for category, stats in category_breakdown.items():
            output.append(f"  {category}: {stats['tool_count']} tools, {stats['total_usage']} uses")
    
    return "\n".join(output)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Manage the enhanced tool registry",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List tools")
    list_parser.add_argument("--category", help="Filter by category")
    list_parser.add_argument("--tag", help="Filter by tag")
    list_parser.add_argument("--requires-auth", action="store_true", help="Filter tools requiring auth")
    list_parser.add_argument("--sort", choices=["name", "usage", "success_rate", "response_time"], 
                           default="name", help="Sort criteria")
    list_parser.add_argument("--limit", type=int, default=50, help="Limit results")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get tool information")
    info_parser.add_argument("tool_name", help="Tool name")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search tools")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--category", help="Filter by category")
    search_parser.add_argument("--min-success-rate", type=float, help="Minimum success rate")
    search_parser.add_argument("--max-response-time", type=float, help="Maximum response time")
    
    # Analytics command
    analytics_parser = subparsers.add_parser("analytics", help="Show analytics")
    analytics_parser.add_argument("--days", type=int, default=30, help="Days to include")
    analytics_parser.add_argument("--export", help="Export to file")
    analytics_parser.add_argument("--detailed", action="store_true", help="Show detailed analytics")
    
    # Categories command
    categories_parser = subparsers.add_parser("categories", help="List categories")
    categories_parser.add_argument("--stats", action="store_true", help="Show category statistics")
    
    # Dependencies command
    deps_parser = subparsers.add_parser("dependencies", help="Show dependencies")
    deps_parser.add_argument("--validate", action="store_true", help="Validate dependencies")
    deps_parser.add_argument("--tool", help="Show dependencies for specific tool")
    
    # Performance command
    perf_parser = subparsers.add_parser("performance", help="Show performance metrics")
    perf_parser.add_argument("--top", type=int, default=10, help="Number of top tools to show")
    perf_parser.add_argument("--sort", choices=["usage", "success_rate", "response_time"], 
                           default="usage", help="Sort criteria")
    
    # Recommendations command
    rec_parser = subparsers.add_parser("recommendations", help="Get tool recommendations")
    rec_parser.add_argument("--context", default="general", 
                          help="Context for recommendations (e.g., data_analysis, web_scraping)")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate from legacy registry")
    migrate_parser.add_argument("--confirm", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "list":
            # Apply filters
            category = ToolCategory(args.category) if args.category else None
            
            tools = enhanced_registry.search_tools(
                category=category,
                tags={args.tag} if args.tag else None,
                requires_auth=args.requires_auth if args.requires_auth else None,
            )
            
            # Sort tools
            if args.sort == "usage":
                tools.sort(key=lambda t: t.performance.usage_count, reverse=True)
            elif args.sort == "success_rate":
                tools.sort(key=lambda t: t.performance.success_rate, reverse=True)
            elif args.sort == "response_time":
                tools.sort(key=lambda t: t.performance.average_response_time)
            else:  # name
                tools.sort(key=lambda t: t.name)
            
            # Limit results
            tools = tools[:args.limit]
            
            print(f"Found {len(tools)} tools:")
            print(format_tool_list(tools))
        
        elif args.command == "info":
            tool = enhanced_registry.get_tool(args.tool_name)
            print(format_tool_info(tool))
        
        elif args.command == "search":
            category = ToolCategory(args.category) if args.category else None
            
            tools = enhanced_registry.search_tools(
                query=args.query,
                category=category,
                min_success_rate=args.min_success_rate,
                max_response_time=args.max_response_time,
            )
            
            print(f"Search results for '{args.query}':")
            print(format_tool_list(tools))
        
        elif args.command == "analytics":
            analytics_data = analytics.generate_usage_report(args.days)
            
            if args.export:
                report_path = analytics.export_analytics_report(args.export)
                print(f"Analytics exported to: {report_path}")
            
            if args.detailed:
                print(json.dumps(analytics_data, indent=2))
            else:
                print(format_analytics_summary(analytics_data))
        
        elif args.command == "categories":
            registry_analytics = enhanced_registry.get_tool_analytics()
            categories = registry_analytics.get("categories", {})
            
            print("Tool Categories:")
            for category, count in categories.items():
                if args.stats:
                    # Get category tools for stats
                    try:
                        cat_enum = ToolCategory(category)
                        cat_tools = enhanced_registry.get_tools_by_category(cat_enum)
                        total_usage = sum(t.performance.usage_count for t in cat_tools)
                        avg_success = sum(t.performance.success_rate for t in cat_tools) / len(cat_tools) if cat_tools else 0
                        
                        print(f"  {category:<15} {count:<3} tools  "
                              f"Usage: {total_usage:<5}  Success: {avg_success:.1%}")
                    except ValueError:
                        print(f"  {category:<15} {count:<3} tools")
                else:
                    print(f"  {category}: {count} tools")
        
        elif args.command == "dependencies":
            if args.validate:
                errors = enhanced_registry.validate_dependencies()
                if errors:
                    print("Dependency validation errors:")
                    for error in errors:
                        print(f"  ❌ {error}")
                else:
                    print("✅ All dependencies are satisfied")
            
            elif args.tool:
                deps = enhanced_registry.get_tool_dependencies(args.tool)
                dependents = enhanced_registry.get_dependent_tools(args.tool)
                
                print(f"Dependencies for '{args.tool}':")
                if deps:
                    for dep in deps:
                        print(f"  → {dep}")
                else:
                    print("  None")
                
                print(f"\nTools that depend on '{args.tool}':")
                if dependents:
                    for dep in dependents:
                        print(f"  ← {dep}")
                else:
                    print("  None")
            
            else:
                # Show all dependencies
                print("Tool Dependencies:")
                for tool_name, tool in enhanced_registry.tools.items():
                    if tool.metadata.dependencies:
                        deps = ", ".join(tool.metadata.dependencies)
                        print(f"  {tool_name} → {deps}")
        
        elif args.command == "performance":
            if args.sort == "usage":
                tools = enhanced_registry.get_popular_tools(args.top)
                print(f"Top {args.top} most used tools:")
            elif args.sort == "success_rate":
                tools = enhanced_registry.get_best_performing_tools(args.top)
                print(f"Top {args.top} best performing tools:")
            else:  # response_time
                all_tools = list(enhanced_registry.tools.values())
                tools = sorted(all_tools, key=lambda t: t.performance.average_response_time)[:args.top]
                print(f"Top {args.top} fastest tools:")
            
            for i, tool in enumerate(tools, 1):
                metrics = tool.performance
                print(f"{i:2d}. {tool.name:<25} "
                      f"Usage: {metrics.usage_count:<5} "
                      f"Success: {metrics.success_rate:.1%} "
                      f"Time: {metrics.average_response_time:.2f}s")
        
        elif args.command == "recommendations":
            recommendations = analytics.get_tool_recommendations(args.context)
            
            print(f"Tool recommendations for '{args.context}':")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i:2d}. {rec['name']:<25} ({rec['category']})")
                print(f"    {rec['description']}")
                print(f"    Success: {rec['success_rate']:.1%}, "
                      f"Time: {rec['average_response_time']:.2f}s, "
                      f"Usage: {rec['usage_count']}")
                print(f"    Reason: {rec['reason']}")
                print()
        
        elif args.command == "migrate":
            if not args.confirm:
                response = input("Migrate tools from legacy registry? This will update the enhanced registry. (y/N): ")
                if response.lower() != 'y':
                    print("Migration cancelled")
                    return
            
            print("Migrating tools from legacy registry...")
            enhanced_registry.migrate_from_legacy_registry()
            print("Migration completed")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()