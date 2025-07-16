#!/usr/bin/env python3
"""
Command-line interface for testing tool search functionality.
"""

import argparse
import sys
from typing import List, Tuple
from .tool_search import (
    smart_tool_search,
    get_tools_by_category,
    get_all_categories,
    suggest_similar_tools,
    get_tool_info,
    fuzzy_search_tools,
    search_tools_by_description
)


def format_search_results(results: List[Tuple[str, float, str]]) -> str:
    """Format search results for display."""
    if not results:
        return "No results found."
    
    output = []
    for tool_name, score, match_type in results:
        output.append(f"  {tool_name:<30} (score: {score:.2f}, type: {match_type})")
    
    return "\n".join(output)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Search and explore iointel tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for tools")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Max results")
    search_parser.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy search")
    search_parser.add_argument("--no-desc", action="store_true", help="Disable description search")
    
    # Categories command
    categories_parser = subparsers.add_parser("categories", help="List tool categories")
    categories_parser.add_argument("--category", help="Show tools in specific category")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get tool information")
    info_parser.add_argument("tool_name", help="Tool name")
    
    # Similar command
    similar_parser = subparsers.add_parser("similar", help="Find similar tools")
    similar_parser.add_argument("tool_name", help="Tool name")
    similar_parser.add_argument("--limit", type=int, default=5, help="Max results")
    
    # Fuzzy command
    fuzzy_parser = subparsers.add_parser("fuzzy", help="Fuzzy search tools")
    fuzzy_parser.add_argument("query", help="Search query")
    fuzzy_parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "search":
            results = smart_tool_search(
                args.query,
                search_descriptions=not args.no_desc,
                search_fuzzy=not args.no_fuzzy,
                limit=args.limit
            )
            print(f"Search results for '{args.query}':")
            print(format_search_results(results))
        
        elif args.command == "categories":
            if args.category:
                tools = get_tools_by_category(args.category)
                print(f"Tools in '{args.category}' category:")
                for tool in tools:
                    print(f"  {tool}")
            else:
                categories = get_all_categories()
                print("Available tool categories:")
                for category, tools in categories.items():
                    print(f"  {category}: {len(tools)} tools")
        
        elif args.command == "info":
            info = get_tool_info(args.tool_name)
            if info:
                print(f"Tool: {info['name']}")
                print(f"Description: {info['description'] or 'No description'}")
                print(f"Category: {info['category'] or 'Unknown'}")
                print(f"Parameters: {len(info['parameters'])} parameters")
                print(f"Similar tools: {', '.join(info['similar_tools'])}")
            else:
                print(f"Tool '{args.tool_name}' not found.")
        
        elif args.command == "similar":
            similar = suggest_similar_tools(args.tool_name, args.limit)
            print(f"Similar tools to '{args.tool_name}':")
            for tool in similar:
                print(f"  {tool}")
        
        elif args.command == "fuzzy":
            results = fuzzy_search_tools(args.query, args.threshold)
            print(f"Fuzzy search results for '{args.query}':")
            for tool_name, score in results:
                print(f"  {tool_name:<30} (score: {score:.2f})")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()