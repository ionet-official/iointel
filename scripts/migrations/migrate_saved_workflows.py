#!/usr/bin/env python3
"""
Simple migration script to update saved workflows from old 'agent' type to new semantic types.
"""

import json
import sys
from pathlib import Path

def migrate_node_type(node):
    """Map old 'agent' type to appropriate semantic type based on context."""
    if node.get('type') != 'agent':
        return node
    
    # Look for clues in the node to determine semantic type
    data = node.get('data', {})
    tools = data.get('tools') or []  # Handle None case
    instructions = (data.get('agent_instructions') or '').lower()  # Handle None case
    label = (node.get('label') or '').lower()  # Handle None case
    
    # Decision agents - have routing tools
    routing_tools = ['conditional_gate', 'threshold_gate', 'conditional_multi_gate']
    if any(tool in routing_tools for tool in tools):
        node['type'] = 'decision'
        print(f"  🎯 {node['label']} -> decision (has routing tools: {[t for t in tools if t in routing_tools]})")
        return node
    
    # Data fetchers - have external API tools or fetch/get in name/instructions
    api_keywords = ['fetch', 'get', 'retrieve', 'search', 'crawl', 'scrape']
    if any(keyword in instructions or keyword in label for keyword in api_keywords):
        if tools:  # Must have tools to be data_fetcher
            node['type'] = 'data_fetcher' 
            print(f"  📊 {node['label']} -> data_fetcher (has tools + fetch keywords)")
            return node
    
    # Executors - have action tools or execute/send/save in name/instructions  
    action_keywords = ['execute', 'send', 'save', 'write', 'create', 'delete', 'trade', 'buy', 'sell']
    if any(keyword in instructions or keyword in label for keyword in action_keywords):
        if tools:  # Must have tools to be executor
            node['type'] = 'executor'
            print(f"  ⚡ {node['label']} -> executor (has tools + action keywords)")
            return node
    
    # Analyzers - have analysis keywords or tools
    analysis_keywords = ['analyz', 'evaluat', 'process', 'assess', 'review', 'examine']
    if any(keyword in instructions or keyword in label for keyword in analysis_keywords):
        node['type'] = 'analyzer'
        print(f"  🔍 {node['label']} -> analyzer (has analysis keywords)")
        return node
    
    # Default to conversational for simple chat/help agents
    node['type'] = 'conversational'
    print(f"  💬 {node['label']} -> conversational (default)")
    return node

def migrate_workflow_file(file_path):
    """Migrate a single workflow file."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        
        print(f"\n📄 Migrating: {file_path.name}")
        
        # Check if migration is needed
        needs_migration = False
        for node in workflow_data.get('nodes', []):
            if node.get('type') == 'agent':
                needs_migration = True
                break
        
        if not needs_migration:
            print("  ✅ Already up to date")
            return True
            
        # Migrate each node
        for node in workflow_data.get('nodes', []):
            migrate_node_type(node)
        
        # Write back the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(workflow_data, f, indent=2, ensure_ascii=False)
        
        print("  ✅ Migration completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Error migrating {file_path.name}: {e}")
        return False

def main():
    """Run the migration on all saved workflows."""
    saved_workflows_dir = Path("saved_workflows/json")
    
    if not saved_workflows_dir.exists():
        print("❌ saved_workflows/json directory not found")
        return False
    
    json_files = list(saved_workflows_dir.glob("*.json"))
    
    if not json_files:
        print("ℹ️  No JSON workflow files found")
        return True
    
    print(f"🚀 Starting migration of {len(json_files)} workflow files...")
    
    success_count = 0
    for json_file in json_files:
        if migrate_workflow_file(json_file):
            success_count += 1
    
    print("\n🎉 Migration completed!")
    print(f"✅ Successfully migrated: {success_count}/{len(json_files)} files")
    
    if success_count < len(json_files):
        print(f"⚠️  Failed to migrate: {len(json_files) - success_count} files")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)