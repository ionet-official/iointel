#!/usr/bin/env python3
"""
Migration script: tool nodes → data_source nodes

This script migrates workflows from the old 'tool' type to the new ontology:
- Pure data sources (user_input, prompt_tool) → type: "data_source" with source_name
- Tool-using operations → type: "agent" with tools list
"""

import os
import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Data sources that should become data_source nodes
DATA_SOURCES = {'user_input', 'prompt_tool'}

def migrate_workflow_file(file_path: str) -> bool:
    """Migrate a single workflow file. Returns True if changes were made."""
    print(f"📝 Processing: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        # 1. Replace type="tool" with appropriate type
        def replace_tool_type(match):
            nonlocal changes_made
            full_match = match.group(0)
            
            # Look for tool_name in the context
            tool_name_match = re.search(r'tool_name["\']?\s*[=:]\s*["\']([^"\']+)["\']', full_match, re.DOTALL)
            
            if tool_name_match:
                tool_name = tool_name_match.group(1)
                
                if tool_name in DATA_SOURCES:
                    # Convert to data_source
                    result = full_match.replace('type="tool"', 'type="data_source"')
                    result = result.replace("type='tool'", "type='data_source'")
                    result = result.replace('tool_name=', 'source_name=')
                    result = result.replace('"tool_name":', '"source_name":')
                    changes_made = True
                    print(f"  ✅ Converted {tool_name} to data_source")
                    return result
                else:
                    # Convert to agent with tools
                    result = full_match.replace('type="tool"', 'type="agent"')
                    result = result.replace("type='tool'", "type='agent'")
                    
                    # Add agent_instructions before tool_name
                    if 'agent_instructions' not in result:
                        tool_instruction = f'agent_instructions="Use the {tool_name} tool to complete this task",'
                        result = re.sub(r'(data=NodeData\([^)]*?)', r'\1\n                    ' + tool_instruction, result)
                    
                    # Convert tool_name to tools array
                    result = re.sub(r'tool_name\s*[=:]\s*["\']([^"\']+)["\']', r'tools=["\1"]', result)
                    
                    changes_made = True
                    print(f"  ✅ Converted {tool_name} to agent with tools")
                    return result
            
            # Fallback: convert to agent
            result = full_match.replace('type="tool"', 'type="agent"')
            result = result.replace("type='tool'", "type='agent'")
            changes_made = True
            print("  ⚠️  Converted unknown tool to agent (needs manual review)")
            return result
        
        # Find all NodeSpec constructions with type="tool"
        pattern = r'NodeSpec\s*\([^)]*?type\s*=\s*["\']tool["\'][^)]*?\)'
        content = re.sub(pattern, replace_tool_type, content, flags=re.DOTALL)
        
        # 2. Update any remaining tool_name references to source_name in data_source contexts
        # This handles cases where the type was already changed but tool_name wasn't
        def replace_tool_name_in_data_source(match):
            nonlocal changes_made
            full_match = match.group(0)
            if 'type="data_source"' in full_match or "type='data_source'" in full_match:
                result = full_match.replace('tool_name=', 'source_name=')
                result = result.replace('"tool_name":', '"source_name":')
                if result != full_match:
                    changes_made = True
                return result
            return full_match
        
        # Apply tool_name → source_name replacement in data_source contexts
        content = re.sub(r'NodeSpec\s*\([^)]*?\)', replace_tool_name_in_data_source, content, flags=re.DOTALL)
        
        # 3. Update validation error messages
        content = content.replace('tool nodes', 'data_source nodes')
        content = content.replace('Tool node', 'Data source node')
        content = content.replace('TOOL HALLUCINATION', 'SOURCE HALLUCINATION')
        
        # 4. Write back if changes were made
        if changes_made:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"  💾 Saved changes to {file_path}")
            return True
        else:
            print("  ➡️  No changes needed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False

def main():
    """Run the migration on all relevant files."""
    print("🚀 Starting tool → data_source migration")
    print("=" * 50)
    
    # Files to migrate
    files_to_migrate = [
        "iointel/src/test_workflows/workflow_examples.py",
        "tests/test_workflow_spec.py", 
        "tests/test_workflow_converter.py",
        "tests/test_workflow_spec_conversions.py",
        "tests/test_workflow_cli.py",
        "tests/workflows/planning/test_workflow_planner_updates.py",
        "examples/workflow_planner_example.py",
    ]
    
    # Also find JSON workflow files
    json_files = []
    for json_dir in ["saved_workflows/json", "iointel/src/test_workflows"]:
        if os.path.exists(json_dir):
            for root, dirs, files in os.walk(json_dir):
                for file in files:
                    if file.endswith('.json'):
                        json_files.append(os.path.join(root, file))
    
    files_to_migrate.extend(json_files)
    
    changed_files = []
    
    for file_path in files_to_migrate:
        if os.path.exists(file_path):
            if migrate_workflow_file(file_path):
                changed_files.append(file_path)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print("\n" + "=" * 50)
    print("🎉 Migration completed!")
    print(f"📊 Files processed: {len(files_to_migrate)}")
    print(f"📝 Files changed: {len(changed_files)}")
    
    if changed_files:
        print("\n📋 Changed files:")
        for file_path in changed_files:
            print(f"  - {file_path}")
        
        print("\n⚠️  Please review the changes and test the system!")
        print("💡 Consider running tests: uv run python -m pytest tests/")
    
    return len(changed_files)

if __name__ == "__main__":
    sys.exit(main())