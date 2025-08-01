#!/usr/bin/env python3
"""
Migration script to update saved workflow files from semantic node types to simple types.

This script converts the old semantic node types:
- analyzer -> agent
- executor -> agent  
- data_fetcher -> agent
- conversational -> agent

Tool and decision types remain unchanged.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

def migrate_workflow_file(file_path: Path) -> Tuple[bool, Dict]:
    """
    Migrate a single workflow file from semantic types to simple types.
    
    Args:
        file_path: Path to the workflow JSON file
        
    Returns:
        Tuple of (changes_made, migration_report)
    """
    print(f"📂 Processing: {file_path.name}")
    
    # Read the workflow file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
    except Exception as e:
        return False, {"error": f"Failed to read file: {e}"}
    
    # Track changes
    changes = []
    nodes_updated = 0
    
    # Define the semantic types that need to be converted to 'agent'
    semantic_types_to_migrate = {
        'analyzer': 'agent',
        'executor': 'agent', 
        'data_fetcher': 'agent',
        'conversational': 'agent'
    }
    
    # Process nodes
    if 'nodes' in workflow_data and isinstance(workflow_data['nodes'], list):
        for i, node in enumerate(workflow_data['nodes']):
            if isinstance(node, dict) and 'type' in node:
                old_type = node['type']
                
                # Check if this node type needs migration
                if old_type in semantic_types_to_migrate:
                    new_type = semantic_types_to_migrate[old_type]
                    node['type'] = new_type
                    nodes_updated += 1
                    
                    node_label = node.get('label', node.get('id', f'node_{i}'))
                    changes.append({
                        'node_id': node.get('id', f'node_{i}'),
                        'node_label': node_label,
                        'old_type': old_type,
                        'new_type': new_type
                    })
                    
                    print(f"  ✅ Updated node '{node_label}': {old_type} -> {new_type}")
    
    # If no changes were made, return early
    if nodes_updated == 0:
        print(f"  ℹ️  No semantic types found - file already uses simple types")
        return False, {"message": "No changes needed"}
    
    # Create backup
    backup_path = file_path.with_suffix('.json.backup')
    shutil.copy2(file_path, backup_path)
    print(f"  💾 Backup created: {backup_path.name}")
    
    # Write updated workflow back to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(workflow_data, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Updated file written successfully")
    except Exception as e:
        # Restore from backup if write fails
        shutil.copy2(backup_path, file_path)
        return False, {"error": f"Failed to write updated file: {e}"}
    
    migration_report = {
        "file": file_path.name,
        "nodes_updated": nodes_updated,
        "changes": changes,
        "backup_created": backup_path.name
    }
    
    return True, migration_report

def migrate_all_saved_workflows(workflows_dir: Path = None) -> Dict:
    """
    Migrate all saved workflow files in the specified directory.
    
    Args:
        workflows_dir: Directory containing workflow JSON files
        
    Returns:
        Migration summary report
    """
    if workflows_dir is None:
        workflows_dir = Path(__file__).parent / "saved_workflows" / "json"
    
    print(f"🚀 WORKFLOW SEMANTIC TYPE MIGRATION")
    print(f"================================")
    print(f"Directory: {workflows_dir}")
    print(f"Target: Convert analyzer/executor/data_fetcher/conversational -> agent")
    print()
    
    if not workflows_dir.exists():
        return {"error": f"Directory not found: {workflows_dir}"}
    
    # Find all JSON files
    json_files = list(workflows_dir.glob("*.json"))
    print(f"📁 Found {len(json_files)} JSON files to process")
    print()
    
    # Process each file
    migration_summary = {
        "total_files": len(json_files),
        "files_migrated": 0,
        "files_unchanged": 0,
        "files_with_errors": 0,
        "total_nodes_updated": 0,
        "migration_reports": [],
        "errors": []
    }
    
    for json_file in json_files:
        try:
            changes_made, report = migrate_workflow_file(json_file)
            
            if changes_made:
                migration_summary["files_migrated"] += 1
                migration_summary["total_nodes_updated"] += report.get("nodes_updated", 0)
                migration_summary["migration_reports"].append(report)
            else:
                migration_summary["files_unchanged"] += 1
                if "error" in report:
                    migration_summary["files_with_errors"] += 1
                    migration_summary["errors"].append({
                        "file": json_file.name,
                        "error": report["error"]
                    })
            
        except Exception as e:
            migration_summary["files_with_errors"] += 1
            migration_summary["errors"].append({
                "file": json_file.name,
                "error": str(e)
            })
            print(f"  ❌ Error processing {json_file.name}: {e}")
        
        print()  # Blank line between files
    
    return migration_summary

def print_migration_summary(summary: Dict):
    """Print a detailed migration summary report."""
    print("📊 MIGRATION SUMMARY")
    print("===================")
    print(f"Total files processed: {summary['total_files']}")
    print(f"Files migrated: {summary['files_migrated']}")
    print(f"Files unchanged: {summary['files_unchanged']}")
    print(f"Files with errors: {summary['files_with_errors']}")
    print(f"Total nodes updated: {summary['total_nodes_updated']}")
    print()
    
    if summary["migration_reports"]:
        print("📋 DETAILED MIGRATION REPORT")
        print("============================")
        for report in summary["migration_reports"]:
            print(f"\n📄 {report['file']}")
            print(f"   Nodes updated: {report['nodes_updated']}")
            print(f"   Backup: {report['backup_created']}")
            
            if report.get("changes"):
                print("   Changes:")
                for change in report["changes"]:
                    print(f"     - {change['node_label']}: {change['old_type']} -> {change['new_type']}")
    
    if summary["errors"]:
        print("\n❌ ERRORS ENCOUNTERED")
        print("====================")
        for error in summary["errors"]:
            print(f"  - {error['file']}: {error['error']}")
    
    print("\n✅ Migration completed!")
    if summary["files_migrated"] > 0:
        print(f"   Successfully migrated {summary['files_migrated']} files")
        print(f"   Updated {summary['total_nodes_updated']} nodes total")
        print("   Backup files created with .backup extension")
    else:
        print("   No files needed migration (all already use simple types)")

if __name__ == "__main__":
    # Run migration
    summary = migrate_all_saved_workflows()
    
    # Print detailed report
    print_migration_summary(summary)
    
    # Create migration log
    log_file = Path(__file__).parent / "migration_log.json"
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n📝 Migration log saved to: {log_file}")
    except Exception as e:
        print(f"\n⚠️  Failed to save migration log: {e}")