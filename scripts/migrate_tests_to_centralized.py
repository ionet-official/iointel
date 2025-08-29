#!/usr/bin/env python3
"""
Test Migration Script
=====================

This script helps migrate existing workflow tests to use the new centralized
test repository system.

Usage:
    python scripts/migrate_tests_to_centralized.py

This will:
1. Scan existing test files for workflow-related tests
2. Extract test patterns and data
3. Generate migration suggestions
4. Optionally apply automatic migrations where safe
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
import argparse

from iointel.src.utilities.workflow_test_repository import (
    WorkflowTestRepository,
    TestLayer,
    get_test_repository
)


class TestFileMigrator:
    """Migrates individual test files to use centralized fixtures."""
    
    def __init__(self, test_repo: WorkflowTestRepository):
        self.test_repo = test_repo
        self.migration_suggestions = []
    
    def scan_test_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan a test file and extract migration information."""
        if not file_path.exists():
            return {}
        
        content = file_path.read_text()
        suggestions = {
            'file_path': str(file_path),
            'fixture_replacements': [],
            'test_classifications': [],
            'extracted_test_data': []
        }
        
        # Find fixture definitions that should be replaced
        fixture_patterns = [
            (r'@pytest\.fixture\s*\ndef\s+tool_catalog\s*\(', 'tool_catalog → mock_tool_catalog'),
            (r'@pytest\.fixture\s*\ndef\s+real_tool_catalog\s*\(', 'real_tool_catalog → already centralized'),
            (r'@pytest\.fixture\s*\ndef\s+workflow_spec\s*\(', 'workflow_spec → sample_workflow_spec'),
        ]
        
        for pattern, suggestion in fixture_patterns:
            if re.search(pattern, content):
                suggestions['fixture_replacements'].append(suggestion)
        
        # Find tests that could be classified by layer
        test_classifications = [
            (r'def test.*generate.*workflow', TestLayer.AGENTIC, 'LLM workflow generation'),
            (r'def test.*validation', TestLayer.LOGICAL, 'Workflow validation'),
            (r'def test.*routing', TestLayer.LOGICAL, 'Conditional routing'),
            (r'def test.*execution', TestLayer.ORCHESTRATION, 'Workflow execution'),
            (r'def test.*feedback', TestLayer.FEEDBACK, 'Chat feedback'),
        ]
        
        for pattern, layer, description in test_classifications:
            if re.search(pattern, content, re.IGNORECASE):
                suggestions['test_classifications'].append({
                    'pattern': pattern,
                    'layer': layer.value,
                    'description': description
                })
        
        # Extract hardcoded test data
        hardcoded_data = self._extract_hardcoded_data(content)
        suggestions['extracted_test_data'] = hardcoded_data
        
        return suggestions
    
    def _extract_hardcoded_data(self, content: str) -> List[Dict[str, Any]]:
        """Extract hardcoded workflow data that could be centralized."""
        extracted = []
        
        # Look for workflow spec patterns
        workflow_patterns = [
            r'WorkflowSpec\s*\([^}]+\)',
            r'"nodes"\s*:\s*\[[^]]+\]',
            r'"edges"\s*:\s*\[[^]]+\]',
        ]
        
        for pattern in workflow_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                extracted.append({
                    'type': 'workflow_data',
                    'content': match.group(0)[:100] + '...' if len(match.group(0)) > 100 else match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return extracted
    
    def generate_migration_plan(self, test_dir: Path) -> Dict[str, Any]:
        """Generate a comprehensive migration plan for a test directory."""
        plan = {
            'summary': {},
            'files': {},
            'recommendations': []
        }
        
        test_files = list(test_dir.rglob('test_*.py'))
        
        for test_file in test_files:
            file_suggestions = self.scan_test_file(test_file)
            if any(file_suggestions.values()):
                plan['files'][str(test_file)] = file_suggestions
        
        # Generate summary statistics
        total_files = len(test_files)
        files_needing_migration = len(plan['files'])
        
        plan['summary'] = {
            'total_test_files': total_files,
            'files_needing_migration': files_needing_migration,
            'migration_percentage': round((files_needing_migration / total_files) * 100, 1) if total_files > 0 else 0
        }
        
        # Generate recommendations
        plan['recommendations'] = self._generate_recommendations(plan['files'])
        
        return plan
    
    def _generate_recommendations(self, files: Dict[str, Any]) -> List[str]:
        """Generate actionable migration recommendations."""
        recommendations = []
        
        fixture_count = sum(len(f.get('fixture_replacements', [])) for f in files.values())
        test_count = sum(len(f.get('test_classifications', [])) for f in files.values())
        data_count = sum(len(f.get('extracted_test_data', [])) for f in files.values())
        
        if fixture_count > 0:
            recommendations.append(f"Replace {fixture_count} local fixtures with centralized ones")
        
        if test_count > 0:
            recommendations.append(f"Classify {test_count} tests into appropriate layers")
        
        if data_count > 0:
            recommendations.append(f"Move {data_count} hardcoded test data items to test repository")
        
        recommendations.extend([
            "Run tests before and after migration to ensure compatibility",
            "Update CI/CD pipelines to use new test structure",
            "Train team on new centralized testing patterns"
        ])
        
        return recommendations


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(description='Migrate tests to centralized system')
    parser.add_argument('--test-dir', default='tests', help='Test directory to scan')
    parser.add_argument('--output', default='migration_plan.json', help='Output file for migration plan')
    parser.add_argument('--apply', action='store_true', help='Apply safe migrations automatically')
    
    args = parser.parse_args()
    
    # Initialize test repository
    test_repo = get_test_repository()
    migrator = TestFileMigrator(test_repo)
    
    # Generate migration plan
    test_dir = Path(args.test_dir)
    plan = migrator.generate_migration_plan(test_dir)
    
    # Save migration plan
    with open(args.output, 'w') as f:
        json.dump(plan, f, indent=2, default=str)
    
    # Print summary
    print("🔍 Test Migration Analysis Complete")
    print("=" * 40)
    print(f"📁 Scanned directory: {test_dir}")
    print(f"📊 Total test files: {plan['summary']['total_test_files']}")
    print(f"🔧 Files needing migration: {plan['summary']['files_needing_migration']}")
    print(f"📈 Migration coverage: {plan['summary']['migration_percentage']}%")
    print(f"💾 Migration plan saved to: {args.output}")
    
    print("\n🎯 Recommendations:")
    for i, rec in enumerate(plan['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n📋 Next Steps:")
    print("  1. Review the migration plan JSON file")
    print("  2. Start with high-impact, low-risk files")
    print("  3. Run tests frequently during migration")
    print("  4. Use the layered test architecture for new tests")
    
    if args.apply:
        print("\n⚠️  --apply flag not yet implemented (safety first!)")
        print("   Manual migration recommended for initial rollout")


if __name__ == '__main__':
    main()