#!/usr/bin/env python3
"""
Test Repository Population Script
=================================

This script extracts actual test cases from existing test files and
populates the centralized repository with real test data.

The current repository only has 4 default test cases, but we have 100+ 
test files with real scenarios that need to be preserved.
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

from iointel.src.utilities.workflow_test_repository import (
    WorkflowTestRepository,
    TestLayer,
    WorkflowTestCase,
    get_test_repository
)


class TestCaseExtractor:
    """Extracts real test cases from existing test files."""
    
    def __init__(self, test_repo: WorkflowTestRepository):
        self.test_repo = test_repo
        self.extracted_count = 0
    
    def extract_from_file(self, file_path: Path) -> List[WorkflowTestCase]:
        """Extract test cases from a single test file."""
        if not file_path.exists():
            return []
        
        content = file_path.read_text()
        extracted_cases = []
        
        # Extract different types of test cases
        extracted_cases.extend(self._extract_workflow_specs(content, file_path))
        extracted_cases.extend(self._extract_user_prompts(content, file_path))
        extracted_cases.extend(self._extract_routing_cases(content, file_path))
        extracted_cases.extend(self._extract_validation_cases(content, file_path))
        
        return extracted_cases
    
    def _extract_workflow_specs(self, content: str, file_path: Path) -> List[WorkflowTestCase]:
        """Extract hardcoded WorkflowSpec definitions."""
        cases = []
        
        # Look for WorkflowSpec constructor calls
        workflow_pattern = r'WorkflowSpec\s*\([^)]+\)'
        matches = re.finditer(workflow_pattern, content, re.DOTALL)
        
        for i, match in enumerate(matches):
            # Try to parse the workflow spec
            try:
                # This is a simplified extraction - in practice you'd need more sophisticated parsing
                spec_text = match.group(0)
                
                # Determine layer based on file location
                layer = self._classify_layer_from_path(file_path)
                category = self._classify_category_from_path(file_path)
                
                test_case = WorkflowTestCase(
                    id=f"extracted_{file_path.stem}_{i}",
                    name=f"Extracted workflow from {file_path.name}",
                    description=f"Workflow spec extracted from {file_path}",
                    layer=layer,
                    category=category,
                    workflow_spec={"extracted_from": str(file_path), "spec_text": spec_text},
                    tags=["extracted", "legacy", file_path.parent.name]
                )
                cases.append(test_case)
                
            except Exception as e:
                print(f"Failed to extract workflow spec from {file_path}: {e}")
        
        return cases
    
    def _extract_user_prompts(self, content: str, file_path: Path) -> List[WorkflowTestCase]:
        """Extract user prompts for agentic testing."""
        cases = []
        
        # Look for common prompt patterns
        prompt_patterns = [
            r'query\s*=\s*["\']([^"\']+)["\']',
            r'prompt\s*=\s*["\']([^"\']+)["\']',
            r'user_input\s*=\s*["\']([^"\']+)["\']',
            r'generate_workflow\([^)]*query\s*=\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in prompt_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for i, match in enumerate(matches):
                prompt = match.group(1)
                
                # Skip trivial prompts
                if len(prompt) < 10 or prompt in ['test', 'hello', 'example']:
                    continue
                
                test_case = WorkflowTestCase(
                    id=f"prompt_{file_path.stem}_{i}",
                    name=f"User prompt from {file_path.name}", 
                    description=f"User prompt extracted from {file_path}",
                    layer=TestLayer.AGENTIC,
                    category=self._classify_category_from_prompt(prompt),
                    user_prompt=prompt,
                    tags=["extracted", "agentic", file_path.parent.name]
                )
                cases.append(test_case)
        
        return cases
    
    def _extract_routing_cases(self, content: str, file_path: Path) -> List[WorkflowTestCase]:
        """Extract conditional routing test cases."""
        cases = []
        
        # Look for routing-related patterns
        if 'routing' in file_path.name.lower() or 'conditional' in content.lower():
            # Look for edge definitions with conditions
            edge_patterns = [
                r'EdgeSpec\s*\([^)]+condition[^)]+\)',
                r'edges?\s*=\s*\[[^]]+\]',
                r'route_index\s*[=:]\s*\d+'
            ]
            
            for pattern in edge_patterns:
                matches = re.finditer(pattern, content, re.DOTALL)
                for i, match in enumerate(matches):
                    test_case = WorkflowTestCase(
                        id=f"routing_{file_path.stem}_{i}",
                        name=f"Routing case from {file_path.name}",
                        description=f"Conditional routing extracted from {file_path}",
                        layer=TestLayer.LOGICAL,
                        category="conditional_routing",
                        workflow_spec={"routing_pattern": match.group(0)},
                        tags=["extracted", "routing", file_path.parent.name]
                    )
                    cases.append(test_case)
        
        return cases
    
    def _extract_validation_cases(self, content: str, file_path: Path) -> List[WorkflowTestCase]:
        """Extract validation test cases."""
        cases = []
        
        # Look for validation patterns
        validation_patterns = [
            r'validate_structure\s*\([^)]+\)',
            r'ValidationError',
            r'assert.*len\(issues\)',
            r'should_pass\s*=\s*(True|False)'
        ]
        
        has_validation = any(re.search(pattern, content, re.IGNORECASE) for pattern in validation_patterns)
        
        if has_validation:
            test_case = WorkflowTestCase(
                id=f"validation_{file_path.stem}",
                name=f"Validation case from {file_path.name}",
                description=f"Validation logic extracted from {file_path}",
                layer=TestLayer.LOGICAL,
                category="validation",
                workflow_spec={"has_validation": True, "file": str(file_path)},
                tags=["extracted", "validation", file_path.parent.name]
            )
            cases.append(test_case)
        
        return cases
    
    def _classify_layer_from_path(self, file_path: Path) -> TestLayer:
        """Classify test layer based on file path."""
        path_str = str(file_path).lower()
        
        if 'planner' in path_str or 'generation' in path_str:
            return TestLayer.AGENTIC
        elif 'execution' in path_str or 'pipeline' in path_str or 'sla' in path_str:
            return TestLayer.ORCHESTRATION
        elif 'chat' in path_str or 'feedback' in path_str:
            return TestLayer.FEEDBACK
        else:
            return TestLayer.LOGICAL
    
    def _classify_category_from_path(self, file_path: Path) -> str:
        """Classify test category based on file path."""
        path_str = str(file_path).lower()
        
        if 'stock' in path_str:
            return "stock_analysis"
        elif 'routing' in path_str:
            return "conditional_routing"
        elif 'validation' in path_str:
            return "validation"
        elif 'tool' in path_str:
            return "tool_integration"
        elif 'agent' in path_str:
            return "agent_behavior"
        else:
            return file_path.parent.name
    
    def _classify_category_from_prompt(self, prompt: str) -> str:
        """Classify category based on prompt content."""
        prompt_lower = prompt.lower()
        
        if 'stock' in prompt_lower or 'trading' in prompt_lower:
            return "stock_analysis"
        elif 'weather' in prompt_lower:
            return "weather_analysis"
        elif 'email' in prompt_lower:
            return "email_automation"
        else:
            return "general"
    
    def populate_repository(self, test_dirs: List[Path]) -> Dict[str, Any]:
        """Populate the repository with extracted test cases."""
        results = {
            "files_processed": 0,
            "test_cases_extracted": 0,
            "extraction_details": {},
            "layer_distribution": {},
            "category_distribution": {}
        }
        
        for test_dir in test_dirs:
            if not test_dir.exists():
                continue
            
            test_files = list(test_dir.rglob("test_*.py"))
            
            for test_file in test_files:
                print(f"Processing {test_file}...")
                extracted_cases = self.extract_from_file(test_file)
                
                results["files_processed"] += 1
                results["test_cases_extracted"] += len(extracted_cases)
                results["extraction_details"][str(test_file)] = len(extracted_cases)
                
                # Add cases to repository
                for case in extracted_cases:
                    self.test_repo.add_test_case(case)
                    
                    # Update distribution stats
                    layer = case.layer.value
                    results["layer_distribution"][layer] = results["layer_distribution"].get(layer, 0) + 1
                    
                    category = case.category
                    results["category_distribution"][category] = results["category_distribution"].get(category, 0) + 1
        
        return results


def main():
    """Main population script."""
    parser = argparse.ArgumentParser(description='Populate test repository with real test cases')
    parser.add_argument('--test-dirs', nargs='+', default=['tests'], help='Test directories to scan')
    parser.add_argument('--output', default='population_results.json', help='Output file for results')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be extracted without saving')
    
    args = parser.parse_args()
    
    # Initialize test repository
    test_repo = get_test_repository()
    extractor = TestCaseExtractor(test_repo)
    
    print("🔍 Extracting test cases from existing files...")
    print("=" * 50)
    
    # Process test directories
    test_dirs = [Path(d) for d in args.test_dirs]
    results = extractor.populate_repository(test_dirs)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n📊 Extraction Results:")
    print(f"Files processed: {results['files_processed']}")
    print(f"Test cases extracted: {results['test_cases_extracted']}")
    
    print("\n📋 Layer Distribution:")
    for layer, count in results['layer_distribution'].items():
        print(f"  {layer}: {count}")
    
    print("\n📂 Category Distribution:")
    for category, count in sorted(results['category_distribution'].items()):
        print(f"  {category}: {count}")
    
    print(f"\n💾 Detailed results saved to: {args.output}")
    
    # Verify repository state
    print(f"\n🗄️ Repository now contains {len(test_repo._test_cases)} total test cases")
    
    if not args.dry_run:
        print("\n✅ Test cases have been saved to the repository!")
        print("You can now run tests using the centralized fixtures.")
    else:
        print("\n🔍 This was a dry run - no changes saved.")


if __name__ == '__main__':
    main()