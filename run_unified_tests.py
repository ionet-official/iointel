#!/usr/bin/env python3
"""
Unified Test Runner
==================

This is THE single file you run to test the whole fucking stack.
It uses our unified smart test repository to run tests by layer, category, or tag.

Usage:
    python run_unified_tests.py                    # Run all tests
    python run_unified_tests.py --layer logical    # Run only logical tests
    python run_unified_tests.py --layer agentic    # Run only agentic tests  
    python run_unified_tests.py --tags route_index # Run only route_index tests
    python run_unified_tests.py --category routing_validation # Run specific category
"""

import asyncio
import sys
import os
import argparse
from typing import List, Optional
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from iointel.src.utilities.workflow_test_repository import (
    TestLayer, 
    WorkflowTestCase,
    get_test_repository
)
from iointel.src.utilities.workflow_helpers import generate_only, plan_and_execute
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.tool_registry_utils import create_tool_catalog
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.utilities.test_executors import get_test_executor, list_test_types
from iointel.src.web.conversation_storage import ConversationStorage
import subprocess

class UnifiedTestRunner:
    """Runs tests from the unified smart test repository."""
    
    def __init__(self):
        self.repo = get_test_repository()
        self.conversation_storage = ConversationStorage()
        self.tool_catalog = None
        self.results = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "total": 0
        }
        self.git_state = self._get_git_state()
    
    def _get_git_state(self) -> dict:
        """Get current git state for test context tracking."""
        try:
            # Get current branch
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                cwd=os.getcwd(), 
                text=True
            ).strip()
            
            # Get current commit hash
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=os.getcwd(), 
                text=True
            ).strip()[:8]
            
            # Get short status
            status = subprocess.check_output(
                ["git", "status", "--porcelain"], 
                cwd=os.getcwd(), 
                text=True
            ).strip()
            
            return {
                "branch": branch,
                "commit": commit,
                "has_changes": bool(status),
                "status_lines": len(status.split('\n')) if status else 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _create_test_conversation(self, test: WorkflowTestCase, layer: str) -> str:
        """Create a conversation for test isolation with git context."""
        git_context = f"{self.git_state['branch']}_{self.git_state['commit']}"
        if self.git_state.get('has_changes'):
            git_context += f"_dirty{self.git_state['status_lines']}"
        
        conversation_id = self.conversation_storage.create_conversation(
            session_type=f"test_{layer}",
            notes=f"Test: {test.name} | Git: {git_context} | Category: {test.category}",
            version=f"git_{git_context}"
        )
        
        return conversation_id
    
    async def setup(self):
        """Setup test environment."""
        print("🔧 Setting up test environment...")
        
        # Display git state for context
        if 'error' not in self.git_state:
            print(f"📋 Git Context: {self.git_state['branch']}@{self.git_state['commit']}")
            if self.git_state['has_changes']:
                print(f"   ⚠️  {self.git_state['status_lines']} uncommitted changes")
            else:
                print("   ✅ Clean working tree")
        else:
            print(f"   ⚠️  Git state error: {self.git_state['error']}")
        
        # AgenticTestExecutor now auto-registered in TestExecutorRegistry
        
        load_tools_from_env()
        self.tool_catalog = create_tool_catalog()
        print(f"✅ Loaded {len(self.tool_catalog)} tools")
    
    async def run_logical_test(self, test: WorkflowTestCase) -> bool:
        """Run a logical layer test (structure validation)."""
        print(f"\n📋 LOGICAL TEST: {test.name}")
        print(f"   Description: {test.description}")
        
        try:
            # Logical tests validate WorkflowSpec structure
            if not test.workflow_spec:
                print("   ❌ No workflow spec provided")
                return False
            
            # Convert dict to WorkflowSpec if needed
            if isinstance(test.workflow_spec, dict):
                spec = WorkflowSpec.model_validate(test.workflow_spec)
            else:
                spec = test.workflow_spec
            
            # Use standard utility to get LLM prompt
            llm_prompt = spec.to_llm_prompt()
            print(f"   📋 Generated LLM prompt: {len(llm_prompt)} chars")
            
            # Validate structure
            issues = spec.validate_structure(self.tool_catalog)
            
            if test.should_pass:
                if issues:
                    print(f"   ❌ Unexpected validation issues: {issues}")
                    return False
                else:
                    print("   ✅ Structure validation passed")
                    return True
            else:
                if issues:
                    print(f"   ✅ Expected validation failures found: {issues}")
                    return True
                else:
                    print("   ❌ Expected validation to fail but it passed")
                    return False
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    async def run_agentic_test(self, test: WorkflowTestCase) -> bool:
        """Run an agentic layer test (LLM workflow generation)."""
        print(f"\n🤖 AGENTIC TEST: {test.name}")
        print(f"   Description: {test.description}")
        print(f"   User prompt: {test.user_prompt}")
        
        try:
            # Create conversation with git context tracking
            test_conversation_id = self._create_test_conversation(test, "agentic")
            print(f"   📞 Conversation: {test_conversation_id}")
            
            # Generate workflow from prompt
            spec = await generate_only(
                test.user_prompt, 
                self.tool_catalog,
                conversation_id=test_conversation_id
            )
            
            if not spec:
                print("   ❌ Failed to generate workflow")
                return False
            
            print(f"   ✅ Generated: {spec.title} ({len(spec.nodes)} nodes, {len(spec.edges)} edges)")
            
            # Validate against expected results
            if test.expected_result:
                return self._validate_against_expected(spec, test.expected_result)
            else:
                print("   ✅ Workflow generated successfully (no specific validation)")
                return True
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    async def run_orchestration_test(self, test: WorkflowTestCase) -> bool:
        """Run an orchestration layer test (end-to-end execution)."""
        print(f"\n🎭 ORCHESTRATION TEST: {test.name}")
        print(f"   Description: {test.description}")
        print(f"   User prompt: {test.user_prompt}")
        
        try:
            # Create conversation with git context tracking
            test_conversation_id = self._create_test_conversation(test, "orchestration")
            print(f"   📞 Conversation: {test_conversation_id}")
            
            # Execute full workflow
            result = await plan_and_execute(
                test.user_prompt,
                conversation_id=test_conversation_id,
                tool_catalog=self.tool_catalog
            )
            
            if not result["success"]:
                print(f"   ❌ Workflow execution failed: {result.get('error')}")
                return False
            
            spec = result["workflow_spec"]
            final_state = result["execution_result"]
            stats = result["execution_stats"]
            
            print(f"   ✅ Executed: {spec.title}")
            print(f"   📊 Stats: {stats['executed_nodes']}/{stats['total_nodes']} nodes, efficiency: {stats['execution_efficiency']}")
            
            # Validate against expected execution results
            if test.expected_result:
                return self._validate_execution_result(result, test.expected_result)
            else:
                print("   ✅ Workflow executed successfully")
                return True
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def _validate_against_expected(self, spec: WorkflowSpec, expected: dict) -> bool:
        """Validate generated workflow against expected results."""
        all_passed = True
        
        # Check workflow generated
        if "workflow_generated" in expected:
            actual = spec is not None
            expected_val = expected["workflow_generated"]
            passed = actual == expected_val
            print(f"   {'✅' if passed else '❌'} workflow_generated: expected={expected_val}, actual={actual}")
            if not passed: all_passed = False
        
        # Check for decision nodes
        if "has_decision_nodes" in expected:
            decision_nodes = [n for n in spec.nodes if n.type == "decision" or 
                            (hasattr(n.data, 'tools') and n.data.tools and 'conditional_gate' in n.data.tools)]
            actual = len(decision_nodes) > 0
            expected_val = expected["has_decision_nodes"]
            passed = actual == expected_val
            print(f"   {'✅' if passed else '❌'} has_decision_nodes: expected={expected_val}, actual={actual} (found {len(decision_nodes)})")
            if not passed: all_passed = False
        
        # Check decision edges have route_index
        if "decision_edges_have_route_index" in expected:
            decision_nodes = [n for n in spec.nodes if n.type == "decision" or 
                            (hasattr(n.data, 'tools') and n.data.tools and 'conditional_gate' in n.data.tools)]
            edges_with_route_index = 0
            total_decision_edges = 0
            
            for decision_node in decision_nodes:
                node_edges = [e for e in spec.edges if e.source == decision_node.id]
                total_decision_edges += len(node_edges)
                
                for edge in node_edges:
                    if edge.data and edge.data.route_index is not None:
                        edges_with_route_index += 1
                        print(f"   🔀 Edge {edge.source} → {edge.target}: route_index={edge.data.route_index}")
            
            actual = edges_with_route_index > 0
            expected_val = expected["decision_edges_have_route_index"]
            passed = actual == expected_val
            print(f"   {'✅' if passed else '❌'} decision_edges_have_route_index: expected={expected_val}, actual={actual} ({edges_with_route_index}/{total_decision_edges})")
            if not passed: all_passed = False
        
        # Check route index values are valid
        if "route_index_values_valid" in expected:
            decision_nodes = [n for n in spec.nodes if n.type == "decision" or 
                            (hasattr(n.data, 'tools') and n.data.tools and 'conditional_gate' in n.data.tools)]
            total_decision_edges = 0
            edges_with_route_index = 0
            
            for decision_node in decision_nodes:
                node_edges = [e for e in spec.edges if e.source == decision_node.id]
                total_decision_edges += len(node_edges)
                edges_with_route_index += len([e for e in node_edges if e.data and e.data.route_index is not None])
            
            actual = edges_with_route_index == total_decision_edges if total_decision_edges > 0 else True
            expected_val = expected["route_index_values_valid"]
            passed = actual == expected_val
            print(f"   {'✅' if passed else '❌'} route_index_values_valid: expected={expected_val}, actual={actual}")
            if not passed: all_passed = False
        
        # Check SLA enforcement
        if "sla_enforcement_present" in expected:
            sla_nodes = [n for n in spec.nodes if hasattr(n, 'sla') and n.sla and 
                        hasattr(n.sla, 'enforce_usage') and n.sla.enforce_usage]
            actual = len(sla_nodes) > 0
            expected_val = expected["sla_enforcement_present"]
            passed = actual == expected_val
            print(f"   {'✅' if passed else '❌'} sla_enforcement_present: expected={expected_val}, actual={actual} (found {len(sla_nodes)})")
            if not passed: all_passed = False
        
        return all_passed
    
    def _validate_execution_result(self, result: dict, expected: dict) -> bool:
        """Validate execution results against expected."""
        all_passed = True
        
        # Basic execution validation
        if "workflow_executed" in expected:
            actual = result["success"]
            expected_val = expected["workflow_executed"]
            passed = actual == expected_val
            print(f"   {'✅' if passed else '❌'} workflow_executed: expected={expected_val}, actual={actual}")
            if not passed: all_passed = False
        
        # Check routing efficiency
        if "routing_efficiency" in expected:
            stats = result.get("execution_stats", {})
            actual_efficiency = stats.get("execution_efficiency", 0)
            expected_str = expected["routing_efficiency"]
            
            if expected_str.startswith(">="):
                threshold = float(expected_str[2:])
                passed = actual_efficiency >= threshold
            else:
                threshold = float(expected_str)
                passed = actual_efficiency == threshold
            
            print(f"   {'✅' if passed else '❌'} routing_efficiency: expected={expected_str}, actual={actual_efficiency}")
            if not passed: all_passed = False
        
        return all_passed
    
    async def run_test(self, test: WorkflowTestCase) -> bool:
        """Run a single test using the appropriate executor."""
        self.results["total"] += 1
        
        try:
            # Fix test type classification based on test structure
            test_type = test.test_type
            
            # Auto-correct misclassified tests in agentic layer
            if test.layer == "agentic":
                if test.user_prompt and not test.workflow_spec:
                    # This is workflow generation testing - should be agentic_generation
                    if test_type != "agentic_generation":
                        print(f"   🔧 Correcting test_type: '{test_type}' → 'agentic_generation' (has user_prompt)")
                        test_type = "agentic_generation"
                elif test.workflow_spec:
                    # This is pre-built workflow validation - should be workflow_validation
                    if test_type != "workflow_validation":
                        print(f"   🔧 Correcting test_type: '{test_type}' → 'workflow_validation' (has workflow_spec)")
                        test_type = "workflow_validation"
                elif not test_type:
                    # No test_type specified - infer from structure
                    test_type = "agentic_generation"  # default for agentic layer
                    print("   🔍 Defaulting to: agentic_generation")
            
            # Get test executor based on test type
            executor = get_test_executor(test_type)
            if not executor:
                print(f"   ⚠️ No executor found for test type: {test_type}")
                print(f"   Available types: {list(list_test_types().keys())}")
                success = False
            else:
                # Execute test using the appropriate executor
                context = {
                    'tool_catalog': self.tool_catalog,
                    'layer': test.layer
                }
                
                print(f"   🔧 Using executor: {executor.__class__.__name__}")
                result = await executor.execute_test(test, context)
                success = result.get('success', False)
                
                # Print additional result info
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                if 'result' in result and isinstance(result['result'], dict):
                    for key, value in result['result'].items():
                        if isinstance(value, bool):
                            icon = "✅" if value else "❌"
                            print(f"   {icon} {key}: {value}")
            
            if success:
                self.results["passed"] += 1
                print("   🎉 TEST PASSED")
            else:
                self.results["failed"] += 1
                print("   ❌ TEST FAILED")
            
            return success
            
        except Exception as e:
            self.results["errors"] += 1
            print(f"   💥 TEST ERROR: {e}")
            return False
    
    async def run_tests(self, 
                       layer: Optional[str] = None,
                       category: Optional[str] = None, 
                       tags: Optional[List[str]] = None,
                       test_numbers: Optional[List[int]] = None) -> dict:
        """Run tests based on filters."""
        
        print("🚀 UNIFIED TEST RUNNER")
        print("=" * 50)
        
        await self.setup()
        
        # Get tests based on filters
        if tags:
            tests = self.repo.get_tests_by_tags(tags)
            filter_desc = f"tags: {tags}"
        elif category:
            tests = self.repo.get_tests_by_category(category)
            filter_desc = f"category: {category}"
        elif layer:
            layer_enum = TestLayer(layer)
            tests = self.repo.get_tests_by_layer(layer_enum)
            filter_desc = f"layer: {layer}"
        else:
            # Get all tests
            tests = []
            for test_layer in TestLayer:
                tests.extend(self.repo.get_tests_by_layer(test_layer))
            filter_desc = "all tests"
        
        print(f"📊 Running {len(tests)} tests ({filter_desc})")
        
        if not tests:
            print("⚠️ No tests found matching criteria")
            return self.results
        
        # Filter by test numbers if specified
        if test_numbers:
            original_count = len(tests)
            # Convert 1-based indices to 0-based and filter
            filtered_tests = []
            for test_num in test_numbers:
                if 1 <= test_num <= len(tests):
                    filtered_tests.append((test_num, tests[test_num - 1]))
                else:
                    print(f"⚠️ Test number {test_num} is out of range (1-{len(tests)})")
            
            if not filtered_tests:
                print("⚠️ No valid test numbers specified")
                return self.results
                
            print(f"🎯 Running {len(filtered_tests)} specific tests from {original_count} total")
            
            # Run specific tests
            for test_num, test in filtered_tests:
                print(f"\n{'='*60}")
                print(f"🧪 TEST {test_num}/{original_count}")
                await self.run_test(test)
        else:
            # Run all tests
            for i, test in enumerate(tests, 1):
                print(f"\n{'='*60}")
                print(f"🧪 TEST {i}/{len(tests)}")
                await self.run_test(test)
        
        # Show summary
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"   Total:  {self.results['total']}")
        print(f"   ✅ Passed: {self.results['passed']}")
        print(f"   ❌ Failed: {self.results['failed']}")
        print(f"   💥 Errors: {self.results['errors']}")
        
        success_rate = (self.results['passed'] / self.results['total']) * 100 if self.results['total'] > 0 else 0
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        
        if self.results['failed'] == 0 and self.results['errors'] == 0:
            print("\n🎉 ALL TESTS PASSED! The stack is working perfectly!")
        else:
            print("\n⚠️ Some tests failed. Check output above for details.")
        
        return self.results

async def main():
    # Set testing mode to suppress non-test output
    os.environ["TESTING_MODE"] = "true"
    
    parser = argparse.ArgumentParser(description="Run unified tests for the whole fucking stack")
    parser.add_argument("--layer", choices=["logical", "agentic", "orchestration", "feedback"],
                       help="Run tests from specific layer only")
    parser.add_argument("--category", help="Run tests from specific category only")
    parser.add_argument("--tags", nargs="+", help="Run tests with specific tags only")
    parser.add_argument("-n", "--test-numbers", type=int, nargs="+", help="Run specific test numbers (1-based index), e.g. -n 5 12 19")
    
    args = parser.parse_args()
    
    runner = UnifiedTestRunner()
    results = await runner.run_tests(
        layer=args.layer,
        category=args.category,
        tags=args.tags,
        test_numbers=args.test_numbers
    )
    
    # Exit with appropriate code
    exit_code = 0 if results["failed"] == 0 and results["errors"] == 0 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())