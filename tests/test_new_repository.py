"""
Test the new unified repository
"""

import json
from pathlib import Path
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.utilities.tool_registry_utils import create_tool_catalog

def test_new_repository():
    """Test our new test cases."""
    
    # Load the new repository
    repo_path = Path("unified_test_repository/tests.json")
    with open(repo_path) as f:
        repository = json.load(f)
    
    tests = repository["tests"]
    print(f"📋 Loaded {len(tests)} tests from unified repository\n")
    
    # Get tool catalog for validation
    tool_catalog = create_tool_catalog()
    
    # Run each test
    passed = 0
    failed = 0
    
    for i, test_wrapper in enumerate(tests, 1):
        test = test_wrapper["data"]
        print(f"Test {i}/{len(tests)}: {test['name']}")
        print(f"  Category: {test['category']}")
        print(f"  Should Pass: {test['should_pass']}")
        
        try:
            # Create WorkflowSpec from test data
            spec = WorkflowSpec(**test["workflow_spec"])
            
            # Validate structure
            issues = spec.validate_structure(tool_catalog)
            
            if test["should_pass"]:
                if len(issues) == 0:
                    print("  ✅ PASSED - Validation succeeded as expected")
                    passed += 1
                else:
                    print("  ❌ FAILED - Expected to pass but got issues:")
                    for issue in issues:
                        print(f"     - {issue}")
                    failed += 1
            else:
                # Test should fail
                if len(issues) > 0:
                    print("  ✅ PASSED - Validation failed as expected")
                    print(f"     Issues: {issues[0][:50]}...")
                    passed += 1
                else:
                    print("  ❌ FAILED - Expected to fail but passed validation")
                    failed += 1
                    
        except Exception as e:
            if test["should_pass"]:
                print(f"  ❌ FAILED - Exception: {str(e)[:100]}")
                failed += 1
            else:
                print(f"  ✅ PASSED - Failed with exception as expected: {str(e)[:50]}...")
                passed += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print(f"   Total:  {len(tests)}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/len(tests)*100:.1f}%")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = test_new_repository()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed")