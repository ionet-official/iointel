#!/usr/bin/env python3
"""
Quick Route Index Test Runner
============================

Run just the route index tests we added to verify the critical routing bug fix.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from run_unified_tests import UnifiedTestRunner

async def test_route_index():
    """Run only the route index tests."""
    
    print("🎯 ROUTE INDEX TEST RUNNER")
    print("=" * 40)
    print("Testing the critical routing bug fix...")
    
    runner = UnifiedTestRunner()
    results = await runner.run_tests(tags=["route_index"])
    
    if results["failed"] == 0 and results["errors"] == 0:
        print("\n🎉 ROUTE INDEX TESTS PASSED!")
        print("   The routing bug fix is working correctly!")
    else:
        print("\n❌ ROUTE INDEX TESTS FAILED!")
        print("   There are issues with the routing system!")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_route_index())