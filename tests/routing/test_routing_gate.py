#!/usr/bin/env python
"""
Test the simplified routing_gate tool.
Shows that we can route without complex conditional JSON.
"""

import asyncio
from iointel.src.agent_methods.tools.conditional_gate import routing_gate, GateResult

def test_routing_gate():
    """Test that routing_gate works for simple routing decisions."""
    
    print("Testing routing_gate - simple routing without conditions")
    print("=" * 60)
    
    # Test 1: Route to System & Shell (index 4)
    result = routing_gate(
        data="pwd",
        route_index=4,
        route_name="System & Shell"
    )
    
    print(f"\nTest 1: Route 'pwd' command")
    print(f"  Input: 'pwd'")
    print(f"  Route Index: {result.route_index}")
    print(f"  Routed To: {result.routed_to}")
    print(f"  Action: {result.action}")
    print(f"  Decision: {result.decision_reason}")
    assert isinstance(result, GateResult)
    assert result.route_index == 4
    assert result.routed_to == "System & Shell"
    assert result.action == "branch"
    
    # Test 2: Route to Finance (index 5)
    result = routing_gate(
        data="TSLA stock price",
        route_index=5,
        route_name="Finance"
    )
    
    print(f"\nTest 2: Route stock query")
    print(f"  Input: 'TSLA stock price'")
    print(f"  Route Index: {result.route_index}")
    print(f"  Routed To: {result.routed_to}")
    print(f"  Action: {result.action}")
    print(f"  Decision: {result.decision_reason}")
    assert result.route_index == 5
    assert result.routed_to == "Finance"
    
    # Test 3: Route without name (auto-generate)
    result = routing_gate(
        data="calculate 2+2",
        route_index=3
    )
    
    print(f"\nTest 3: Route with auto-generated name")
    print(f"  Input: 'calculate 2+2'")
    print(f"  Route Index: {result.route_index}")
    print(f"  Routed To: {result.routed_to}")
    print(f"  Action: {result.action}")
    assert result.route_index == 3
    assert result.routed_to == "route_3"
    
    # Test 4: Terminate action
    result = routing_gate(
        data="stop",
        route_index=-1,
        route_name="terminate",
        action="terminate"
    )
    
    print(f"\nTest 4: Terminate routing")
    print(f"  Input: 'stop'")
    print(f"  Route Index: {result.route_index}")
    print(f"  Routed To: {result.routed_to}")
    print(f"  Action: {result.action}")
    assert result.route_index == -1
    assert result.action == "terminate"
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! routing_gate works for simple routing.")
    print("\nKey differences from conditional_gate:")
    print("  - No complex JSON conditions to evaluate")
    print("  - Agent directly specifies route_index")
    print("  - Much simpler and more predictable")
    print("  - Perfect for decision nodes that analyze input and pick a route")

if __name__ == "__main__":
    test_routing_gate()