#!/usr/bin/env python3
"""
Test the frontend execution feedback integration.

This script will:
1. Start the workflow server
2. Create a workflow that will generate execution feedback
3. Execute the workflow
4. Verify that execution feedback is displayed in the UI
"""

import asyncio
import aiohttp
from uuid import uuid4

# Test workflow that will execute and generate feedback
TEST_WORKFLOW = {
    "user_message": "Create a simple workflow to analyze Bitcoin price and send results via email",
    "current_workflow": None
}

async def test_frontend_feedback():
    """Test the complete frontend feedback integration."""
    
    print("🧪 Testing Frontend Execution Feedback Integration")
    print("=" * 50)
    
    # Server URL
    base_url = "http://localhost:8080"
    
    try:
        async with aiohttp.ClientSession() as session:
            # 1. Create workflow
            print("\n1️⃣ Creating test workflow...")
            async with session.post(
                f"{base_url}/api/process", 
                json=TEST_WORKFLOW
            ) as resp:
                result = await resp.json()
                workflow_spec = result.get("workflow_spec")
                
                if not workflow_spec:
                    print("❌ Failed to create workflow")
                    return
                
                print(f"✅ Created workflow: {workflow_spec.get('title', 'Untitled')}")
                print(f"   Nodes: {len(workflow_spec.get('nodes', []))}")
                print(f"   Edges: {len(workflow_spec.get('edges', []))}")
            
            # 2. Execute the workflow
            print("\n2️⃣ Executing workflow...")
            execution_id = f"test-feedback-{uuid4()}"
            
            execute_payload = {
                "workflow_spec": workflow_spec,
                "execution_id": execution_id,
                "user_inputs": {
                    "analysis_type": "technical",
                    "email_recipient": "test@example.com"
                }
            }
            
            async with session.post(
                f"{base_url}/api/execute",
                json=execute_payload
            ) as resp:
                if resp.status != 200:
                    print(f"❌ Execution failed with status {resp.status}")
                    return
                
                print("✅ Workflow execution started")
                print(f"   Execution ID: {execution_id}")
            
            # 3. Wait for execution to complete
            print("\n3️⃣ Waiting for execution to complete...")
            await asyncio.sleep(5)  # Give it time to execute
            
            # 4. Check for execution feedback
            print("\n4️⃣ Checking for execution feedback...")
            async with session.get(
                f"{base_url}/api/execution/{execution_id}/feedback"
            ) as resp:
                if resp.status == 200:
                    feedback = await resp.json()
                    print("✅ Execution feedback received!")
                    print("\n📋 FEEDBACK SUMMARY:")
                    print("-" * 30)
                    
                    if "curated_summary" in feedback:
                        summary_preview = feedback["curated_summary"][:500]
                        print(summary_preview + "..." if len(feedback["curated_summary"]) > 500 else summary_preview)
                    
                    if "ai_analysis" in feedback:
                        print("\n🤖 AI ANALYSIS:")
                        print("-" * 30)
                        analysis_preview = feedback["ai_analysis"][:500]
                        print(analysis_preview + "..." if len(feedback["ai_analysis"]) > 500 else analysis_preview)
                else:
                    print(f"⚠️ No feedback available yet (status: {resp.status})")
            
            # 5. Instructions for manual verification
            print("\n5️⃣ Manual Verification Steps:")
            print("-" * 30)
            print("1. Open http://localhost:8080 in your browser")
            print("2. Look for the execution feedback in the chat interface")
            print("3. You should see:")
            print("   - A system message with the execution report")
            print("   - Formatted execution details with status indicators")
            print("   - AI analysis and improvement suggestions")
            print("4. The feedback should appear automatically after workflow execution")
            
            print("\n✅ Frontend feedback integration test completed!")
            print("🎯 The system should now display execution feedback in the UI")
            
    except aiohttp.ClientError as e:
        print(f"\n❌ Error connecting to server: {e}")
        print("Make sure the workflow server is running:")
        print("  cd iointel && uv run python -m src.web.workflow_server")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


async def simulate_feedback_display():
    """Simulate what the frontend should display."""
    
    print("\n" + "=" * 50)
    print("📱 SIMULATED FRONTEND DISPLAY")
    print("=" * 50)
    
    # Simulate execution report
    execution_report = """
🤖 SYSTEM EXECUTION REPORT
=============================
Workflow: Bitcoin Price Analysis and Email Notification
Execution ID: test-feedback-123
Status: PARTIAL
Duration: 3.45s
Timestamp: 2025-01-19T10:45:23

📊 EXECUTION OVERVIEW
--------------------
Total Nodes: 3
Executed: 2
Skipped: 1
Success Rate: 50.0%

🔍 NODE EXECUTION DETAILS
-------------------------
✅ Get Bitcoin Price (tool) (0.85s)
   Tools: get_coin_quotes
   Result: BTC: $67,234.56 (+2.3% 24h)
❌ Email Notification Agent (agent) (1.20s)
   Tools: send_email
   Error: KeyError: 'self' - Agent initialization failed
⏭️ Analysis Summary Agent (agent)
   Skipped due to upstream failure

🚨 ERROR ANALYSIS
----------------
Email agent failed due to type hint issue in tool registration
"""
    
    print("💬 System Message (appears in chat):")
    print("-" * 40)
    print(execution_report)
    
    # Simulate AI analysis
    ai_analysis = """
🤖 Execution Analysis Summary

I've analyzed your Bitcoin price analysis workflow execution:

**What Happened:**
- Successfully retrieved Bitcoin price data ($67,234.56)
- Failed to send email notification due to technical issue
- Analysis summary was skipped as a result

**Issues Identified:**
1. **Critical Bug**: The 'self' parameter in type hints prevents email agent initialization
2. **Workflow Brittleness**: Linear dependency caused cascade failure

**Improvement Suggestions:**
1. Fix the type hint issue in the email agent factory
2. Add error recovery mechanisms
3. Consider parallel execution where possible
4. Implement fallback notification methods

Would you like me to help fix these issues?
"""
    
    print("\n🤖 AI Analysis (appears after report):")
    print("-" * 40)
    print(ai_analysis)
    
    print("\n📱 This is how it should appear in the web UI with proper formatting!")


if __name__ == "__main__":
    print("Starting frontend feedback integration test...")
    asyncio.run(test_frontend_feedback())
    asyncio.run(simulate_feedback_display())