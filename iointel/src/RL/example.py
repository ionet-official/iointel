import asyncio
from typing import Dict, Any
from .environment import RLEnvironment, State, Action
from .oracle_agent import OracleAgent
from .task_manager import TaskManager
from .critic import Critic
from .tools.splunk import SplunkTool

async def main():
    # Initialize components
    oracle = OracleAgent(model="gpt-4", temperature=0.0)
    task_manager = TaskManager()
    critic = Critic()
    
    # Create tools
    splunk_tool = SplunkTool({
        "base_url": "https://splunk.example.com",
        "api_key": "your-api-key"
    })
    tools = [splunk_tool]
    
    # Create environment
    env = RLEnvironment(
        oracle=oracle,
        tools=tools,
        max_steps=10
    )
    
    # Get a task
    task = task_manager.get_task(difficulty=0.5)
    print(f"Task: {task.description}")
    
    # Reset environment for new task
    state = await env.reset(
        task_description=task.description,
        ground_truth=task.ground_truth,
        expected_tools=task.required_tools
    )
    
    # Run episode
    done = False
    total_reward = 0
    actions_taken = []
    
    while not done:
        # Agent decides action (simplified for example)
        action = Action(
            tool_name="splunk_query",
            parameters={
                "query": "source_ip=192.168.1.100",
                "earliest_time": "-1h",
                "latest_time": "now"
            },
            reasoning="Querying Splunk for recent activity from the suspicious IP"
        )
        
        # Execute action
        next_state, reward, done, info = await env.step(action)
        
        # Store results
        total_reward += reward.value
        actions_taken.append({
            "action": action.dict(),
            "reward": reward.value,
            "feedback": reward.feedback
        })
        
        # Update state
        state = next_state
        
        # Print progress
        print(f"\nAction: {action.tool_name}")
        print(f"Parameters: {action.parameters}")
        print(f"Reward: {reward.value}")
        print(f"Feedback: {reward.feedback}")
    
    # Get final evaluation
    evaluation = critic.evaluate_performance(
        task=task,
        agent_actions=actions_taken,
        tool_results=[],  # Would be populated with actual tool results
        final_response=state.agent_response,
        ground_truth=task.ground_truth
    )
    
    # Print results
    print("\n=== Episode Complete ===")
    print(f"Total Reward: {total_reward}")
    print(f"Final Score: {evaluation.score}")
    print(f"Feedback: {evaluation.feedback}")
    print("\nSuggestions for improvement:")
    for suggestion in evaluation.suggestions:
        print(f"- {suggestion}")
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for metric, value in evaluation.metrics.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    asyncio.run(main()) 