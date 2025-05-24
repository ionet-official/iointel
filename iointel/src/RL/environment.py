from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel
from .oracle import Oracle, EvaluationResult
from .tools.base import BaseTool, ToolResult

class State(BaseModel):
    """Current state of the environment"""
    task_description: str
    available_tools: List[str]
    tool_results: List[ToolResult]
    agent_response: Optional[Any] = None
    step_count: int = 0
    context: Dict[str, Any] = {}

class Action(BaseModel):
    """Agent's action in the environment"""
    tool_name: str
    tool_params: Dict[str, Any]
    reasoning: str

class Reward(BaseModel):
    """Reward signal for the agent"""
    value: float
    feedback: str
    details: Dict[str, Any] = {}

class RLEnvironment:
    """Reinforcement learning environment for goal-seeking agents"""
    
    def __init__(
        self,
        oracle: Oracle,
        tools: List[BaseTool],
        max_steps: int = 10,
        reward_weights: Dict[str, float] = None
    ):
        self.oracle = oracle
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.reward_weights = reward_weights or {
            "correctness": 0.6,
            "tool_usage": 0.3,
            "efficiency": 0.1
        }
        self.current_state = None
        self.ground_truth = None
        self.expected_tools = None
    
    def reset(
        self,
        task_description: str,
        ground_truth: Any,
        expected_tools: List[str],
        context: Dict[str, Any] = None
    ) -> State:
        """Reset the environment for a new task"""
        self.ground_truth = ground_truth
        self.expected_tools = expected_tools
        
        self.current_state = State(
            task_description=task_description,
            available_tools=list(self.tools.keys()),
            tool_results=[],
            context=context or {}
        )
        
        return self.current_state
    
    async def step(self, action: Action) -> Tuple[State, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: The agent's action to take
            
        Returns:
            Tuple of (new_state, reward, done, info)
        """
        if self.current_state.step_count >= self.max_steps:
            return self.current_state, Reward(value=0.0, feedback="Max steps reached"), True, {}
        
        # Execute the tool
        if action.tool_name not in self.tools:
            return (
                self.current_state,
                Reward(value=-1.0, feedback=f"Invalid tool: {action.tool_name}"),
                True,
                {"error": "Invalid tool"}
            )
        
        tool = self.tools[action.tool_name]
        if not tool.validate_params(**action.tool_params):
            return (
                self.current_state,
                Reward(value=-1.0, feedback="Invalid tool parameters"),
                True,
                {"error": "Invalid parameters"}
            )
        
        # Execute tool and update state
        result = await tool.execute(**action.tool_params)
        result.metadata["tool_name"] = action.tool_name
        result.metadata["reasoning"] = action.reasoning
        
        self.current_state.tool_results.append(result)
        self.current_state.step_count += 1
        
        # Calculate reward
        reward = await self._calculate_reward()
        
        # Check if done
        done = (
            self.current_state.step_count >= self.max_steps or
            reward.value >= 0.95  # Success threshold
        )
        
        return self.current_state, reward, done, {
            "tool_result": result,
            "step_count": self.current_state.step_count
        }
    
    async def _calculate_reward(self) -> Reward:
        """Calculate reward based on current state"""
        if not self.current_state.agent_response:
            return Reward(
                value=0.0,
                feedback="No agent response yet",
                details={"step": self.current_state.step_count}
            )
        
        # Evaluate response correctness
        response_eval = await self.oracle.evaluate(
            self.current_state.agent_response,
            self.ground_truth,
            self.current_state.task_description,
            self.current_state.context
        )
        
        # Evaluate tool usage
        tool_eval = await self.oracle.evaluate_tool_usage(
            self.current_state.tool_results,
            self.expected_tools,
            self.current_state.task_description
        )
        
        # Calculate efficiency (penalize for extra steps)
        efficiency = 1.0 - (self.current_state.step_count / self.max_steps)
        
        # Combine rewards
        total_reward = (
            self.reward_weights["correctness"] * response_eval.score +
            self.reward_weights["tool_usage"] * tool_eval.score +
            self.reward_weights["efficiency"] * efficiency
        )
        
        return Reward(
            value=total_reward,
            feedback=f"Response: {response_eval.feedback}, Tools: {tool_eval.feedback}",
            details={
                "response_eval": response_eval.details,
                "tool_eval": tool_eval.details,
                "efficiency": efficiency
            }
        )
    
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get list of available actions for the agent"""
        actions = []
        for tool_name, tool in self.tools.items():
            actions.append({
                "tool_name": tool_name,
                "description": tool.get_description(),
                "schema": tool.get_schema()
            })
        return actions 