from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import numpy as np

class CriticFeedback(BaseModel):
    """Feedback from the critic on agent performance"""
    score: float  # 0.0 to 1.0
    feedback: str
    suggestions: List[str]
    metrics: Dict[str, float]

class Critic:
    """Evaluates agent performance and provides feedback"""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, float]] = []
    
    def evaluate_performance(
        self,
        task: Any,
        agent_actions: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        final_response: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> CriticFeedback:
        """Evaluate agent performance on a task"""
        
        # Calculate various metrics
        metrics = {
            "tool_usage_efficiency": self._calculate_tool_efficiency(agent_actions),
            "response_accuracy": self._calculate_response_accuracy(final_response, ground_truth),
            "action_relevance": self._calculate_action_relevance(agent_actions, task),
            "response_completeness": self._calculate_completeness(final_response, ground_truth)
        }
        
        # Calculate overall score
        score = np.mean(list(metrics.values()))
        
        # Generate feedback
        feedback = self._generate_feedback(metrics)
        suggestions = self._generate_suggestions(metrics)
        
        # Store metrics for historical analysis
        self.metrics_history.append(metrics)
        
        return CriticFeedback(
            score=score,
            feedback=feedback,
            suggestions=suggestions,
            metrics=metrics
        )
    
    def _calculate_tool_efficiency(self, actions: List[Dict[str, Any]]) -> float:
        """Calculate how efficiently tools were used"""
        if not actions:
            return 0.0
        
        # Count unique tools used
        unique_tools = len(set(a.get("tool_name") for a in actions))
        total_actions = len(actions)
        
        # Efficiency decreases with redundant tool usage
        return min(1.0, unique_tools / total_actions)
    
    def _calculate_response_accuracy(
        self,
        response: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """Calculate how accurate the response is compared to ground truth"""
        if not response or not ground_truth:
            return 0.0
        
        # Count matching key-value pairs
        matches = 0
        total = len(ground_truth)
        
        for key, value in ground_truth.items():
            if key in response and response[key] == value:
                matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def _calculate_action_relevance(
        self,
        actions: List[Dict[str, Any]],
        task: Any
    ) -> float:
        """Calculate how relevant the actions were to the task"""
        if not actions:
            return 0.0
        
        # Simple relevance score based on task description keywords
        task_keywords = set(task.description.lower().split())
        relevant_actions = 0
        
        for action in actions:
            action_str = str(action).lower()
            if any(keyword in action_str for keyword in task_keywords):
                relevant_actions += 1
        
        return relevant_actions / len(actions)
    
    def _calculate_completeness(
        self,
        response: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """Calculate how complete the response is"""
        if not response or not ground_truth:
            return 0.0
        
        # Count how many ground truth fields are present in response
        present_fields = sum(1 for key in ground_truth if key in response)
        return present_fields / len(ground_truth)
    
    def _generate_feedback(self, metrics: Dict[str, float]) -> str:
        """Generate feedback based on metrics"""
        feedback_parts = []
        
        if metrics["tool_usage_efficiency"] < 0.5:
            feedback_parts.append("The agent used tools inefficiently, with many redundant calls.")
        elif metrics["tool_usage_efficiency"] > 0.8:
            feedback_parts.append("The agent used tools very efficiently.")
        
        if metrics["response_accuracy"] < 0.5:
            feedback_parts.append("The response was not very accurate compared to the expected result.")
        elif metrics["response_accuracy"] > 0.8:
            feedback_parts.append("The response was highly accurate.")
        
        if metrics["action_relevance"] < 0.5:
            feedback_parts.append("Many actions were not relevant to the task.")
        elif metrics["action_relevance"] > 0.8:
            feedback_parts.append("Actions were highly relevant to the task.")
        
        if metrics["response_completeness"] < 0.5:
            feedback_parts.append("The response was incomplete, missing important information.")
        elif metrics["response_completeness"] > 0.8:
            feedback_parts.append("The response was very complete.")
        
        return " ".join(feedback_parts) if feedback_parts else "No specific feedback available."
    
    def _generate_suggestions(self, metrics: Dict[str, float]) -> List[str]:
        """Generate improvement suggestions based on metrics"""
        suggestions = []
        
        if metrics["tool_usage_efficiency"] < 0.7:
            suggestions.append("Try to minimize redundant tool usage and plan actions more carefully.")
        
        if metrics["response_accuracy"] < 0.7:
            suggestions.append("Double-check responses against expected results before submitting.")
        
        if metrics["action_relevance"] < 0.7:
            suggestions.append("Focus on actions that directly contribute to solving the task.")
        
        if metrics["response_completeness"] < 0.7:
            suggestions.append("Ensure all required information is included in the response.")
        
        return suggestions
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Get historical performance trends"""
        if not self.metrics_history:
            return {}
        
        trends = {}
        for metric in self.metrics_history[0].keys():
            trends[metric] = [m[metric] for m in self.metrics_history]
        
        return trends 