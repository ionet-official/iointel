#!/usr/bin/env python3
"""
Example client for the RL Model Evaluation API
"""

import requests
import time
import json
import os
from typing import Optional
from dotenv import load_dotenv


class RLEvaluationClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def health_check(self):
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_models(self):
        """Get recommended models"""
        response = requests.get(f"{self.base_url}/models", headers=self.headers)
        return response.json()
    
    def evaluate_sync(self, models: list, num_tasks: int = 3, timeout: int = 120, api_key: str = None, base_url: str = None):
        """Run synchronous evaluation"""
        payload = {
            "models": models,
            "num_tasks": num_tasks,
            "timeout": timeout
        }
        if api_key:
            payload["api_key"] = api_key
        if base_url:
            payload["base_url"] = base_url
            
        response = requests.post(
            f"{self.base_url}/evaluate",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def evaluate_async(self, models: list, num_tasks: int = 3, timeout: int = 120, api_key: str = None, base_url: str = None):
        """Start asynchronous evaluation"""
        payload = {
            "models": models,
            "num_tasks": num_tasks,
            "timeout": timeout
        }
        if api_key:
            payload["api_key"] = api_key
        if base_url:
            payload["base_url"] = base_url
            
        response = requests.post(
            f"{self.base_url}/evaluate/async",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_status(self, task_id: str):
        """Get evaluation status"""
        response = requests.get(
            f"{self.base_url}/evaluate/{task_id}/status",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_results(self, task_id: str):
        """Get evaluation results"""
        response = requests.get(
            f"{self.base_url}/evaluate/{task_id}/results",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, task_id: str, poll_interval: int = 5, max_wait: int = 600):
        """Wait for evaluation to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = self.get_status(task_id)
            print(f"Status: {status['status']}, Models: {status['models_completed']}/{status['total_models']}")
            
            if status["status"] in ["completed", "failed"]:
                return status
            
            if status.get("current_model"):
                print(f"Currently evaluating: {status['current_model']}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Evaluation did not complete within {max_wait} seconds")


def main():
    """Example usage of the API client"""
    
    # Load environment variables from creds.env
    load_dotenv("creds.env")
    
    # Get IO API credentials from environment
    io_api_key = os.getenv("IO_API_KEY")
    io_base_url = os.getenv("IO_BASE_URL")
    
    if not io_api_key or not io_base_url:
        print("Warning: IO_API_KEY and/or IO_BASE_URL not found in environment variables")
        print("The evaluation will fail unless these are provided")
        print("\nYou can set them by adding to creds.env:")
        print("IO_API_KEY=your-api-key-here")
        print("IO_BASE_URL=https://api.intelligence-dev.io.solutions/api/v1")
    
    # Initialize client
    client = RLEvaluationClient()
    
    # Check health
    print("\n=== Health Check ===")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Get available models
    print("\n=== Available Models ===")
    models_info = client.get_models()
    print(f"Recommended models: {models_info['recommended_models'][:3]}")
    
    # Example 1: Synchronous evaluation (small, quick)
    print("\n=== Synchronous Evaluation ===")
    try:
        result = client.evaluate_sync(
            models=["microsoft/phi-4"],
            num_tasks=2,
            timeout=60,
            api_key=io_api_key,
            base_url=io_base_url
        )
        print(f"Completed! Total tasks: {result['total_tasks']}")
        print(f"Summary: {json.dumps(result['summary']['models'], indent=2)}")
    except Exception as e:
        print(f"Sync evaluation failed: {e}")
    
    # Example 2: Asynchronous evaluation (multiple models)
    print("\n=== Asynchronous Evaluation ===")
    try:
        # Start evaluation
        task = client.evaluate_async(
            models=["microsoft/phi-4", "meta-llama/Llama-3.3-70B-Instruct"],
            num_tasks=3,
            timeout=120,
            api_key=io_api_key,
            base_url=io_base_url
        )
        print(f"Started evaluation task: {task['task_id']}")
        
        # Wait for completion
        final_status = client.wait_for_completion(task['task_id'])
        
        if final_status["status"] == "completed":
            # Get results
            results = client.get_results(task['task_id'])
            print("\nEvaluation completed!")
            print(f"Total models: {results['total_models']}")
            print(f"Total tasks: {results['total_tasks']}")
            print(f"Success rate: {results['summary']['successful_evaluations']}/{results['summary']['total_evaluations']}")
            
            # Print model-specific results
            for model, stats in results['summary']['models'].items():
                print(f"\n{model}:")
                print(f"  - Tasks: {stats['total_tasks']}")
                print(f"  - Success: {stats['successful']}")
                print(f"  - Oracle accuracy: {stats['oracle_accuracy']:.2%}")
                print(f"  - Avg execution time: {stats['average_execution_time']:.2f}s")
        else:
            print(f"Evaluation failed: {final_status.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Async evaluation failed: {e}")


if __name__ == "__main__":
    main()