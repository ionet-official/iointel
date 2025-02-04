import requests
import os
from typing import Optional, List

BASE_URL=os.getenv("BASE_URL")
BASE_MCP_URL=os.getenv("BASE_MCP_URL")


def schedule_task(task: str) -> dict:
    raise NotImplementedError()
    # payload = {"task": task}
    # response = requests.post(f"{BASE_URL}/api/v1/agents/schedule", json=payload)
    # response.raise_for_status()
    # return response.json()

def run_council_task(task: str) -> dict:
    raise NotImplementedError()
    # response = requests.post(f"{BASE_URL}/api/v1/workflows/run", json=payload)
    # response.raise_for_status()
    # return response.json()

def run_reasoning_task(text: str) -> dict:
    payload = {
        "text": text,
        "agent_names": ["reasoning_agent"],
        "args": {
            "type": "solve_with_reasoning"
        }
    }
    response = requests.post(f"{BASE_URL}/api/v1/workflows/run", json=payload)
    response.raise_for_status()
    return response.json()

def summarize_task(text: str) -> dict:
    payload = {
        "text": text,
        "agent_names": ["summary_agent"],
        "args": {
            "type": "summarize_text"
        }
    }
    response = requests.post(f"{BASE_URL}/api/v1/workflows/run", json=payload)
    response.raise_for_status()
    return response.json()

def sentiment_analysis(text: str) -> dict:
    payload = {
        "text": text,
        "agent_names": ["sentiment_analysis_agent"],
        "args": {
            "type": "sentiment"
        }
    }
    response = requests.post(f"{BASE_URL}/api/v1/workflows/run", json=payload)
    response.raise_for_status()
    return response.json()

def extract_entities(text: str) -> dict:
    payload = {
        "text": text,
        "agent_names": ["extractor"],
        "args": {
            "type": "extract_categorized_entities"
        }
    }
    response = requests.post(f"{BASE_URL}/api/v1/workflows/run", json=payload)
    response.raise_for_status()
    return response.json()

def translate_text_task(text: str, target_language: str) -> dict:
    payload = {
        "text": text,
        "agent_names": ["translation_agent"],
        "args": {
            "type": "translate_text",
            "target_language": target_language
        }
    }
    response = requests.post(f"{BASE_URL}/api/v1/workflows/run", json=payload)
    response.raise_for_status()
    return response.json()

def classify_text(text: str, classify_by: list[str]) -> dict:
    # Note: The endpoint just uses req.text right now, if classify_by is needed, update payload accordingly.
    payload = {
        "text": text,
        "agent_names": ["classification_agent"],
        "args": {
            "type": "classify",
            "classify_by": classify_by
        }
    }
    response = requests.post(f"{BASE_URL}/api/v1/workflows/run", json=payload)
    response.raise_for_status()
    return response.json()

def moderation_task(text: str, threshold: float = 0.5) -> dict:
    payload = {
        "text": text,
        "agent_names": ["moderation_agent"],
        "args": {
            "type": "moderation",
            "threshold": threshold
        }
    }
    response = requests.post(f"{BASE_URL}/api/v1/workflows/run", json=payload)
    response.raise_for_status()
    return response.json()

def custom_workflow(
        text: str,
        name: str, 
        objective: str, 
        instructions: str = "", 
        agents: Optional[List[str]] = None,
        context: Optional[dict] = None
        ) -> dict:

    payload = {
        "text": text,
        "agent_names": agents or ["custom_agent"],
        "args": {
            "type": "custom",
            "name": name,
            "objective": objective,
            "instructions": instructions,
            "context": context or {},
        }
    }
    response = requests.post(f"{BASE_URL}/api/v1/workflows/run", json=payload)
    response.raise_for_status()
    return response.json()

def get_tools() -> dict:
    response = requests.get(f"{BASE_MCP_URL}/mcp/tools")
    response.raise_for_status()
    return response.json()

def get_servers() -> dict:
    response = requests.get(f"{BASE_MCP_URL}/mcp/servers")
    response.raise_for_status()
    return response.json()

def get_agents() -> dict:
    response = requests.get(f"{BASE_URL}/api/v1/agents/get-available-agents")
    response.raise_for_status()
    return response.json()

def upload_workflow_file(file_path: str) -> dict:
    """
    Uploads a workflow file to the server.
    This file may contain either JSON or YAML content, which the server
    will parse and validate as a WorkflowDefinition.

    :param file_path: Local path to the file to upload.
    :return: JSON response from the server as a dict.
    :raises: HTTPError if the request fails.
    """

    with open(file_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/api/v1/workflows/run-file",
            files={"yaml_file": 
                   (os.path.basename(file_path), 
                    f, 
                    "application/octet-stream")}
        )
    response.raise_for_status()
    return response.json()
