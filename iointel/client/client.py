import requests
import os
import time
from typing import Optional, List
from functools import partial

BASE_URL = os.getenv("OPENAI_API_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_MCP_URL = os.getenv("BASE_MCP_URL")
try:
    SLOW_MODE_SLEEP = int(os.getenv("SLOW_MODE_SLEEP"))
except (ValueError, TypeError):
    SLOW_MODE_SLEEP = -1


def __make_api_call(method, **kwargs) -> dict:
    start = time.time()
    url = kwargs.pop("url", f"{BASE_URL}/workflows/run")
    headers = kwargs.pop("headers", {})
    if "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {API_KEY}"
    response = requests.request(method, url, headers=headers, **kwargs)
    response.raise_for_status()
    result = response.json()
    if (remain := SLOW_MODE_SLEEP - (time.time() - start)) > 0:
        # HACK avoid triggering rate limit protection if told so
        # by making sure each `__make_api_call()` takes at least `SLOW_MODE_SLEEP` seconds
        time.sleep(remain)
    return result


__make_post_call = partial(__make_api_call, method="POST")
__make_get_call = partial(__make_api_call, method="GET")


def schedule_task(task: str) -> dict:
    raise NotImplementedError()
    # payload = {"task": task}
    # return __make_post_call(json=payload)


def run_council_task(task: str) -> dict:
    raise NotImplementedError()
    # return __make_post_call(json=payload)


def run_reasoning_task(text: str) -> dict:
    payload = {
        "text": text,
        "agent_names": ["reasoning_agent"],
        "args": {"type": "solve_with_reasoning"},
    }
    return __make_post_call(json=payload)


def summarize_task(text: str, max_words: int = 100) -> dict:
    payload = {
        "text": text,
        "agent_names": ["summary_agent"],
        "args": {
            "type": "summarize_text",
            "max_words": max_words,
        },
    }
    return __make_post_call(json=payload)


def sentiment_analysis(text: str) -> dict:
    payload = {
        "text": text,
        "agent_names": ["sentiment_analysis_agent"],
        "args": {"type": "sentiment"},
    }
    return __make_post_call(json=payload)


def extract_entities(text: str) -> dict:
    payload = {
        "text": text,
        "agent_names": ["extractor"],
        "args": {"type": "extract_categorized_entities"},
    }
    return __make_post_call(json=payload)


def translate_text_task(text: str, target_language: str) -> dict:
    payload = {
        "text": text,
        "agent_names": ["translation_agent"],
        "args": {"type": "translate_text", "target_language": target_language},
    }
    return __make_post_call(json=payload)


def classify_text(text: str, classify_by: list[str]) -> dict:
    payload = {
        "text": text,
        "agent_names": ["classification_agent"],
        "args": {"type": "classify", "classify_by": classify_by},
    }
    return __make_post_call(json=payload)


def moderation_task(text: str, threshold: float = 0.5) -> dict:
    payload = {
        "text": text,
        "agent_names": ["moderation_agent"],
        "args": {"type": "moderation", "threshold": threshold},
    }
    return __make_post_call(json=payload)


def custom_workflow(
    text: str,
    name: str,
    objective: str,
    instructions: str = "",
    agents: Optional[List[str]] = None,
    context: Optional[dict] = None,
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
        },
    }
    return __make_post_call(json=payload)


def get_tools() -> dict:
    response = requests.get(f"{BASE_MCP_URL}/mcp/tools")
    response.raise_for_status()
    return response.json()


def get_servers() -> dict:
    response = requests.get(f"{BASE_MCP_URL}/mcp/servers")
    response.raise_for_status()
    return response.json()


def get_agents() -> dict:
    return __make_get_call(url=f"{BASE_URL}/agents/")


def upload_workflow_file(file_path: str) -> dict:
    """
    Uploads a workflow file to the server.
    This file may contain either JSON or YAML content, which the server
    will parse and validate as a WorkflowDefinition.

    :param file_path: Local path to the file to upload.
    :return: JSON response from the server as a dict.
    :raises: HTTPError if the request fails.
    """

    raise NotImplementedError()
    # with open(file_path, "rb") as f:
    #     return __make_post_call(
    #         url=f"{BASE_URL}/workflows/run-file",
    #         files={"yaml_file":
    #                (os.path.basename(file_path),
    #                 f,
    #                 "application/octet-stream")})
