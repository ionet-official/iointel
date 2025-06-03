import os
import asyncio
import csv
import json
from iointel.src.RL.task_manager import TaskManager
from iointel.src.RL.critic import CriticAgent
from iointel.src.RL.oracle import OracleAgent
from iointel.src.RL.training import RLEnvironment
from iointel import Agent
from iointel.src.RL.example_tools import (
    add,
    subtract,
    multiply,
    divide,
    get_weather,
    square_root,
)
from iointel.src.RL.utils import tool_usage_results_to_string

MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Qwen/Qwen3-235B-A22B-FP8",
    "deepseek-ai/DeepSeek-R1",
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "databricks/dbrx-instruct",
    "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
    "microsoft/phi-4",
    "nvidia/AceMath-7B-Instruct",
    "google/gemma-3-27b-it",
    "mistralai/Mistral-Large-Instruct-2411",
    "watt-ai/watt-tool-70B",
    "meta-llama/Llama-3.1-405B-Instruct-FP8",
    "SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B",
    "tiiuae/Falcon3-10B-Instruct",
    "bespokelabs/Bespoke-Stratos-32B",
    "netease-youdao/Confucius-o1-14B",
    "CohereForAI/aya-expanse-32b",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "NovaSky-AI/Sky-T1-32B-Preview",
    "THUDM/glm-4-9b-chat",
    "mistralai/Ministral-8B-Instruct-2410",
    "jinaai/ReaderLM-v2",
    "openbmb/MiniCPM3-4B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "ibm-granite/granite-3.1-8b-instruct",
    "ozone-ai/0x-lite",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "BAAI/bge-multilingual-gemma2",
    "mixedbread-ai/mxbai-embed-large-v1",
]

MODELS = [
    # "deepseek-ai/DeepSeek-R1-0528",
    # "Qwen/Qwen3-235B-A22B-FP8",
    # "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "meta-llama/Llama-3.3-70B-Instruct"
]

MODELS_THAT_REQUIRE_MODEL_SETTINGS = [
    "deepseek-ai/DeepSeek-R1-0528",
    "meta-llama/Llama-3.3-70B-Instruct",
]

REPORT_CSV = "rl_model_report.csv"


def linearize_tool_usage(tool_usage_results):
    if not tool_usage_results:
        return ""
    try:
        return tool_usage_results_to_string(tool_usage_results, prefix="")
    except Exception:
        return json.dumps(
            [
                tur.model_dump() if hasattr(tur, "model_dump") else dict(tur)
                for tur in tool_usage_results
            ]
        )


def linearize_dict(d):
    if d is None:
        return ""
    try:
        return json.dumps(d, ensure_ascii=False)
    except Exception:
        return str(d)


def linearize_list(lst):
    if lst is None:
        return ""
    return ", ".join(str(x) for x in lst)


def linearize_agent_result(agent_result):
    if agent_result is None:
        return ""
    try:
        return json.dumps(agent_result, ensure_ascii=False)
    except Exception:
        return str(agent_result)


async def evaluate_model(model_name, num_tasks=3, timeout=120):
    api_key = os.getenv("IO_API_KEY")
    base_url = os.getenv(
        "IO_BASE_URL", "https://api.intelligence-dev.io.solutions/api/v1"
    )
    tools = [add, subtract, multiply, divide, get_weather, square_root]
    critic = CriticAgent(model=model_name, api_key=api_key, base_url=base_url)
    task_manager = TaskManager(model=model_name, api_key=api_key, base_url=base_url)
    oracle = OracleAgent(model=model_name, api_key=api_key, base_url=base_url)
    environment = RLEnvironment(
        name=f"RL-{model_name}",
        agent_instructions="You are a tool-using assistant.",
        task_manager=task_manager,
        critic=critic,
        oracle=oracle,
        tools=tools,
        max_steps=3,
        agent_class=Agent,
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        needs_model_settings=MODELS_THAT_REQUIRE_MODEL_SETTINGS,
    )
    environment.load_tasks(verbose=False)
    tasks = environment.task_manager.get_all_tasks()
    if num_tasks:
        import random

        tasks = random.sample(tasks, min(num_tasks, len(tasks)))
    rows = []
    for task in tasks:
        row = {
            "model": model_name,
            "task_id": task.id,
            "task_description": task.description,
            "task_difficulty": task.difficulty,
            "task_required_tools": linearize_list(task.required_tools),
            "task_ground_truth": linearize_dict(task.ground_truth),
            "task_context": linearize_dict(task.context),
            "task_goal_seek": task.goal_seek or "",
        }
        try:
            state = await asyncio.wait_for(
                environment.run_episode(task=task, verbose=False), timeout=timeout
            )
            row["step_count"] = getattr(state, "step_count", "")
            row["agent_result"] = linearize_agent_result(
                getattr(state, "agent_result", None)
            )
            # Tool usage results (from agent_result if present)
            agent_result = getattr(state, "agent_result", None)
            tool_usage_results = None
            if agent_result and isinstance(agent_result, dict):
                tool_usage_results = agent_result.get("tool_usage_results")
            row["tool_usage_results"] = linearize_tool_usage(tool_usage_results)
            row["best_query"] = getattr(state, "best_query", "")
            row["best_instructions"] = getattr(state, "best_instructions", "")
            # Critic feedback
            critic_feedback = getattr(state, "critic_feedback", None)
            if critic_feedback:
                row["critic_score"] = getattr(critic_feedback, "score", "")
                row["critic_better_query"] = getattr(
                    critic_feedback, "better_query", ""
                )
                row["critic_metrics"] = linearize_dict(
                    getattr(critic_feedback, "metrics", None)
                )
                row["critic_agent_prompt_instructions"] = getattr(
                    critic_feedback, "agent_prompt_instructions", ""
                )
            else:
                row["critic_score"] = row["critic_better_query"] = row[
                    "critic_metrics"
                ] = row["critic_agent_prompt_instructions"] = ""
            # Oracle result
            oracle_result = getattr(state, "oracle_result", None)
            if oracle_result:
                row["oracle_correct"] = getattr(oracle_result, "correct", "")
                row["oracle_score"] = getattr(oracle_result, "score", "")
                row["oracle_feedback"] = getattr(oracle_result, "feedback", "")
                details = getattr(oracle_result, "details", None)
                if details:
                    row["oracle_matching_fields"] = linearize_list(
                        details.get("matching_fields")
                    )
                    row["oracle_missing_fields"] = linearize_list(
                        details.get("missing_fields")
                    )
                    row["oracle_incorrect_values"] = linearize_dict(
                        details.get("incorrect_values")
                    )
                    row["oracle_additional_insights"] = details.get(
                        "additional_insights", ""
                    )
                else:
                    row["oracle_matching_fields"] = row["oracle_missing_fields"] = row[
                        "oracle_incorrect_values"
                    ] = row["oracle_additional_insights"] = ""
            else:
                row["oracle_correct"] = row["oracle_score"] = row["oracle_feedback"] = (
                    ""
                )
                row["oracle_matching_fields"] = row["oracle_missing_fields"] = row[
                    "oracle_incorrect_values"
                ] = row["oracle_additional_insights"] = ""
            row["error"] = ""
        except asyncio.TimeoutError:
            row["error"] = f"Timeout after {timeout}s"
        except Exception as e:
            row["error"] = f"Error: {str(e)}"
        rows.append(row)
    return rows


async def main():
    all_rows = []
    for model in MODELS:
        print("================================================")
        print(f"\n=== Evaluating model: {model} ===")
        try:
            rows = await evaluate_model(model)
            for row in rows:
                print(f"Result: {row}")
            all_rows.extend(rows)
        except Exception as e:
            print(f"Error evaluating {model}: {e}")
            all_rows.append({"model": model, "error": str(e)})
    # Write CSV report
    fieldnames = [
        "model",
        "task_id",
        "task_description",
        "task_difficulty",
        "task_required_tools",
        "task_ground_truth",
        "task_context",
        "task_goal_seek",
        "step_count",
        "agent_result",
        "tool_usage_results",
        "best_query",
        "best_instructions",
        "critic_score",
        "critic_better_query",
        "critic_metrics",
        "critic_agent_prompt_instructions",
        "oracle_correct",
        "oracle_score",
        "oracle_feedback",
        "oracle_matching_fields",
        "oracle_missing_fields",
        "oracle_incorrect_values",
        "oracle_additional_insights",
        "error",
    ]
    with open(REPORT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            for key in fieldnames:
                if key not in row:
                    row[key] = ""
            writer.writerow(row)
    print(f"\nReport written to {REPORT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
