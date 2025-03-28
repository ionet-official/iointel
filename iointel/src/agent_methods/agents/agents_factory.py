import os

from iointel import Agent
from ..data_models.datamodels import AgentParams
from typing import Optional


AGENT_SPECS = {
    "leader": {
        "name": "Leader",
        "instructions": """
            You are the council leader, 
            you lead the council and provide guidance, 
            and administer the voting process.
        """,
    },
    "council_member1": {
        "name": "Council Member 1",
        "instructions": "You are a council member who provides input and votes on decisions.",
    },
    "council_member2": {
        "name": "Council Member 2",
        "instructions": "You are a council member who provides input and votes on decisions.",
    },
    "council_member3": {
        "name": "Council Member 3",
        "instructions": "You are a council member who provides input and votes on decisions.",
    },
    "coder": {
        "name": "Coder",
        "instructions": "You are an expert python coder who provides code for the task.",
    },
    "agent_maker": {
        "name": "Agent Maker",
        "instructions": "You create agents that can perform tasks from the provided code.",
    },
    "reasoning_agent": {
        "name": "Reasoning Agent",
        "instructions": "You are an agent that performs reasoning steps.",
    },
    "docker_sandbox_agent": {
        "name": "Docker Sandbox Agent",
        "instructions": "You are an agent that runs code in a docker sandbox.",
    },
    "summary_agent": {
        "name": "Summary Agent",
        "instructions": "You are an agent that summarizes text.",
    },
    "sentiment_analysis_agent": {
        "name": "Sentiment Analysis Agent",
        "instructions": "You are an agent that performs unbiased sentiment analysis on text.",
    },
    "extractor": {
        "name": "Named Entity Recognizer",
        "instructions": "You are an agent that extracts named entities from text.",
    },
    "default_agent": {
        "name": "Default Agent",
        "instructions": "You are an agent that does a lot of different things, you are dynamic.",
    },
    "moderation_agent": {
        "name": "Moderation Agent",
        "instructions": "You are an agent that moderates content.",
    },
    "classification_agent": {
        "name": "Classification Agent",
        "instructions": "You are an agent that is an expert in classifying things.",
    },
    "translation_agent": {
        "name": "Translation Agent",
        "instructions": "You are an agent that is a polyglot of multiple languages and is an expert in translating text.",
    },
}


def create_agent(params: AgentParams) -> Agent:
    """
    Create a Agent instance from the given AgentParams.
    """
    # return Agent(name=params.name, instructions=params.instructions, persona=params.persona, tools=params.tools)
    return Agent(**params.model_dump(exclude_none=True))


def create_agents():
    return {
        agent_key: create_agent(
            AgentParams(name=spec["name"], instructions=spec["instructions"])
        )
        for agent_key, spec in AGENT_SPECS.items()
    }


AGENTS = create_agents() if os.environ.get("LIBRARY_MODE", "").lower() != "true" else {}


def get_agent(agent_name: str, agent_params: Optional[AgentParams] = None) -> Agent:
    if agent_params is None:
        agent_params = AgentParams(name="", instructions="")

    # Merge or set the default name/instructions
    spec = AGENT_SPECS.get(agent_name, AGENT_SPECS["default_agent"])
    agent_params.name = agent_params.name or spec["name"]
    agent_params.instructions = agent_params.instructions or spec["instructions"]

    return create_agent(agent_params)
