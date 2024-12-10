import controlflow as cf

from agents import (leader, council_member1, council_member2, 
                    council_member3, coder, agent_maker, reminder_agent)

from datamodels import AgentParams
from tools import create_agent
from models import run_agents

@cf.flow
def schedule_reminder_flow(command: str, delay: int = 0) -> str:
    """
    A flow that schedules a reminder after a given delay.
    Instead of a persistent task manager, we just call this flow whenever we need.
    """
    
    reminder = run_agents(
        objective = "Schedule a reminder",
        instructions="""
            Schedule a reminder and use the tool to track the time for the reminder.
        """,
        agents=[reminder_agent],
        context={"command": command},
        result_type=str,
    )
    return reminder


@cf.flow
def council_flow(task: str):
    # step 1: deliberation and voting process
    deliberate = run_agents(
        "Deliberate and vote on the best way to complete the task.",
        agents=[leader, council_member1, council_member2, council_member3],
        completion_agents=[leader],
        #turn_strategy=cf.orchestration.turn_strategies.Moderated(moderator=leader),
        instructions="""
            Deliberate with other council members on the best way to complete the task.
            Allow each council member to provide input before voting.
            Vote on the best answer.
            Show the entire deliberation, voting process, final decision, and reasoning.
        """,
        context={"task": task},
        result_type=str
    )
    print(deliberate)

    # Step 2: write code for the task
    codes = run_agents(
        "Write code for the task",
        agents=[coder],
        instructions="""
            Provide Python code to accomplish the task.
            Return code in a format that can be parsed by python (as a string).
        """,
        context={"deliberation": deliberate},
        result_type=str
    )

    # Step 3: generate an agent to run the code
    custom_agent_params = run_agents(
        "Create a ControlFlow agent using the provided code.",
        agents=[agent_maker],
        context={"code": codes},
        result_type=AgentParams,
    )


    # step 4: run the agent
    result = run_agents(
        "Execute the agent to complete the task",
        agents=[create_agent(custom_agent_params)],
        result_type=str
    )
    return result
