import logging
from datetime import datetime
import time
import controlflow as cf
from models import AgentParams

logger = logging.getLogger(__name__)

@cf.tool
def create_agent(params: AgentParams) -> cf.Agent:
    """
    Create a controlflow.Agent instance from the given AgentParams.
    """
    return cf.Agent(name=params.name, instructions=params.instructions)


@cf.tool
def get_current_datetime() -> str:
    """
    Return the current datetime as a string in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Define a common tools list so every agent has the date-time tool
tools = [get_current_datetime]

# Define agents with access to the get_current_datetime tool
leader = cf.Agent(
    name="Leader",
    instructions="""
    You are the council leader, 
    you lead the council and provide guidance, 
    and administer the voting process.
    """,
    tools=tools
)

council_member1 = cf.Agent(
    name="Council Member 1",
    instructions="You are a council member who provides input and votes on decisions.",
    tools=tools
)

council_member2 = cf.Agent(
    name="Council Member 2",
    instructions="You are a council member who provides input and votes on decisions.",
    tools=tools
)

council_member3 = cf.Agent(
    name="Council Member 3",
    instructions="You are a council member who provides input and votes on decisions.",
    tools=tools
)

coder = cf.Agent(
    name="Coder",
    instructions="You are an expert python coder who provides code for the task.",
    tools=tools
)

agent_maker = cf.Agent(
    name="Agent",
    instructions="You create agents that can perform tasks from the provided code.",
    tools=tools
)

@cf.task
def remind_after_delay(command: str, delay: int = 0) -> str:
    """
    A simple task that waits for `delay` seconds and then returns a reminder message.
    """
    if delay > 0:
        time.sleep(delay)
    return f"Reminder: {command}"

@cf.flow
def schedule_reminder_flow(command: str, delay: int = 0) -> str:
    """
    A flow that schedules a reminder after a given delay.
    Instead of a persistent task manager, we just call this flow whenever we need.
    """
    result = cf.run(
        "Set a reminder",
        tasks=[(remind_after_delay, {"command": command, "delay": delay})],
        result_type=str
    )
    return result

@cf.flow
def council_task(task: str):
    # step 1: deliberation and voting process
    deliberate = cf.run(
        "Deliberate and vote on the best way to complete the task.",
        agents=[leader, council_member1, council_member2, council_member3],
        completion_agents=[leader],
        instructions="""
            Deliberate with other council members on the best way to complete the task.
            Allow each council member to provide input before voting.
            Vote on the best answer.
            Show the entire deliberation, voting process, final decision, and reasoning.
        """,
        context={"task": task},
        #turn_strategy=cf.orchestration.turn_strategies.Moderated(moderator=leader),
        result_type=str
    )
    print(deliberate)

    # Step 2: write code for the task
    codes = cf.run(
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
    custom_agent_params = cf.run(
        "Create a ControlFlow agent using the provided code.",
        agents=[agent_maker],
        context={"code": codes},
        result_type={"name": str, "instructions": str}
    )
    custom_agent = cf.Agent(name=custom_agent_params["name"], instructions=custom_agent_params["instructions"], tools=tools)

    # step 4: run the agent
    result = cf.run(
        "Execute the agent to complete the task",
        agents=[custom_agent],
        result_type=str
    )
    return result