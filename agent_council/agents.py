import controlflow as cf
from tools import get_current_datetime, remind_after_delay
from models import Agent

tools = [get_current_datetime, remind_after_delay]


reminder_agent = Agent(
    name="Reminder Agent",
    instructions="A simple agent that sends reminders.",
    tools=tools,
)

# Define agents with access to the get_current_datetime tool
leader = cf.Agent(
    name="Leader",
    instructions="""
    You are the council leader, 
    you lead the council and provide guidance, 
    and administer the voting process.
    """,
    tools=tools,

)

council_member1 = cf.Agent(
    name="Council Member 1",
    instructions="You are a council member who provides input and votes on decisions.",
    tools=tools,

)

council_member2 = cf.Agent(
    name="Council Member 2",
    instructions="You are a council member who provides input and votes on decisions.",
    tools=tools,

)

council_member3 = cf.Agent(
    name="Council Member 3",
    instructions="You are a council member who provides input and votes on decisions.",
    tools=tools,

)

coder = cf.Agent(
    name="Coder",
    instructions="You are an expert python coder who provides code for the task.",
    tools=tools,

)

agent_maker = cf.Agent(
    name="Agent",
    instructions="You create agents that can perform tasks from the provided code.",
    tools=tools,

)