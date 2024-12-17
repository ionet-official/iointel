from agent_methods.tools.tools import get_current_datetime
from framework.src.agents import Agent




tools = []


reminder_agent = Agent(
    name="Reminder Agent",
    instructions="A simple agent that sends reminders.",
    tools=[get_current_datetime],
)

# Define agents with access to the get_current_datetime tool
leader = Agent(
    name="Leader",
    instructions="""
    You are the council leader, 
    you lead the council and provide guidance, 
    and administer the voting process.
    """,
    #tools=tools,

)

council_member1 = Agent(
    name="Council Member 1",
    instructions="You are a council member who provides input and votes on decisions.",
    #tools=tools,

)

council_member2 = Agent(
    name="Council Member 2",
    instructions="You are a council member who provides input and votes on decisions.",
    #tools=tools,

)

council_member3 = Agent(
    name="Council Member 3",
    instructions="You are a council member who provides input and votes on decisions.",
    #tools=tools,

)

coder = Agent(
    name="Coder",
    instructions="You are an expert python coder who provides code for the task.",
    #tools=tools,

)

agent_maker = Agent(
    name="Agent Maker",
    instructions="You create agents that can perform tasks from the provided code.",
    #tools=tools,

)

reasoning_agent = Agent(
    name="Reasoning Agent",
    instructions="You are an agent that performs reasoning steps.",
    #tools=tools,

)

docker_sandbox_agent = Agent( #WIP - need to add docker sandbox tool
    name="Docker Sandbox Agent",
    instructions="You are an agent that runs code in a docker sandbox.",
    #tools=[],

)

sentiment_analysis_agent = Agent(
    name= "Sentiment Analysis Agent",
    instructions="You are an agent that performs unbiased sentiment analysis on text.",
    )

extractor = Agent(
    name="Named Entity Recognizer",
    instructions="You are an agent that extracts named entities from text.",
)

default_agent = Agent(
    name="Default Agent",
    instructions="You are an agent that does alot of different things, you are dynamic.",
)

moderation_agent = Agent(
    name="Moderation Agent",
    instructions="You are an agent that moderates content.",
)
