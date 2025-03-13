from datetime import datetime

from iointel import Agent

toolcall_happened = None


def get_current_datetime(args: None) -> str:
    global toolcall_happened
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    toolcall_happened = {
        'datetime': current_datetime,
    }
    return current_datetime


a = Agent(name="RunAgent", instructions="Return current datetime.",
          tools=[get_current_datetime])
result = a.run("Return current datetime. Use the tool provided")
assert result is not None, "Expected a result from the agent run."
assert toolcall_happened is not None, "Make sure the tool was actually called"
assert toolcall_happened['datetime'] in result, "Make sure the result of the toolcall matches the return value of the agent"
