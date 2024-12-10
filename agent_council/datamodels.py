from pydantic import BaseModel


# Request models
class ScheduleRequest(BaseModel):
    command: str
    delay: int = 0 # Delay in seconds


##agent params###
class AgentParams(BaseModel):
    name: str
    instructions: str
