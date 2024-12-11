from pydantic import BaseModel


# Request models
class ScheduleRequest(BaseModel):
    task: str


##agent params###
class AgentParams(BaseModel):
    name: str
    instructions: str
