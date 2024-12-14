from pydantic import BaseModel


##agent params###
class AgentParams(BaseModel):
    name: str
    instructions: str
