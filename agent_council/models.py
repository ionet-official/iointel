from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# Request models
class ScheduleRequest(BaseModel):
    command: str
    delay: int = 0 # Delay in seconds


##agent params###
class AgentParams(BaseModel):
    name: str
    instructions: str
