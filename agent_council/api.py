from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI
import uvicorn
from datamodels import ScheduleRequest
from agent_flows import council_flow, schedule_reminder_flow


app = FastAPI()

@app.post("/schedule")
async def schedule_task(req: ScheduleRequest):
    # Directly invoke the schedule_reminder_flow defined in agents.py
    result = schedule_reminder_flow(req.task)
    return {"message": result}



@app.post("/council")
async def run_council_task(req: ScheduleRequest):
    # Directly invoke the council_task flow
    result = council_flow(req.task)
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)