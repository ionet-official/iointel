from fastapi import FastAPI
import uvicorn
from datamodels import ScheduleRequest
from agent_flows import council_task, schedule_reminder_flow


app = FastAPI()

@app.post("/schedule")
async def schedule_task(req: ScheduleRequest):
    # Directly invoke the schedule_reminder_flow defined in agents.py
    result = schedule_reminder_flow(req.command, req.delay)
    return {"message": result}

@app.post("/council_task")
async def run_council_task(task: str):
    # Directly invoke the council_task flow
    result = council_task(task)
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)