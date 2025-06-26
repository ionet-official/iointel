import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn
from dotenv import load_dotenv

# Load environment variables from creds.env
load_dotenv("creds.env")

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid

from .models import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationStatus,
    ErrorResponse,
)
from .service import EvaluationService
from .config import settings
from .middleware import RateLimitMiddleware, APIKeyMiddleware, TaskCleanupMiddleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


# Global storage for background task status
evaluation_tasks = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RL Model Evaluation API")
    yield
    # Shutdown
    logger.info("Shutting down RL Model Evaluation API")


app = FastAPI(
    title=settings.app_name,
    description="API for evaluating reinforcement learning models with tools",
    version=settings.version,
    lifespan=lifespan,
)

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middlewares
app.add_middleware(TaskCleanupMiddleware, evaluation_tasks=evaluation_tasks)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)

# Initialize service
evaluation_service = EvaluationService()


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ).model_dump()
    )


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "RL Model Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "evaluate": "/evaluate",
            "evaluate_async": "/evaluate/async",
            "status": "/evaluate/{task_id}/status",
            "models": "/models",
            "health": "/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "rl-model-evaluation-api",
        "environment": {
            "io_api_key_set": bool(os.getenv("IO_API_KEY")),
            "io_base_url_set": bool(os.getenv("IO_BASE_URL"))
        }
    }


@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
    tags=["Evaluation"],
    summary="Evaluate models synchronously",
    description="Evaluate one or more models synchronously. This endpoint will wait for all evaluations to complete before returning results."
)
async def evaluate_models(request: EvaluationRequest):
    try:
        response = await evaluation_service.evaluate_models(
            models=request.models,
            num_tasks=request.num_tasks,
            timeout=request.timeout,
            api_key=request.api_key,
            base_url=request.base_url
        )
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


async def run_evaluation_async(
    task_id: str,
    models: list,
    num_tasks: int,
    timeout: int,
    api_key: Optional[str],
    base_url: Optional[str]
):
    try:
        evaluation_tasks[task_id]["status"] = "running"
        evaluation_tasks[task_id]["models_completed"] = 0
        
        for i, model in enumerate(models):
            evaluation_tasks[task_id]["current_model"] = model
            try:
                results = await evaluation_service.evaluate_single_model(
                    model_name=model,
                    num_tasks=num_tasks,
                    timeout=timeout,
                    api_key=api_key,
                    base_url=base_url
                )
                evaluation_tasks[task_id]["results"].extend(results)
                evaluation_tasks[task_id]["models_completed"] = i + 1
            except Exception as e:
                logger.error(f"Error evaluating model {model}: {e}")
                evaluation_tasks[task_id]["errors"].append({
                    "model": model,
                    "error": str(e)
                })
        
        evaluation_tasks[task_id]["status"] = "completed"
        evaluation_tasks[task_id]["current_model"] = None
        
        # Build final response
        response = await evaluation_service.evaluate_models(
            models=models,
            num_tasks=num_tasks,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url
        )
        evaluation_tasks[task_id]["response"] = response
        
    except Exception as e:
        logger.error(f"Async evaluation failed: {e}", exc_info=True)
        evaluation_tasks[task_id]["status"] = "failed"
        evaluation_tasks[task_id]["error"] = str(e)


@app.post(
    "/evaluate/async",
    response_model=EvaluationStatus,
    tags=["Evaluation"],
    summary="Evaluate models asynchronously",
    description="Start an asynchronous evaluation of one or more models. Returns a task ID that can be used to check status and retrieve results."
)
async def evaluate_models_async(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    task_id = str(uuid.uuid4())
    
    # Initialize task tracking
    evaluation_tasks[task_id] = {
        "status": "pending",
        "models_completed": 0,
        "total_models": len(request.models),
        "current_model": None,
        "results": [],
        "errors": [],
        "response": None,
        "error": None,
        "created_at": datetime.utcnow()
    }
    
    # Add to background tasks
    background_tasks.add_task(
        run_evaluation_async,
        task_id,
        request.models,
        request.num_tasks,
        request.timeout,
        request.api_key,
        request.base_url
    )
    
    return EvaluationStatus(
        task_id=task_id,
        status="pending",
        total_models=len(request.models),
        models_completed=0
    )


@app.get(
    "/evaluate/{task_id}/status",
    tags=["Evaluation"],
    summary="Get evaluation status",
    description="Check the status of an asynchronous evaluation task"
)
async def get_evaluation_status(task_id: str):
    if task_id not in evaluation_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task = evaluation_tasks[task_id]
    
    response = {
        "task_id": task_id,
        "status": task["status"],
        "models_completed": task["models_completed"],
        "total_models": task["total_models"],
        "current_model": task["current_model"]
    }
    
    if task["status"] == "completed" and task["response"]:
        response["results"] = task["response"]
    elif task["status"] == "failed":
        response["error"] = task.get("error", "Unknown error")
    
    if task["errors"]:
        response["partial_errors"] = task["errors"]
    
    return response


@app.get(
    "/evaluate/{task_id}/results",
    response_model=EvaluationResponse,
    tags=["Evaluation"],
    summary="Get evaluation results",
    description="Retrieve the results of a completed asynchronous evaluation"
)
async def get_evaluation_results(task_id: str):
    if task_id not in evaluation_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task = evaluation_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task {task_id} is not completed. Current status: {task['status']}"
        )
    
    if not task.get("response"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Results not available"
        )
    
    return task["response"]


@app.get(
    "/models",
    tags=["Models"],
    summary="Get available models",
    description="Get a list of recommended models for evaluation"
)
async def get_models():
    return {
        "recommended_models": settings.recommended_models,
        "models_requiring_settings": settings.models_requiring_settings,
        "note": "You can use any model available through your API provider"
    }


@app.delete(
    "/evaluate/{task_id}",
    tags=["Evaluation"],
    summary="Cancel evaluation task",
    description="Cancel a running or pending evaluation task"
)
async def cancel_evaluation(task_id: str):
    if task_id not in evaluation_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    # In a real implementation, you would implement actual cancellation logic
    # For now, we'll just mark it as cancelled
    if evaluation_tasks[task_id]["status"] in ["pending", "running"]:
        evaluation_tasks[task_id]["status"] = "cancelled"
        return {"message": f"Task {task_id} cancelled"}
    else:
        return {"message": f"Task {task_id} cannot be cancelled (status: {evaluation_tasks[task_id]['status']})"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )