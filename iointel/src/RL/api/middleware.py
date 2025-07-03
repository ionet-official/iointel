import time
from typing import Dict
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.request_counts: Dict[str, list] = defaultdict(list)
        
    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = request.client.host
        
        # Clean up old requests
        current_time = time.time()
        cutoff_time = current_time - settings.rate_limit_period
        
        self.request_counts[client_ip] = [
            timestamp for timestamp in self.request_counts[client_ip]
            if timestamp > cutoff_time
        ]
        
        # Check rate limit
        if len(self.request_counts[client_ip]) >= settings.rate_limit_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {settings.rate_limit_requests} requests per {settings.rate_limit_period} seconds."
            )
        
        # Record this request
        self.request_counts[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)
        
    async def dispatch(self, request: Request, call_next):
        if not settings.require_api_key:
            return await call_next(request)
        
        # Skip auth for health endpoints
        if request.url.path in ["/", "/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # Get API key from header
        api_key = request.headers.get(settings.api_key_header)
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        if api_key not in settings.api_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
        
        # Process request
        response = await call_next(request)
        return response


class TaskCleanupMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, evaluation_tasks: dict):
        super().__init__(app)
        self.evaluation_tasks = evaluation_tasks
        self.last_cleanup = datetime.utcnow()
        
    async def dispatch(self, request: Request, call_next):
        # Perform cleanup if needed
        current_time = datetime.utcnow()
        if current_time - self.last_cleanup > timedelta(hours=1):
            self._cleanup_old_tasks()
            self.last_cleanup = current_time
        
        response = await call_next(request)
        return response
    
    def _cleanup_old_tasks(self):
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=settings.task_retention_hours)
        
        tasks_to_remove = []
        for task_id, task_data in self.evaluation_tasks.items():
            if "created_at" in task_data:
                if task_data["created_at"] < cutoff_time:
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.evaluation_tasks[task_id]
        
        # Also enforce max stored tasks
        if len(self.evaluation_tasks) > settings.max_stored_tasks:
            # Remove oldest tasks
            sorted_tasks = sorted(
                self.evaluation_tasks.items(),
                key=lambda x: x[1].get("created_at", datetime.min)
            )
            
            tasks_to_keep = settings.max_stored_tasks
            for task_id, _ in sorted_tasks[:-tasks_to_keep]:
                del self.evaluation_tasks[task_id]