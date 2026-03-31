from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio

from .orchestrator.core import LLMOrchestrator, LLMTask

app = FastAPI(title="LLM Deploy Orchestrator API")
orchestrator = LLMOrchestrator()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(orchestrator.start())

@app.on_event("shutdown")
async def shutdown_event():
    await orchestrator.stop()

class TaskRequest(BaseModel):
    model_name: str
    prompt: str
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    model_name: str
    prompt: str
    priority: int
    status: str
    result: Optional[str]
    error: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    metadata: Dict[str, Any]

@app.post("/tasks/", response_model=TaskStatusResponse)
async def create_task(request: TaskRequest):
    task = await orchestrator.submit_task(
        model_name=request.model_name,
        prompt=request.prompt,
        priority=request.priority,
        metadata=request.metadata
    )
    return TaskStatusResponse(**task.to_dict())

@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str):
    task = orchestrator.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(**task.to_dict())

@app.get("/tasks/", response_model=List[TaskStatusResponse])
async def list_tasks():
    tasks = orchestrator.get_all_tasks()
    return [TaskStatusResponse(**task.to_dict()) for task in tasks]
