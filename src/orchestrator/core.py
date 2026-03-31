import asyncio
import uuid
import time
import logging
from collections import deque
from typing import Dict, Any, Optional, Deque, List
from datetime import datetime

# Assuming Config is defined elsewhere or imported from a config file
# For this example, we'll define a minimal Config class
class Config:
    WORKER_POOL_SIZE: int = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LLMTask:
    """Represents a single LLM inference task."""
    def __init__(
        self,
        model_name: str,
        prompt: str,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = str(uuid.uuid4())
        self.model_name = model_name
        self.prompt = prompt
        self.priority = priority
        self.metadata = metadata if metadata is not None else {}
        self.status = "PENDING"
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts the task object to a dictionary."""
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "priority": self.priority,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

class LLMOrchestrator:
    """Orchestrates LLM inference tasks, managing a worker pool and task queue."""
    def __init__(self, worker_pool_size: int = Config.WORKER_POOL_SIZE):
        self.task_queue: Deque[LLMTask] = deque()
        self.active_tasks: Dict[str, LLMTask] = {}
        self.completed_tasks: Dict[str, LLMTask] = {}
        self.worker_pool_size = worker_pool_size
        self.workers = []
        self._running = False
        logging.info(f"LLMOrchestrator initialized with {self.worker_pool_size} workers.")

    async def start(self):
        """Starts the orchestrator and its worker pool."""
        if self._running:
            logging.warning("Orchestrator is already running.")
            return
        self._running = True
        logging.info("Starting orchestrator worker pool...")
        for i in range(self.worker_pool_size):
            worker = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker)
        logging.info(f"Orchestrator started with {len(self.workers)} workers.")

    async def stop(self):
        """Stops the orchestrator and gracefully shuts down workers."""
        if not self._running:
            logging.warning("Orchestrator is not running.")
            return
        self._running = False
        logging.info("Stopping orchestrator worker pool...")
        for worker in self.workers:
            worker.cancel() # Request workers to stop
        await asyncio.gather(*self.workers, return_exceptions=True) # Wait for workers to finish
        logging.info("Orchestrator stopped.")

    async def submit_task(self, model_name: str, prompt: str, priority: int = 0, metadata: Optional[Dict[str, Any]] = None) -> LLMTask:
        """Submits a new LLM task to the orchestrator."""
        task = LLMTask(model_name, prompt, priority, metadata)
        # For simplicity, we'll just append. A real system might use a priority queue.
        self.task_queue.append(task)
        self.active_tasks[task.task_id] = task
        logging.info(f"Task {task.task_id} submitted with priority {task.priority}.")
        return task

    def get_task_status(self, task_id: str) -> Optional[LLMTask]:
        """Retrieves the current status of a task."""
        return self.active_tasks.get(task_id) or self.completed_tasks.get(task_id)

    def get_all_tasks(self) -> List[LLMTask]:
        """Returns a list of all active and completed tasks."""
        return list(self.active_tasks.values()) + list(self.completed_tasks.values())

    async def _worker_loop(self, worker_id: int):
        """Worker loop that processes tasks from the queue."""
        logging.info(f"Worker {worker_id} started.")
        try:
            while self._running:
                if self.task_queue:
                    task = self.task_queue.popleft() # Get the next task
                    logging.info(f"Worker {worker_id} picked up task {task.task_id}.")
                    task.status = "RUNNING"
                    task.started_at = datetime.now().isoformat()
                    
                    try:
                        # Simulate LLM inference
                        await asyncio.sleep(random.uniform(1, 5)) # Simulate network latency and processing time
                        task.result = f"Generated response for '{task.prompt[:50]}...' using {task.model_name}"
                        task.status = "COMPLETED"
                        logging.info(f"Task {task.task_id} completed by worker {worker_id}.")
                    except asyncio.CancelledError:
                        task.status = "CANCELLED"
                        task.error = "Task cancelled during processing."
                        logging.warning(f"Task {task.task_id} cancelled by worker {worker_id}.")
                        raise # Re-raise to exit worker loop
                    except Exception as e:
                        task.status = "FAILED"
                        task.error = str(e)
                        logging.error(f"Task {task.task_id} failed: {e}")
                    finally:
                        task.completed_at = datetime.now().isoformat()
                        self.completed_tasks[task.task_id] = task
                        if task.task_id in self.active_tasks:
                            del self.active_tasks[task.task_id] # Move from active to completed
                else:
                    await asyncio.sleep(0.1) # Wait a bit if no tasks
        except asyncio.CancelledError:
            logging.info(f"Worker {worker_id} received cancellation request and is shutting down.")
        except Exception as e:
            logging.error(f"Worker {worker_id} encountered an unexpected error: {e}")
