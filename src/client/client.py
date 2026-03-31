import httpx
import asyncio
from typing import Dict, Any, Optional, List

class LLMOrchestratorClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def submit_task(self, model_name: str, prompt: str, priority: int = 0, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/tasks/",
                json={
                    "model_name": model_name,
                    "prompt": prompt,
                    "priority": priority,
                    "metadata": metadata
                }
            )
            response.raise_for_status()
            return response.json()

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/tasks/{task_id}")
            response.raise_for_status()
            return response.json()

    async def list_tasks(self) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/tasks/")
            response.raise_for_status()
            return response.json()

async def main():
    client = LLMOrchestratorClient()

    print("Submitting a task...")
    task = await client.submit_task("gpt-4", "Generate a short story about a space-faring cat.", priority=1)
    print(f"Task submitted: {task["task_id"]}")

    print("Submitting another task...")
    task2 = await client.submit_task("llama-70b", "Summarize the history of AI.")
    print(f"Task submitted: {task2["task_id"]}")

    print("Fetching task status...")
    status = await client.get_task_status(task["task_id"])
    print(f"Task {status["task_id"]} status: {status["status"]}")

    print("Listing all tasks...")
    all_tasks = await client.list_tasks()
    for t in all_tasks:
        print(f"- {t["task_id"]}: {t["status"]}")

if __name__ == "__main__":
    asyncio.run(main())
