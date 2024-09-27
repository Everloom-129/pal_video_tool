import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from api_client import APIClient

class MultiThreadedAPIClient(APIClient):
    def __init__(self, num_workers: int = 3, queue_size: int = 100):
        super().__init__()
        self.queue = queue.Queue(maxsize=queue_size)
        self.results = {}
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.stop_event = threading.Event()

    def start_workers(self):
        for _ in range(self.num_workers):
            self.executor.submit(self._worker)

    def stop_workers(self):
        self.stop_event.set()
        self.executor.shutdown(wait=True)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                task_id, image_path, prompts = self.queue.get(timeout=1)
                result = self.detect_objects(image_path, prompts)
                self.results[task_id] = result
            except queue.Empty:
                continue
            finally:
                self.queue.task_done()

    def add_task(self, task_id: int, image_path: str, prompts: List[str]):
        self.queue.put((task_id, image_path, prompts))

    def get_result(self, task_id: int) -> TaskResult:
        return self.results.get(task_id)

    def process_batch(self, tasks: List[Tuple[int, str, List[str]]]) -> List[TaskResult]:
        for task in tasks:
            self.add_task(*task)
        
        self.queue.join()  # Wait for all tasks to be processed

        results = []
        for task_id, _, _ in tasks:
            results.append(self.get_result(task_id))
        
        return results