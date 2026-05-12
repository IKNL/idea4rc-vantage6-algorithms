from vantage6.algorithm.tools.exceptions import AlgorithmExecutionError
from vantage6.algorithm.tools.util import info


def create_child_task(client, **kwargs) -> int:
    """Create a child task and validate that task_id is returned.

    Raises AlgorithmExecutionError if task creation fails or returns no id.
    Returns the task_id directly.
    """
    task = client.task.create(**kwargs)
    task_id = task.get("id") if isinstance(task, dict) else None
    if not task_id:
        raise AlgorithmExecutionError(f"Child task creation failed: {task}")
    info(f"Child task created with id={task_id}")
    return task_id
