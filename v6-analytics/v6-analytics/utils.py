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


def assert_results_complete(results: list, task_description: str = "child task") -> None:
    """Raise AlgorithmExecutionError if any node returned None (crashed run).

    vantage6 sets a node's result to None when its run has failed or crashed.
    Continuing with a partial result set would silently corrupt aggregations that
    assume contributions from every included organization.
    """
    failed_indices = [i for i, r in enumerate(results) if r is None]
    if failed_indices:
        raise AlgorithmExecutionError(
            f"Task '{task_description}': {len(failed_indices)} node(s) at position(s) "
            f"{failed_indices} returned None (crashed or failed run). "
            "Stopping execution to prevent corrupt aggregation. "
            "Check the node logs."
        )
