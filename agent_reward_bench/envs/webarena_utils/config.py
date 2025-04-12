# TASK_IDS = range(100)
import pkg_resources
import json
from typing import List

def get_task_ids(package='agent_reward_bench') -> List[int]:
    task_ids_path = pkg_resources.resource_filename(package, 'data/webarena.task_ids.json')
    
    with open(task_ids_path) as f:
        task_ids = json.load(f)
    
    assert isinstance(task_ids, list), f"Expected a list of task ids, got {task_ids}. This is an internal error that should be reported."

    return task_ids

TASK_IDS: List[int] = get_task_ids()