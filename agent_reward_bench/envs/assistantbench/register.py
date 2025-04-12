import json
import os
import logging
from browsergym.core.registration import register_task

from . import task

logger = logging.getLogger(__name__)

TOY_AB_TASK_IDS = []
VALID_AB_TASK_IDS = []
TEST_AB_TASK_IDS = []

# START_URL = os.getenv("ASSISTANTBENCH_START_URL", "https://www.google.com")
START_URL = "https://duckduckgo.com"

print(f"Using AssistantBench start URL: {START_URL}")

# register a toy easy task for testing implementation
gym_id = f"assistantbench.improved.imp.0"
register_task(
    gym_id,
    task.ImprovedAssistantBenchTask,
    task_kwargs={
        "task_id": f"imp.0",
        "start_url": START_URL,
    },
    default_task_kwargs={
        "save_predictions": False,  # can be overriden
    },
)
TOY_AB_TASK_IDS.append(gym_id)

# register the AssistantBench dev set
for task_id in range(33):
    print("Registering AssistantBench task", task_id)
    gym_id = f"assistantbench.improved.validation.{task_id}"
    register_task(
        gym_id,
        task.ImprovedAssistantBenchTask,
        task_kwargs={
            "task_id": f"validation.{task_id}",
            "start_url": START_URL,
        },
        default_task_kwargs={
            "save_predictions": False,  # can be overriden
        },
    )
    VALID_AB_TASK_IDS.append(gym_id)

# register the AssistantBench test set
for task_id in range(181):
    gym_id = f"assistantbench.improved.test.{task_id}"
    register_task(
        gym_id,
        task.AssistantBenchTask,
        task_kwargs={
            "task_id": f"test.{task_id}",
        },
        default_task_kwargs={
            "save_predictions": True,  # can be overriden
        },
    )
    TEST_AB_TASK_IDS.append(gym_id)

ALL_AB_TASK_IDS = TOY_AB_TASK_IDS + VALID_AB_TASK_IDS + TEST_AB_TASK_IDS
