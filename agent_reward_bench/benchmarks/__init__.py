
import csv
import pkg_resources
import json
from typing import List
import os

from browsergym.experiments.benchmark.metadata.utils import (
    task_metadata,
)
from browsergym.experiments.loop import EnvArgs
from browsergym.experiments.benchmark.utils import (
    make_env_args_list_from_fixed_seeds,
)
from browsergym.experiments.benchmark.utils import (
    make_env_args_list_from_fixed_seeds,
    make_env_args_list_from_repeat_tasks,
    make_env_args_list_from_workarena_curriculum,
)
from browsergym.experiments.benchmark.base import Benchmark, HighLevelActionSetArgs
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS, make_env_args_list_from_repeat_tasks, task_list_from_metadata
import numpy as np

from .base import ImprovedBenchmark


class ModifiedEnvArgs(EnvArgs):
    def make_env(self, action_mapping, exp_dir, exp_task_kwargs: dict = {}):
        # register here for the parallel backends (joblib, ray)
        task_name = self.task_name
        if task_name.startswith("visualwebarena.resized"):
            print("Registering visualwebarena.resized")
            import agent_reward_bench.envs.visualwebarena.register
        elif task_name.startswith("assistantbench.improved"):
            print("Registering assistantbench.improved")
            import agent_reward_bench.envs.assistantbench.register

        # part below is copied from browsergym.experiments.loop.EnvArgs.make_env
        import gymnasium as gym
        from browsergym.experiments.loop import _get_env_name

        extra_kwargs = {}
        if self.record_video:
            extra_kwargs["record_video_dir"] = exp_dir
        if self.viewport:
            extra_kwargs["viewport"] = self.viewport
        if self.slow_mo is not None:
            extra_kwargs["slow_mo"] = self.slow_mo
        if self.storage_state:
            extra_kwargs["pw_context_kwargs"] = {"storage_state": self.storage_state}
        if self.task_kwargs is not None:
            extra_kwargs["task_kwargs"] = self.task_kwargs
        if exp_task_kwargs:
            extra_kwargs["task_kwargs"] = extra_kwargs.get("task_kwargs", {}) | exp_task_kwargs

        # assistantbench hack, write the task output (agent prediction) to a file in the experiment's directory
        # TODO: find a better way to deal with this
        if self.task_name.startswith("assistantbench.test"):
            extra_kwargs["task_kwargs"] = extra_kwargs.get("task_kwargs", {}) | {
                "output_file": exp_dir / "assistantbench-prediction.json"
            }

        return gym.make(
            _get_env_name(self.task_name),
            disable_env_checker=True,
            max_episode_steps=self.max_steps,
            headless=self.headless,
            wait_for_user_message=self.wait_for_user_message,
            action_mapping=action_mapping,  # action mapping is provided by the agent
            **extra_kwargs,
        )
    
def modified_make_env_args_list_from_fixed_seeds(
    task_list: list[str], max_steps: int, fixed_seeds: list[int]
):
    """
    Generates a list of `len(task_list)` time `n_repeats` environments arguments, using randomly generated seeds.
    """
    env_args_list = []
    for task in task_list:
        for seed in fixed_seeds:
            env_args_list.append(
                ModifiedEnvArgs(
                    task_name=task,
                    task_seed=int(seed),
                    max_steps=max_steps,
                    headless=True,
                    record_video=False,
                    wait_for_user_message=False,
                    viewport=None,
                    slow_mo=None,
                    storage_state=None,
                    task_kwargs=None,
                )
            )

    return env_args_list


def get_task_ids_sampled_wa(package='agent_reward_bench') -> List[int]:
    task_ids_path = pkg_resources.resource_filename(package, 'data/webarena.task_ids.json')
    
    with open(task_ids_path) as f:
        task_ids = json.load(f)
    
    assert isinstance(task_ids, list), f"Expected a list of task ids, got {task_ids}. This is an internal error that should be reported."

    return list(sorted(task_ids))

def get_task_ids_sampled_vwa(package='agent_reward_bench') -> List[int]:
    task_ids_path = pkg_resources.resource_filename(package, 'data/visualwebarena.task_ids.json')
    
    with open(task_ids_path) as f:
        task_ids = json.load(f)
    
    assert isinstance(task_ids, list), f"Expected a list of task ids, got {task_ids}. This is an internal error that should be reported."

    return list(sorted(task_ids))

def get_task_ids_sampled_workarena_l2(package='agent_reward_bench') -> List[str]:
    task_ids_path = pkg_resources.resource_filename(package, 'data/workarena_l2.task_ids.json')
    
    with open(task_ids_path) as f:
        task_ids = json.load(f)
    
    assert isinstance(task_ids, list), f"Expected a list of task ids, got {task_ids}. This is an internal error that should be reported."

    return list(sorted(task_ids))

TASK_IDS: List[int] = get_task_ids_sampled_wa()
VWA_TASK_IDS: List[int] = get_task_ids_sampled_vwa()
WORKARENA_L2_TASK_IDS: List[int] = get_task_ids_sampled_workarena_l2()

def get_workarena_l1_split(split="test", num_repeat=1):
    # https://github.com/ServiceNow/BrowserGym/blob/ec6b802cd655f2c6a84ebd66a22a4435d8147272/browsergym/experiments/src/browsergym/experiments/benchmark/configs.py#L94
    b = Benchmark(
        name="workarena_l1",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["workarena"],
        is_multi_tab=False,
        supports_parallel_seeds=True,
        backends=["workarena"],
        env_args_list=make_env_args_list_from_workarena_curriculum(
            level="l1",
            task_category_filter=None,
            meta_seed=42,  # meta seed for evaluation curriculum
            max_steps=30,
            curriculum_type="agent",
            seeds_l1=num_repeat,
        ),
        task_metadata=task_metadata("workarena"),
    )

    b_split = b.subset_from_split(split)

    return b_split


def get_assistantbench_split(split='valid', start_url=None):
    from agent_reward_bench.envs.assistantbench import AssistantbenchImprovedBackend
    if start_url is not None:
        os.environ['ASSISTANTBENCH_START_URL'] = start_url

    task_df = task_metadata("assistantbench")
    # replace all in task_name col from assistantbench to assistantbench.improved
    task_df['task_name'] = task_df['task_name'].str.replace('assistantbench', 'assistantbench.improved')
    task_df_split = task_df[task_df['browsergym_split'] == split].reset_index(drop=True)

    benchmark = ImprovedBenchmark(
        name="assistantbench_improved",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["assistantbench"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=[AssistantbenchImprovedBackend()],
        env_args_list=modified_make_env_args_list_from_fixed_seeds(
            task_list=task_list_from_metadata(metadata=task_df_split),
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=task_df_split,
    )
    print("Finished creating AssistantBench benchmark.")

    return benchmark

def get_webarena_100_benchmark():
    return Benchmark(
        name="webarena_100",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["webarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=[f"webarena.{task_id}" for task_id in TASK_IDS],
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=task_metadata("webarena"),
    )

def get_visualwebarena_100_benchmark():
    return Benchmark(
        name="visualwebarena_100",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["visualwebarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=['visualwebarena'],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=[f"visualwebarena.{task_id}" for task_id in VWA_TASK_IDS],
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=task_metadata("visualwebarena"),
    )

def get_visualwebarena_100_benchmark_resized():
    from agent_reward_bench.envs.visualwebarena import ResizedVWABackend, resized_task_metadata

    task_df = resized_task_metadata()

    return ImprovedBenchmark(
        name="visualwebarena_100_resized",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["visualwebarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=[ResizedVWABackend()],
        env_args_list=modified_make_env_args_list_from_fixed_seeds(
            task_list=[f"visualwebarena.resized.{task_id}" for task_id in VWA_TASK_IDS],
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=task_df,
    )

def get_workarena_100_l2_benchmark():
    return Benchmark(
        name="workarena_l2_100",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["workarena++"],
        is_multi_tab=True,
        supports_parallel_seeds=True,
        backends=["workarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=WORKARENA_L2_TASK_IDS,
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=task_metadata("workarena"),
    )

def get_visualwebarena_benchmark_failed_tasks():
    from agent_reward_bench.envs.visualwebarena import ResizedVWABackend, resized_task_metadata

    tasks = (
        "visualwebarena.resized.244",
        "visualwebarena.resized.245",
        "visualwebarena.resized.247",
        "visualwebarena.resized.345",
        "visualwebarena.resized.372",
        "visualwebarena.resized.569",
        "visualwebarena.resized.570",
        "visualwebarena.resized.598",
        "visualwebarena.resized.600",
        "visualwebarena.resized.602",
        "visualwebarena.resized.608",
        "visualwebarena.resized.614",
        "visualwebarena.resized.730",
        "visualwebarena.resized.739",
    )

    task_df = resized_task_metadata()

    return ImprovedBenchmark(
        name="visualwebarena_100_failed_tasks_resized",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["visualwebarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=[ResizedVWABackend()],
        env_args_list=modified_make_env_args_list_from_fixed_seeds(
            task_list=tasks,
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=task_df,
    )
