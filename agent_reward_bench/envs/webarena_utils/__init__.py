# # modified from browsergym/webarena/__init__.py
# from . import config, instance

# def register_webarena_100():
#     from . import task

#     import nltk

#     from browsergym.core.registration import register_task
#     # download necessary tokenizer ressources
#     # note: deprecated punkt -> punkt_tab https://github.com/nltk/nltk/issues/3293
#     try:
#         nltk.data.find("tokenizers/punkt_tab")
#     except:
#         nltk.download("punkt_tab", quiet=True, raise_on_error=True)

#     ALL_WEBARENA_TASK_IDS = []

#     # register all WebArena benchmark
#     for task_id in config.TASK_IDS:
#         gym_id = f"webarena.{task_id}"
#         register_task(
#             gym_id,
#             task.GenericWebArenaTask,
#             task_kwargs={"task_id": task_id},
#         )
#         ALL_WEBARENA_TASK_IDS.append(gym_id)

import csv

from browsergym.experiments.benchmark.metadata.utils import (
    task_metadata,
)
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

# TASK_IDS = range(100)
import numpy as np
import pkg_resources
import json
from typing import List

def get_task_ids_old(package='agent_reward_bench') -> List[int]:
    task_ids_path = pkg_resources.resource_filename(package, 'data/webarena.task_ids.json')
    
    with open(task_ids_path) as f:
        task_ids = json.load(f)
    
    assert isinstance(task_ids, list), f"Expected a list of task ids, got {task_ids}. This is an internal error that should be reported."

    return list(sorted(task_ids))


def get_task_ids_wa(package='agent_reward_bench', browsergym_split='test') -> List[int]:
    wa_csv = pkg_resources.resource_filename(package, 'data/webarena.csv')

    task_ids = []

    with open(wa_csv) as f:
        # get header as keys
        reader = csv.DictReader(f)
    
        for row in reader:
            if row['browsergym_split'] == browsergym_split:
                task_ids.append(int(row['task_id']))
    
    return list(sorted(task_ids))

def get_task_ids_vwa(package='agent_reward_bench', browsergym_split='test') -> List[int]:
    vwa_csv = pkg_resources.resource_filename(package, 'data/visualwebarena.csv')

    task_ids = []

    with open(vwa_csv) as f:
        # get header as keys
        reader = csv.DictReader(f)
    
        for row in reader:
            if row['browsergym_split'] == browsergym_split:
                task_ids.append(int(row['task_id']))
    
    return list(sorted(task_ids))

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

# class WebArenaBenchmarkWithoutReset(Benchmark):
#     def prepare_backends(self):
#         print("Preparing backends for WebArenaBenchmarkWithoutReset")
#         for backend in self.backends:
#             match backend:
#                 case "webarena":
#                     # register environments
#                     import browsergym.webarena

#                     # full reset the instance (requires environment variables properly set up)
#                     from browsergym.webarena.instance import WebArenaInstance

#                     default_instance = WebArenaInstance()
                    
#                     # default_instance.full_reset()  # no reset

#                 case _:
#                     raise ValueError(f"Unknown benchmark backend {repr(backend)}. Note this is the class BenchmarkWithoutReset, which is a subclass of Benchmark that does not support reset, and only supports the webarena backend.")
                
def get_webarena_100_benchmark():
    # TODO: Might want to switch back to `Backend` when WA_FULL_RESET issue is resolved
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

def get_webarena_100_benchmark_20steps():
    # TODO: Might want to switch back to `Backend` when WA_FULL_RESET issue is resolved
    return Benchmark(
        name="webarena_100_20_steps",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["webarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=[f"webarena.{task_id}" for task_id in TASK_IDS],
            max_steps=20,
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
        backends=["visualwebarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=[f"visualwebarena.{task_id}" for task_id in VWA_TASK_IDS],
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=task_metadata("visualwebarena"),
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
        # make_env_args_list_from_workarena_curriculum(
        #     level="l2",
        #     task_category_filter=None,
        #     meta_seed=42,  # meta seed for evaluation curriculum
        #     max_steps=50,
        #     curriculum_type="agent",
        # ),
        task_metadata=task_metadata("workarena"),
    )


def get_workarena_l1_split(split="test", num_repeat=10):
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

    

def get_webarena_gitlab_failing_cases():
    # The following URL matches have positive rewards on aws: {156: 1.0, 669: 1.0, 357: 1.0, 106: 1.0}
    task_list = [156, 669, 357, 106]

    return Benchmark(
        name="webarena_gitlab_failing_cases",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["webarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=[f"webarena.{task_id}" for task_id in task_list],
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=task_metadata("webarena"),
    )


def get_webarena_benchmark_split(split='test'):
    benchmark = Benchmark(
        name="webarena",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["webarena"],
        env_args_list=make_env_args_list_from_repeat_tasks(
            task_list=task_list_from_metadata(metadata=task_metadata("webarena")),
            max_steps=30,
            n_repeats=1,
            seeds_rng=np.random.RandomState(42),
        ),
        task_metadata=task_metadata("webarena"),
    )

    benchmark_split = benchmark.subset_from_split(split) # type: ignore

    return benchmark_split

def get_assistantbench_split(split='valid'):
    benchmark = Benchmark(
        name="assistantbench",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["assistantbench"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["assistantbench"],
        env_args_list=make_env_args_list_from_repeat_tasks(
            task_list=task_list_from_metadata(metadata=task_metadata("assistantbench")),
            max_steps=30,
            n_repeats=1,
            seeds_rng=np.random.RandomState(42),
        ),
        task_metadata=task_metadata("assistantbench"),
    )

    benchmark_split = benchmark.subset_from_split(split) # type: ignore

    return benchmark_split

def get_visualwebarena_benchmark_split(split='test'):
    benchmark = Benchmark(
        name="visualwebarena",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["visualwebarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["visualwebarena"],
        env_args_list=make_env_args_list_from_repeat_tasks(
            task_list=task_list_from_metadata(metadata=task_metadata("visualwebarena")),
            max_steps=30,
            n_repeats=1,
            seeds_rng=np.random.RandomState(42),
        ),
        task_metadata=task_metadata("visualwebarena"),
    )

    benchmark_split = benchmark.subset_from_split(split) # type: ignore

    return benchmark_split
