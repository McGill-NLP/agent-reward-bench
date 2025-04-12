import logging
import typing

import pandas as pd
from browsergym.experiments.benchmark.base import Benchmark, BenchmarkBackend, prepare_backend

logger = logging.getLogger(__name__)

class ImprovedBenchmark(Benchmark):
    def __post_init__(self):
        # if no metadata is present, generate a dataframe with single "task_name" column
        if self.task_metadata is None:
            unique_task_names = list(set([env_args.task_name for env_args in self.env_args_list]))
            self.task_metadata = pd.DataFrame(
                [{"task_name": task_name} for task_name in unique_task_names]
            )
        # make sure all tasks in env_args are in the metadata
        metadata_tasks = list(self.task_metadata["task_name"])
        assert all([env_args.task_name in metadata_tasks for env_args in self.env_args_list])
        # check backend values
        for backend in self.backends:
            if not hasattr(backend, 'prepare') and backend not in typing.get_args(BenchmarkBackend):
                raise ValueError(
                    f"Unknown Benchmark backend {repr(backend)}. Available backends: {typing.get_args(BenchmarkBackend)}"
                )
    
    def prepare_backends(self):
        for backend in self.backends:
            if hasattr(backend, 'prepare'):
                logger.info(f"Found prepare method for {backend} backend...")
                backend.prepare()
            else:
                logger.info(f"Preparing {backend} backend...")
                prepare_backend(backend)
            logger.info(f"{backend} backend ready")