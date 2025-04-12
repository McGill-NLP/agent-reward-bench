"""
The goal of this script is to convert the trajectories in the agentlab_results from pickles for each step into a single json file for each trajectory.
"""

import json
import orjson
from pathlib import Path

from tqdm.auto import tqdm

from agent_reward_bench.trajectories import TrajectoriesManager, list_experiments, Trajectory, Step

# this file is in scripts/ so we get the parent of the parent to get the project root
project_root_dir = Path(__file__).resolve().parent.parent

results_base_dir = project_root_dir / "agentlab_results"
base_save_dir = project_root_dir / "trajectories" / "cleaned"


trajectories_manager = TrajectoriesManager()
experiments = list_experiments(base_dir=results_base_dir)
trajectories_manager.add_trajectories_from_dirs(experiments)
trajectories_manager.build_index()

for benchmark in tqdm(trajectories_manager.get_benchmarks(), desc="Benchmarks"):
    for model in tqdm(trajectories_manager.get_model_names(benchmark), desc="Models", leave=False):
        exps = trajectories_manager.get_exp_names(benchmark, model)
        # if there's more than one experiment, or less than one, we have a problem
        assert len(exps) == 1, f"Expected 1 experiment for {benchmark} {model}, got {len(exps)}"

        exp = exps[0]
        trajectories = trajectories_manager.get_trajectories(benchmark, model, exp)

        for traj in tqdm(trajectories, desc="Trajectories", leave=False):
            traj_save_dir: Path = base_save_dir / benchmark / model / exp
            traj_save_dir.mkdir(parents=True, exist_ok=True)
            save_path = traj_save_dir / f"{traj.task_id}.json"
            
            # if the save path already exists, skip
            if save_path.exists():
                continue
            
            traj_dict = traj.to_dict()
            traj_dict['steps'] = [Step.from_mini_dict(s).to_dict(prune_axtree=True) for s in traj]

            with open(save_path, 'wb') as fb:
               # use orjson for faster serialization
                fb.write(orjson.dumps(traj_dict))
