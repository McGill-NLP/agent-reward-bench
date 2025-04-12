import argparse
import itertools
import os
from pathlib import Path

try:
    import orjson
except ImportError:
    import json as orjson

import openai
from tqdm.auto import tqdm

from agent_reward_bench.judge import create_chat_messages_from_trajectory
from agent_reward_bench.judge.existing import aer, nnetnav
from agent_reward_bench.judge.args import default_judge_args, judge_args

from agent_reward_bench.utils import (
    get_api_key_from_env_var,
    get_base_url_from_env_var,
    CostEstimator,
)

benchmarks = ["assistantbench", "webarena", "visualwebarena", "workarena"]
agents = [
    "GenericAgent-gpt-4o-2024-11-20",
    "GenericAgent-anthropic_claude-3.7-sonnet",
    "GenericAgent-meta-llama_Llama-3.3-70B-Instruct",
    "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct",
]


parser = argparse.ArgumentParser(
    description="Run the judge on a set of trajectories",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--judge",
    type=str,
    default="gpt-4o-screen",
    choices=judge_args.keys(),
    help="The judge model to use",
)
parser.add_argument(
    "--base_dir",
    type=str,
    default="trajectories/cleaned",
    help="The base directory where the trajectories are stored",
)
parser.add_argument(
    "--base_save_dir",
    type=str,
    default="trajectories/judgments",
    help="The base directory where the judgments will be saved",
)
parser.add_argument(
    "--agents",
    nargs="+",
    default=agents,
    help="Select the agent to use",
)
args = parser.parse_args()

agents = args.agents
judge = args.judge
base_dir = Path(args.base_dir)
base_save_dir = Path(args.base_save_dir)

max_completion_tokens = judge_args[judge].get(
    "max_completion_tokens", default_judge_args["max_completion_tokens"]
)
temperature = judge_args[judge].get("temperature", default_judge_args["temperature"])
seed = judge_args[judge].get("seed", default_judge_args["seed"])
use_screenshot = judge_args[judge]["use_screenshot"]
use_axtree = judge_args[judge]["use_axtree"]
invert_sys_prompt = judge_args[judge].get("invert_sys_prompt", False)
provider = judge_args[judge]["provider"]
judge_model_name = judge_args[judge]["model_name"]


base_url = get_base_url_from_env_var(provider)
if "llama" in judge_model_name.lower():
    base_url = os.getenv("VLLM_LLAMA_BASE_URL", base_url)
client = openai.OpenAI(api_key=get_api_key_from_env_var(provider), base_url=base_url)

cost_across_runs = 0
for agent, benchmark in itertools.product(agents, benchmarks):
    # get all json files that in the path tree of the benchmark and model
    traj_dir = Path(base_dir, benchmark, agent)
    save_dir = Path(base_save_dir, benchmark, agent, judge.replace("/", "_"))

    trajectories_paths = list(traj_dir.glob("**/*.json"))

    estimator = CostEstimator.from_model_name(judge_model_name)
    print("\nAgent:", agent)
    print("Benchmark:", benchmark)
    print("Judge:", judge)
    print("Judge Model Name:", judge_model_name)
    print("Provider:", provider)

    pbar = tqdm(trajectories_paths, desc="Running judge")

    for path in pbar:
        save_path = save_dir.joinpath(f"{path.stem}.json")
        # if the file already exists, skip
        if save_path.exists():
            continue

        # with open(path, "r") as f:
        #     trajectory = json.load(f)
        with open(path, "rb") as f:
            trajectory = orjson.loads(f.read())

        # if it's not valid, skip
        if not trajectory["valid"]:
            print(f"Skipping {path} because it's not valid")
            continue

        # show the trajectory id in the progress bar
        pbar.set_postfix(
            task_id=path.stem.split(".")[-1], estimator=estimator.total_cost
        )

        if judge == "functional":
            results = {
                "benchmark": trajectory["benchmark"],
                "goal": trajectory["goal"],
                "agent": trajectory["agent"],
                "judge": judge,
                "judge_model_name": "judge",
                "provider": provider,
                "cost": 0,
                "trajectory_info": {
                    "valid": trajectory["valid"],
                    "model": trajectory["model"],
                    "trajectory_dir": trajectory["trajectory_dir"],
                    "seed": trajectory["seed"],
                    "model_args": trajectory["model_args"],
                    "flags": trajectory["flags"],
                    "summary_info": trajectory["summary_info"],
                },
            }
            save_dir.mkdir(parents=True, exist_ok=True)
            # with open(save_path, "w") as f:
            #     json.dump(results, f)
            with open(save_path, "wb") as f:
                f.write(orjson.dumps(results))
            continue

        if judge == "aer":
            chat_messages = aer.create_aer_chat_messages_from_trajectory(
                client=client,
                trajectory=trajectory,
                traj_dir=traj_dir,
                path=path,
            )
        elif judge == "aerv":
            chat_messages = aer.create_aer_chat_messages_from_trajectory_vis(
                client=client,
                trajectory=trajectory,
                traj_dir=traj_dir,
            )
        elif judge == "nnetnav":
            chat_messages = nnetnav.create_nnetnav_chat_messages_from_trajectory(
                trajectory=trajectory,
                path=path,
                client=client,
            )
        else:
            chat_messages = create_chat_messages_from_trajectory(
                trajectory=trajectory,
                traj_dir=traj_dir,
                use_screenshot=use_screenshot,
                use_axtree=use_axtree,
                invert_system_prompt=invert_sys_prompt,
            )

        try:
            response = client.chat.completions.create(
                model=judge_model_name,
                messages=chat_messages["regular"],
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                seed=seed,
            )
        except openai.BadRequestError as e:
            try:
                response = client.chat.completions.create(
                    model=judge_model_name,
                    messages=chat_messages["pruned"],
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    seed=seed,
                )
            except openai.BadRequestError as e:
                # if the request fails, create an error file and continue
                save_dir.mkdir(parents=True, exist_ok=True)
                # change save_path to add a suffix to indicate that it's an error file
                save_path = save_dir.joinpath(f"{path.stem}_error.json")
                results = {
                    "error": str(e),
                    "chat_messages": chat_messages,
                    "trajectory_info": {
                        "valid": trajectory["valid"],
                        "model": trajectory["model"],
                        "trajectory_dir": trajectory["trajectory_dir"],
                        "seed": trajectory["seed"],
                        "model_args": trajectory["model_args"],
                        "flags": trajectory["flags"],
                        "summary_info": trajectory["summary_info"],
                    },
                }
                # with open(save_path, "w") as f:
                #     json.dump(results, f)
                with open(save_path, "wb") as f:
                    f.write(orjson.dumps(results))
                
                print(f"Error in {path}: {e}")
                continue

        results = {
            "benchmark": trajectory["benchmark"],
            "goal": trajectory["goal"],
            "agent": trajectory["agent"],
            "judge": judge,
            "judge_model_name": judge_model_name,
            "provider": provider,
            "judge_args": {"use_screenshot": use_screenshot, "use_axtree": use_axtree, "invert_sys_prompt": invert_sys_prompt},
            "completion_args": {
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
                "seed": seed,
            },
            "cost": estimator.estimate_cost(response, finegrained=True),
            "response": response.model_dump(),
            "chat_messages": chat_messages,
            "trajectory_info": {
                "valid": trajectory["valid"],
                "model": trajectory["model"],
                "trajectory_dir": trajectory["trajectory_dir"],
                "seed": trajectory["seed"],
                "model_args": trajectory["model_args"],
                "flags": trajectory["flags"],
                "summary_info": trajectory["summary_info"],
            },
        }

        save_dir.mkdir(parents=True, exist_ok=True)
        # with open(save_path, "w") as f:
        #     json.dump(results, f)
        with open(save_path, "wb") as f:
            f.write(orjson.dumps(results))

        estimator.increment_cost(response)

    cost_across_runs += estimator.total_cost

print(f"Total cost across runs: ${round(cost_across_runs, 6)}")
