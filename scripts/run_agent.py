"""
To run:

```bash
export SUFFIX="xl-0"

source vars/set_cf_vars.sh
python run_agent.py -b "webarena_100" -m "gpt-4o-mini"
"""
import argparse
import logging
from pathlib import Path
import os

# this needs to be set before importing agentlab
default_exp_root = str(Path(__file__).parent / "agentlab_results")
os.environ["AGENTLAB_EXP_ROOT"] = os.getenv("AGENTLAB_EXP_ROOT", default_exp_root)

from agentlab.experiments.study import Study

import agent_reward_bench.modeling as arbm
import agent_reward_bench.benchmarks as arbb

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(
    description="Run a generic agent on a benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-r",
    "--relaunch",
    type=str,
    help="Relaunch an existing study with a string that matches the study name",
)

parser.add_argument(
    "-b",
    "--benchmark",
    choices=[
        "assistantbench_valid",
        "workarena_l1",
        "workarena_100_l2",
        "webarena_100",
        "visualwebarena_100",
        "visualwebarena_100_resized",
        "vwa_failed",
    ],
    default="webarena_100",
    help="Select the benchmark to run on",
)

parser.add_argument(
    "-m",
    "--models",
    choices=[
        "claude-3.7-sonnet",
        "gpt-4o",
        "gpt-4o-mini",
        "qwen-2.5-vl",  # only vllm
        "llama-3.3-70b",
    ],
    default="gpt-4o",
    help="Select the model to use",
    # allow multiple models, but at least one
    nargs="+",
)
parser.add_argument(
    "-n",
    "--n_jobs",
    type=int,
    default=4,
    help="Number of parallel jobs to run",
)
parser.add_argument(
    "--parallel",
    choices=["ray", "joblib", "sequential"],
    default="ray",
    help="Select the parallel backend to use",
)
args = parser.parse_args()

# first, select benchmark:

if args.benchmark == "workarena_l1":
    benchmark = arbb.get_workarena_l1_split(split="test")
elif args.benchmark == "webarena_100":
    benchmark = arbb.get_webarena_100_benchmark()
elif args.benchmark == "visualwebarena_100":
    benchmark = arbb.get_visualwebarena_100_benchmark()
elif args.benchmark == "visualwebarena_100_resized":
    benchmark = arbb.get_visualwebarena_100_benchmark_resized()
elif args.benchmark == "workarena_100_l2":
    benchmark = arbb.get_workarena_100_l2_benchmark()
elif args.benchmark == "assistantbench_valid":
    # duckduckgo might have fewer captcha issues
    benchmark = arbb.get_assistantbench_split(split="valid", start_url="https://duckduckgo.com")
elif args.benchmark == "vwa_failed":
    benchmark = arbb.get_visualwebarena_benchmark_failed_tasks()
else:
    raise ValueError(f"Unknown benchmark {args.benchmark}")

# then, select model:
# since args.models is a list now, we'll make a list of agent_args
agent_args = []
for model in args.models:
    if model == "gpt-4o":
        # must set OPENAI_API_KEY in environment
        agent_args.append(arbm.prepare_gpt(model_name='gpt-4o-2024-11-20'))
    elif model == "gpt-4o-mini":
        # must set OPENAI_API_KEY in environment
        agent_args.append(arbm.prepare_gpt(model_name='gpt-4o-mini-2024-07-18'))
    elif model == "claude-3.7-sonnet":
        # must set OPENROUTER_API_KEY and OPENROUTER_BASE_URL in environment
        agent_args.append(arbm.prepare_claude(model_name="anthropic/claude-3.7-sonnet"))
    elif model == "qwen-2.5-vl":
        # must set VLLM_API_KEY and VLLM_BASE_URL in environment
        agent_args.append(arbm.prepare_vllm_model(
            model_name="Qwen/Qwen2.5-VL-72B-Instruct",
        ))
    elif model == "llama-3.3-70b":
        # must set VLLM_API_KEY and VLLM_BASE_URL in environment
        agent_args.append(arbm.prepare_vllm_model(
            model_name="meta-llama/Llama-3.3-70B-Instruct",
            use_vision=False,
            enable_chat=False,
            base_url="https://vllm-llama.mcgill-nlp.org/v1"
        ))
    else:
        raise ValueError(f"Unknown model {model}")

reproducibility_mode = True
strict_reproducibility = False
n_relaunch = 3
parallel_backend = args.parallel
n_jobs = args.n_jobs


if __name__ == "__main__":  # necessary for dask backend
    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if args.relaunch is not None:
        print("Relaunching study from directory containing:", args.relaunch)
        study = Study.load_most_recent(contains=args.relaunch, root_dir=Path(os.environ["AGENTLAB_EXP_ROOT"]))
        study.find_incomplete(include_errors=True)
    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.INFO)  # type: ignore

    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=strict_reproducibility,
        n_relaunch=n_relaunch,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=strict_reproducibility)
