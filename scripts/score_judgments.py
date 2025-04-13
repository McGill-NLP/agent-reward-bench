import argparse
from collections import defaultdict
import json
import datetime as dt
import os
from pathlib import Path
import csv
import importlib.resources

import agent_reward_bench.eval.metrics as arbmetrics
from agent_reward_bench.eval.metrics import mean, numerize, calculate_agreement_rate
from agent_reward_bench.judge import (
    parse_judgment,
    get_response_msg,
    parse_aer_judgment,
    parse_nnetnav_judgment,
)
from agent_reward_bench.judge.utils import (
    is_unsure,
    dictify,
    rename_records,
    get_renames,
    get_judges,
    save_as_csv,
    flatten_dict_to_records,
    normalize_task_id,
    filter_combined_records_by_split,
    infer_annotator_type,
    is_valid_judgment,
)


def create_annotator_pairs(combined_records):
    # now, create pairs of primary and secondary annotators
    pairs = {}

    for r in combined_records:
        cat = (r["human"]["benchmark"], r["human"]["model_name"], r["human"]["task_id"])
        if cat not in pairs:
            pairs[cat] = {"primary": None, "secondary": None}

        if r["annotator"] == "primary":
            pairs[cat]["primary"] = r

        if r["annotator"] == "secondary":
            pairs[cat]["secondary"] = r

    # remove pairs that do not have both primary and secondary annotators
    pairs = {
        k: v
        for k, v in pairs.items()
        if v["primary"] is not None and v["secondary"] is not None
    }

    return pairs


def combine_annotations_into_records(
    annotations, judgments_base_dir, judge, skipped_agents
):
    splits_path = (
        importlib.resources.files("agent_reward_bench") / "data" / "splits.csv"
    )
    with open(splits_path, "r") as f:
        splits = list(csv.DictReader(f))

    # convert splits to a dictionary
    splits_dict = {split["task_id"]: split["split"] for split in splits}

    combined_records = []
    models_seen = set()
    datasets_seen = set()
    num_skipped_missing_judgments = 0
    num_skipped_missing_responses = 0

    existing_annotations = set()
    for annotation in annotations:
        if annotation["model_name"] in skipped_agents:
            continue

        judgment_path = Path(
            judgments_base_dir,
            annotation["benchmark"],
            annotation["model_name"],
            judge,
            f"{annotation['task_id']}.json",
        )

        if not judgment_path.exists():
            print(
                f"\tSkipping missing judgment for {judge}, {annotation['benchmark']}, {annotation['model_name']}, {annotation['task_id']}"
            )
            num_skipped_missing_judgments += 1
            continue

        with open(judgment_path, "r") as f:
            judgment = json.load(f)

        cum_reward = judgment["trajectory_info"]["summary_info"]["cum_reward"]

        if not is_valid_judgment(judgment):
            num_skipped_missing_responses += 1
            parsed_judgment = {
                "reasoning": None,
                "trajectory_success": "n/a",
                "trajectory_side_effect": "n/a",
                "trajectory_optimality": "n/a",
                "trajectory_looping": "n/a",
            }

        elif judge == "functional":
            success = "success" if cum_reward > 0.5 else "failure"
            # just return the cum_reward
            parsed_judgment = {
                "reasoning": None,
                "trajectory_success": success,
                "trajectory_side_effect": "n/a",
                "trajectory_optimality": "n/a",
                "trajectory_looping": "n/a",
            }
        elif judge in ["aer", "aerv"]:
            parsed_judgment = parse_aer_judgment(get_response_msg(judgment["response"]))
        elif judge == "nnetnav":
            parsed_judgment = parse_nnetnav_judgment(
                get_response_msg(judgment["response"])
            )
        else:
            parsed_judgment = parse_judgment(get_response_msg(judgment["response"]))

        annotator_type = infer_annotator_type(annotation, existing_annotations)

        combined_records.append(
            {
                "judge": parsed_judgment,
                "human": annotation,
                "cum_reward": cum_reward,
                "annotator": annotator_type,
                "split": splits_dict[normalize_task_id(annotation["task_id"])],
            }
        )

        models_seen.add(annotation["model_name"])
        datasets_seen.add(annotation["benchmark"])

    print(f"Skipped {num_skipped_missing_judgments} missing judgments")
    print(f"Found {num_skipped_missing_responses} judgments without responses")

    # check for test and dev splits only for primary annotators
    num_test = len(
        [
            r
            for r in combined_records
            if r["split"] == "test" and r["annotator"] == "primary"
        ]
    )
    num_dev = len(
        [
            r
            for r in combined_records
            if r["split"] == "dev" and r["annotator"] == "primary"
        ]
    )
    print(f"Number of test trajectories: {num_test}")
    print(f"Number of dev trajectories: {num_dev}")

    return combined_records, models_seen, datasets_seen


def process_results_and_metrics(combined_records, benchmarks, models, judge):
    results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    success_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    num_missing = 0

    for benchmark in benchmarks + ["all"]:
        for model in list(models) + ["all"]:
            for cat in [
                "trajectory_success",
                "trajectory_side_effect",
                "trajectory_looping",
                "trajectory_optimality",
            ]:
                results[judge][benchmark][model][cat] = {"y_human": {}, "y_judge": {}}
                recs_filt = [
                    r
                    for r in combined_records
                    if not is_unsure(r["human"][cat])
                    and r["annotator"] == "primary"
                    and (r["human"]["model_name"] == model or model == "all")
                ]

                if benchmark == "workarena":
                    recs_filt = [
                        r
                        for r in recs_filt
                        if (
                            r["human"]["benchmark"] == "workarena"
                            and "l2" not in r["human"]["task_id"].lower()
                        )
                    ]
                elif benchmark == "workarena++":
                    recs_filt = [
                        r
                        for r in recs_filt
                        if (
                            r["human"]["benchmark"] == "workarena"
                            and "l2" in r["human"]["task_id"].lower()
                        )
                    ]
                else:
                    recs_filt = [
                        r
                        for r in recs_filt
                        if (r["human"]["benchmark"] == benchmark or benchmark == "all")
                    ]

                y_true = [
                    numerize(r["human"][cat], raise_if_none=True) for r in recs_filt
                ]
                y_pred = [
                    numerize(r["judge"][cat], warn_if_none=False) for r in recs_filt
                ]

                results[judge][benchmark][model][cat]["y_human"] = y_true
                results[judge][benchmark][model][cat]["y_judge"] = y_pred

                if cat == "trajectory_optimality":
                    val = {
                        "accuracy": arbmetrics.accuracy(y_true, y_pred),
                        "unsures": arbmetrics.calculate_unsures(combined_records, cat),
                        "total": len(y_true),
                    }
                else:
                    # get accuracy, precision, recall, f1 score
                    val = {
                        "accuracy": arbmetrics.accuracy(
                            y_true, y_pred, ndigits=1, percentage=True
                        ),
                        "precision": arbmetrics.precision(
                            y_true, y_pred, ndigits=1, percentage=True
                        ),
                        "recall": arbmetrics.recall(
                            y_true, y_pred, ndigits=1, percentage=True
                        ),
                        "f1": arbmetrics.f1(y_true, y_pred, ndigits=1, percentage=True),
                        "npv": arbmetrics.npv(
                            y_true, y_pred, ndigits=1, percentage=True
                        ),
                        "tnr": arbmetrics.tnr(
                            y_true, y_pred, ndigits=1, percentage=True
                        ),
                        "unsures": arbmetrics.calculate_unsures(combined_records, cat),
                        "total": len(y_true),
                    }

                metrics[judge][benchmark][model][cat] = val

                if cat == "trajectory_success":
                    y_human = results[judge][benchmark][model][cat]["y_human"]
                    y_judge = results[judge][benchmark][model][cat]["y_judge"]
                    y_funct = [r["cum_reward"] for r in recs_filt]

                    success_rates[judge][benchmark][model] = {
                        "human_sr": mean(y_human),
                        "judge_sr": mean(y_judge),
                        "function_sr": mean(y_funct),
                        "count": len(y_human),
                    }

                num_missing += len([r for r in recs_filt if r["judge"][cat] is None])

    results = dictify(results)
    metrics = dictify(metrics)
    success_rates = dictify(success_rates)

    print(f"Found {num_missing} missing judgments, set to 0")

    return results, metrics, success_rates


def filter_mismatches(combined_records):
    """
    Filter out records where the human and judge judgments do not match, only trajectory_success
    """

    filtered = {
        "false_positive": [],
        "false_negative": [],
    }
    for r in combined_records:
        human = numerize(r["human"]["trajectory_success"])
        judge = numerize(r["judge"]["trajectory_success"])
        if human == 0 and judge == 1:
            filtered["false_positive"].append(r)
        elif human == 1 and judge == 0:
            filtered["false_negative"].append(r)

    return filtered


def get_judge_records(judgments_base_dir):
    judge_records = []
    # first find all .json files in the judgments_base_dir
    judge_paths = list(Path(judgments_base_dir).glob("**/*.json"))

    for judge_path in judge_paths:
        with open(judge_path, "r") as f:
            judgment = json.load(f)

        cum_reward = judgment["trajectory_info"]["summary_info"]["cum_reward"]

        # if it's a functional judge, we do not skip
        if judgment.get("judge") == "functional":
            response = success = "success" if cum_reward > 0.5 else "failure"
            parsed_judgment = {
                "reasoning": None,
                "trajectory_success": success,
                "trajectory_side_effect": "n/a",
                "trajectory_optimality": "n/a",
                "trajectory_looping": "n/a",
            }
        else:
            if "response" not in judgment:
                print(f"Skipping judgment without response: {judge_path}")
                continue
            if judgment["response"] is None:
                print(f"Skipping judgment with None response: {judge_path}")
                continue
            if judgment["response"]["choices"] is None:
                print(f"Skipping judgment with None choices: {judge_path}")
                continue
            if len(judgment["response"]["choices"]) == 0:
                print(f"Skipping judgment with empty choices: {judge_path}")
                continue

            response = get_response_msg(judgment["response"])
            parsed_judgment = parse_judgment(response)

        judge_records.append(
            {
                "benchmark": judgment["benchmark"],
                "agent": judgment["agent"],
                "judge": judgment["judge"],
                "response": response,
                "parsed_judgment": parsed_judgment,
                "cum_reward": cum_reward,
            }
        )

    return judge_records


parser = argparse.ArgumentParser(
    description="Score judgments and create metrics and success rates",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--split",
    type=str,
    default="test",
    choices=["test", "dev"],
    help="The split to analyze",
)
parser.add_argument(
    "--judgments_base_dir",
    type=str,
    default="trajectories/judgments/",
    help="The base directory where the judgments are stored",
)
parser.add_argument(
    "--results_save_dir",
    type=str,
    default="artifacts/",
    help="The base directory where the results will be saved",
)

args = parser.parse_args()
split = args.split

judgments_base_dir = args.judgments_base_dir
results_save_dir = args.results_save_dir

judges = get_judges()

arb_path = importlib.resources.files("agent_reward_bench")
with open(arb_path / "data" / "annotations.csv") as f:
    reader = csv.DictReader(f)
    annotations = list(reader)

benchmarks = [
    "assistantbench",
    "webarena",
    "visualwebarena",
    "workarena",
    "workarena++",
]

agent_models = [
    "GenericAgent-anthropic_claude-3.7-sonnet",
    "GenericAgent-gpt-4o-2024-11-20",
    "GenericAgent-meta-llama_Llama-3.3-70B-Instruct",
    "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct",
]

print(f"Analyzing benchmarks: {benchmarks}\n")

full_results = {}
full_metrics_recs = []
full_success_rates_recs = []
full_mismatched_dict = defaultdict(list)
judge_records = get_judge_records(judgments_base_dir)

for judge in judges:
    print("-" * 80)
    print("Processing judge:", judge)
    combined_records, models_seen, datasets_seen = combine_annotations_into_records(
        annotations, judgments_base_dir, judge, skipped_agents=[]
    )
    combined_records = filter_combined_records_by_split(combined_records, split)

    results, metrics, success_rates = process_results_and_metrics(
        combined_records, benchmarks=benchmarks, models=agent_models, judge=judge
    )

    metrics_recs = flatten_dict_to_records(
        metrics,
        level_names=["judge", "benchmark", "agent", "category"],
        expand_final_dict=True,
    )
    success_rates_recs = flatten_dict_to_records(
        success_rates,
        level_names=["judge", "benchmark", "agent"],
        expand_final_dict=True,
    )

    # get judge records
    full_results.update(results)
    full_metrics_recs.extend(metrics_recs)
    full_success_rates_recs.extend(success_rates_recs)

    mismatched = filter_mismatches(combined_records)
    print(
        f"Found {len(mismatched['false_positive'])} false positives and {len(mismatched['false_negative'])} false negatives"
    )

    # save the mismatched records
    full_mismatched_dict[judge] = mismatched

    print(f"Added {len(metrics_recs)} metrics records")
    print(f"Added {len(success_rates_recs)} success rate records")

# calculate the agreement rate
annotator_pairs = create_annotator_pairs(combined_records)
agreement_rate = calculate_agreement_rate(
    y_prim=arbmetrics.get_annotator_scores(annotator_pairs, "primary"), 
    y_sec=arbmetrics.get_annotator_scores(annotator_pairs, "secondary")
)

# save as json
results_save_dir = Path(results_save_dir)
results_save_dir.mkdir(parents=True, exist_ok=True)
with open(results_save_dir / "results.json", "w") as f:
    json.dump(full_results, f, indent=2)

# save agreement rate
with open(results_save_dir / "agreement_rate.json", "w") as f:
    json.dump(agreement_rate, f, indent=2)

renames = get_renames()
rename_records(full_metrics_recs, "agent", renames['agents'])
rename_records(full_success_rates_recs, "agent", renames['agents'])
rename_records(full_metrics_recs, "category", renames['labels'])
rename_records(full_metrics_recs, "benchmark", renames['benchmarks'])
rename_records(full_metrics_recs, "judge", renames['judges'])

save_as_csv(full_metrics_recs, results_save_dir / f"metrics_{split}.csv")
save_as_csv(full_success_rates_recs, results_save_dir / f"success_rates_{split}.csv")

# show the trajectory_success metrics (f1, etc.) for "all" benchmarks and "all" models
print("-" * 80)
print("Metrics for Success")
for metric in full_metrics_recs:
    if (
        metric["category"] == "Success"
        and metric["benchmark"] == "Overall"
        and metric["agent"] == "All"
    ):
        print(metric)
