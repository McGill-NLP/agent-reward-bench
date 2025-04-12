import base64
import csv
import datetime
import json
from pathlib import Path
from typing import List


def is_unsure(label):
    return label == "Unsure" or label is None


def dictify(obj):
    # use json to convert object to str to dict
    return json.loads(json.dumps(obj))


def rename_records(records, key, mapping):
    for rec in records:
        if rec[key] in mapping:
            rec[key] = mapping[rec[key]]


def image_to_base64(image_path: Path) -> str:
    """Convert an image file to a Base64-encoded string."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as image_file:
        x = base64.b64encode(image_file.read()).decode("utf-8")
    # if it's a png image, change the format to png
    if image_path.suffix == ".png":
        return f"data:image/png;base64,{x}"
    elif image_path.suffix.lower() in [".jpg", ".jpeg"]:
        return f"data:image/jpeg;base64,{x}"
    else:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")


def normalize_task_id(task_id: str):
    remove_lst = [".resized", ".improved"]
    for remove in remove_lst:
        if remove in task_id:
            task_id = task_id.replace(remove, "")
    return task_id


def get_list_of_keys(records: List[dict]):
    if not isinstance(records, list):
        raise ValueError("Input must be a list of dictionaries")
    if not isinstance(records[0], dict):
        raise ValueError("Input must be a list of dictionaries")

    keys = set()
    lst = []

    for record in records:
        for k in record.keys():
            if k not in keys:
                keys.add(k)
                lst.append(k)

    return lst


def save_as_csv(
    records, filename, extrasaction="ignore", write_mode="w", fieldnames=None
):
    with open(filename, mode=write_mode) as f:
        if fieldnames is None:
            fieldnames = get_list_of_keys(records)

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
        writer.writeheader()
        writer.writerows(records)


def has_later_timestamp(new, prev):
    # e.g.  "timestamp": "2025-02-28T15:05:49.056721",
    # use datetime.fromisoformat() to parse the timestamp, then compare
    from datetime import datetime

    current_timestamp = datetime.fromisoformat(new["timestamp"])
    other_timestamp = datetime.fromisoformat(prev["timestamp"])

    return current_timestamp > other_timestamp


def flatten_dict_to_records(
    d, level_names, expand_final_dict=False, enable_override=False, prefix="", suffix=""
):
    """
    Flatten a dictionary to a list of records, where each record is a dictionary
    with keys from level_names and values from the original dictionary.
    """

    for name in level_names:
        if name == "_value":
            raise ValueError("_value is a reserved key for the value of the dictionary")

        if name == "value":
            final_val = "_value"
        else:
            final_val = "value"

    def get_max_depth(d):
        if not isinstance(d, dict):
            return 0
        return 1 + max(get_max_depth(v) for v in d.values())

    # check if level_names match the depth of the dictionary
    max_d = get_max_depth(d)

    if level_names is None:
        level_names = [f"level_{i}" for i in range(max_d)]

    records = []

    def flatten_dict(d, level_names, record):
        if len(level_names) == 0:
            if expand_final_dict:
                for k, v in d.items():
                    new_key = prefix + k + suffix
                    if not enable_override and new_key in record:
                        raise ValueError(
                            f"Key {new_key} already exists in record. Set enable_override=True to override"
                        )

                    record[new_key] = v
            else:
                record[final_val] = d
            records.append(record)
            return

        level_name = level_names[0]
        for k, v in d.items():
            new_record = record.copy()
            new_record[level_name] = k
            flatten_dict(v, level_names[1:], new_record)

    flatten_dict(d, level_names, {})
    return records


def filter_combined_records_by_split(combined_records, split):
    return [r for r in combined_records if r["split"] == split]



def infer_annotator_type(annotation, existing_annotations):
    """
    Warning: This function modifies the existing_annotations set
    """
    unique_key = (
        annotation["benchmark"],
        annotation["model_name"],
        annotation["task_id"],
    )

    if unique_key in existing_annotations:
        return "secondary"
    else:
        existing_annotations.add(unique_key)
        return "primary"

def remove_duplicate_annotations(annotations, verbose=True, sort_by_timestamp=True):
    if verbose:
        show = print
    else:
        show = lambda *args: None

    unique_annotations_dict = {}
    for annotation in annotations:
        key = tuple(
            annotation[k]
            for k in ["annotator_name", "benchmark", "model_name", "task_id"]
        )
        if key not in unique_annotations_dict:
            unique_annotations_dict[key] = annotation
        else:
            show("Found duplicate annotation with later timestamp")
            show("Prev annotation:", unique_annotations_dict[key]["timestamp"])
            show("New annotation:", annotation["timestamp"])
            show(key)

            # find the latest timestamp and keep that one
            if has_later_timestamp(new=annotation, prev=unique_annotations_dict[key]):
                show("Replacing previous annotation with new annotation\n")
                unique_annotations_dict[key] = annotation
            else:
                show("Keeping previous annotation\n")

    unique_annotations = list(unique_annotations_dict.values())

    show(
        f"\nRemoved {len(annotations) - len(unique_annotations)} duplicate annotations\n"
    )

    if sort_by_timestamp:
        unique_annotations = sorted(
            unique_annotations,
            key=lambda x: datetime.datetime.strptime(x["timestamp"], "%Y-%m-%dT%H:%M:%S.%f"),
            reverse=True,
        )

    return unique_annotations


def is_valid_judgment(judgment):
    if judgment.get("judge") == "functional":
        return True
    if "response" not in judgment:
        return False
    if judgment["response"] is None:
        return False
    if judgment["response"]["choices"] is None:
        return False
    if len(judgment["response"]["choices"]) == 0:
        return False
    return True


def get_renames():

    renames_agents = {
        "GenericAgent-anthropic_claude-3.7-sonnet": "Claude 3.7 Sonnet",
        "GenericAgent-gpt-4o-2024-11-20": "GPT-4o",
        "GenericAgent-meta-llama_Llama-3.3-70B-Instruct": "Llama 3.3",
        "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL",
        "all": "All",
    }
    renames_benchmarks = {
        "all": "Overall",
        "assistantbench": "AssistantBench",
        "webarena": "WebArena",
        "visualwebarena": "VisualWebArena",
        "workarena": "WorkArena",
        "workarena++": "WorkArena++",
    }

    renames_judges = {
        "aer": "AER-C",
        "aerv": "AER-V",
        "nnetnav": "NNetNav",
        "claude-3.7-sonnet-noscreen": "Claude 3.7 Sonnet (Axtree)",
        "claude-3.7-sonnet-noaxtree": "Claude 3.7 Sonnet (Screen)",
        "gpt-4o-mini": "GPT-4o Mini (Both)",
        "gpt-4o-mini-noaxtree": "GPT-4o Mini (Screen)",
        "gpt-4o-mini-noscreen": "GPT-4o Mini (Axtree)",
        "gpt-4o-mini-noscreen-noaxtree": "GPT-4o Mini (Neither)",
        "gpt-4o-noaxtree": "GPT-4o (Screen)",
        "gpt-4o-noscreen": "GPT-4o (Axtree)",
        "qwen-2.5-vl-noaxtree": "Qwen2.5-VL (Screen)",
        "qwen-2.5-vl-noscreen": "Qwen2.5-VL (Axtree)",
        "llama-3.3-70b-noscreen": "Llama 3.3",
    }

    renames_labels = {
        "trajectory_success": "Success",
        "trajectory_side_effect": "Side Effect",
        "trajectory_optimality": "Optimality",
        "trajectory_looping": "Repetition",
    }

    return {
        "agents": renames_agents,
        "benchmarks": renames_benchmarks,
        "judges": renames_judges,
        "labels": renames_labels,
    }

def get_judges():
    return [
        "functional",
        "aer",
        "aerv",
        "nnetnav",
        "claude-3.7-sonnet-noscreen",
        "claude-3.7-sonnet-noaxtree",
        "gpt-4o-mini",
        "gpt-4o-mini-noaxtree",
        "gpt-4o-mini-noscreen",
        "gpt-4o-mini-noscreen-noaxtree",
        "gpt-4o-noaxtree",
        "gpt-4o-noscreen",
        "qwen-2.5-vl-noaxtree",
        "qwen-2.5-vl-noscreen",
        "llama-3.3-70b-noscreen",
    ]


