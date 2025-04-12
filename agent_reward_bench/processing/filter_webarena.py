"""
This filters webarena's test.raw.json to only get 100 representative tasks
"""

from collections import defaultdict
import json
import random
from pathlib import Path
import re
import pkg_resources
import csv

def beautify(d: dict) -> str:
    # show as Key (value) pairs
    return ", ".join([f"{k.capitalize()} ({v})" for k, v in d.items()])

def get_site_freq(site_eval_freq) -> dict:
    site_freq = defaultdict(int)

    for (site, _), freq in site_eval_freq.items():
        site_freq[site] += freq
    
    return site_freq

def get_unique_sites(records, browsergym_split='test'):
    unique_sites = set()

    for instance in records:
        if instance['browsergym_split'] == browsergym_split:
            unique_sites.update(instance['sites'].split())
    
    return unique_sites

def get_unique_eval_types(test_raw, browsergym_split='test'):
    unique_eval_types = set()

    for instance in test_raw:
        if instance['browsergym_split'] == browsergym_split:
            unique_eval_types.update(instance['eval_types'].split())
    
    return unique_eval_types


def get_templates(test_raw):
    templates = {}

    for instance in test_raw:
        template = instance['intent_template']
        site = instance['sites'][0]
        eval_types: list = instance['eval']['eval_types']
        
        templates[instance['intent_template_id']] = {
            'template': template,
            'site': site,
            'intent_template_id': instance['intent_template_id'],
            'eval_types': eval_types
        }
    
    return templates

def get_site_eval_map(records, browsergym_split='test'):
    site_eval_map = defaultdict(set)

    for instance in records:
        if instance['browsergym_split'] != browsergym_split:
            continue

        for eval_type in instance['eval_types'].split():
            site = instance['sites'].split()[0]
            site_eval_map[(site, eval_type)].add(instance['task_id'])

    return site_eval_map

def get_site_eval_frequency(site_eval_map) -> dict:
    site_eval_freq = {
        (site, eval_type): len(value) for (site, eval_type), value in site_eval_map.items()
    }
    site_eval_freq = dict(sorted(site_eval_freq.items(), key=lambda x: x[0]))

    return site_eval_freq

def get_site_subsample_size(site_freq: dict, num_templates=100) -> dict:
    # calculate how many to attribute to each site first
    remaining_templates = num_templates
    remaining_sites = len(site_freq)
    site_subsample_size = {}

    for site in sorted(site_freq, key=lambda x: site_freq[x]):
        max_template = remaining_templates // remaining_sites
        site_subsample_size[site] = min(max_template, site_freq[site])
        remaining_templates -= site_subsample_size[site]
        remaining_sites -= 1

    # add the remaining templates to the site with the highest frequency
    max_site = max(site_freq, key=lambda x: site_freq[x])
    site_subsample_size[max_site] += remaining_templates

    return site_subsample_size


def get_site_eval_subsample_size(site_subsample_size, site_eval_freq, unique_eval_types):
    # Now, for each site, given the subsample size, we select the subsample size per eval type
    site_eval_subsample_size = defaultdict(dict)

    for target_site, subsample_size in site_subsample_size.items():
        eval_type_freq_for_site = {}
        for eval_type in unique_eval_types:
            if (target_site, eval_type) in site_eval_freq:
                eval_type_freq_for_site[eval_type] = site_eval_freq[(target_site, eval_type)]
        
        remaining_subsample_size = subsample_size
        remaining_eval_types = len(eval_type_freq_for_site)

        for eval_type in sorted(eval_type_freq_for_site, key=lambda x: eval_type_freq_for_site[x]):
            max_template = remaining_subsample_size // remaining_eval_types
            site_eval_subsample_size[target_site][eval_type] = min(max_template, eval_type_freq_for_site[eval_type])
            remaining_subsample_size -= site_eval_subsample_size[target_site][eval_type]
            remaining_eval_types -= 1
        
        max_eval_type = max(eval_type_freq_for_site, key=lambda x: eval_type_freq_for_site[x])
        site_eval_subsample_size[target_site][max_eval_type] += remaining_subsample_size
        
    site_eval_subsample_size: dict = dict(site_eval_subsample_size)

    return site_eval_subsample_size

def show_site_eval_subsample_size(site_eval_subsample_size):
    for site, eval_types in site_eval_subsample_size.items():
        eval_types: dict

        print(site)
        for eval_type, freq in eval_types.items():
            print(f"\t{eval_type}: {freq}")
        print("\t(Total:", sum(eval_types.values()), ")")
    print(f"Total: {sum(sum(eval_types.values()) for eval_types in site_eval_subsample_size.values())}")

def get_sample_tasks(site_eval_subsample_size, site_eval_map, seed=0):
    seed = 0

    random.seed(seed)
    sampled_templates_subset = set()

    for site, eval_types in site_eval_subsample_size.items():
        eval_types: dict
        for eval_type, freq in eval_types.items():
            dedup_ids_list = [
                x for x in site_eval_map[(site, eval_type)]
                if x not in sampled_templates_subset
            ]

            random_ids = random.sample(dedup_ids_list, freq)
            sampled_templates_subset.update(random_ids)

    sampled_templates_subset = list(sampled_templates_subset)

    return sampled_templates_subset


if __name__ == "__main__":
    package = "agent_reward_bench"
    wa_csv = pkg_resources.resource_filename(package, 'data/webarena.csv')
    vwa_csv = pkg_resources.resource_filename(package, 'data/visualwebarena.csv')

    with open(wa_csv) as f:
        reader = csv.DictReader(f)
        wa_records = list(reader)
    
    with open(vwa_csv) as f:
        reader = csv.DictReader(f)
        vwa_records = list(reader)
    
    benchmarks = {
        'webarena': {
            'records': wa_records,
            'task_ids_package_path': "agent_reward_bench/data/webarena.task_ids.json"
        },
        'visualwebarena': {
            'records': vwa_records,
            'task_ids_package_path': "agent_reward_bench/data/visualwebarena.task_ids.json"
        }
    }
    for benchmark_name, benchmark in benchmarks.items():
        records = benchmark['records']
        task_ids_path = benchmark['task_ids_package_path']

        print(f"Processing {benchmark_name}")
        print("records:", len(records))
        print("task_ids_path:", task_ids_path)

        unique_sites = get_unique_sites(records)
        print(f"Number of unique sites: {len(unique_sites)}")

        unique_eval_types = get_unique_eval_types(records)
        print(f"Number of unique eval types: {len(unique_eval_types)}")

        site_eval_map: dict = get_site_eval_map(records)
        site_eval_freq = get_site_eval_frequency(site_eval_map)
        site_freq = get_site_freq(site_eval_freq)

        # we want to get 100 representative tasks, one from each templates, equally distributed across sites
        site_subsample_size = get_site_subsample_size(site_freq)

        print("site_subsample_size:", beautify(site_subsample_size))
        site_eval_subsample_size = get_site_eval_subsample_size(site_subsample_size, site_eval_freq, unique_eval_types)
        show_site_eval_subsample_size(site_eval_subsample_size)

        # now, randomly select the tasks based on the frequency we calculated in site_eval_subsample_size
        sampled_task_ids = get_sample_tasks(site_eval_subsample_size, site_eval_map, seed=0)
        print(f"Sampled {len(sampled_task_ids)} tasks")

        # create parent of task_ids_package_path
        Path(task_ids_path).parent.mkdir(parents=True, exist_ok=True)
        with open(task_ids_path, "w") as f:
            json.dump(sampled_task_ids, f)
