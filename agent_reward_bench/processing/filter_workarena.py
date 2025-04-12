import json
import numpy as np
import pandas as pd

def get_sample_count_per_cat(cat_vc: pd.DataFrame, num_cats, num_to_keep=100, assert_sum=True):
    cat_vc = cat_vc.sort_values()
    num_to_keep = 100
    num_cats_remaining = num_cats
    sample_counts_per_cat = {}

    # for each category and count, we decrement the num_to_keep by the count and then
    # either take the count or num_to_keep/num_cats, whichever is smaller
    # we then add this to the total number of samples to keep
    for cat, count in cat_vc.items():
        if num_to_keep == 0:
            break
        if count < num_to_keep // num_cats_remaining:
            sample_counts_per_cat[cat] = count
            num_to_keep -= count
        else:
            sample_counts_per_cat[cat] = num_to_keep // num_cats_remaining
            num_to_keep -= num_to_keep // num_cats_remaining
        num_cats_remaining -= 1

    if assert_sum:
        assert sum(sample_counts_per_cat.values()) == 100

    return sample_counts_per_cat

def sample_from_df(df: pd.DataFrame, sample_counts_per_cat: dict, seed=42, sort_index=True):
    samples = []
    np.random.seed(seed)
    
    for cat, count in sample_counts_per_cat.items():
        samples.append(df[df['category'] == cat].sample(count))
    df_sampled = pd.concat(samples)

    # sort by index
    if sort_index:
        df_sampled.sort_index(inplace=True)

    return df_sampled

save_path = 'agent_reward_bench/data/workarena_l2.task_ids.json'
df = pd.read_csv('agent_reward_bench/data/workarena.csv')

df = df[df['level'].isin(['l2'])]
df = df[df['browsergym_split'] == 'test']

num_cats = len(df['category'].unique())
# get distribution of categories
cat_vc = df['category'].value_counts()

sample_counts_per_cat = get_sample_count_per_cat(cat_vc, num_cats)
df_sampled = sample_from_df(df, sample_counts_per_cat)

def beautify(sample_counts_per_cat):
    return ", ".join(f"{k.replace("_", " ").capitalize()} ({v})" for k, v in sample_counts_per_cat.items())

print(beautify(sample_counts_per_cat))

# save as json of task names to list of task ids
task_ids = df_sampled['task_name'].tolist()
with open(save_path, 'w') as f:
    json.dump(task_ids, f)