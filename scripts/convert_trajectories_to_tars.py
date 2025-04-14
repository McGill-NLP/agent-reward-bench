"""
This converts trajectories/cleaned/<benchmark>/<agent> to
trajectories_tars/cleaned/<benchmark>/<agent>.tar

e.g. trajectories/cleaned/webarena/GenericAgent-anthropic_claude-3.7-sonnet --> trajectories_tars/cleaned/webarena/GenericAgent-anthropic_claude-3.7-sonnet.tar
"""

import os
import tarfile
import argparse
import glob
import shutil
import tqdm
import tempfile
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def convert_trajectories_to_tars(benchmark: str, agent: str) -> None:
    """
    Convert trajectories/cleaned/<benchmark>/<agent> to trajectories_tars/cleaned/<benchmark>/<agent>.tar
    """
    src_dir = Path(f"trajectories/cleaned/{benchmark}/{agent}")
    dst_dir = Path(f"trajectories_tars/cleaned/{benchmark}/{agent}.tar")
    if dst_dir.exists():
        print(f"Skipping {benchmark}/{agent} as tar already exists.")
        return
    
    os.makedirs(dst_dir.parent, exist_ok=True)
    with tarfile.open(dst_dir, "w") as tar:
        # add the subdirectory to the tar file
        # tar.add(src_dir, arcname=os.path.basename(src_dir))
        tar.add(src_dir)
    print(f"Created tar for {benchmark}/{agent} at {dst_dir}")

def main() -> None:
    # iterate over all directories in trajectories/cleaned
    # and convert them to tars
    cleaned_dir = Path("trajectories/cleaned")

    for benchmark_dir in cleaned_dir.iterdir():
        # skip hidden files
        if benchmark_dir.name.startswith("."):
            continue

        if not benchmark_dir.is_dir():
            continue
        
        for agent_dir in benchmark_dir.iterdir():
            # only focus on folders tarting with GenericAgent
            if not agent_dir.is_dir():
                continue
            if not agent_dir.name.startswith("GenericAgent"):
                continue

            convert_trajectories_to_tars(benchmark_dir.name, agent_dir.name)
    
    print("All trajectories converted to tars.")

if __name__ == "__main__":
    main()
