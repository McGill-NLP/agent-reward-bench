"""
This script goes through each json in trajectories/processed, then look for key steps -> <list item> -> screenshot_path, copies the image 
to the corresponding folder in trajectories/screenshots/<benchmark>/<agent>/<judge>/<json file stem>/<image name>, then update the json file to point to the new location and save that json file to 
a trajectories/cleaned directory.

"""
import os
import orjson
import shutil
from tqdm import tqdm
from pathlib import Path


def copy_screenshots(json_path, screenshot_path):
    """
    Copy the screenshots to the corresponding folder in trajectories/screenshots/<benchmark>/<agent>/<json file stem>/<image name>
    """
    # Get the json file stem
    # json_file_stem = os.path.splitext(os.path.basename(json_path))[0]
    json_file_stem = str(Path(json_path).stem)
    
    # load the json file
    with open(json_path, 'rb') as f:
        data = orjson.loads(f.read())
    
    # Get the benchmark, agent, and judge from the json file
    benchmark = data['benchmark']
    agent = data['agent']
    # for each step, copy the screenshot to the corresponding folder
    for step in data['steps']:
        # Get the screenshot path
        screenshot_path = step['screenshot_path']
        screenshot_fname = os.path.basename(screenshot_path)
        # Get the new screenshot path
        new_screenshot_path = os.path.join('trajectories', 'screenshots', benchmark, agent, json_file_stem, screenshot_fname)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(new_screenshot_path), exist_ok=True)
        # Copy the screenshot to the new location, only if it doesn't exist
        if not os.path.exists(new_screenshot_path):    
            try:
                shutil.copy2(screenshot_path, new_screenshot_path)
            except:
                breakpoint()
        
        # Update the json file to point to the new location
        step['screenshot_path'] = new_screenshot_path
    # Save the json file to the cleaned directory
    # remove the following keys: trajectory_dir, logs

    keys_to_remove = ['trajectory_dir', 'logs']
    for key in keys_to_remove:
        if key in data:
            del data[key]
            
    cleaned_json_path = str(json_path).replace('processed', 'cleaned')
    os.makedirs(os.path.dirname(cleaned_json_path), exist_ok=True)
    with open(cleaned_json_path, 'wb') as f:
        f.write(orjson.dumps(data))

def main():
    # Get the list of json files in the trajectories/processed directory
    json_files = list(Path('trajectories', 'processed').glob('**/*.json'))
    
    # For each json file, copy the screenshots and update the json file
    for json_file in tqdm(json_files, desc='Processing json files'):
        # check if the cleaned json file already exists
        cleaned_json_path = os.path.join('trajectories', 'cleaned', os.path.basename(json_file))
        if os.path.exists(cleaned_json_path):
            print(f'Skipping {json_file} as it has already been cleaned')
            continue

        # Get the screenshot path
        screenshot_path = os.path.join('trajectories', 'screenshots', os.path.basename(json_file))
        copy_screenshots(json_file, screenshot_path)

if __name__ == '__main__':
    main()