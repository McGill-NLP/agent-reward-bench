import base64
from collections import defaultdict
from copy import deepcopy
from functools import cache, cached_property
from io import BytesIO
import logging
from pathlib import Path
import json
import pickle
import gzip

from tqdm.auto import tqdm

def prune_axtree_dict(axtree_obj):
    # for each node in the axtree_obj, remove a child if ignored=True, and recurse
    if 'nodes' not in axtree_obj:
        return axtree_obj

    pruned_nodes = []
    pruned_ids = set()

    for node in axtree_obj['nodes']:
        if node.get('ignored', False) == True:
            pruned_ids.add(node['nodeId'])
            continue

        new_node = deepcopy(node)
        new_node['childIds'] = []

        for child_id in node['childIds']:
            if child_id in pruned_ids:
                continue
            new_node['childIds'].append(child_id)
        
        pruned_nodes.append(new_node)
    
    return {
        'nodes': pruned_nodes
    }
    

def remove_image_urls(chat_messages):
    for msg in chat_messages:
        if isinstance(msg['content'], str):
            continue
        
        for part in msg['content']:
            if part['type'] == 'image_url':
                part['image_url']['url'] = 'REMOVED'
    return chat_messages

def list_experiments(base_dir):
    base_dir = Path(base_dir)
    lst =  [d for d in Path(base_dir).iterdir() if d.is_dir()]

    # check if there's a summary_info.json as a child or grandchild
    lst = [d for d in lst if any(d.rglob("**/summary_info.json"))]

    return lst

def get_trajectories(experiment_dir, filter_invalid=True, max_num=None):
    # get all the subdir of experiment_dir and return Trajectory objects
    dirs = [d for d in Path(experiment_dir).iterdir() if d.is_dir()]
    if max_num is not None:
        dirs = dirs[:max_num]
    
    trajs = [Trajectory(d) for d in dirs]

    if filter_invalid:
        trajs = [t for t in trajs if t.is_valid()]

    return trajs

def get_number_from_path(path: Path):
    stem = path.stem
    # remove by the first period
    stem = stem.split(".")[0]
    # rpartition by _ and take the second part
    num = stem.rpartition("_")[2]
    return int(num)

def base64_to_bytes(b64_data):
    return BytesIO(base64.b64decode(b64_data))

def base64_to_pillow(b64_data):
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required to load images into PIL.Image format. Please install Pillow using `pip install Pillow`")

    return Image.open(base64_to_bytes(b64_data))

def set_all_bounding_boxes(bounding_boxes: dict, clickable: bool, set_of_marks: bool) -> dict:
    bounding_boxes = deepcopy(bounding_boxes)

    for k in bounding_boxes:
        bounding_boxes[k]['clickable'] = clickable
        bounding_boxes[k]['set_of_marks'] = set_of_marks

    return bounding_boxes

class Trajectory:
    def __init__(self, trajectory_dir: Path):
        self.trajectory_dir = Path(trajectory_dir)
        self.exp_dir = self.trajectory_dir.parent

    def __repr__(self):
        return f"Trajectory({self.trajectory_dir})"

    def __str__(self):
        return f"Trajectory({self.trajectory_dir})"

    def __eq__(self, other):
        return self.trajectory_dir == other.trajectory_dir

    def __len__(self):
        return self.count_step_pickles()

    def __getitem__(self, idx):
        step_pickle_paths = self.list_step_pickles()
        # only chosoe png, otherwise jpg
        if self.has_png_images():
            image_paths = self.list_png_images()
        elif self.has_jpg_images():
            image_paths = self.list_jpg_images()
        else:
            raise FileNotFoundError("No images found")
        
        if int(idx) >= len(step_pickle_paths):
            raise IndexError("Index out of range")
        
        step_pickle_path = step_pickle_paths[idx]
        image_path = image_paths[idx]

        return {
            "step_pickle": step_pickle_path,
            "image": image_path,
        }

    def has_summary_info(self):
        return self.trajectory_dir.joinpath("summary_info.json").exists()

    def has_exp_args(self):
        return self.trajectory_dir.joinpath("exp_args.pkl").exists()

    def has_goal_object(self):
        return self.trajectory_dir.joinpath("goal_object.pkl.gz").exists()

    def has_experiment_logs(self):
        return self.trajectory_dir.joinpath("experiment.log").exists()

    def has_package_versions(self):
        return self.trajectory_dir.joinpath("package_versions.txt").exists()

    def has_step_pickles(self):
        return any(self.trajectory_dir.glob("step*.pkl.gz"))

    def has_jpg_images(self):
        return any(self.trajectory_dir.glob("*.jpg"))

    def has_png_images(self):
        return any(self.trajectory_dir.glob("*.png"))

    def has_images(self):
        return self.has_jpg_images() or self.has_png_images()

    def has_leading_underscore(self):
        return self.trajectory_dir.name.startswith("_")
    
    def is_valid(self):
        return (
            self.has_summary_info()
            and self.has_exp_args()
            and self.has_goal_object()
            and self.has_images()
            and not self.has_leading_underscore()
        )

    def list_files(self, pattern, as_str=False):
        files = list(self.trajectory_dir.glob(pattern))

        if as_str:
            return [str(p) for p in files]
        else:
            return files

    def list_file_names(self):
        file_names = [f.name for f in self.trajectory_dir.iterdir() if f.is_file()]
        return file_names
    
    def list_jpg_images(self, as_str=False):
        return self.list_files("*.jpg", as_str=as_str)

    def list_png_images(self, as_str=False):
        return self.list_files("*.png", as_str=as_str)

    def list_images(self, as_str=False):
        return self.list_jpg_images(as_str=as_str) + self.list_png_images(as_str=as_str)

    def list_step_pickles(self, as_str=False):
        lst = self.list_files("step*.pkl.gz", as_str=as_str)
        lst = sorted(lst, key=lambda x: get_number_from_path(x))
        
        return lst

    def count_jpg_images(self):
        return len(self.list_jpg_images())

    def count_png_images(self):
        return len(self.list_png_images())

    def count_images(self):
        return len(self.list_images())

    def count_step_pickles(self):
        return len(self.list_step_pickles())

    def get_images(self, with_pillow=False):
        image_paths = []
        if self.has_png_images():
            image_paths.extend(self.list_png_images())
        if self.has_jpg_images():
            image_paths.extend(self.list_jpg_images())

        if not with_pillow:
            return [str(path) for path in image_paths]
        else:
            # load with pillow if with_pillow is True
            try:
                from PIL import Image
            except ImportError:
                raise ImportError(
                    "Pillow is required to load images with with_pillow=True"
                )

            return [Image.open(i) for i in image_paths]
    
    def get_step_pickles(self):
        pickles = []

        for p in self.list_step_pickles():
            with gzip.open(p, "rb") as f:
                pickles.append(pickle.load(f))

        return pickles

    def to_dict(self):
        d =  {
            'benchmark': self.benchmark,
            'agent': self.agent_name,
            'model': self.model_name,
            'valid': self.is_valid(),
            'experiment': self.exp_name,
            'trajectory_dir': str(self.trajectory_dir),
            'goal': self.goal_text,
            'seed': self.task_seed,
            'model_args': self.model_args.__dict__,
            'flags': self.flags,
            'summary_info': self.summary_info,
            'package_version': self.package_versions,
        }

        if self.has_experiment_logs():
            d['logs'] = self.experiment_log

        return d

    @cached_property
    def exp_args(self):
        with open(Path(self.trajectory_dir, "exp_args.pkl"), "rb") as f:
            return pickle.load(f)
        # example: ExpArgs(agent_args=GenericAgentArgs(agent_name='GenericAgent-gpt-4o-mini-2024-07-18', chat_model_args=OpenAIModelArgs(model_name='gpt-4o-mini-2024-07-18', max_total_tokens=128000, max_input_tokens=128000, max_new_tokens=16384, temperature=0.1, vision_support=True), flags=GenericPromptFlags(obs=ObsFlags(use_html=False, use_ax_tree=True, use_tabs=True, use_focused_element=True, use_error_logs=True, use_history=True, use_past_error_logs=False, use_action_history=True, use_think_history=False, use_diff=False, html_type='pruned_html', use_screenshot=True, use_som=False, extract_visible_tag=True, extract_clickable_tag=True, extract_coords='False', filter_visible_elements_only=False, openai_vision_detail='auto', filter_with_bid_only=False, filter_som_only=False), action=ActionFlags(action_set=HighLevelActionSetArgs(subsets=('visualwebarena',), multiaction=False, strict=False, retry_with_force=True, demo_mode='off'), long_description=False, individual_examples=False, multi_actions=None, is_strict=None), use_plan=False, use_criticise=False, use_thinking=True, use_memory=False, use_concrete_example=True, use_abstract_example=True, use_hints=True, enable_chat=False, max_prompt_tokens=40000, be_cautious=True, extra_instructions=None, add_missparsed_messages=True, max_trunc_itr=20, flag_group=None), max_retry=4), env_args=EnvArgs(task_name='visualwebarena.8', task_seed=23, max_steps=30, headless=True, record_video=False, wait_for_user_message=False, viewport=None, slow_mo=None, storage_state=None, task_kwargs=None), exp_dir=PosixPath('/home/nlp/users/xlu41/agentlab_results/2024-11-27_00-57-06_genericagent-gpt-4o-mini-2024-07-18-on-visualwebarena-test/2024-11-27_01-23-11_GenericAgent-gpt-4o-mini-2024-07-18_on_visualwebarena.8_23'), exp_name='GenericAgent-gpt-4o-mini-2024-07-18_on_visualwebarena.8_23', enable_debug=True, err_msg=None, stack_trace=None, order=4, logging_level=10, logging_level_stdout=30, exp_id='780f342b-167b-4d17-885d-9a27400c4990', depends_on=('b126f35a-4794-437e-b7cd-0ac6d9d9a4db',), save_screenshot=True, save_som=False)

    @cached_property
    def goal_object(self):
        path = Path(self.trajectory_dir, "goal_object.pkl.gz")

        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    @cached_property
    def summary_info(self):
        with open(Path(self.trajectory_dir, "summary_info.json")) as f:
            return json.load(f)

    @cached_property
    def experiment_log(self):
        with open(Path(self.trajectory_dir, "experiment.log")) as f:
            return f.read()

    @cached_property
    def package_versions(self):
        with open(Path(self.trajectory_dir, "package_versions.txt")) as f:
            return f.read()
    
    @property
    def agent_name(self):
        return self.exp_args.agent_args.agent_name

    @property
    def model_name(self):
        return self.exp_args.agent_args.chat_model_args.model_name
    
    @property
    def model_args(self):
        return self.exp_args.agent_args.chat_model_args


    @property
    def task_id(self):
        return self.exp_args.env_args.task_name
    
    @property
    def benchmark(self):
        return self.task_id.split('.')[0]

    @property
    def task_seed(self):
        return self.exp_args.env_args.task_seed

    @property
    def flags(self):
        return self.exp_args.agent_args.flags.asdict()

    @property
    def exp_name(self):
        exp_name = self.exp_args.exp_name
        exp_name, sep, task_id_and_seed = exp_name.rpartition(".")
        return exp_name

    @property
    def goal_text(self):
        text = ""
        for d in self.goal_object:
            if d["type"] == "text":
                text += d["text"] + "\n"

        return text

    def get_goal_images(self, open_as="b64"):
        if open_as.lower() not in ["b64", "bytes", "pillow", "pil"]:
            raise ValueError('open_as must be one of "b64", "bytes", "pillow"')

        image_b64_data = []
        for d in self.goal_object:
            if d["type"] == "image_url":
                url = d["image_url"]["url"]
                if url.startswith("data:image"):
                    # remove that part and decode the base64
                    # split with first ,
                    header, data = url.split(",", 1)
                    image_b64_data.append(data)

        if open_as == "b64":
            return image_b64_data
        elif open_as == "bytes":
            return [base64_to_bytes(data) for data in image_b64_data]
        elif open_as.lower() in ["pillow", "pil"]:
            return [base64_to_pillow(data) for data in image_b64_data]
        else:
            raise ValueError("Invalid open_as")


class Step:
    def __init__(self, step_pickle_path: Path):
        self.step_pickle_path = step_pickle_path

        if not step_pickle_path.exists():
            raise FileNotFoundError(f"{step_pickle_path} not found")
        
        if str(step_pickle_path).endswith(".gz"):
            with gzip.open(step_pickle_path, "rb") as f:
                self.info = pickle.load(f)
        elif str(step_pickle_path).endswith(".pkl") or str(step_pickle_path).endswith(".pickle"):
            with open(step_pickle_path, "rb") as f:
                self.info = pickle.load(f)
        else:
            raise ValueError(f"Invalid file type: {step_pickle_path}")
    
    @classmethod
    def from_mini_dict(cls, d):
        return cls(Path(d["step_pickle"]))
    
    def __str__(self):
        return f"Step({self.step_pickle_path})"

    def __repr__(self):
        return f"Step({self.step_pickle_path})"

    @property
    def action(self):
        return self.info.action
    
    @property
    def num(self):
        return self.info.step
    
    @property
    def url(self):
        return self.info.obs["url"]

    @property
    def goal(self):
        return self.info.obs['goal']
    
    @property
    def chat_messages(self):
        if not self.has_agent_info():
            return []
        
        return self.info.agent_info.chat_messages.to_openai()

    @property
    def cost(self):
        if self.has_agent_info():
            return self.info.agent_info.extra_info['cost']
    
    @property
    def reasoning(self):
        if self.has_agent_info():
            return self.info.agent_info.think
    
    @property
    def model_name(self):
        if self.has_agent_info():
            return self.info.agent_info.extra_info['chat_model_args']['model_name']
    
    @property
    def stats(self):
        if self.has_agent_info():
            return self.info.agent_info.stats

    @property
    def temperature(self):
        if self.has_agent_info():
            return self.info.agent_info.extra_info['chat_model_args']['temperature']
    
    @property
    def vision_support(self):
        if self.has_agent_info():
            return self.info.agent_info.extra_info['chat_model_args']['vision_support']
    
    @property
    def max_total_tokens(self):
        if self.has_agent_info():
            return self.info.agent_info.extra_info['chat_model_args']['max_total_tokens']
    
    @property
    def max_input_tokens(self):
        if self.has_agent_info():
            return self.info.agent_info.extra_info['chat_model_args']['max_input_tokens']
    
    @property
    def max_new_tokens(self):
        if self.has_agent_info():
            return self.info.agent_info.extra_info['chat_model_args']['max_new_tokens']
    
    @property
    def num_input_tokens(self):
        if self.has_agent_info():
            return self.info.agent_info.extra_info['input_tokens']
    
    @property
    def num_output_tokens(self):
        if self.has_agent_info():
            return self.info.agent_info.extra_info['output_tokens']

    @property
    def focused_element(self):
        return self.info.obs['focused_element_bid']

    @property
    def axtree_text(self):
        return self.info.obs['axtree_txt']
    
    @property
    def axtree_obj(self):
        return self.info.obs['axtree_object']

    @property
    def dom_text(self):
        return self.info.obs['dom_txt']
    
    @property
    def html_pruned(self):
        return self.info.obs['pruned_html']
    
    @property
    def urls(self):
        return self.info.obs['open_pages_urls']
    
    @property
    def bounding_boxes(self):
        return self.info.obs['extra_element_properties']
    
    @property
    def last_action_error(self):
        return self.info.obs.get('last_action_error')
    
    @property
    def extra_element_properties(self):
        return self.info.obs['extra_element_properties']

    def has_agent_info(self):
        return hasattr(self.info, "agent_info") and self.info.agent_info is not None and self.info.agent_info != {}
    
    def filter_bounding_boxes(self, element_id) -> dict:
        if element_id is None or not isinstance(element_id, (str, int)):
            return {}
        
        filtered_boxes = {
            k: v for k, v in self.bounding_boxes.items() if k == element_id
        }

        return filtered_boxes

    def get_image_path(self, prefix='screenshot_step_', extension='png'):
        if not extension.startswith("."):
            extension = "." + extension
        
        # use step num to get the image path
        step_num = self.num

        # get the trajectory dir
        trajectory_dir = self.step_pickle_path.parent

        # get the image path
        image_path = trajectory_dir.joinpath(f"{prefix}{step_num}{extension}")

        if not image_path.exists():
            # warn that the image does not exist
            logging.warning(f"Image {image_path} does not exist. Returning none")

            return None

        return str(image_path)
    

    def to_dict(self, image_urls=False, prune_axtree=False):
        cm = self.chat_messages
        if not image_urls:
            cm = remove_image_urls(cm)
        
        d = {
            'num': self.num,
            'reasoning': self.reasoning,
            'action': self.action,
            'screenshot_path': str(self.get_image_path()),
            'url': self.url,
            'open_pages_urls': self.urls,
            'focused_element': self.focused_element,
            'last_action_error': self.last_action_error,
            'stats': self.stats,
            'axtree': self.axtree_text,
            'axtree_obj': self.axtree_obj,
            'chat_messages': cm,
            'bounding_boxes': self.bounding_boxes,
            'extra_element_properties': self.extra_element_properties,
        }

        if prune_axtree:
            try:
                from browsergym.utils.obs import flatten_axtree_to_str
            except:
                raise ImportError("browsergym is required to use this method. Please install browsergym using `pip install browsergym`")
            
            d = self.to_dict(image_urls=image_urls)
            d['axtree_pruned'] = flatten_axtree_to_str(
                deepcopy(self.axtree_obj), filter_visible_only=True, filter_with_bid_only=True, extra_properties=self.extra_element_properties
            )
        
        return d

class TrajectoriesManager:
    def __init__(self):
        self.trajectories: 'list[Trajectory]' = []
    
    def add_trajectories_from_dir(self, experiment_dir, filter_invalid=True, max_num=None):
        self.trajectories.extend(get_trajectories(experiment_dir, filter_invalid=filter_invalid, max_num=max_num))

    def add_trajectories_from_dirs(self, experiment_dirs, filter_invalid=True, max_num=None):
        for d in experiment_dirs:
            self.add_trajectories_from_dir(d, filter_invalid=filter_invalid, max_num=max_num)
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx]

    def __iter__(self):
        return iter(self.trajectories)
    
    def build_index(self, exp_dir_suffix_parser=None):
        # index consists of dictionary of benchmark -> model_name -> exp_name -> list of indices of all experiments
        index = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for i, t in enumerate(self.trajectories):
            t: Trajectory
            benchmark = t.benchmark
            model_name = t.agent_name
            exp_name = t.exp_name
            task_id = t.task_id
            if exp_dir_suffix_parser is not None:
                if not callable(exp_dir_suffix_parser):
                    raise ValueError("exp_suffix_parser must be callable")
                exp_dir_suffix = exp_dir_suffix_parser(t.trajectory_dir.parent)
                benchmark = f"{benchmark}_{exp_dir_suffix}"

            index[benchmark][model_name][exp_name][task_id] = i

        # convert to normal dict recursively
        index = json.loads(json.dumps(index))
        self.index = index
    
    def get_benchmarks(self):
        """
        Get all benchmarks
        """
        return list(self.index.keys())
    
    def get_model_names(self, benchmark=None):
        """
        Get all model_names for a given benchmark
        """
        if benchmark is None:
            # return all the model names by merging the values of all the benchmarks
            lst = set()
            for b in self.index:
                lst.update(self.index[b].keys())
            return list(lst)
        
        if benchmark not in self.index:
            raise ValueError(f"Benchmark {benchmark} not found. The following benchmarks are available: {self.get_benchmarks()}")
        
        return list(self.index[benchmark].keys())
        
    def get_exp_names(self, benchmark, model_name):
        """
        Get all exp_names for a given benchmark and model_name
        """
        return list(self.index[benchmark][model_name].keys())

    def assert_exp_limit(self, benchmark, model_name, min_exp=1, max_exp=1):
        """
        Enforce an experiment limit for a given benchmark and model_name
        and raise an error if the limit is exceeded or not met
        """
        exp_names = self.get_exp_names(benchmark, model_name)

        if len(exp_names) < min_exp or len(exp_names) > max_exp:
            raise ValueError(f"Experiment limit exceeded for benchmark={benchmark}, model_name={model_name}, limits={min_exp}-{max_exp}, Found={len(exp_names)}")

    def get_trajectories(self, benchmark, model_name, exp_name) -> 'list[Trajectory]':
        """
        Get a trajectory by benchmark, model_name, and exp_name
        """
        traj_indices_filtered = self.index[benchmark][model_name][exp_name].values()
        return [self.trajectories[i] for i in traj_indices_filtered]

    def get_trajectory(self, benchmark, model_name, exp_name, task_id) -> Trajectory:
        """
        Get a trajectory by benchmark, model_name, exp_name, and task_id
        """
        traj_idx = self.index[benchmark][model_name][exp_name][task_id]
        return self.trajectories[traj_idx]

    def has_trajectory(self, benchmark, model_name, exp_name, task_id):
        """
        Check if a trajectory exists by benchmark, model_name, exp_name, and task_id
        """
        return task_id in self.index[benchmark][model_name][exp_name]
    
    def model_is_benchmarked(self, benchmark, model_name):
        """
        Check if a model is benchmarked on a given benchmark
        """
        return model_name in self.index[benchmark]
    
    def infer_exp_name(self, benchmark, model_name):
        """
        Infer the exp_name from the benchmark and model_name
        """
        exp_names = self.get_exp_names(benchmark, model_name)

        if len(exp_names) == 0:
            raise ValueError(f"No experiments found for benchmark={benchmark}, model_name={model_name}")

        if len(exp_names) > 1:
            raise ValueError(f"Multiple experiments found for benchmark={benchmark}, model_name={model_name}")

        return exp_names[0]
    
    def get_task_ids(self, benchmark, model_name, exp_name, sort=True):
        """
        Get the task names of the trajectory
        """
        lst = [t.task_id for t in self.get_trajectories(benchmark, model_name, exp_name)]

        # sort if needed
        if sort:
            # sort by task_id
            def sort_func(x):
                r = x.split('.')[-1]
                if r.isdigit():
                    return int(r)
                else:
                    return r
                
            lst = sorted(lst, key=sort_func)
        
        return lst
