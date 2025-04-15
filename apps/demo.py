import ast
import csv
from textwrap import dedent
import pyparsing as pp
from dataclasses import dataclass
from typing import Any
import json
from pathlib import Path
import logging

import orjson
from PIL import Image
import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)

benchmarks_dict = {
    "assistantbench": "AssistantBench",
    "visualwebarena": "VisualWebArena",
    "webarena": "WebArena",
    "workarena": "WorkArena",
}

tasks_dict = {
    "assistantbench": "assistantbench.improved.validation",
    "visualwebarena": "visualwebarena.resized",
    "webarena": "webarena",
    "workarena": "workarena.servicenow",
}

agents_dict = {
    "GenericAgent-anthropic_claude-3.7-sonnet": "Claude 3.7 Sonnet",
    "GenericAgent-gpt-4o-2024-11-20": "GPT-4o",
    "GenericAgent-meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3 70B",
    "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL 72B",
}

judges_dict = {
    "aer": "AER-C",
    "nnetnav": "NNetNav",
    "claude-3.7-sonnet-noaxtree": "Claude 3.7 Sonnet (Screen)",
    "claude-3.7-sonnet-noscreen": "Claude 3.7 Sonnet (Axtree)",
    "gpt-4o-noaxtree": "GPT-4o (Screen)",
    "gpt-4o-noscreen": "GPT-4o (Axtree)",
    "qwen-2.5-vl-noaxtree": "Qwen 2.5 VL (Screen)",
    "qwen-2.5-vl-noscreen": "Qwen 2.5 VL (Axtree)",
    "llama-3.3-70b-noscreen": "Llama 3.3 70B",
    "functional": "Rule-based",
}

default_judges = [
    "AER-C",
    "NNetNav",
    "Claude 3.7 Sonnet (Screen)",
    "GPT-4o (Screen)",
    "Qwen 2.5 VL (Screen)",
    "Llama 3.3 70B",
]

benchmarks_inverse = {v: k for k, v in benchmarks_dict.items()}
agents_inverse = {v: k for k, v in agents_dict.items()}
tasks_inverse = {v: k for k, v in tasks_dict.items()}
judges_inverse = {v: k for k, v in judges_dict.items()}


@dataclass
class NamedArgument:
    """
    Source: https://github.com/ServiceNow/BrowserGym/blob/c3336ef61781ce39166ee6a9551dbfc8fac32ddc/browsergym/core/src/browsergym/core/action/parsers.py#L9
    """

    name: str
    value: Any

    def __repr__(self):
        return f"{self.name}={repr(self.value)}"


def overlay_som(
    screenshot: np.typing.ArrayLike,
    extra_properties: dict,
    fontsize: int = 12,
    linewidth: int = 2,
    tag_margin: int = 2,
):
    """
    Source: https://github.com/ServiceNow/BrowserGym/blob/c3336ef61781ce39166ee6a9551dbfc8fac32ddc/browsergym/core/src/browsergym/utils/obs.py#L429
    """
    from PIL import Image, ImageDraw, ImageFont
    import math

    img = Image.fromarray(screenshot).copy()  # make a copy
    img = img.convert(mode="RGBA")
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default(size=fontsize)

    # Adapted from https://stackoverflow.com/questions/51908563/dotted-or-dashed-line-with-python-pillow/58885306#58885306
    def linedashed(
        draw: ImageDraw.Draw,
        x0,
        y0,
        x1,
        y1,
        fill,
        width,
        dash_length=4,
        nodash_length=8,
    ):
        line_dx = x1 - x0  # delta x (can be negative)
        line_dy = y1 - y0  # delta y (can be negative)
        line_length = math.hypot(line_dx, line_dy)  # line length (positive)
        if line_length == 0:
            return  # Avoid division by zero in case the line length is 0
        pixel_dx = line_dx / line_length  # x add for 1px line length
        pixel_dy = line_dy / line_length  # y add for 1px line length
        dash_start = 0
        while dash_start < line_length:
            dash_end = dash_start + dash_length
            if dash_end > line_length:
                dash_end = line_length
            draw.line(
                (
                    round(x0 + pixel_dx * dash_start),
                    round(y0 + pixel_dy * dash_start),
                    round(x0 + pixel_dx * dash_end),
                    round(y0 + pixel_dy * dash_end),
                ),
                fill=fill,
                width=width,
            )
            dash_start += dash_length + nodash_length

    for bid, properties in extra_properties.items():
        if properties["set_of_marks"] and properties["bbox"]:
            x, y, width, height = properties["bbox"]
            x0, y0 = x, y
            x1, y1 = x + width, y + height

            # skip small boxes
            area = (x1 - x0) * (y1 - y0)
            if area < 20:
                logger.warning(
                    f'som overlay: skipping bid "{bid}" due to bbox too small (area={area})'
                )
                continue

            # draw bounding box with dashed lines
            linedashed(draw, x0, y0, x1, y0, fill=(0, 0, 0, 255), width=linewidth)
            linedashed(draw, x1, y0, x1, y1, fill=(0, 0, 0, 255), width=linewidth)
            linedashed(draw, x1, y1, x0, y1, fill=(0, 0, 0, 255), width=linewidth)
            linedashed(draw, x0, y1, x0, y0, fill=(0, 0, 0, 255), width=linewidth)

            # get text box size (left, top, right, bottom)
            tag_box = font.getbbox(
                bid,
            )

            # set tag size, including margins
            tag_size = (
                (tag_box[2] - tag_box[0] + 2 * (tag_margin + 1)),
                (tag_box[3] - tag_box[1] + 2 * (tag_margin + 1)),
            )

            # create tag image with correct size and black background
            tag_img = Image.new("RGBA", tag_size, "black")
            tag_draw = ImageDraw.Draw(tag_img)
            # write text with 1px horizontal margin
            tag_draw.text(
                (-tag_box[0] + tag_margin + 1, -tag_box[1] + tag_margin + 1),
                bid,
                font=font,
                fill=(255, 255, 255, 255),
                spacing=0,
            )
            tag_draw.rectangle(
                (0, 0, tag_size[0] - 1, tag_size[1] - 1),
                fill=None,
                outline=(255, 255, 255, 255),
                width=1,
            )

            # draw tag in the source image, upper left of the bounding box
            tag_pos = (x + 0, y - tag_size[1] / 2 + 4)
            tag_pos = list(map(round, tag_pos))
            img.paste(tag_img, tag_pos)

    # convert to RGB (3 channels)
    img = img.convert(mode="RGB")
    # convert to a numpy array
    img = np.array(img)

    return img


def apply_overlay_to_image(im, step, highlevel_action_parser=None):
    action = step.get("action", None)
    if action is None:
        return im

    # get the element from the action string
    element = get_element_from_action_str(
        action, highlevel_action_parser=highlevel_action_parser
    )
    if element is None:
        return im

    # overlay the extra properties on the image
    extra_properties = step.get("extra_element_properties", {})
    if element not in extra_properties:
        return im

    # get the extra properties for the element
    extra_properties = {element: extra_properties[element]}

    im_arr = np.array(im)
    im_overlayed = overlay_som(im_arr, extra_properties=extra_properties)
    im = Image.fromarray(im_overlayed)

    return im


def _build_highlevel_action_parser() -> pp.ParserElement:
    """
    SOURCE: https://github.com/ServiceNow/BrowserGym/blob/c3336ef61781ce39166ee6a9551dbfc8fac32ddc/browsergym/core/src/browsergym/core/action/parsers.py#L17
    ---------------

    Returns:
        An action parser that accepts Python-like function calls with string, number, list or dict literals as arguments.
        Example:
            func("a", 42, None, True, [2, 4, "s"], {"a_key": "a_value"}, )
        The parser is loose and accepts multi-line or single-line combinations af calls.
        Example:
            func() func()
            \tfunc()
        Python comments are ignored.
        Example:
            # this is a comment
            func()    # this function call will be parsed
            # func()  # this one will not
        The parser will return a list of (function_name, function_args) tuples, one for each function call in the input.
        The parser will raise exceptions

    """

    def make_keyword(kwd_str, kwd_value):
        return pp.Keyword(kwd_str).set_parse_action(pp.replace_with(kwd_value))

    TRUE = make_keyword("True", True)
    FALSE = make_keyword("False", False)
    NONE = make_keyword("None", None)

    LBRACK, RBRACK, LBRACE, RBRACE, LPAREN, RPAREN, COLON = map(pp.Suppress, "[]{}():")

    def literal_eval(toks):
        return ast.literal_eval(toks[0])

    string = pp.python_quoted_string().set_parse_action(literal_eval)
    number = pp.pyparsing_common.number()
    dict = pp.Forward().set_name("dict")  # will be defined later
    list = pp.Forward().set_name("list")  # will be defined later
    _tuple = pp.Forward().set_name("tuple")  # will be defined later
    element = (string | number | dict | list | _tuple | TRUE | FALSE | NONE).set_name(
        "element"
    )

    list_items = pp.DelimitedList(element, allow_trailing_delim=True).set_name(None)
    list << pp.Group(LBRACK + pp.Optional(list_items) + RBRACK, aslist=True)
    _tuple << pp.Group(
        LPAREN + pp.Optional(list_items) + RPAREN, aslist=True
    ).set_parse_action(lambda tokens: tuple(tokens[0]))

    dict_item = pp.Group(string + COLON + element, aslist=True).set_name("dict item")
    dict_items = pp.DelimitedList(dict_item, allow_trailing_delim=True).set_name(None)
    dict << pp.Dict(LBRACE + pp.Optional(dict_items) + RBRACE, asdict=True)

    arg = element
    list_args = pp.DelimitedList(arg, allow_trailing_delim=True).set_name(None)
    named_arg = (
        pp.pyparsing_common.identifier() + pp.Literal("=") + element
    ).set_parse_action(lambda tokens: NamedArgument(name=tokens[0], value=tokens[2]))
    list_named_args = pp.DelimitedList(named_arg, allow_trailing_delim=True).set_name(
        None
    )
    function_call = pp.pyparsing_common.identifier() + pp.Group(
        LPAREN + pp.Optional(list_args) + pp.Optional(list_named_args) + RPAREN,
        aslist=True,
    )

    multiple_function_calls = pp.DelimitedList(pp.Group(function_call), delim="")
    multiple_function_calls.ignore(pp.python_style_comment())

    parser = multiple_function_calls

    return parser


def replace_string_content(s, start="https://", end=".png", replacement="<URL>"):
    # erase everything between start and end
    # example: https://www.example.com/image.png
    # becomes: replaced

    # find the start and end indices
    start_index = s.find(start)
    end_index = s.find(end, start_index) + len(end)
    if start_index == -1 or end_index == -1:
        return s
    # replace the content
    return s[:start_index] + replacement + s[end_index:]


def infer_task_name(base_traj_dir, benchmark, agent):
    agent_full = agents_inverse[agent]
    benchmark_full = benchmarks_inverse[benchmark]
    traj_dir = Path(
        base_traj_dir,
        benchmark_full,
        agent_full,
        f"{agent_full}_on_{benchmark_full}",
    )
    traj_dir = traj_dir.resolve()
    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")
    # get one json file in the directory
    json_files = list(traj_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {traj_dir}")

    # get the first json file
    json_file = json_files[0]
    # task_name is the part of the filename before the last dot
    task_name = json_file.stem.split(".")[:-1]
    # join the task name with the benchmark name
    task_name = ".".join(task_name)

    return task_name


def get_element_from_action_str(action_str, highlevel_action_parser=None):
    import pyparsing

    if highlevel_action_parser is not None:
        highlevel_action_parser = _build_highlevel_action_parser()

    try:
        function_calls = highlevel_action_parser.parse_string(
            action_str, parse_all=True
        )
        action_function, action_args = function_calls[0]
    except pyparsing.exceptions.ParseException:
        action_function = "UNKNOWN"
        action_args = []

    if len(action_args) > 0:
        # first argument is the element
        element = action_args[0]
    else:
        element = None

    return element


def get_trajectory_path(base_traj_dir, benchmark, agent, task_id):
    agent_full = agents_inverse[agent]
    benchmark_full = benchmarks_inverse[benchmark]
    task_full = tasks_dict[benchmark_full]

    traj_path = Path(
        base_traj_dir,
        benchmark_full,
        agent_full,
        f"{agent_full}_on_{task_full}",
        f"{task_full}.{task_id}.json",
    )
    traj_path = traj_path.resolve()

    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    return traj_path


def get_judgment_path(base_judgments_dir, benchmark, agent, judge, task_id):
    agent_full = agents_inverse[agent]
    benchmark_full = benchmarks_inverse[benchmark]
    task_full = tasks_dict[benchmark_full]
    judge_full = judges_inverse[judge]

    judgment_path = Path(
        base_judgments_dir,
        benchmark_full,
        agent_full,
        judge_full,
        f"{task_full}.{task_id}.json",
    )
    judgment_path = judgment_path.resolve()

    if not judgment_path.exists():
        raise FileNotFoundError(f"Judgment file not found: {judgment_path}")

    return judgment_path


def list_benchmarks(base_traj_dir):
    benchmarks_all = list(benchmarks_dict.values())
    # filter by the benchmarks that are in the base_traj_dir
    benchmarks = []
    for benchmark in benchmarks_all:
        traj_dir = Path(base_traj_dir, benchmarks_inverse[benchmark])
        traj_dir = traj_dir.resolve()
        if traj_dir.exists():
            benchmarks.append(benchmark)
    # sort the benchmarks
    benchmarks.sort()
    return benchmarks



def list_agents(base_traj_dir, benchmark):
    # show only the agents that are in the base_traj_dir
    benchmark_full = benchmarks_inverse[benchmark]
    traj_dir = Path(base_traj_dir, benchmark_full)
    traj_dir = traj_dir.resolve()
    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")
    # list all dirs that are not hidden
    subdirs = [
        f for f in traj_dir.iterdir() if f.is_dir() and not f.name.startswith(".")
    ]
    agent_names = [agents_dict[s.name] for s in subdirs if s.name in agents_dict]

    # sort the agent names
    agent_names.sort()

    return agent_names


def list_task_ids(base_traj_dir, benchmark, agent):
    # example: trajectories/cleaned/workarena/GenericAgent-anthropic_claude-3.7-sonnet/GenericAgent-anthropic_claude-3.7-sonnet_on_workarena.servicenow
    agent_full = agents_inverse[agent]
    benchmark_full = benchmarks_inverse[benchmark]
    task_full = tasks_dict[benchmark_full]

    traj_dir = Path(
        base_traj_dir,
        benchmark_full,
        agent_full,
        f"{agent_full}_on_{task_full}",
    )
    traj_dir = traj_dir.resolve()

    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    task_ids = [f.stem.split(".")[-1] for f in traj_dir.glob("*.json")]

    # sort as integer if possible, otherwise as string
    task_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

    return task_ids


def get_message_from_judgment(judgment):
    try:
        output = judgment['response']['choices'][0]['message']['content']
    except:
        output = "No judgment found"
    return output

def get_message_from_rule_based(judgment):
    try:
        r = judgment['trajectory_info']['summary_info']['cum_reward']
        output = "Success" if r > 0.5 else "Failure"
    except:
        output = "No judgment found"
    
    return output

def records_to_dict(records, key_order: list = ['benchmark', 'model_name', 'task_id']):
    """
    Convert a list of records to a nested dict, with key order
    The depth of the dict is determined by the number of keys in key_order.
    """
    
    result = {}

    for record in records:
        # get the keys in the order of key_order
        keys = [record[key] for key in key_order]
        # create a nested dict
        d = result
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        # set the value
        d[keys[-1]] = record
    
    return result

def format_annotation(annotation):
    annotation_str = dedent(f"""
    Success: {annotation['trajectory_success']}
    Side Effect: {annotation['trajectory_side_effect']}
    Looping: {annotation['trajectory_looping']}
    """)
    return annotation_str.strip()


base_traj_dir = "trajectories/cleaned"
base_screenshot_dir = "trajectories/screenshots"
base_judgments_dir = "trajectories/judgments"
annotations_path = "./annotations.csv"

base_traj_dir = Path(base_traj_dir)
base_screenshot_dir = Path(base_screenshot_dir)

hl_action_parser = _build_highlevel_action_parser()

# load annotations as records via csv
with open(annotations_path, "r") as f:
    annotations = list(csv.DictReader(f))
annotations_dict = records_to_dict(annotations, key_order=['benchmark', 'model_name', 'task_id'])

# convert the annotations to a dict, with key order 

with gr.Blocks(title="AgentRewardBench Demo") as demo:
    gr.Markdown(
        """
        # AgentRewardBench Demo ([paper](https://arxiv.org/abs/2504.08942))
        """
    )
    with gr.Row():
        with gr.Column(scale=4):
            benchmark_default = "WebArena"
            benchmark_dd = gr.Dropdown(
                label="Benchmark", choices=list_benchmarks(base_traj_dir), value=benchmark_default
            )

            agents = list_agents(base_traj_dir, benchmark_default)
            model_dd = gr.Dropdown(label="Agent", choices=agents, value=agents[0])

            task_ids = list_task_ids(base_traj_dir, benchmark_default, agents[0])
            task_id_dd = gr.Dropdown(label="Task ID", choices=task_ids, value=task_ids[0])

        @benchmark_dd.change(inputs=[benchmark_dd], outputs=[model_dd])
        def update_agents(benchmark):
            agents = list_agents(base_traj_dir, benchmark)
            return gr.Dropdown(label="Agent", choices=agents, value=agents[0])

        @model_dd.change(inputs=[benchmark_dd, model_dd], outputs=[task_id_dd])
        def update_task_ids(benchmark, agent):
            task_ids = list_task_ids(base_traj_dir, benchmark, agent)

            return gr.Dropdown(choices=task_ids, value=task_ids[0])

        with gr.Column(scale=8):
            @gr.render(inputs=[benchmark_dd, model_dd, task_id_dd])
            def render_trajectory(benchmark, agent, task_id):
                traj_path = get_trajectory_path(base_traj_dir, benchmark, agent, task_id)
                with open(traj_path, "rb") as f:
                    traj = orjson.loads(f.read())

                goal = replace_string_content(traj["goal"])

                gr.Textbox(label="Goal", value=goal, visible=True)

                for step in traj["steps"]:
                    num = step["num"]
                    action = step["action"]
                    reasoning = step["reasoning"]
                    screenshot_path = step["screenshot_path"]

                    gr.Markdown(f"# Step {num}")
                    with gr.Group():
                        im = Image.open(screenshot_path)
                        im = apply_overlay_to_image(
                            im, step, highlevel_action_parser=hl_action_parser
                        )
                        format_ = "webp" if im.format is None else im.format
                        gr.Image(im, label="Screenshot", format=format_)
                        if reasoning is not None:
                            gr.Textbox(reasoning, label="Reasoning", lines=4)
                        if action is not None:
                            gr.Textbox(action, label="Action", lines=2)

            # multi-choices dropdown for judges
            judge_dd = gr.Dropdown(
                label="Judges",
                choices=list(judges_dict.values()),
                multiselect=True,
                value=default_judges,
            )

            # get annotation for the task from annotations_dict
            @gr.render(inputs=[benchmark_dd, model_dd, task_id_dd])
            def render_annotation(benchmark, agent, task_id):
                bench_full = benchmarks_inverse[benchmark]
                agent_full = agents_inverse[agent]
                task_full = tasks_dict[bench_full]
                task_id_full = f"{task_full}.{task_id}"
                # get the annotation
                annotation = annotations_dict[bench_full][agent_full][task_id_full]
                annotation_str = format_annotation(annotation)

                gr.Textbox(label="Expert Annotation", value=annotation_str, lines=3)
                

            @gr.render(inputs=[benchmark_dd, model_dd, task_id_dd, judge_dd])
            def render_judge(benchmark, agent, task_id, judge_choices):                
                # load judgments
                for judge in judges_dict.values():
                    if judge not in judge_choices:
                        continue

                    judgment_path = get_judgment_path(
                        base_judgments_dir, benchmark, agent, judge, task_id
                    )
                    if not judgment_path.exists():
                        continue

                    with open(judgment_path, "rb") as f:
                        judgment = orjson.loads(f.read())
                    if judge == "Rule-based":
                        msg = get_message_from_rule_based(judgment)
                    else:
                        msg = get_message_from_judgment(judgment)

                    gr.Textbox(label=judge, value=msg, lines=4)

demo.launch()
