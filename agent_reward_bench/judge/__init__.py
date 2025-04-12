import base64
import json
from pathlib import Path

from . import defaults, utils


def format_content_for_image(b64_url: str) -> list:
    content = [
        {"type": "text", "text": "Here is the screenshot of the last step."},
        {"type": "image_url", "image_url": {"url": b64_url}},
    ]

    return content


def format_steps(steps: list, step_template: str = None) -> str:
    if step_template is None:
        step_template = defaults.STEP_TEMPLATE

    steps_str = ""
    for i, step in enumerate(steps):
        steps_str += step_template.format(
            step_number=i + 1,
            url=step["url"],
            action=step["action"],
            reasoning=step["reasoning"],
        )
    return steps_str


def format_chat_messages_for_judge(
    sys_prompt: str,
    goal_msg: str,
    action_msg: str,
    axtree_msg: str,
    img_msg_content: list,
    final_msg: str = None,
):
    if axtree_msg is None:
        axtree_msg_content = []
    else:
        axtree_msg_content = [{"type": "text", "text": axtree_msg}]

    if final_msg is None:
        final_msg = defaults.FINAL_MSG

    user_content_lst = [
        {"type": "text", "text": goal_msg},
        {"type": "text", "text": action_msg},
        *axtree_msg_content,
        *img_msg_content,
        {"type": "text", "text": final_msg},
    ]

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content_lst},
    ]


def get_response_msg(response: dict):
    return response["choices"][0]["message"]["content"]


def get_content_inside_tag(tag: str, response_msg: str):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"

    start_idx = response_msg.find(start_tag)
    end_idx = response_msg.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        return None
    return response_msg[start_idx + len(start_tag) : end_idx]


def parse_judgment(response_msg: dict):
    # get the content between <reasoning>, <success>, <side>, <optimal>, <loop>
    reasoning = get_content_inside_tag("reasoning", response_msg)
    success = get_content_inside_tag("success", response_msg)
    side = get_content_inside_tag("side", response_msg)
    optimal = get_content_inside_tag("optimal", response_msg)
    loop = get_content_inside_tag("loop", response_msg)

    return {
        "reasoning": reasoning,
        "trajectory_success": success,
        "trajectory_side_effect": side,
        "trajectory_optimality": optimal,
        "trajectory_looping": loop,
    }


def parse_aer_judgment(response_msg: dict):
    judgment = {
        "reasoning": None,
        "trajectory_success": None,
        "trajectory_side_effect": "n/a",
        "trajectory_optimality": "n/a",
        "trajectory_looping": "n/a",
    }
    for line in response_msg.split("\n"):
        if line.startswith("Thoughts:"):
            splitted = line.split(":", 1)
            if len(splitted) == 2:
                judgment["reasoning"] = splitted[1].strip()
        elif line.startswith("Status:"):
            splitted = line.split(":", 1)
            if len(splitted) == 2:
                judgment["trajectory_success"] = splitted[1].strip()

    return judgment

def convert_likert5_to_likert4(score_on_5: str) -> str:
    # first, convert to integers
    try:
        score_on_5 = int(score_on_5.strip())
    except ValueError:
        raise ValueError(f"Invalid Likert-5 value: {score_on_5}")
    
    if score_on_5 == 1:
        return 1
    elif score_on_5 == 2:
        return 1
    elif score_on_5 == 3:
        return 2
    elif score_on_5 == 4:
        return 3
    elif score_on_5 == 5:
        return 4

def convert_likert5_to_binary(score_on_5: str) -> str:
    # first, convert to integers
    try:
        score_on_5 = int(score_on_5.strip())
    except ValueError:
        raise ValueError(f"Invalid Likert-5 value: {score_on_5}")
    
    if score_on_5 >= 4:
        return 1
    else:
        return 0
        
def parse_nnetnav_judgment(response_msg: dict):
    judgment = {
        "reasoning": None,
        "trajectory_success": None,
        "trajectory_side_effect": "n/a",
        "trajectory_optimality": "n/a",
        "trajectory_looping": "n/a",
    }
    if "Reward:" in response_msg:
        # split by "Success:"
        reasoning_line, success_line = response_msg.split("Reward:", 1)
        success_line = success_line.strip()
        reasoning_line = reasoning_line.strip()

        # if success_line is an integer, convert it to a boolean
        if success_line.isdigit():
            judgment["trajectory_optimality"] = convert_likert5_to_likert4(success_line)
            judgment["trajectory_success"] = convert_likert5_to_binary(success_line)
        else:
            breakpoint()
            raise ValueError(f"Invalid success line: {success_line}")
        
        if "Thought:" in reasoning_line:
            # split by "Thoughts:"
            _, reasoning_line = reasoning_line.split("Thought:", 1)
        reasoning_line = reasoning_line.strip()
        judgment["reasoning"] = reasoning_line
    
    if judgment["trajectory_success"] is None or judgment["reasoning"] is None:
        breakpoint()
    return judgment


def create_chat_messages_from_trajectory(
    trajectory, traj_dir, use_screenshot=True, use_axtree=True, invert_system_prompt=False
):
    last_step = trajectory["steps"][-1]

    if use_screenshot:
        img_msg_content = format_content_for_image(
            utils.image_to_base64(traj_dir.joinpath(last_step["screenshot_path"]))
        )
    else:
        img_msg_content = []

    if use_axtree:
        axtree_msg = defaults.AXTREE_TEMPLATE.format(axtree=last_step["axtree"])
        axtree_pruned = defaults.AXTREE_TEMPLATE.format(
            axtree=last_step["axtree_pruned"]
        )
    else:
        axtree_msg = None
        axtree_pruned = None
    
    if invert_system_prompt:
        sys_prompt = defaults.INVERTED_SYSTEM_PROMPT
    else:
        sys_prompt = defaults.SYSTEM_PROMPT
    
    action_msg = defaults.ACTION_TEMPLATE.format(
        steps=format_steps(trajectory["steps"])
    )
    
    chat_messages = format_chat_messages_for_judge(
        sys_prompt=sys_prompt,
        goal_msg=defaults.GOAL_TEMPLATE.format(goal=trajectory["goal"]),
        action_msg=action_msg,
        axtree_msg=axtree_msg,
        img_msg_content=img_msg_content,
    )

    chat_messages_pruned = format_chat_messages_for_judge(
        sys_prompt=sys_prompt,
        goal_msg=defaults.GOAL_TEMPLATE.format(goal=trajectory["goal"]),
        action_msg=action_msg,
        axtree_msg=axtree_pruned,
        img_msg_content=img_msg_content,
    )

    return {
        "regular": chat_messages,
        "pruned": chat_messages_pruned,
    }
