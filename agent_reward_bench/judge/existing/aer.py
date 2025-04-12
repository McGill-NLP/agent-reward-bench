import json
from pathlib import Path
from typing import TYPE_CHECKING

from .. import format_steps
from ..utils import image_to_base64

if TYPE_CHECKING:
    import openai


def format_msg_for_captioning(b64_url: str) -> list:
    prompt = "You are an advanced GUI captioner. Please describe this GUI interface in details and don’t miss anything. Your response should be hierarchical and in Markdown format. Don't do paraphrase. Don't wrap your response in a code block."
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": b64_url}},
    ]

    return content


def get_caption(
    client: "openai.OpenAI",
    trajectory,
    traj_dir,
    path,
    screenshot_cache_dir="trajectories/captions/",
    captioner_model="gpt-4o-2024-11-20",
    temperature=0,
    max_completion_tokens=1024,
    seed=0,
):
    last_step = trajectory["steps"][-1]
    benchmark = trajectory["benchmark"]
    agent = trajectory["agent"].replace("/", "_")

    caption_save_path = Path(
        screenshot_cache_dir,
        benchmark,
        agent,
        captioner_model.replace("/", "_"),
        f"{path.stem}.last_screenshot_caption.json",
    )

    if caption_save_path.exists():
        with open(caption_save_path, "r") as f:
            result = json.load(f)
            caption = result["caption"]
    else:
        content = format_msg_for_captioning(
            image_to_base64(traj_dir.joinpath(last_step["screenshot_path"]))
        )
        response = client.chat.completions.create(
            model=captioner_model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            seed=seed,
        )

        caption_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(caption_save_path, "w") as f:
            caption = response.choices[0].message.content
            result = {
                "caption": caption,
                "response": response.model_dump(),
            }
            json.dump(result, f)

    return caption


sys_prompt = """
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.
There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, map info, comparison of map routes, etc. The bot's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the bot encounters an exception and respond with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked to list the top-searched items in a shop, the agent should order the items by the number of searches, and then return the top items. If the ordering action is missing, the task is likely to fail.
2. Site navigation: The user wants to navigate to a specific page. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>"

Status: "success" or "failure"
"""

user_prompt_template = """
User Intent: {intent}
Action History:
{last_actions}

The detailed final state of the webpage:
```md
{cap}
```
"""

user_prompt_template_vis = """
User Intent: {intent}
Action History: {last_actions}
The last snapshot of the web page is shown in the image.
"""


def create_aer_chat_messages_from_trajectory(
    trajectory,
    traj_dir,
    path,
    client=None,
    captioner_kwargs=None,
):
    if client is None:
        raise ValueError("client must be provided")

    if captioner_kwargs is None:
        captioner_kwargs = {}

    cap = get_caption(
        client=client,
        trajectory=trajectory,
        traj_dir=traj_dir,
        path=path,
        **captioner_kwargs,
    )
    user_msg = user_prompt_template.strip().format(
        intent=trajectory["goal"],
        last_actions=format_steps(trajectory["steps"]),
        cap=cap,
    )

    chat_msg = [
        {"role": "system", "content": sys_prompt.strip()},
        {"role": "user", "content": user_msg},
    ]

    return {"regular": chat_msg, "pruned": chat_msg}

def create_aer_chat_messages_from_trajectory_vis(
    trajectory,
    traj_dir,
    client=None,
):
    if client is None:
        raise ValueError("client must be provided")

    user_msg = user_prompt_template_vis.strip().format(
        intent=trajectory["goal"],
        last_actions=format_steps(trajectory["steps"])
    )

    traj_dir = Path(traj_dir)
    last_step = trajectory["steps"][-1]
    b64_url = image_to_base64(traj_dir.joinpath(last_step["screenshot_path"]))

    chat_msg = [
        {"role": "system", "content": sys_prompt.strip()},
        {"role": "user", "content": [
            {'type': 'text', 'text': user_msg},
            {"type": "image_url", "image_url": {"url": b64_url}},
        ]},
    ]

    return {"regular": chat_msg, "pruned": chat_msg}