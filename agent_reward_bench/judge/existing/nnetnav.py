import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

if TYPE_CHECKING:
    import openai


JUDGE_SYS_PROMPT = """
An autonomous intelligent agent navigating a web browser is given an instruction by a user. Your objective is to give a score to the agent based on how well it completed its task. Your score must be on the scale of 1 to 5. Give a score of 5 only when there are no errors. To do this task you are provided with the following information:

Instruction: This is the natural language instruction given to the agent.
Trajectory: This is a sequence of natural language descriptions of the agent's interaction with the web-browser.

To be successful, it is very important to follow the following rules:
1. Explictly think about what is needed to follow the instruction correctly on the website and if the trajectory reflects these steps.
2 Give a score of 4 if there are minor errors, or if the task was more than 70% completed. Give a score of 3 (or below) if the model made very little progress towards the given instruction.
3. Start by thinking by outputing Thought: <your-reasoning>.
4. End your answer by strictly following the format "Reward: <your-answer>" for your output
"""

JUDGE_USER_PROMPT_TEMPLATE = """
Instruction: {goal}
Trajectory: {descriptions}
"""

DELTA_LM_SYS_PROMPT = """
"You are given the output of an action taken by an autonomous intelligent agent navigating a web browser. Your objective is to produce a description of the changes made to the state of the browser.

Here's the information you'll have:

Initial state of the browser as an accessibility tree: This is a simplified representation of the webpage, providing key information.
Final state of the browser: This is the accessibility tree representation after the agent took the action

The action taken by the web agent: The agent can take actions that fall under the following categories: {action_str}

To be successful, it is very important to follow the following rules:
1. Explictly think about the various features on the website and how the interaction with the website changed
these features
2. Provide the description of changes in one or two sentences.
3. Strictly follow the format "State change: <your-answer>" for your output"
"""

DELTA_LM_USER_PROMPT = """
Initial state of the browser as an accessibility tree:
{prev_axtree}
Final state of the browser:
{curr_axtree}
The action taken by the web agent: {action}
    """

webarena_action_str = """
Page Operation Actions:
'click [id]': This action clicks on an element with a specific id on the webpage.
'type [id] [content] [press_enter_after=0|1]': Use this to type the content into the field with id. By default,
the "Enter" key is pressed after typing unless press_enter_after is set to 0.
'hover [id]': Hover over an element with id.
'press [key_comb]': Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
'scroll [direction=down|up]': Scroll the page up or down.
Tab Management Actions:
'new_tab': Open a new, empty browser tab.
'tab_focus [tab_index]': Switch the browser's focus to a specific tab using its index.
'close_tab': Close the currently active tab.
URL Navigation Actions:
'goto [url]': Navigate to a specific URL.
'go_back': Navigate to the previously viewed page.
'go_forward': Navigate to the next page (if a previous 'go_back' action was performed).
Completion Action:
'stop ["done"]': Issue this action when you are done.
"""

JUDGE_SYS_PROMPT = JUDGE_SYS_PROMPT.strip()
DELTA_LM_SYS_PROMPT = DELTA_LM_SYS_PROMPT.strip()
DELTA_LM_USER_PROMPT = DELTA_LM_USER_PROMPT.strip()
JUDGE_USER_PROMPT_TEMPLATE = JUDGE_USER_PROMPT_TEMPLATE.strip()
webarena_action_str = webarena_action_str.strip()


def prepare_chat_messages_for_delta_lm(prev_step, curr_step, pruned_axtree=False):
    # get the axtree from prev step and curr step
    # compare the axtree and describe the changes
    if pruned_axtree:
        prev_axtree = prev_step["axtree_pruned"]
        curr_axtree = curr_step["axtree_pruned"]
    else:
        prev_axtree = prev_step["axtree"]
        curr_axtree = curr_step["axtree"]

    action = curr_step["action"]
    # USE openai client to get the description of the changes

    sys_prompt = DELTA_LM_SYS_PROMPT.format(action_str=webarena_action_str)
    user_prompt = DELTA_LM_USER_PROMPT.format(
        prev_axtree=prev_axtree,
        curr_axtree=curr_axtree,
        action=action,
    )

    chat_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return chat_messages


def get_descriptions(
    client: "openai.OpenAI",
    trajectory,
    path,
    verbose=True,
    description_cache_dir="trajectories/descriptions/",
    delta_lm_name="meta-llama/Llama-3.3-70B-Instruct",
    temperature=0,
    max_completion_tokens=2048,
    seed=0,
):
    import openai

    agent = trajectory["agent"].replace("/", "_")
    benchmark = trajectory["benchmark"]

    descriptions_save_path = Path(
        description_cache_dir,
        benchmark,
        agent,
        delta_lm_name.replace("/", "_"),
        f"{path.stem}.descriptions.json",
    )

    if descriptions_save_path.exists():
        with open(descriptions_save_path, "r") as f:
            result = json.load(f)

        return result
    else:
        descriptions = []
        task_id = path.stem.split(".")[-1]
        
        for i in tqdm(
            range(len(trajectory["steps"]) - 1),
            desc="Getting descriptions for task {}".format(task_id),
            disable=not verbose,
            leave=False,
        ):
            prev_step = trajectory["steps"][i]
            curr_step = trajectory["steps"][i + 1]

            chat_messages = prepare_chat_messages_for_delta_lm(
                prev_step, curr_step, pruned_axtree=False
            )
            try:
                response = client.chat.completions.create(
                    model=delta_lm_name,
                    messages=chat_messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    seed=seed,
                )
            except openai.BadRequestError as e:
                chat_messages = prepare_chat_messages_for_delta_lm(
                    prev_step, curr_step, pruned_axtree=True
                )
                try:
                    response = client.chat.completions.create(
                        model=delta_lm_name,
                        messages=chat_messages,
                        max_completion_tokens=max_completion_tokens,
                        temperature=temperature,
                        seed=seed,
                    )
                except openai.BadRequestError as e:
                    logging.warning(f"Error in getting descriptions for {path}")
                    continue

            result = {
                "step": i,
                "description": response.choices[0].message.content,
                "response": response.model_dump(),
            }
            descriptions.append(result)

        descriptions_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(descriptions_save_path, "w") as f:
            json.dump(descriptions, f)

    return descriptions


def create_nnetnav_chat_messages_from_trajectory(
    trajectory,
    path,
    client=None,
    delta_lm_kwargs=None,
    verbose=True,
):
    if client is None:
        raise ValueError("client must be provided")

    if delta_lm_kwargs is None:
        delta_lm_kwargs = {}

    descriptions = get_descriptions(
        client, trajectory, path, verbose, **delta_lm_kwargs
    )
    descriptions_text = "\n".join([d["description"] for d in descriptions])

    user_msg = JUDGE_USER_PROMPT_TEMPLATE.format(
        goal=trajectory["goal"],
        descriptions=descriptions_text,
    )

    chat_messages = [
        {"role": "system", "content": JUDGE_SYS_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    return {
        "regular": chat_messages,
        "pruned": chat_messages,
    }
