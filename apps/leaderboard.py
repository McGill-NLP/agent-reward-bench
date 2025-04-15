"""
gradio app that reads results.csv and display it in a table, title is "AgentRewardBench Leaderboard"
"""
import gradio as gr

import pandas as pd

def load_data():
    # read the csv file
    df = pd.read_csv("./results.csv")
    # remove Recall and F1 columns
    df = df.drop(columns=["Recall", "F1"])
    # return the dataframe
    return df

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # AgentRewardBench Leaderboard


        | [**🤗Dataset**](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench) | **📄Paper (TBA)** | [**🌐Website**](https://agent-reward-bench.github.io) | [**🏆Leaderboard**](https://huggingface.co/spaces/McGill-NLP/agent-reward-bench-leaderboard) | [**💻Demo**](https://huggingface.co/spaces/McGill-NLP/agent-reward-bench-demo)
        | :--: | :--: | :--: | :--: | :--: |

        
        This is the leaderboard for the AgentRewardBench. The scores are based on the results of the agents on the benchmark. We report the *precision* score.
        [Open an issue to submit your results to the leadeboard](https://github.com/McGill-NLP/agent-reward-bench/issues/new?template=add-results-to-leaderboard.yml). We will review your results and add them to the leaderboard.
        """
    )
    df = load_data()
    table = gr.DataFrame(df, show_label=False)

demo.queue(default_concurrency_limit=40).launch()