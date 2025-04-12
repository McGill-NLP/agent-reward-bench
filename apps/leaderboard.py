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

with gr.Blocks() as app:
    gr.Markdown(
        """
        # AgentRewardBench Leaderboard
        
        This is the leaderboard for the AgentRewardBench. The scores are based on the results of the agents on the benchmark. We report the *precision* score.
        """
    )
    df = load_data()
    table = gr.DataFrame(df, show_label=False)
    
# launch the app

if __name__ == "__main__":
    # launch the app    
    app.launch()