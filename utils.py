import pandas

def return_empty_dataframe_benchmark() -> pandas.DataFrame:
    return pandas.DataFrame(
        columns=[
            "learning_rate",
            "initial_epsilon",
            "epsilon_decay",
            "final_epsilon",
            "discount_factor",
            "episode",
            "total_reward",
            "total_steps",
            "total_time",
        ]
    )

def write_dataframe_into_csv(df: pandas.DataFrame, path: str):
    df.to_csv(path, index=False)