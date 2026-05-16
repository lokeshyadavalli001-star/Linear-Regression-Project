import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame):

    df = df.copy()

    # Feature Engineering

    if "AveRooms" in df.columns and "AveBedrms" in df.columns:

        df["room_bedroom_ratio"] = (
            df["AveRooms"] /
            (df["AveBedrms"] + 1e-5)
        )

    if "Population" in df.columns:

        df["Population_log"] = np.log1p(
            df["Population"]
        )

    return df
