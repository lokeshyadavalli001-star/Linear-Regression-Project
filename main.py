from sklearn.datasets import (
    fetch_california_housing
)

from src.train import (
    train_pipeline
)


def load_dataset():

    dataset = (
        fetch_california_housing(
            as_frame=True
        )
    )

    df = dataset.frame

    df.rename(
        columns={
            "MedHouseVal": "price"
        },
        inplace=True
    )

    return df


if __name__ == "__main__":

    

    print(
        "ADVANCED LINEAR "
        "REGRESSION PIPELINE"
    )

    

    df = load_dataset()

    train_pipeline(df)
