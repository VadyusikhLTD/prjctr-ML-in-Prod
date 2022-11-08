from datetime import datetime
from pathlib import Path

import feast
import pandas as pd
import typer


# Connect to your local feature store
fs = feast.FeatureStore(repo_path="features/")


def get_dataset(file_path: Path = Path("app/data/data_for_13_SEP_2022_a.parquet")) -> pd.DataFrame:
    entity_df = pd.read_parquet(file_path)
    return entity_df


def add_features(training_df: pd.DataFrame) -> pd.DataFrame:
    training_df_with_features = fs.get_historical_features(
        entity_df=training_df,
        features=[
            "bus_decimal_stats_source:lat",
            "bus_decimal_stats_source:lng",
            "bus_decimal_stats_source:routeName",
        ],
    ).to_df()
    return training_df_with_features


def train_model(
        dataset_path,
        model_resutl_path: Path = Path("driver_model.bin"), ):
    training_df = get_dataset(dataset_path)
    training_df_with_features = add_features(training_df=training_df)
    print(f"training_df = {training_df_with_features.head()}")
    # Train model
    # target = "trip_completed"

    # reg = LinearRegression()
    # train_X = training_df_with_features[training_df_with_features.columns.drop(target).drop("event_timestamp")]
    # train_Y = training_df_with_features.loc[:, target]
    # reg.fit(train_X[sorted(train_X)], train_Y)
    #
    # # Save model
    # dump(reg, model_resutl_path)


if __name__ == "__main__":
    train_model(dataset_path = Path("data/data_for_13_SEP_2022_a.parquet"))
    # typer.run(
    #     train_model,
    #     dataset_path = Path("data/data_for_13_SEP_2022_a.parquet")
    # )
