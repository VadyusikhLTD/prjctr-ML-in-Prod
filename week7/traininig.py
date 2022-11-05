from datetime import datetime
from pathlib import Path

import feast
import pandas as pd
import typer
from joblib import dump
from sklearn.linear_model import LinearRegression

# Connect to your local feature store
fs = feast.FeatureStore(repo_path="features/")


def get_dataset() -> pd.DataFrame:
    entity_df = pd.read_parquet("data/data_for_21_SEP_to_30_SEP_2022.parquet")
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


def train_model(model_resutl_path: Path = Path("driver_model.bin")):
    training_df = get_dataset()
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
    typer.run(train_model)
