from datetime import datetime
from pathlib import Path

import feast
import pandas as pd
import typer
from joblib import dump
from sklearn.naive_bayes import CategoricalNB
from typing import Tuple

# Connect to your local feature store
fs = feast.FeatureStore(repo_path="features/")


def get_dataset(file_path: Path = Path("app/data/data_for_13_SEP_2022.parquet")) -> pd.DataFrame:
    entity_df = pd.read_parquet(file_path)
    return entity_df


def add_features(training_df: pd.DataFrame) -> Tuple[feast.infra.offline_stores.file.FileRetrievalJob, pd.DataFrame]:
    hist_fs = fs.get_historical_features(
        entity_df=training_df[:1000],
        features=[
            "bus_decimal_stats_source:lat",
            "bus_decimal_stats_source:lng",
            "bus_decimal_stats_source:routeName",
        ],
    )
    return hist_fs, hist_fs.to_df()


def train_model(
        dataset_path,
        model_resutl_path: Path = Path("driver_model.bin")
):
    training_df = get_dataset(dataset_path)
    print(f"training_df = {training_df.head()}")

    hist_fs, hist_fs_df = add_features(training_df=training_df)
    features_name_list = [f.split(':')[-1] + '__' for f in hist_fs.metadata.features]
    print(f"features_name_list = {features_name_list}")
    # Train model
    target = "routeName__"
    features_name_list.remove(target)

    clf = CategoricalNB()
    train_X = hist_fs_df[features_name_list]
    train_Y = hist_fs_df.loc[:, target]
    clf.fit(train_X, train_Y)

    # Save model
    dump(clf, model_resutl_path)


if __name__ == "__main__":
    print("P start")
    train_model(
        dataset_path = Path("data/data_for_13_SEP_2022.parquet"),
        model_resutl_path=Path("data/driver_model.bin"))
    print("P end")
