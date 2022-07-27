import pickle
from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd
from autosklearn.classification import AutoSklearnClassifier
from feast import FeatureStore


class NotARealTrainer:
    def __init__(
        self,
        name: str,
        work_dir: Path,
        memory: int,
        time_limit_in_seconds: int,
        metric: Callable,
        fs_feature_names: List[str],
        n_jobs: int = -1,
    ):
        self.name = name
        self.work_dir = work_dir
        self.memory = memory
        self.time_limit_in_seconds = time_limit_in_seconds
        self.metric = metric
        self.fs_feature_names = [f"space_titanic:{name}" for name in fs_feature_names]
        self.n_jobs = n_jobs

    def fit(
        self,
        train_data: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        self.model = AutoSklearnClassifier(
            time_left_for_this_task=self.time_limit_in_seconds,
            memory_limit=self.memory,
            n_jobs=self.n_jobs,
            metric=self.metric,
        )
        train_features, train_labels = train_data
        self.feature_names = train_features.columns
        self.model.fit(train_features, train_labels)
        self.save_model()

    def save_model(self) -> None:
        with open(self.work_dir / f"{self.name}.pkl", "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self) -> None:
        with open(self.work_dir / f"{self.name}.pkl", "rb") as file:
            self.model = pickle.load(file)

    def predict(self, store: FeatureStore, model_input: pd.DataFrame) -> None:
        # In real life you will already get the data in the predict method parameters
        entity_rows = [
            {"PassengerId": pid} for pid in model_input["PassengerId"].values
        ]
        fs_features = store.get_online_features(
            features=self.fs_feature_names, entity_rows=entity_rows
        ).to_df()
        all_features = pd.merge(model_input, fs_features, on="PassengerId")

        return self.model.predict(all_features[self.feature_names], n_jobs=self.n_jobs)
