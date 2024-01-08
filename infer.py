import os

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig


def load_and_prepare_data(path, columns_for_drop, columns_to_float):
    X_test = pd.read_csv(path)
    X_test = X_test.drop(columns_for_drop.split(), axis=1)
    for column in columns_to_float.split():
        X_test[column] = (
            X_test[column].apply(lambda x: 0 if x == " " else x).astype(float)
        )
    X_test = X_test.fillna(0)
    X_test = pd.get_dummies(X_test, drop_first=True)
    return X_test


def predict(X_test, model_name):
    model = joblib.load(f"{model_name}.joblib")
    pred = pd.DataFrame(model.predict(X_test))
    return pred


@hydra.main(config_path="configs", config_name="config", version_base=None)
def infer(cfg: DictConfig):
    os.system("dvc pull --remote myremote")
    X_test = load_and_prepare_data(
        cfg.data.test_x_path, cfg.data.columns_for_drop, cfg.data.columns_to_float
    )
    pred = predict(X_test, cfg.model_name)
    pred.to_csv("data/predict.csv", index=False)
    os.system("dvc add data/predict.csv")
    os.system("dvc push")


if __name__ == "__main__":
    infer()
