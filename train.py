import os

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter


def load_and_prepare_train_data(
    train_path, target_column, columns_for_drop, columns_to_float
):
    X_train = pd.read_csv(train_path)
    y_train = X_train[target_column]
    X_train = X_train.drop(columns_for_drop.split() + [target_column], axis=1)
    for column in columns_to_float.split():
        X_train[column] = (
            X_train[column].apply(lambda x: 0 if x == " " else x).astype(float)
        )
    X_train = X_train.fillna(0)
    X_train = pd.get_dummies(X_train, drop_first=True)
    return X_train, y_train


def fit_and_predict(X_train, y_train, params):
    gb = GradientBoostingClassifier()
    gb.set_params(**params)
    gb.fit(X_train, y_train)
    pred_train = gb.predict(X_train)
    precision = precision_score(y_train, pred_train)
    recall = recall_score(y_train, pred_train)
    f1 = f1_score(y_train, pred_train)
    sw = SummaryWriter("mlops_logs")
    sw.add_scalar("Precision on train dataset:", precision, global_step=0)
    sw.add_scalar("Recall on train dataset:", recall, global_step=0)
    sw.add_scalar("F1-score on train dataset:", f1, global_step=0)
    return gb


def save_model(model, model_name):
    with open(f"{model_name}.joblib", "wb") as file:
        joblib.dump(model, file)
    os.system(f"dvc add {model_name}.joblib")
    os.system(f"dvc push {model_name}.joblib.dvc")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    os.system("dvc pull --remote myremote")
    X_train, y_train = load_and_prepare_train_data(
        cfg.data.train_path,
        cfg.data.target_column,
        cfg.data.columns_for_drop,
        cfg.data.columns_to_float,
    )
    params = OmegaConf.to_container(cfg["model_params"])
    model = fit_and_predict(X_train, y_train, params)
    save_model(model, cfg.model_name)


if __name__ == "__main__":
    train()
