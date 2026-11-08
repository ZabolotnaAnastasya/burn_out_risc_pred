import json
import os
import uuid
import datetime as dt

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from app.database import crud
from app.services.utils import split_features, to_class

RANDOM_STATE = 42
TRAIN_SIZE = 0.9          # 90% на train, 10% залишається як new_input
MODEL_NAME = "SVR_unbalanced_tuned"
MODELS_DIR = "models"

# Найкращі параметри з Optuna з ЛР3 (SVR без балансування)
BEST_PARAMS = {
    "C": 5.43670510259403,
    "epsilon": 0.014424574668842602,
    "gamma": 0.06991333531567578,
    "pca_components": 0.9838987819681129,
}


def build_model(params: dict = BEST_PARAMS) -> Pipeline:
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=params["pca_components"], random_state=RANDOM_STATE)),
        ("model", SVR(kernel="rbf", C=params["C"],
                      epsilon=params["epsilon"], gamma=params["gamma"])),
    ])


def evaluate(y_true, y_raw) -> dict:
    y_cls = to_class(y_raw)
    return {
        "mae": float(mean_absolute_error(y_true, y_raw)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_raw))),
        "r2": float(r2_score(y_true, y_raw)),
        "accuracy": float(accuracy_score(y_true, y_cls)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_cls)),
        "qwk": float(cohen_kappa_score(y_true, y_cls, weights="quadratic")),
    }


def run_training() -> dict:
    run_id = uuid.uuid4().hex[:8]
    created_at = dt.datetime.now()

    # 1. Читаємо всі записи з таблиці ознак
    df = crud.load_features()

    # 2. Ознаки, ціль, поділ 90:10 (стратифіковано за класом)
    X, y, ids = split_features(df)
    X_tr, X_new, y_tr, y_new, ids_tr, ids_new = train_test_split(
        X, y, ids, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=y)

    # 3. Тренування
    model = build_model().fit(X_tr, y_tr)

    # 4. Збереження моделі та метаданих
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_{run_id}.joblib")
    joblib.dump(model, model_path)

    # new_input зберігаємо окремо, щоб далі тестувати /predict
    new_input = df.loc[X_new.index].copy()
    new_input_path = os.path.join(MODELS_DIR, f"new_input_{run_id}.csv")
    new_input.to_csv(new_input_path, index=False)

    meta = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "params": BEST_PARAMS,
        "features": list(X.columns),
        "classes": {0: "Low", 1: "Medium", 2: "High"},
        "postprocess": "clip(round(pred), 0, 2)",
        "train_size": int(len(X_tr)),
        "new_input_size": int(len(X_new)),
        "created_at": created_at.isoformat(),
    }
    with open(os.path.join(MODELS_DIR, f"{MODEL_NAME}_{run_id}_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 5. Передбачення на train-даних у predictions (source = "train")
    raw_tr = np.clip(model.predict(X_tr), 0, 2)
    crud.save_predictions(pd.DataFrame({
        "run_id": run_id,
        "created_at": created_at,
        "model_name": MODEL_NAME,
        "source": "train",
        "student_id": np.asarray(ids_tr),
        "y_true": np.asarray(y_tr),
        "y_pred_raw": raw_tr,
        "y_pred_class": to_class(raw_tr),
    }))

    # 6. Метрики на train в model_metrics
    metrics = evaluate(y_tr, raw_tr)
    crud.save_metrics({
        "run_id": run_id, "created_at": created_at, "model_name": MODEL_NAME,
        "stage": "train", "split": "train", **metrics,
        "params": json.dumps(BEST_PARAMS),
    })

    return {
        "run_id": run_id,
        "model_path": model_path,
        "train_rows": len(X_tr),
        "new_input_rows": len(X_new),
        "new_input_path": new_input_path,
        "train_metrics": metrics,
    }