import numpy as np
import pandas as pd

TARGET = "Burnout_Risk_Level"
DROP_COLS = [TARGET, "Student_ID", "timestamp"]
DERIVED_COLS = ["GPA_Diff", "Anxiety_Squared", "Total_Study_Load",
                "Exhaustion_Index", "AI_Inefficiency"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["GPA_Diff"] = d["Post_Semester_GPA"] - d["Pre_Semester_GPA"]
    d["Anxiety_Squared"] = d["Anxiety_Level_During_Exams"] ** 2
    d["Total_Study_Load"] = d["Weekly_GenAI_Hours"] + d["Traditional_Study_Hours"]
    d["Exhaustion_Index"] = d["Anxiety_Level_During_Exams"] * d["Perceived_AI_Dependency"]
    d["AI_Inefficiency"] = d["Weekly_GenAI_Hours"] / (d["Skill_Retention_Score"] + 1)
    return d


def split_features(df: pd.DataFrame):
    """Повертає X (лише числові ознаки), y та Student_ID."""
    d = add_features(df)
    ids = d["Student_ID"]
    y = d[TARGET].astype(int)
    X = d.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include="number")
    return X, y, ids

def build_model_input(raw: dict, feature_names: list) -> pd.DataFrame:
    """Перетворює вхідні ознаки на DataFrame з тим самим порядком колонок, що й під час тренування."""
    base_needed = [c for c in feature_names if c not in DERIVED_COLS]
    missing = [c for c in base_needed if c not in raw]
    if missing:
        raise ValueError(f"Відсутні ознаки: {missing}")
    d = add_features(pd.DataFrame([raw]))
    return d[feature_names].astype(float)


def to_class(pred) -> np.ndarray:
    return np.clip(np.round(pred), 0, 2).astype(int)