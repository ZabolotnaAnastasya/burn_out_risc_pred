import pandas as pd
from app.database.database import engine

import json
from typing import Optional

from sqlalchemy import text


def load_student_features(student_id: int) -> Optional[dict]:
    df = pd.read_sql(
        text('SELECT * FROM stress_data_processed WHERE "Student_ID" = :sid'),
        engine, params={"sid": student_id})
    if df.empty:
        return None
    row = df.iloc[0].drop(labels=["Student_ID", "Burnout_Risk_Level", "timestamp"], errors="ignore")
    return {k: float(v) for k, v in row.items()}


def save_inference_input(request_id: str, student_id: Optional[int], payload: dict) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO inference_inputs (request_id, created_at, "Student_ID", raw_payload)
            VALUES (:rid, now(), :sid, CAST(:payload AS jsonb))
        """), {"rid": request_id, "sid": student_id,
               "payload": json.dumps(payload, default=str)})

def load_features() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM stress_data_processed;", engine)


def save_predictions(df: pd.DataFrame) -> None:
    df.to_sql("predictions", engine, if_exists="append", index=False,
              chunksize=1000, method="multi")


def save_metrics(row: dict) -> None:
    pd.DataFrame([row]).to_sql("model_metrics", engine, if_exists="append", index=False)