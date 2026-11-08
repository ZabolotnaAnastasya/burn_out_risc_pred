import pandas as pd
from app.database.database import engine


def load_features() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM stress_data_processed;", engine)


def save_predictions(df: pd.DataFrame) -> None:
    df.to_sql("predictions", engine, if_exists="append", index=False,
              chunksize=1000, method="multi")


def save_metrics(row: dict) -> None:
    pd.DataFrame([row]).to_sql("model_metrics", engine, if_exists="append", index=False)