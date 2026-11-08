import datetime as dt
import uuid
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.database import crud
from app.schemas import PredictRequest, PredictResponse
from app.services.model import latest_model_path, load_meta, load_model
from app.services.utils import build_model_input, to_class

router = APIRouter()
LABELS = {0: "Low", 1: "Medium", 2: "High"}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1. Модель з диска
    try:
        model_path = latest_model_path()
        model = load_model(model_path)
        meta = load_meta(model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # 2. Ознаки: з БД за student_id або з JSON
    if req.student_id is not None:
        raw = crud.load_student_features(req.student_id)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"Student_ID {req.student_id} не знайдено")
        student_id: Optional[int] = req.student_id
    else:
        raw = req.features
        student_id = None

    # 3. Підготовка ознак у тому самому порядку, що й під час тренування
    try:
        X = build_model_input(raw, meta["features"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 4. Передбачення (та сама постобробка, що й у train)
    raw_pred = float(np.clip(model.predict(X)[0], 0, 2))
    y_cls = int(to_class(np.array([raw_pred]))[0])

    # 5. Запис вхідних даних, далі передбачення (inference_inputs має бути раніше через FK)
    request_id = uuid.uuid4().hex
    created_at = dt.datetime.now()
    crud.save_inference_input(request_id, student_id, req.model_dump())
    crud.save_predictions(pd.DataFrame([{
        "run_id": meta["run_id"],
        "created_at": created_at,
        "model_name": meta["model_name"],
        "source": "inference",
        "student_id": student_id,
        "y_true": None,
        "y_pred_raw": raw_pred,
        "y_pred_class": y_cls,
        "request_id": request_id,
    }]))

    return PredictResponse(
        request_id=request_id,
        run_id=meta["run_id"],
        model_name=meta["model_name"],
        source="inference",
        student_id=student_id,
        y_pred_raw=raw_pred,
        y_pred_class=y_cls,
        label=LABELS[y_cls],
    )