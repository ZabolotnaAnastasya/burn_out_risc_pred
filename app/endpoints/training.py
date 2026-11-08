from fastapi import APIRouter, HTTPException
from app.services.model import run_training

router = APIRouter()


@router.post("/train-model")
def train_model():
    try:
        return run_training()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))