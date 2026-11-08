from typing import Optional

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    student_id: Optional[int] = Field(
        None, description="Ознаки беруться з stress_data_processed за цим Student_ID")
    features: Optional[dict[str, float]] = Field(
        None, description="Ознаки у форматі stress_data_processed (без Burnout_Risk_Level, Student_ID, timestamp)")

    @model_validator(mode="after")
    def exactly_one_source(self):
        if (self.student_id is None) == (self.features is None):
            raise ValueError("Передайте рівно одне: student_id або features")
        return self


class PredictResponse(BaseModel):
    request_id: str
    run_id: str
    model_name: str
    source: str
    student_id: Optional[int]
    y_pred_raw: float
    y_pred_class: int
    label: str