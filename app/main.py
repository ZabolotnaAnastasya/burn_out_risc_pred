from fastapi import FastAPI
from app.endpoints import training
from app.endpoints import inference


app = FastAPI(title="Burnout Risk API")
app.include_router(training.router)
app.include_router(inference.router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Burnout Risk API. Documentation: /docs"}
