from fastapi import FastAPI
from api.app import router


app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts Remaining Useful Life (RUL) of aircraft engines using NASA CMAPSS dataset",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}