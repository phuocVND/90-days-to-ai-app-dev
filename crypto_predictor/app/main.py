from fastapi import FastAPI
from app.controllers import predict_router


app = FastAPI(title="Crypto Price Predictor API", version="0.1.0")

# Đăng ký router
app.include_router(predict_router, prefix="/api", tags=["Prediction"])

@app.get("/")
async def root():
    return {"message": "Welcome to Crypto Predictor API"}
