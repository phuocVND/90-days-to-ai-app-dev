from fastapi import APIRouter
from rag_backend.app.schemas.schema import PriceRequest, PriceResponse
from app.services.predictor import Predictor
from rag_backend.app.services.service import DataFetcher

router = APIRouter()
predictor = Predictor()
fetcher = DataFetcher()

@router.post("/predict", response_model=PriceResponse)
async def predict_price(request: PriceRequest):
    prediction = predictor.predict(request)
    return prediction


@router.get("/predict_live", response_model=PriceResponse)
async def predict_live(symbol: str = "BTCUSDT", timeframe: str = "15m"):

    price_request = fetcher.fetch_as_price_request(symbol, timeframe)
    prediction = predictor.predict(price_request)
    return prediction
