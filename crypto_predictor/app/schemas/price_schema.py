from pydantic import BaseModel
from datetime import datetime

class PriceRequest(BaseModel):
    symbol: str  # ví dụ: BTCUSDT
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_time: datetime
    close_time: datetime

class PriceResponse(BaseModel):
    symbol: str
    predicted_close: float
    predicted_high: float
    predicted_low: float
    timestamp: datetime
    current_close: float
    open_time: datetime
    close_time: datetime
