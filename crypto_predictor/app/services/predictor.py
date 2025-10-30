from datetime import datetime, timedelta
from app.schemas.price_schema import PriceRequest, PriceResponse
from app.models.price_model import PriceModel

class Predictor:
    def __init__(self):
        self.model = PriceModel()

    def predict(self, data: PriceRequest) -> PriceResponse:
        features = [data.open, data.high, data.low, data.close, data.volume]
        predicted_close = self.model.predict(features)

        # Tạm ước lượng high/low từ close
        predicted_high = predicted_close * 1.02
        predicted_low = predicted_close * 0.98

        # return PriceResponse(
        #     symbol=data.symbol,
        #     predicted_close=predicted_close,
        #     predicted_high=predicted_high,
        #     predicted_low=predicted_low,
        #     timestamp=datetime.utcnow(),
        #     current_close=data.close
        # )
        open_time = data.close_time
        timeframe_seconds = 15 * 60  # 15 phút
        close_time = open_time + timedelta(seconds=timeframe_seconds)
        return PriceResponse(
            symbol=data.symbol,
            predicted_close=data.close,
            predicted_high=data.high,
            predicted_low=data.low,
            timestamp=datetime.utcnow(),
            current_close=data.close,
            open_time=open_time,
            close_time=close_time
        )
