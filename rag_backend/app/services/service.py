from rag_backend.app.schemas.schema import PriceRequest
import requests
from datetime import datetime

class DataFetcher:
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def fetch_last_closed_candle(self, symbol: str = "BTCUSDT", timeframe: str = "15m") -> dict:
        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": 2
        }
        response = requests.get(self.BASE_URL, params=params)
        data = response.json()
        last_closed = data[-2]

        return {
            "symbol": symbol.upper(),
            "open": float(last_closed[1]),
            "high": float(last_closed[2]),
            "low": float(last_closed[3]),
            "close": float(last_closed[4]),
            "volume": float(last_closed[5]),
            "open_time": int(last_closed[0]),
            "close_time": int(last_closed[6])
        }

    def fetch_as_price_request(self, symbol: str = "BTCUSDT", timeframe: str = "15m") -> PriceRequest:
        candle = self.fetch_last_closed_candle(symbol, timeframe)

        open_time = datetime.utcfromtimestamp(candle["open_time"] / 1000)
        close_time = datetime.utcfromtimestamp(candle["close_time"] / 1000)

        return PriceRequest(
            symbol=candle["symbol"],
            open=candle["open"],
            high=candle["high"],
            low=candle["low"],
            close=candle["close"],
            volume=candle["volume"],
            open_time=open_time,
            close_time=close_time
        )
