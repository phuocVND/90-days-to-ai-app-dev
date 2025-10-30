from pydantic import BaseModel

class Weather(BaseModel):
    city: str
    temp: float | None
    humidity: int | None
    wind_speed: float | None
    description: str | None
    error: str | None = None
