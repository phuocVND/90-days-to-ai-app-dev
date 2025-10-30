from dataclasses import dataclass

@dataclass
class Weather:
    city: str
    temp: float
    humidity: int
    wind_speed: float
    description: str