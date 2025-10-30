import requests
import os
from dotenv import load_dotenv
from models.weather_data import Weather

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

class WeatherModel:
    def __init__(self, city: str):
        self.city = city
        
    def get_weather_data(self) -> dict:
        params = {
            "q": self.city,
            "appid": API_KEY,
            "units": "metric",
            "lang": "en"
        }
        try:
            response = requests.get(BASE_URL, params=params)
            if response.status_code == 200:
                data = response.json()

                weather = Weather(
                    city=data["name"],
                    temp=data["main"]["temp"],
                    humidity=data["main"]["humidity"],
                    wind_speed=data["wind"]["speed"],
                    description=data["weather"][0]["description"].capitalize()
                )
                return weather
            
            elif response.status_code == 404:
                weather = Weather(
                    city="City not found",
                    temp=None,
                    humidity=None,
                    wind_speed=None,
                    description=None
                )
                return weather
            else:
                return {"error": "Fault when calling API"}
        except requests.RequestException as e:
            return {"error": str(e)}