from models.weather_model import WeatherModel
from views.weather_view import WeatherView


class WeatherController:
    def __init__(self, city: str):
        self.city = city

    def fetch_and_display_weather(self):
        model = WeatherModel(self.city)
        view = WeatherView()
        weather_data = model.get_weather_data()
        if weather_data:
            view.display_weather(weather_data)
