from models.weather_data import Weather

class WeatherView:
    def display_weather(self, weather: Weather):
        print(f"Weather in {weather.city}:")
        print(f"Temperature: {weather.temp}°C")
        print(f"Condition: {weather.description}")
        print(f"Humidity: {weather.humidity}%")
        print(f"Wind Speed: {weather.wind_speed} m/s")
        print("-*---------------------------*-")
    def display_error(self, message):
        print(f"Error: {message}")