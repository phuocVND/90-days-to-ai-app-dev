from  controllers.weather_controller  import WeatherController
import sys

if __name__ == "__main__":
    print("Welcome to the Weather CLI App!")
    print("Get current weather information for any city.")
    while True:
        print("Please enter the city name below:")
        city = input("Enter city name: ")
        controller = WeatherController(city)
        controller.fetch_and_display_weather()
        print("\nDo you want to check another city? (yes/no)")
        choice = input().strip().lower()
        if choice != 'yes':
            print("Thank you for using the Weather CLI App. Goodbye!")
            break
