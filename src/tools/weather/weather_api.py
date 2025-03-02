import requests
from typing import Dict, Any

class WeatherAPI:
    GEO_URL = "https://nominatim.openstreetmap.org/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    def get_coordinates(self, location: str) -> Dict[str, float]:
        """Convert location name to latitude and longitude"""
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {"q": location, "format": "json", "limit": 1}

        try:
            response = requests.get(self.GEO_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data:
                return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
            else:
                return {"error": f"Location '{location}' not found"}

        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    def execute(self, location: str, unit: str = "celsius") -> Dict[str, Any]:
        """Get weather for the specified location"""
        coords = self.get_coordinates(location)

        if "error" in coords:
            return coords

        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "current_weather": "true",
            "timezone": "auto"
        }
        try:
            response = requests.get(self.WEATHER_URL, params=params)
            response.raise_for_status()
            weather_data = response.json().get("current_weather", {})

            temperature = weather_data.get("temperature", 0)
            if unit == "fahrenheit":
                temperature = (temperature * 9/5) + 32

            return {
                "weather": "sunny",
                "temperature": round(temperature, 2),
                "location": location,
                "unit": unit,
            }
        except requests.exceptions.RequestException as e:
            return {"error": f"Weather request failed: {str(e)}"}
