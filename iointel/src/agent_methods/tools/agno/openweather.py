from typing import Optional
from agno.tools.openweather import OpenWeatherTools as AgnoOpenWeather
from pydantic import Field

from .common import make_base, wrap_tool


class OpenWeather(make_base(AgnoOpenWeather)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    units: str = Field(default="metric", frozen=True)
    current_weather: bool = Field(default=True, frozen=True)
    forecast: bool = Field(default=True, frozen=True)
    air_pollution: bool = Field(default=True, frozen=True)
    geocoding: bool = Field(default=True, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            units=self.units,
            current_weather=self.current_weather,
            forecast=self.forecast,
            air_pollution=self.air_pollution,
            geocoding=self.geocoding,
        )

    @wrap_tool("geocode_location", AgnoOpenWeather.geocode_location)
    def geocode_location(self, location: str, limit: int = 1) -> str:
        return self._tool.geocode_location(location, limit)

    @wrap_tool("get_current_weather", AgnoOpenWeather.get_current_weather)
    def get_current_weather(self, location: str) -> str:
        return self._tool.get_current_weather(location)

    @wrap_tool("get_forecast", AgnoOpenWeather.get_forecast)
    def get_forecast(self, location: str, days: int = 5) -> str:
        return self._tool.get_forecast(location, days)

    @wrap_tool("get_air_pollution", AgnoOpenWeather.get_air_pollution)
    def get_air_pollution(self, location: str) -> str:
        return self._tool.get_air_pollution(location)
