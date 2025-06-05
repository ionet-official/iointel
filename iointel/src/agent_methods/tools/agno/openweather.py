from agno.tools.openweather import OpenWeatherTools as AgnoOpenWeather

from .common import make_base, wrap_tool


class OpenWeather(make_base(AgnoOpenWeather)):
    def _get_tool(self):
        return self.Inner()

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
