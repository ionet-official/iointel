import os
import json
from typing import Any, Dict, List

import pytest
from iointel.src.agent_methods.tools.agno.openweather import OpenWeather
from pydantic_ai.models.openai import OpenAIModel

from iointel.src.agents import Agent


@pytest.fixture(scope="module")
def ow() -> OpenWeather:  # noqa: D401 – fixture
    """Return a module‑scoped instance of the OpenWeather tool."""
    return OpenWeather()


@pytest.mark.skipif(
    not os.environ.get("OPENWEATHER_API_KEY"),
    reason="OPENWEATHER_API_KEY must be set to run this test",
)
def test_geocode_location_basic(ow: OpenWeather) -> None:
    """The tool should return latitude / longitude coordinates for a city."""
    raw = ow.geocode_location("New York")
    payload: List[Dict[str, Any]] = json.loads(raw)

    assert isinstance(payload, list)
    assert payload, "Empty geocode payload"
    location = payload[0]
    assert {"lat", "lon"} <= location.keys(), location
    assert -90 <= location["lat"] <= 90 and -180 <= location["lon"] <= 180, location


@pytest.mark.skipif(
    not os.environ.get("OPENWEATHER_API_KEY"),
    reason="OPENWEATHER_API_KEY must be set to run this test",
)
def test_get_current_weather_basic(ow: OpenWeather) -> None:
    """We should receive a weather dict with a `weather` list and `main` block."""
    raw = ow.get_current_weather("New York")
    data: Dict[str, Any] = json.loads(raw)

    assert isinstance(data, dict)
    assert "weather" in data and isinstance(data["weather"], list)
    assert "main" in data and "temp" in data["main"], data


@pytest.mark.skipif(
    not os.environ.get("OPENWEATHER_API_KEY"),
    reason="OPENWEATHER_API_KEY must be set to run this test",
)
def test_get_forecast_basic(ow: OpenWeather) -> None:
    """`get_forecast` should return a list of forecast entries (default 5 days)."""
    raw = ow.get_forecast("New York", days=3)
    data: Dict[str, Any] = json.loads(raw)

    assert isinstance(data, dict)
    assert "list" in data and len(data["list"]) > 0, data


@pytest.mark.skipif(
    not os.environ.get("OPENWEATHER_API_KEY"),
    reason="OPENWEATHER_API_KEY must be set to run this test",
)
def test_get_air_pollution_basic(ow: OpenWeather) -> None:
    """Verify we get an Air Quality Index and components breakdown."""
    raw = ow.get_air_pollution("New York")
    data: Dict[str, Any] = json.loads(raw)

    assert isinstance(data, dict)
    assert "list" in data and data["list"], data
    point = data["list"][0]
    assert "main" in point and "aqi" in point["main"]
    assert "components" in point


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENWEATHER_API_KEY"),
    reason="Both OPENAI_API_KEY and OPENWEATHER_API_KEY must be set to run this test",
)
async def test_agent_openweather_with_agent(ow: OpenWeather) -> None:
    """Full agent flow: ensure WeatherAgent can call OpenWeather via tool methods."""
    agent = Agent(
        name="WeatherAgent",
        instructions="You are a weather forecast agent that can provide weather information for locations.",
        tools=[
            ow.get_current_weather,
            ow.get_forecast,
            ow.get_air_pollution,
            ow.geocode_location,
        ],
        model=OpenAIModel(model_name="gpt-4o-mini"),
    )

    prompt = "What is the current temperature in New York in Celsius?"
    answer = (await agent.run(prompt))["result"]

    assert any(char.isdigit() for char in answer)
    assert "°" in answer or "C" in answer
