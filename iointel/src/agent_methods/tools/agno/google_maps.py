from datetime import datetime
from typing import List, Optional
from agno.tools.google_maps import GoogleMapTools as AgnoGoogleMapTools
from .common import make_base, wrap_tool
from pydantic import Field


class GoogleMaps(make_base(AgnoGoogleMapTools)):
    key: Optional[str] = Field(default=None, frozen=True)
    search_places_: bool = Field(default=True, frozen=True)
    get_directions_: bool = Field(default=True, frozen=True)
    validate_address_: bool = Field(default=True, frozen=True)
    geocode_address_: bool = Field(default=True, frozen=True)
    reverse_geocode_: bool = Field(default=True, frozen=True)
    get_distance_matrix_: bool = Field(default=True, frozen=True)
    get_elevation_: bool = Field(default=True, frozen=True)
    get_timezone_: bool = Field(default=True, frozen=True)

    def _get_tool(self):
        return self._tool.Inner(
            key=self.key,
            search_places=self.search_places_,
            get_directions=self.get_directions_,
            validate_address=self.validate_address_,
            geocode_address=self.geocode_address_,
            reverse_geocode=self.reverse_geocode_,
            get_distance_matrix=self.get_distance_matrix_,
            get_elevation=self.get_elevation_,
            get_timezone=self.get_timezone_,
        )

    @wrap_tool("agno__google_maps__search_places", AgnoGoogleMapTools.search_places)
    def search_places(self, query: str) -> str:
        return self._tool.search_places(query)

    @wrap_tool("agno__google_maps__get_directions", AgnoGoogleMapTools.get_directions)
    def get_directions(
        self,
        origin: str,
        destination: str,
        mode: str = "driving",
        departure_time: Optional[datetime] = None,
        avoid: Optional[List[str]] = None,
    ) -> str:
        return self._tool.get_directions(
            origin, destination, mode, departure_time, avoid
        )

    @wrap_tool(
        "agno__google_maps__validate_address", AgnoGoogleMapTools.validate_address
    )
    def validate_address(
        self,
        address: str,
        region_code: str = "US",
        locality: Optional[str] = None,
        enable_usps_cass: bool = False,
    ) -> str:
        return self._tool.validate_address(
            address, region_code, locality, enable_usps_cass
        )

    @wrap_tool("agno__google_maps__geocode_address", AgnoGoogleMapTools.geocode_address)
    def geocode_address(self, address: str, region: Optional[str] = None) -> str:
        return self._tool.geocode_address(address, region)

    @wrap_tool("agno__google_maps__reverse_geocode", AgnoGoogleMapTools.reverse_geocode)
    def reverse_geocode(
        self,
        lat: float,
        lng: float,
        result_type: Optional[List[str]] = None,
        location_type: Optional[List[str]] = None,
    ) -> str:
        return self._tool.reverse_geocode(lat, lng, result_type, location_type)

    @wrap_tool(
        "agno__google_maps__get_distance_matrix", AgnoGoogleMapTools.get_distance_matrix
    )
    def get_distance_matrix(
        self,
        origins: List[str],
        destinations: List[str],
        mode: str = "driving",
        departure_time: Optional[datetime] = None,
        avoid: Optional[List[str]] = None,
    ) -> str:
        return self._tool.get_distance_matrix(
            origins, destinations, departure_time, avoid
        )

    @wrap_tool("agno__google_maps__get_elevation", AgnoGoogleMapTools.get_elevation)
    def get_elevation(self, lat: float, lng: float) -> str:
        return self._tool.get_elevation(lat, lng)

    @wrap_tool("agno__google_maps__get_timezone", AgnoGoogleMapTools.get_timezone)
    def get_timezone(
        self, lat: float, lng: float, timestamp: Optional[datetime] = None
    ) -> str:
        return self._tool.get_timezone(lat, lng, timestamp)
