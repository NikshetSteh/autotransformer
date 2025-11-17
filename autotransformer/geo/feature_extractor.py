from enum import Enum
from typing import Any, Optional, Union

import pandas as pd

from .providers import GeoProviderEnum, get_provider
from .utils import parse_coordinates


def add_geo_features(
    df: pd.DataFrame,
    providers: Optional[dict[Union[GeoProviderEnum, str], dict[str, Any]]] = None,
    place: str = "Moscow, Russia",
    centers: Optional[list[dict[str, Any]]] = None,
    radii: Optional[dict[str, list[int]]] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    coord_col: Optional[str] = None,
    coord_separator: str = ";",
) -> pd.DataFrame:
    """
    Add geographic features to a DataFrame using configured providers.

    Args:
        df: Input DataFrame with geographic coordinates.
        providers: Dictionary mapping GeoProviderEnum (or string) to provider-specific kwargs.
        place: Default location for providers that require a city/place.
        centers: List of dicts [{"name": str, "lon": float, "lat": float}] for distance calculation.
        radii: Optional dictionary of radius lists for providers (e.g., {"subway": [300, 500]}).
        lat_col: Name of latitude column in df.
        lon_col: Name of longitude column in df.
        coord_col: Name of single column with coordinates (lat/lon), separated by coord_separator.
        coord_separator: Separator used in coord_col if provided.

    Returns:
        DataFrame with added columns for each requested geo feature.

    Example:
        df = pd.DataFrame([{"lat": 55.7558, "lon": 37.6173}])
        df = add_geo_features(df, providers={GeoProviderEnum.METRO: {}})
    """
    df = df.copy()

    lon, lat = parse_coordinates(
        df,
        lat_col=lat_col,
        lon_col=lon_col,
        coord_col=coord_col,
        separator=coord_separator,
    )
    df["__parsed_lon__"] = lon
    df["__parsed_lat__"] = lat

    if providers is None:
        providers = {
            GeoProviderEnum.CENTERS: {},
            GeoProviderEnum.METRO: {},
            GeoProviderEnum.TRANSPORT: {},
            GeoProviderEnum.CITY_BOUNDARY: {},
            GeoProviderEnum.DISTRICTS: {},
            GeoProviderEnum.INTERSECTIONS: {},
            GeoProviderEnum.INDUSTRIAL: {},
        }

    common_kwargs = {
        "place": place,
        "lon_col": "__parsed_lon__",
        "lat_col": "__parsed_lat__",
        "centers": centers,
        "radii": radii or {},
    }

    for provider_enum, provider_params in providers.items():
        if isinstance(provider_enum, str):
            provider_enum = GeoProviderEnum(provider_enum)
        provider_class = get_provider(provider_enum)
        provider_instance = provider_class()
        full_kwargs = {**common_kwargs, **provider_params}
        df = provider_instance.apply(df, **full_kwargs)

    df.drop(columns=["__parsed_lon__", "__parsed_lat__"], inplace=True)
    return df
