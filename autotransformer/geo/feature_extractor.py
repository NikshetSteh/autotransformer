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
    Универсальный интерфейс для добавления геофичей.

    Parameters:
        df: исходный DataFrame
        providers: словарь вида {GeoProviderEnum.METRO: {'radii': [300, 500]}}
        place: общий параметр местоположения (для всех провайдеров, где применимо)
        centers: список центров [{"name": "center", "lon": 37.6, "lat": 55.7}]
        radii: например {"subway": [500, 1000]}
        lat_col, lon_col: названия столбцов с координатами
        coord_col: альтернатива — один столбец с координатами (строка или список)
    """
    df = df.copy()

    # --- Парсинг координат ---
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
        }

    # Общие параметры для всех провайдеров
    common_kwargs = {
        "place": place,
        "lon_col": "__parsed_lon__",
        "lat_col": "__parsed_lat__",
        "centers": centers,
        "radii": radii or {},
    }

    # Запуск выбранных провайдеров
    for provider_enum, provider_params in providers.items():
        print(provider_enum, provider_params)
        if isinstance(provider_enum, str):
            provider_enum = GeoProviderEnum(provider_enum)
        provider_class = get_provider(provider_enum)
        provider_instance = provider_class()
        full_kwargs = {**common_kwargs, **provider_params}
        df = provider_instance.apply(df, **full_kwargs)
        print(provider_enum, provider_params)

    # Убираем временные колонки
    df.drop(columns=["__parsed_lon__", "__parsed_lat__"], inplace=True)

    return df
