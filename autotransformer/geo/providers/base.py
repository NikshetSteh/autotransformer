from abc import ABC, abstractmethod

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


class GeoProvider(ABC):
    name: str = "base"

    @abstractmethod
    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    def _prepare_gdf(
        self, df: pd.DataFrame, lon_col: str, lat_col: str
    ) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
            crs="EPSG:4326",
        )
