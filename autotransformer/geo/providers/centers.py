from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .base import GeoProvider


class CentersProvider(GeoProvider):
    name = "centers"

    def apply(
        self,
        df: pd.DataFrame,
        centers: Optional[list] = None,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        **kwargs,
    ) -> pd.DataFrame:
        """
        centers: list of {"name": str, "lon": float, "lat": float}
        """
        if not centers:
            return df

        gdf = self._prepare_gdf(df, lon_col, lat_col)
        gdf_m = gdf.to_crs("EPSG:32637")

        for center in centers:
            print(center)
            name = center["name"]
            point = (
                gpd.GeoDataFrame(
                    geometry=[Point(center["lon"], center["lat"])], crs="EPSG:4326"
                )
                .to_crs("EPSG:32637")
                .geometry.iloc[0]
            )
            df[f"dist_to_{name}"] = gdf_m.distance(point)
            print(point)

        return df
