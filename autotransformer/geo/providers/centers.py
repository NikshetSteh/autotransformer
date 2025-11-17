from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .base import GeoProvider


class CentersProvider(GeoProvider):
    """
    Calculate distances from points in the DataFrame to specified centers.
    """

    name = "centers"

    def apply(
        self,
        df: pd.DataFrame,
        centers: Optional[list[dict]] = None,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Args:
            df: DataFrame with coordinates.
            centers: List of centers [{"name": str, "lon": float, "lat": float}].
            lon_col: Name of longitude column.
            lat_col: Name of latitude column.

        Returns:
            DataFrame with new columns "dist_to_{center_name}" in meters.

        Example:
            df = pd.DataFrame([{"lon": 37.6173, "lat": 55.7558}])
            provider = CentersProvider()
            df = provider.apply(df, centers=[{"name": "kremlin", "lon": 37.6176, "lat": 55.7517}])
        """
        if not centers:
            return df

        gdf = self._prepare_gdf(df, lon_col, lat_col)
        gdf_m = gdf.to_crs("EPSG:32637")

        for center in centers:
            name = center["name"]
            point = (
                gpd.GeoDataFrame(
                    geometry=[Point(center["lon"], center["lat"])], crs="EPSG:4326"
                )
                .to_crs("EPSG:32637")
                .geometry.iloc[0]
            )
            df[f"dist_to_{name}"] = gdf_m.distance(point)

        return df
