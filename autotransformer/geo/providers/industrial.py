import geopandas as gpd
import osmnx as ox
import pandas as pd

from .base import GeoProvider


class IndustrialProvider(GeoProvider):
    name = "industrial"

    def apply(
        self,
        df: pd.DataFrame,
        place: str = "Moscow, Russia",
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        **kwargs,
    ) -> pd.DataFrame:
        gdf = self._prepare_gdf(df, lon_col, lat_col)
        gdf_m = gdf.to_crs("EPSG:32637")

        try:
            industrial = ox.features_from_place(
                place, tags={"landuse": "industrial"}
            ).to_crs("EPSG:32637")
            industrial = industrial[~industrial.geometry.isna()]
        except Exception as e:
            print(f"⚠️ IndustrialProvider: {e}")
            industrial = gpd.GeoDataFrame(geometry=[], crs="EPSG:32637")

        if len(industrial) == 0:
            df["is_industrial"] = False
            return df

        union_geom = industrial.unary_union
        df["is_industrial"] = gdf_m.geometry.apply(lambda p: union_geom.contains(p))

        return df
