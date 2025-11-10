import geopandas as gpd
import osmnx as ox
import pandas as pd

from .base import GeoProvider


class DistrictsProvider(GeoProvider):
    name = "districts"

    def apply(
        self,
        df: pd.DataFrame,
        place: str = "Moscow, Russia",
        admin_level: str = "8",
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        **kwargs,
    ) -> pd.DataFrame:
        gdf = self._prepare_gdf(df, lon_col, lat_col)

        try:
            districts = ox.geometries_from_place(
                place, tags={"admin_level": admin_level}
            )
            districts = districts[["name", "geometry"]].reset_index(drop=True)
            districts = districts.set_crs("EPSG:4326")
        except Exception as e:
            print(f"⚠️ DistrictsProvider: {e}")
            df["district"] = "Unknown"
            return df

        # Spatial join
        gdf_with_district = gpd.sjoin(gdf, districts, how="left", predicate="within")
        df["district"] = gdf_with_district["name"].fillna(
            "Outside " + place.split(",")[0].strip()
        )

        return df
