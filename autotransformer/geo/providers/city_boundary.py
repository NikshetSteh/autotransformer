import geopandas as gpd
import osmnx as ox
import pandas as pd

from .base import GeoProvider


class CityBoundaryProvider(GeoProvider):
    name = "city_boundary"

    def apply(
        self,
        df: pd.DataFrame,
        place: str = "Moscow, Russia",
        fallback_radius_m: float = 15000.0,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        **kwargs,
    ) -> pd.DataFrame:
        gdf = self._prepare_gdf(df, lon_col, lat_col)
        gdf_m = gdf.to_crs("EPSG:32637")

        try:
            boundary_gdf = ox.geocode_to_gdf(place).to_crs("EPSG:32637")
            boundary = boundary_gdf.geometry.unary_union
            df["is_outside_city"] = ~gdf_m.geometry.apply(
                lambda p: boundary.contains(p)
            )
        except Exception as e:
            print(f"⚠️ CityBoundaryProvider: {e}")
            # Fallback: use distance to city center (assumes df already has dist_to_center or similar)
            center_key = None
            # Try to find any "dist_to_..." column as proxy
            dist_cols = [col for col in df.columns if col.startswith("dist_to_")]
            if dist_cols:
                center_key = dist_cols[0]
                df["is_outside_city"] = df[center_key] > fallback_radius_m
            else:
                # If no distance column, compute to default center of place (approximate)
                try:
                    center_point = ox.geocode(place)
                    center_gdf = gpd.GeoDataFrame(
                        geometry=[
                            gpd.points_from_xy([center_point[1]], [center_point[0]])[0]
                        ],
                        crs="EPSG:4326",
                    ).to_crs("EPSG:32637")
                    dist_to_center = gdf_m.distance(center_gdf.geometry.iloc[0])
                    df["is_outside_city"] = dist_to_center > fallback_radius_m
                except:
                    df["is_outside_city"] = True

        return df
