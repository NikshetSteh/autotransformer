import geopandas as gpd
import pandas as pd
import osmnx as ox
from .base import GeoProvider

class DistrictsProvider(GeoProvider):
    """
    Assign administrative names (districts / округа / районы) to points using OSM via OSMnx.
    
    This provider fetches administrative polygons from OSM for a given place
    and assigns to each point the name of the polygon (typically a district)
    in which it is located.
    """

    name = "districts"

    def apply(
        self,
        df: pd.DataFrame,
        place: str = "Moscow, Russia",
        target_level: str = "8",  # usually districts / округа
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Assign the administrative district name to each point in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with point coordinates.
            place (str): The city or area name for which to fetch administrative boundaries.
            target_level (str): OSM admin_level to use (default "8").
            lon_col (str): Name of the longitude column in df.
            lat_col (str): Name of the latitude column in df.

        Returns:
            pd.DataFrame: Original DataFrame with an added 'district' column.
        """
        # Prepare GeoDataFrame for points
        gdf_points = self._prepare_gdf(df, lon_col, lat_col).set_crs("EPSG:4326")

        # Fetch city boundary
        try:
            city_boundary = ox.geocode_to_gdf(place).to_crs("EPSG:4326")
        except Exception:
            city_boundary = None

        try:
            # Fetch administrative polygons
            tags = {"boundary": "administrative"}
            districts = ox.features.features_from_place(place, tags=tags)
            districts = districts[districts.geom_type.isin(["Polygon", "MultiPolygon"])].set_crs("EPSG:4326")

            # Intersect with city boundary if available
            if city_boundary is not None:
                districts = gpd.overlay(districts, city_boundary, how="intersection", keep_geom_type=False)

            # Filter by target admin_level
            districts_level = districts[districts["admin_level"] == target_level]
            if districts_level.empty:
                df["district"] = "Unknown"
                return df

            # Select name column
            name_col = "name_1" if "name_1" in districts_level.columns else next(
                (c for c in districts_level.columns if c.startswith("name")), None
            )
            if name_col is None:
                df["district"] = "Unknown"
                return df

            # Spatial join points with polygons
            joined = gpd.sjoin(gdf_points, districts_level[[name_col, "geometry"]],
                               how="left", predicate="within")
            joined = joined[~joined.index.duplicated(keep="first")]

            # Assign district names to original DataFrame
            df["district"] = joined[name_col].reindex(df.index).fillna("Outside boundary")

        except Exception:
            df["district"] = "Unknown"

        return df
