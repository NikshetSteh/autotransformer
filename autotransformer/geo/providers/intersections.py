import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base import GeoProvider


class IntersectionsProvider(GeoProvider):
    """
    Compute distance from each point to the nearest major intersection.
    """

    name = "intersections"

    def apply(
        self,
        df: pd.DataFrame,
        place: str = "Moscow, Russia",
        min_street_count: int = 4,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Args:
            df: DataFrame with coordinates.
            place: City or area name for OSMnx.
            min_street_count: Minimum number of streets to consider an intersection major.

        Returns:
            DataFrame with new column "dist_to_major_intersection" in meters.
        """
        gdf = self._prepare_gdf(df, lon_col, lat_col)
        gdf_m = gdf.to_crs("EPSG:32637")
        X_points = gdf_m.geometry.apply(lambda p: [p.x, p.y]).tolist()

        try:
            G = ox.graph_from_place(place, network_type="drive")
            nodes = ox.graph_to_gdfs(G, edges=False)  # returns nodes only
            nodes = nodes.to_crs("EPSG:32637")
            major_intersections = nodes[nodes["street_count"] >= min_street_count]
        except Exception as e:
            print(f"⚠️ IntersectionsProvider: {e}")
            major_intersections = gpd.GeoDataFrame(geometry=[], crs="EPSG:32637")

        if major_intersections.empty:
            df["dist_to_major_intersection"] = np.nan
            return df

        X_int = major_intersections.geometry.apply(lambda p: [p.x, p.y]).tolist()
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_int)
        df["dist_to_major_intersection"] = nbrs.kneighbors(X_points)[0].flatten()
        return df
