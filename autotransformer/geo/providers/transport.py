import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base import GeoProvider


class TransportProvider(GeoProvider):
    name = "transport"

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
        X_points = gdf_m.geometry.apply(lambda p: [p.x, p.y]).tolist()

        try:
            aerodromes = ox.features_from_place(place, tags={"aeroway": "aerodrome"})
            railway_stations = ox.features_from_place(
                place, tags={"railway": "station"}
            )
            ferry_terminals = ox.features_from_place(
                place, tags={"amenity": "ferry_terminal"}
            )
            transport = pd.concat(
                [aerodromes, railway_stations, ferry_terminals], ignore_index=True
            ).to_crs("EPSG:32637")
            transport = transport[~transport.geometry.isna()]
            transport["geometry"] = transport.geometry.apply(
                lambda g: (
                    g.centroid if g.geom_type in ["Polygon", "MultiPolygon"] else g
                )
            )
            transport = transport[transport.geometry.type == "Point"]
        except Exception as e:
            print(f"⚠️ TransportProvider: {e}")
            transport = gpd.GeoDataFrame(geometry=[], crs="EPSG:32637")

        if len(transport) == 0:
            df["dist_to_transport_hub"] = np.nan
            return df

        X_trans = transport.geometry.apply(lambda p: [p.x, p.y]).tolist()
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_trans)
        df["dist_to_transport_hub"] = nbrs.kneighbors(X_points)[0].flatten()

        return df
