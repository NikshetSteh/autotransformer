from typing import Optional

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base import GeoProvider


class MetroProvider(GeoProvider):
    name = "metro"

    def apply(
        self,
        df: pd.DataFrame,
        place: str = "Moscow, Russia",
        radii: Optional[list] = None,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        **kwargs,
    ) -> pd.DataFrame:
        print("A")
        if radii is None:
            radii = [500, 1000]

        gdf = self._prepare_gdf(df, lon_col, lat_col)
        gdf_m = gdf.to_crs("EPSG:32637")
        X_points = gdf_m.geometry.apply(lambda p: [p.x, p.y]).tolist()

        # Загрузка станций метро из OSM
        try:
            print("B")
            subway = ox.features_from_place(place, tags={'station': 'subway'}).to_crs('EPSG:32637')
            print("C")
            subway = subway[~subway.geometry.isna()]  # убираем пустые геометрии
            print("D")
            subway = subway[subway.geometry.type == 'Point']  # только точки
            print("E")
        except Exception:
            subway = gpd.GeoDataFrame(geometry=[], crs='EPSG:32637')

        # Вычисление признаков, если станции найдены
        if len(subway) > 0:
            print(subway)
            X_subway = subway.geometry.apply(lambda p: [p.x, p.y]).tolist()
            nbrs_subway = NearestNeighbors(n_neighbors=1).fit(X_subway)
            df['dist_to_subway'] = nbrs_subway.kneighbors(X_points)[0].flatten()

            # Количество станций в каждом заданном радиусе
            for r in radii:
                nbrs_count = NearestNeighbors(radius=r).fit(X_subway)
                counts = nbrs_count.radius_neighbors(X_points, return_distance=False)
                df[f'count_subway_{r}m'] = [len(c) for c in counts]
        else:
            df['dist_to_subway'] = np.nan
            for r in radii:
                df[f'count_subway_{r}m'] = 0

        return df
