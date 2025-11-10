from ast import literal_eval
from typing import Optional

import pandas as pd


def parse_coordinates(
    df: pd.DataFrame,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    coord_col: Optional[str] = None,
    separator: str = ";",
) -> tuple[pd.Series, pd.Series]:
    if coord_col is not None:

        def _parse_single(val):
            if pd.isna(val):
                return (None, None)
            if isinstance(val, str):
                if separator in val:
                    parts = val.split(separator)
                    return float(parts[1].strip()), float(parts[0].strip())  # lon, lat
                else:
                    try:
                        coords = literal_eval(val)
                        return float(coords[0]), float(coords[1])  # как в вашем примере
                    except:
                        return (None, None)
            elif isinstance(val, (list, tuple)):
                return float(val[0]), float(val[1])
            else:
                return (None, None)

        parsed = df[coord_col].apply(_parse_single)
        lon = parsed.apply(lambda x: x[0])
        lat = parsed.apply(lambda x: x[1])
    else:
        lon = df[lon_col].astype(float)
        lat = df[lat_col].astype(float)

    return lon, lat
