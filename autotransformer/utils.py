from typing import List, Set, Tuple

import pandas as pd


def detect_column_types(
    df: pd.DataFrame,
    drop_cols: Set[str],
    target_cols: Set[str],
    forced_num_cols: Set[str],
    forced_cat_cols: Set[str],
    forced_text_cols: Set[str],
    custom_cols: Set[str],
    per_column_transformers: Set[str],
    text_heuristic_min_unique_ratio: float,
    cat_heuristic_max_unique_ratio: float,
    cat_heuristic_max_unique_count: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Detect numerical, categorical, and text columns using configurable heuristics.

    Args:
        df: Input DataFrame to analyze.
        drop_cols: Columns to exclude from consideration.
        target_cols: Columns treated as targets (excluded from feature detection).
        forced_num_cols: Columns explicitly treated as numerical.
        forced_cat_cols: Columns explicitly treated as categorical.
        forced_text_cols: Columns explicitly treated as text.
        custom_cols: Columns with user-defined custom transformers.
        per_column_transformers: Additional per-column transformers (also custom).
        text_heuristic_min_unique_ratio: Minimum unique ratio to classify as text.
            If unique_ratio >= this value → text.
        cat_heuristic_max_unique_ratio: Maximum unique ratio to classify as categorical
            for numeric columns. If unique_ratio <= this value → categorical.
        cat_heuristic_max_unique_count: Maximum unique count to classify as categorical
            regardless of ratio.

    Returns:
        A tuple of three lists:
            - Detected numerical column names
            - Detected categorical column names
            - Detected text column names
    """
    exclude = drop_cols | target_cols
    cols_to_consider = [col for col in df.columns if col not in exclude]

    num_cols, cat_cols, text_cols = [], [], []

    all_custom = custom_cols | per_column_transformers

    for col in cols_to_consider:
        if col in forced_num_cols:
            num_cols.append(col)
            continue
        if col in forced_cat_cols:
            cat_cols.append(col)
            continue
        if col in forced_text_cols:
            text_cols.append(col)
            continue
        if col in all_custom:
            continue

        series = df[col]
        n_total = len(series)
        if n_total == 0:
            continue
        n_unique = series.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= 1:
                num_cols.append(col)
            elif (n_unique / n_total <= cat_heuristic_max_unique_ratio) or (
                n_unique <= cat_heuristic_max_unique_count
            ):
                cat_cols.append(col)
            else:
                num_cols.append(col)
        else:
            if (n_unique / n_total) >= text_heuristic_min_unique_ratio:
                text_cols.append(col)
            else:
                cat_cols.append(col)

    return num_cols, cat_cols, text_cols
