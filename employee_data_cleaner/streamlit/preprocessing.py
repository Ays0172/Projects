"""
preprocessing.py — All data cleaning and transformation operations.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd


MissingStrategy = Literal["drop_rows", "mean", "median", "mode", "custom", "ffill", "bfill"]
OutlierMethod = Literal["zscore", "iqr", "none"]


def detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric column names."""
    return df.select_dtypes(include="number").columns.tolist()


def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return list of categorical/object column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column missing value report.

    Returns:
        DataFrame with columns: Column, Type, Missing Count, Missing %, Sample Values.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    sample_vals = {
        col: ", ".join(str(v) for v in df[col].dropna().head(3).tolist())
        for col in df.columns
    }
    report = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Missing Count": missing.values,
        "Missing %": pct.values,
        "Non-Null Sample": [sample_vals[c] for c in df.columns],
    })
    return report.sort_values("Missing Count", ascending=False).reset_index(drop=True)


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Force specified columns to numeric dtype using pd.to_numeric with coercion.

    Args:
        df: Input DataFrame.
        columns: Column names to convert.

    Returns:
        Tuple of (modified DataFrame, list of successfully converted columns).
    """
    df = df.copy()
    converted: list[str] = []
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted.append(col)
    return df, converted


def fill_missing_values(
    df: pd.DataFrame,
    strategy: MissingStrategy,
    columns: Optional[list[str]] = None,
    custom_value: Union[str, float, None] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Fill missing values in specified columns using the chosen strategy.

    Args:
        df: Input DataFrame.
        strategy: How to fill — 'drop_rows', 'mean', 'median', 'mode', 'custom', 'ffill', 'bfill'.
        columns: Columns to operate on; defaults to all columns with nulls.
        custom_value: Used only when strategy == 'custom'.

    Returns:
        Tuple of (cleaned DataFrame, count of values filled).
    """
    df = df.copy()
    before = int(df.isnull().sum().sum())

    if columns is None:
        columns = [c for c in df.columns if df[c].isnull().any()]

    if strategy == "drop_rows":
        df = df.dropna(subset=columns)
    else:
        for col in columns:
            if col not in df.columns or df[col].isnull().sum() == 0:
                continue
            if strategy == "mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
            elif strategy == "custom" and custom_value is not None:
                fill = custom_value
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        fill = float(custom_value)  # type: ignore[arg-type]
                    except (ValueError, TypeError):
                        pass
                df[col] = df[col].fillna(fill)
            elif strategy == "ffill":
                df[col] = df[col].ffill()
            elif strategy == "bfill":
                df[col] = df[col].bfill()

    after = int(df.isnull().sum().sum())
    return df, before - after


def remove_duplicates(df: pd.DataFrame, subset: Optional[list[str]] = None) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from DataFrame.

    Args:
        df: Input DataFrame.
        subset: Columns to consider; None means all columns.

    Returns:
        Tuple of (deduplicated DataFrame, count removed).
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset or None).reset_index(drop=True)
    return df, before - len(df)


def fix_negative_values(df: pd.DataFrame, column: str, strategy: Literal["mean", "abs", "drop"] = "mean") -> tuple[pd.DataFrame, int]:
    """
    Fix negative values in a numeric column.

    Args:
        df: Input DataFrame.
        column: Target column name.
        strategy: 'mean' → replace with column mean; 'abs' → take absolute value; 'drop' → remove rows.

    Returns:
        Tuple of (modified DataFrame, count of values fixed).
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0
    mask = df[column] < 0
    count = int(mask.sum())
    if count == 0:
        return df, 0
    if strategy == "mean":
        df.loc[mask, column] = df[column][~mask].mean()
    elif strategy == "abs":
        df.loc[mask, column] = df.loc[mask, column].abs()
    elif strategy == "drop":
        df = df[~mask].reset_index(drop=True)
    return df, count


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    method: OutlierMethod = "zscore",
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> tuple[pd.DataFrame, int, float, float]:
    """
    Remove statistical outliers from a numeric column.

    Args:
        df: Input DataFrame.
        column: Target column.
        method: 'zscore' or 'iqr'.
        zscore_threshold: Z-score cutoff (default 3.0).
        iqr_multiplier: IQR fence multiplier (default 1.5).

    Returns:
        Tuple of (cleaned DataFrame, rows_removed, lower_bound, upper_bound).
    """
    df = df.copy()
    if column not in df.columns or method == "none":
        return df, 0, float(df[column].min()), float(df[column].max())

    before = len(df)
    col_data = df[column].dropna()

    if method == "zscore":
        mu, sigma = col_data.mean(), col_data.std()
        lower = mu - zscore_threshold * sigma
        upper = mu + zscore_threshold * sigma
    else:  # iqr
        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

    df = df[(df[column].isna()) | ((df[column] >= lower) & (df[column] <= upper))].reset_index(drop=True)
    return df, before - len(df), round(lower, 2), round(upper, 2)


def cast_column_type(df: pd.DataFrame, column: str, target_type: str) -> tuple[pd.DataFrame, bool]:
    """
    Cast a column to the specified dtype.

    Args:
        df: Input DataFrame.
        column: Column to cast.
        target_type: Pandas dtype string like 'float64', 'int64', 'str', 'datetime64'.

    Returns:
        Tuple of (modified DataFrame, success_flag).
    """
    df = df.copy()
    try:
        if target_type == "datetime64":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif target_type in ("int64", "float64"):
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(target_type)
        else:
            df[column] = df[column].astype(target_type)
        return df, True
    except Exception:
        return df, False


def select_columns(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Return DataFrame with only the specified columns (in given order)."""
    valid = [c for c in keep if c in df.columns]
    return df[valid].copy()