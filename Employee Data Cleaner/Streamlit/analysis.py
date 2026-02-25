"""
analysis.py — Statistical analysis and descriptive computations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_stats(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Compute extended descriptive statistics for numeric columns.

    Args:
        df: Input DataFrame.
        columns: Specific columns; defaults to all numeric columns.

    Returns:
        DataFrame with statistics as rows and columns as columns.
    """
    numeric_df = df[columns] if columns else df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()

    desc = numeric_df.describe().T
    desc["skewness"] = numeric_df.skew()
    desc["kurtosis"] = numeric_df.kurtosis()
    desc["cv%"] = ((numeric_df.std() / numeric_df.mean()) * 100).round(2)
    desc["missing%"] = ((df.shape[0] - numeric_df.count()) / df.shape[0] * 100).round(2)
    return desc.round(4)


def correlation_matrix(df: pd.DataFrame, columns: Optional[list[str]] = None, method: str = "pearson") -> pd.DataFrame:
    """
    Compute a correlation matrix for numeric columns.

    Args:
        df: Input DataFrame.
        columns: Columns to correlate; defaults to all numeric.
        method: 'pearson', 'spearman', or 'kendall'.

    Returns:
        Correlation matrix as a DataFrame.
    """
    numeric = df[columns] if columns else df.select_dtypes(include="number")
    return numeric.corr(method=method).round(4)  # type: ignore[arg-type]


def top_correlations(corr_matrix: pd.DataFrame, target: str, top_n: int = 5) -> pd.DataFrame:
    """
    Return top N columns most correlated with the target column.

    Args:
        corr_matrix: Symmetric correlation matrix.
        target: Target column name.
        top_n: How many columns to return.

    Returns:
        DataFrame with 'Feature' and 'Correlation' columns sorted by absolute value.
    """
    if target not in corr_matrix.columns:
        return pd.DataFrame()
    series = corr_matrix[target].drop(labels=[target]).dropna()
    series_sorted = series.reindex(series.abs().sort_values(ascending=False).index)
    result = series_sorted.head(top_n).reset_index()
    result.columns = pd.Index(["Feature", "Correlation"])
    return result


def group_statistics(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """
    Compute grouped statistics for a value column segmented by a categorical column.

    Args:
        df: Input DataFrame.
        group_col: Categorical column to group by.
        value_col: Numeric column to aggregate.

    Returns:
        DataFrame with group statistics.
    """
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    grp = (
        df.groupby(group_col)[value_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .rename(columns={
            "count": "Count", "mean": "Mean", "median": "Median",
            "std": "Std Dev", "min": "Min", "max": "Max",
        })
        .reset_index()
        .rename(columns={group_col: "Group"})
    )
    grp = grp.round(2)
    return grp.sort_values("Mean", ascending=False).reset_index(drop=True)


def outlier_summary(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Compute the number of outliers in each numeric column using IQR and Z-score methods.

    Args:
        df: Input DataFrame.
        columns: Columns to analyze; defaults to all numeric.

    Returns:
        DataFrame with outlier counts per column.
    """
    numeric = df[columns] if columns else df.select_dtypes(include="number")
    rows = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) == 0:
            continue
        # IQR
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        iqr_out = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        # Z-score
        z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
        z_out = int((z_scores > 3).sum())
        rows.append({
            "Column": col,
            "IQR Outliers": iqr_out,
            "Z-Score Outliers (|z|>3)": z_out,
            "Total Values": len(series),
            "IQR Outlier %": round(iqr_out / len(series) * 100, 2),
        })
    return pd.DataFrame(rows)


def value_counts_table(df: pd.DataFrame, column: str, normalize: bool = True) -> pd.DataFrame:
    """
    Return a value counts table for a categorical column.

    Args:
        df: Input DataFrame.
        column: Column to tabulate.
        normalize: Include percentage column if True.

    Returns:
        DataFrame with 'Value', 'Count', and optionally 'Percentage' columns.
    """
    if column not in df.columns:
        return pd.DataFrame()
    vc = df[column].value_counts(dropna=False).reset_index()
    vc.columns = pd.Index(["Value", "Count"])
    if normalize:
        vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)
    return vc


def compute_analysis_report(df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
    """
    Orchestrate a full analysis pass and return a structured report dictionary.

    Args:
        df: Cleaned DataFrame.
        target_col: Optional target column for correlation ranking.

    Returns:
        Dictionary with keys: 'descriptive', 'correlation', 'outliers', 'top_corr'.
    """
    report: dict = {}
    report["descriptive"] = descriptive_stats(df)
    report["correlation"] = correlation_matrix(df)
    report["outliers"] = outlier_summary(df)
    if target_col and target_col in report["correlation"].columns:
        report["top_corr"] = top_correlations(report["correlation"], target_col)
    else:
        report["top_corr"] = pd.DataFrame()
    return report