"""
data_loader.py — Dataset loading, encoding detection, and sample data generation.
"""

from __future__ import annotations

import io
import random
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ── Sample dataset ─────────────────────────────────────────────────────────────
_DEPARTMENTS = ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations", "Legal", "Product"]
_ROLES = {
    "Engineering": ["Software Engineer", "Senior Engineer", "Tech Lead", "Architect"],
    "Sales": ["Sales Executive", "Account Manager", "Sales Lead", "VP Sales"],
    "HR": ["HR Generalist", "Recruiter", "HR Manager", "CHRO"],
    "Marketing": ["Marketing Analyst", "Brand Manager", "CMO", "Content Lead"],
    "Finance": ["Financial Analyst", "Controller", "CFO", "Accountant"],
    "Operations": ["Operations Analyst", "Ops Manager", "COO", "Logistics Lead"],
    "Legal": ["Legal Counsel", "Compliance Officer", "CLO", "Paralegal"],
    "Product": ["Product Manager", "Senior PM", "VP Product", "Product Analyst"],
}
_CITIES = ["Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"]
_EDUCATION = ["B.Tech", "MBA", "B.Com", "M.Tech", "BCA", "MCA", "B.Sc", "M.Sc"]
_GENDER = ["Male", "Female", "Non-Binary"]


def generate_sample_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Indian employee dataset.

    Args:
        n: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with realistic employee data including injected data quality issues.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    departments = rng.choice(_DEPARTMENTS, n)
    roles = [random.choice(_ROLES[d]) for d in departments]
    ages = rng.integers(22, 62, n).astype(float)
    experience = np.clip(ages - 22 - rng.integers(0, 5, n), 0, 40).astype(float)
    education = rng.choice(_EDUCATION, n)
    gender = rng.choice(_GENDER, n, p=[0.55, 0.42, 0.03])
    city = rng.choice(_CITIES, n)
    performance = rng.uniform(1.0, 5.0, n).round(1)
    salary_base = (
        25_000
        + experience * 8_000
        + (performance - 1) * 5_000
        + rng.normal(0, 8_000, n)
    )
    salary = np.clip(salary_base, 18_000, 500_000).round(-2)
    leaves = rng.integers(0, 30, n).astype(float)
    tenure = np.clip(experience - rng.integers(0, 3, n), 0, 40).astype(float)

    df = pd.DataFrame({
        "Employee ID": [f"EMP{10000 + i}" for i in range(n)],
        "Name": [f"Employee_{i}" for i in range(n)],
        "Age": ages,
        "Gender": gender,
        "Department": departments,
        "Role": roles,
        "City": city,
        "Education": education,
        "Experience (Years)": experience,
        "Tenure (Years)": tenure,
        "Salary (INR)": salary,
        "Performance Rating": performance,
        "Leaves Taken": leaves,
    })

    # ── Inject data quality issues ────────────────────────────────────────────
    # Missing values
    for col, frac in [("Salary (INR)", 0.04), ("Performance Rating", 0.03),
                      ("Experience (Years)", 0.02), ("Age", 0.01), ("City", 0.015)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Negative salaries
    neg_idx = rng.choice(n, 30, replace=False)
    df.loc[neg_idx, "Salary (INR)"] = df.loc[neg_idx, "Salary (INR)"] * -1

    # Salary outliers
    out_idx = rng.choice(n, 15, replace=False)
    df.loc[out_idx, "Salary (INR)"] = rng.choice([5_000_000, 8_000_000, -500_000], 15)

    # Duplicates
    dup_idx = rng.choice(n, 25, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_uploaded_file(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, str]:
    """
    Load a CSV or Excel file from uploaded bytes with automatic encoding detection.

    Args:
        file_bytes: Raw bytes from the uploaded file.
        filename: Original filename used to determine format.

    Returns:
        Tuple of (DataFrame, detected_encoding).

    Raises:
        ValueError: If the file cannot be parsed.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df, "binary"

    # CSV — try encodings in order
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            return df, enc
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError("Could not decode file. Try saving as UTF-8 CSV.")


def dataset_summary(df: pd.DataFrame) -> dict:
    """
    Compute a high-level summary dictionary of a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with rows, columns, missing, duplicates, numeric cols, categorical cols.
    """
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_total": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": df.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": df.select_dtypes(include=["object", "category"]).columns.tolist(),
    }