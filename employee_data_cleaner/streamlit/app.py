"""
app.py — Main Streamlit entry point. UI only — all logic delegated to modules.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── Local modules ──────────────────────────────────────────────────────────────
from data_loader import generate_sample_dataset, load_uploaded_file, dataset_summary
from preprocessing import (
    missing_value_report,
    coerce_numeric_columns,
    fill_missing_values,
    remove_duplicates,
    fix_negative_values,
    remove_outliers,
    cast_column_type,
    select_columns,
    detect_numeric_columns,
    detect_categorical_columns,
)
from analysis import (
    descriptive_stats,
    correlation_matrix,
    group_statistics,
    outlier_summary,
    value_counts_table,
    compute_analysis_report,
    top_correlations,
)
from visualization import (
    render_distribution,
    render_boxplot,
    render_correlation_heatmap,
    render_bar,
    render_scatter,
    interactive_pie,
    fig_to_bytes,
    ENGINE,
)
from utils import (
    inject_css,
    section_header,
    badge,
    status_ok,
    status_warn,
    status_info,
    divider,
    mono,
    GOLD, SURFACE, BORDER, TEXT_PRIMARY, TEXT_MUTED, SUCCESS, WARNING, ERROR,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataForge — Employee Analytics",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
def _init_state() -> None:
    defaults = {
        "raw_df": None,
        "df": None,
        "analysis_report": None,
        "selected_cols": None,
        "target_col": None,
        "step_loaded": False,
        "step_cleaned": False,
        "step_features": False,
        "step_analysed": False,
        "clean_log": [],
        "viz_engine": "interactive",
        "encoding_used": "—",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:1.2rem 0 1rem 0; border-bottom: 1px solid {BORDER}; margin-bottom:1rem;">
        <p style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{TEXT_MUTED};
                  letter-spacing:0.14em;text-transform:uppercase;margin:0;">Enterprise Analytics</p>
        <h1 style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                   color:{GOLD};margin:0.2rem 0 0 0;letter-spacing:-0.02em;">DataForge ⚗️</h1>
        <p style="font-size:0.78rem;color:{TEXT_MUTED};margin:0.3rem 0 0 0;">
            Employee Data Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline status
    st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:{TEXT_MUTED};'
                f'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Pipeline Status</p>',
                unsafe_allow_html=True)

    steps_status = [
        ("01", "Data Loading", st.session_state.step_loaded),
        ("02", "Data Cleaning", st.session_state.step_cleaned),
        ("03", "Feature Selection", st.session_state.step_features),
        ("04", "Analysis", st.session_state.step_analysed),
        ("05", "Visualization", st.session_state.step_analysed),
        ("06", "Export", st.session_state.step_analysed),
    ]

    for num, name, done in steps_status:
        dot_color = SUCCESS if done else BORDER
        text_color = TEXT_PRIMARY if done else TEXT_MUTED
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.6rem;padding:0.3rem 0;">
            <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};flex-shrink:0;"></div>
            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:{TEXT_MUTED};">{num}</span>
            <span style="font-size:0.82rem;color:{text_color};">{name}</span>
        </div>
        """, unsafe_allow_html=True)

    divider()

    # Live metrics
    if st.session_state.df is not None:
        df_live = st.session_state.df
        st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:{TEXT_MUTED};'
                    f'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Current Dataset</p>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(df_live):,}")
        c2.metric("Cols", df_live.shape[1])
        c1.metric("Missing", f"{df_live.isnull().sum().sum():,}")
        c2.metric("Dupes", f"{df_live.duplicated().sum():,}")

    divider()

    # Cleaning log
    if st.session_state.clean_log:
        st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:{TEXT_MUTED};'
                    f'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Cleaning Log</p>',
                    unsafe_allow_html=True)
        for entry in st.session_state.clean_log[-8:]:
            st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:0.71rem;'
                        f'color:{SUCCESS};margin:0.1rem 0;">✓ {entry}</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="display:flex;align-items:flex-end;justify-content:space-between;
            padding-bottom:1rem;border-bottom:1px solid {BORDER};margin-bottom:0.5rem;">
    <div>
        <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:{TEXT_MUTED};
                  letter-spacing:0.12em;text-transform:uppercase;margin:0;">
            Employee Intelligence Platform
        </p>
        <h1 style="font-family:'Syne',sans-serif;font-size:2.1rem;font-weight:800;
                   color:{TEXT_PRIMARY};margin:0.1rem 0 0 0;letter-spacing:-0.03em;">
            DataForge <span style="color:{GOLD};">⚗️</span>
        </h1>
    </div>
    <div style="text-align:right;">
        <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:{TEXT_MUTED};margin:0;">
            Guided Analytics Pipeline
        </p>
        <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:{GOLD};margin:0.2rem 0 0 0;">
            v2.0 · Production Ready
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
section_header(1, "Data Loading", "Upload a CSV/Excel file or explore with the built-in sample dataset.")

load_tab_a, load_tab_b = st.tabs(["📤  Upload Dataset", "🧪  Use Sample Dataset"])

with load_tab_a:
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your CSV or Excel file here",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    if uploaded and st.button("Load Uploaded File", key="btn_load_upload", use_container_width=True):
        with st.spinner("Reading file…"):
            try:
                raw_bytes = uploaded.read()
                df, enc = load_uploaded_file(raw_bytes, uploaded.name)
                st.session_state.raw_df = df.copy()
                st.session_state.df = df.copy()
                st.session_state.encoding_used = enc
                st.session_state.step_loaded = True
                st.session_state.step_cleaned = False
                st.session_state.step_features = False
                st.session_state.step_analysed = False
                st.session_state.clean_log = []
                st.session_state.selected_cols = None
                status_ok(f"Loaded **{uploaded.name}** ({enc}) — {len(df):,} rows × {df.shape[1]} columns")
            except Exception as exc:
                st.error(f"Could not load file: {exc}")

with load_tab_b:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{TEXT_MUTED};font-size:0.85rem;">A synthetic Indian employee dataset with 2,025 rows, '
        f'13 columns, and pre-injected data quality issues — ready to clean and analyse.</p>',
        unsafe_allow_html=True,
    )
    sample_n = st.slider("Number of sample rows", min_value=500, max_value=5000, value=2000, step=500)
    if st.button("Generate & Load Sample Dataset", key="btn_load_sample", use_container_width=True):
        with st.spinner("Generating sample dataset…"):
            df = generate_sample_dataset(n=sample_n)
            st.session_state.raw_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.encoding_used = "synthetic"
            st.session_state.step_loaded = True
            st.session_state.step_cleaned = False
            st.session_state.step_features = False
            st.session_state.step_analysed = False
            st.session_state.clean_log = []
            st.session_state.selected_cols = None
        status_ok(f"Sample dataset generated — {len(df):,} rows × {df.shape[1]} columns")

# ── Dataset Preview ────────────────────────────────────────────────────────────
if st.session_state.step_loaded and st.session_state.df is not None:
    df = st.session_state.df
    summary = dataset_summary(df)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{summary['rows']:,}")
    c2.metric("Columns", summary["columns"])
    c3.metric("Missing Values", f"{summary['missing_total']:,}")
    c4.metric("Duplicates", f"{summary['duplicates']:,}")
    c5.metric("Encoding", st.session_state.encoding_used)

    with st.expander("📋  Dataset Preview", expanded=True):
        prev_tab1, prev_tab2, prev_tab3 = st.tabs(["First 10 Rows", "Column Types", "Missing Report"])
        with prev_tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with prev_tab2:
            dtype_df = pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str).values,
                                     "Non-Null": df.count().values, "Unique": df.nunique().values})
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        with prev_tab3:
            st.dataframe(
                missing_value_report(df).style.background_gradient(subset=["Missing Count"], cmap="YlOrRd"),
                use_container_width=True, hide_index=True,
            )

divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
section_header(2, "Data Cleaning", "Apply cleaning operations interactively — each action updates the live dataset.")

if not st.session_state.step_loaded:
    st.info("Complete Step 1 first.")
else:
    df = st.session_state.df

    # ── 2A: Convert Numeric Columns ──────────────────────────────────────────
    with st.expander("🔢  Convert Columns to Numeric", expanded=False):
        st.caption("Force selected columns to numeric dtype. Non-parseable values become NaN.")
        default_numeric_targets = [c for c in ["Salary (INR)", "Performance Rating",
                                                "Experience (Years)", "Age", "Tenure (Years)"] if c in df.columns]
        convert_cols = st.multiselect(
            "Columns to convert", df.columns.tolist(), default=default_numeric_targets, key="ms_convert"
        )
        if st.button("Apply Conversion", key="btn_convert"):
            with st.spinner("Converting…"):
                df, converted = coerce_numeric_columns(df, convert_cols)
                st.session_state.df = df
                if converted:
                    st.session_state.clean_log.append(f"Numeric conversion: {', '.join(converted)}")
                    status_ok(f"Converted {len(converted)} column(s) to numeric.")

    # ── 2B: Handle Missing Values ────────────────────────────────────────────
    with st.expander("🩹  Handle Missing Values", expanded=False):
        miss_cols_with_nulls = [c for c in df.columns if df[c].isnull().any()]
        if not miss_cols_with_nulls:
            status_ok("No missing values detected.")
        else:
            st.markdown(f"**{len(miss_cols_with_nulls)} columns** have missing values.")
            miss_strategy = st.radio(
                "Fill strategy",
                ["mean", "median", "mode", "custom", "ffill", "bfill", "drop_rows"],
                horizontal=True, key="radio_miss",
            )
            miss_target_cols = st.multiselect(
                "Apply to columns (leave empty for all affected)",
                miss_cols_with_nulls, key="ms_miss_cols"
            )
            custom_val: Optional[str] = None
            if miss_strategy == "custom":
                custom_val = st.text_input("Custom fill value", key="ti_custom_fill")

            if st.button("Fill Missing Values", key="btn_fill_missing"):
                cols_arg = miss_target_cols if miss_target_cols else None
                with st.spinner("Filling…"):
                    df, filled = fill_missing_values(df, miss_strategy, cols_arg, custom_val)  # type: ignore[arg-type]
                    st.session_state.df = df
                    st.session_state.clean_log.append(f"Missing → {miss_strategy}: {filled:,} values filled")
                    status_ok(f"Filled {filled:,} values using **{miss_strategy}**.")
                    remaining = int(df.isnull().sum().sum())
                    if remaining:
                        status_warn(f"{remaining:,} values still missing.")

    # ── 2C: Remove Duplicates ────────────────────────────────────────────────
    with st.expander("🗑️  Remove Duplicate Rows", expanded=False):
        dup_count = int(df.duplicated().sum())
        st.markdown(f"Detected {badge(str(dup_count), WARNING if dup_count > 0 else SUCCESS)} duplicate rows.",
                    unsafe_allow_html=True)
        dup_subset = st.multiselect("Consider only these columns for duplicates (empty = all)",
                                    df.columns.tolist(), key="ms_dup_subset")
        if st.button("Remove Duplicates", key="btn_dedup", disabled=dup_count == 0):
            with st.spinner("Removing…"):
                df, removed = remove_duplicates(df, dup_subset if dup_subset else None)
                st.session_state.df = df
                st.session_state.clean_log.append(f"Removed {removed:,} duplicates")
                status_ok(f"Removed {removed:,} duplicate rows. {len(df):,} rows remain.")

    # ── 2D: Fix Negative Values ──────────────────────────────────────────────
    with st.expander("➖  Fix Negative Values", expanded=False):
        numeric_cols = detect_numeric_columns(df)
        neg_col = st.selectbox("Select numeric column to check", numeric_cols, key="sb_neg_col")
        if neg_col:
            neg_count = int((df[neg_col] < 0).sum())
            st.markdown(f"Found {badge(str(neg_count), WARNING if neg_count > 0 else SUCCESS)} "
                        f"negative values in `{neg_col}`.", unsafe_allow_html=True)
            neg_strategy = st.radio("Fix strategy", ["mean", "abs", "drop"],
                                    horizontal=True, key="radio_neg")
            if st.button("Fix Negatives", key="btn_fix_neg", disabled=neg_count == 0):
                with st.spinner("Fixing…"):
                    df, fixed = fix_negative_values(df, neg_col, neg_strategy)  # type: ignore[arg-type]
                    st.session_state.df = df
                    st.session_state.clean_log.append(f"Fixed {fixed:,} negative values in {neg_col}")
                    status_ok(f"Fixed {fixed:,} negative values in **{neg_col}** using **{neg_strategy}**.")

    # ── 2E: Remove Outliers ──────────────────────────────────────────────────
    with st.expander("📊  Remove Outliers", expanded=False):
        numeric_cols = detect_numeric_columns(df)
        out_col = st.selectbox("Column", numeric_cols, key="sb_out_col")
        out_method = st.radio("Detection method", ["zscore", "iqr"], horizontal=True, key="radio_out")
        if out_method == "zscore":
            z_thresh = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1, key="sl_z")
            iqr_mult = 1.5
        else:
            iqr_mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="sl_iqr")
            z_thresh = 3.0

        if out_col and st.button("Remove Outliers", key="btn_outliers"):
            with st.spinner("Removing outliers…"):
                df, removed, lo, hi = remove_outliers(df, out_col, out_method, z_thresh, iqr_mult)  # type: ignore[arg-type]
                st.session_state.df = df
                st.session_state.clean_log.append(
                    f"Outliers removed from {out_col} ({out_method}): {removed:,} rows"
                )
                status_ok(f"Removed {removed:,} outliers. Bounds: [{lo:,.2f}, {hi:,.2f}].")

    # ── 2F: Change Column Types ──────────────────────────────────────────────
    with st.expander("🔄  Change Column Data Type", expanded=False):
        type_col = st.selectbox("Column to retype", df.columns.tolist(), key="sb_type_col")
        type_target = st.selectbox("Target type", ["float64", "int64", "str", "datetime64"],
                                   key="sb_type_target")
        if st.button("Apply Type Cast", key="btn_cast"):
            with st.spinner("Casting…"):
                df, success = cast_column_type(df, type_col, type_target)
                st.session_state.df = df
                if success:
                    st.session_state.clean_log.append(f"Cast {type_col} → {type_target}")
                    status_ok(f"Column **{type_col}** cast to **{type_target}**.")
                else:
                    status_warn(f"Cast failed. Check for incompatible values in **{type_col}**.")

    # ── 2G: Drop Columns ─────────────────────────────────────────────────────
    with st.expander("🗂️  Drop Columns", expanded=False):
        drop_cols = st.multiselect("Select columns to remove", df.columns.tolist(), key="ms_drop_cols")
        if st.button("Drop Selected Columns", key="btn_drop_cols", disabled=not drop_cols):
            keep = [c for c in df.columns if c not in drop_cols]
            df = select_columns(df, keep)
            st.session_state.df = df
            st.session_state.clean_log.append(f"Dropped {len(drop_cols)} column(s): {', '.join(drop_cols)}")
            status_ok(f"Dropped {len(drop_cols)} column(s).")

    # ── Mark cleaning done ────────────────────────────────────────────────────
    if st.session_state.clean_log:
        if st.button("✅  Mark Cleaning Complete", key="btn_clean_done", type="primary"):
            st.session_state.step_cleaned = True
            status_ok("Cleaning marked complete. Proceed to Feature Selection.")

divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
section_header(3, "Feature Selection", "Choose which columns to include in the analysis and select a target variable.")

if not st.session_state.step_loaded:
    st.info("Complete Step 1 first.")
else:
    df = st.session_state.df
    numeric_cols = detect_numeric_columns(df)
    cat_cols = detect_categorical_columns(df)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(f'<p style="font-size:0.82rem;color:{TEXT_MUTED};">Select columns to include in analysis:</p>',
                    unsafe_allow_html=True)
        all_cols = df.columns.tolist()
        default_sel = st.session_state.selected_cols if st.session_state.selected_cols else all_cols
        default_sel = [c for c in default_sel if c in all_cols]
        selected_cols = st.multiselect("Active Columns", all_cols, default=default_sel, key="ms_feat_cols")

        target_col = st.selectbox(
            "Target Variable (for correlation ranking)",
            ["— None —"] + numeric_cols,
            key="sb_target",
        )
        target_col = None if target_col == "— None —" else target_col

    with col_right:
        st.markdown(f'<p style="font-size:0.82rem;color:{TEXT_MUTED};">Column type breakdown:</p>',
                    unsafe_allow_html=True)
        selected_numeric = [c for c in selected_cols if c in numeric_cols]
        selected_cat = [c for c in selected_cols if c in cat_cols]
        m1, m2, m3 = st.columns(3)
        m1.metric("Selected", len(selected_cols))
        m2.metric("Numeric", len(selected_numeric))
        m3.metric("Categorical", len(selected_cat))

        if selected_numeric and target_col and target_col in df.columns:
            corr = df[selected_numeric].corr()
            if target_col in corr.columns:
                top = top_correlations(corr, target_col, top_n=4)
                if not top.empty:
                    st.markdown(f'<p style="font-size:0.78rem;color:{TEXT_MUTED};margin-top:0.8rem;">Top correlates with {target_col}:</p>',
                                unsafe_allow_html=True)
                    for _, row in top.iterrows():
                        color = SUCCESS if row["Correlation"] > 0 else ERROR
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;font-family:\'DM Mono\','
                            f'monospace;font-size:0.75rem;padding:0.15rem 0;">'
                            f'<span style="color:{TEXT_MUTED};">{row["Feature"]}</span>'
                            f'<span style="color:{color};">{row["Correlation"]:+.3f}</span></div>',
                            unsafe_allow_html=True,
                        )

    if st.button("Confirm Feature Selection", key="btn_feat_confirm", type="primary"):
        if not selected_cols:
            st.error("Select at least one column.")
        else:
            st.session_state.selected_cols = selected_cols
            st.session_state.target_col = target_col
            st.session_state.step_features = True
            status_ok(f"{len(selected_cols)} features confirmed. Target: {target_col or 'None'}.")

divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section_header(4, "Statistical Analysis", "Run a full analysis pass on the cleaned, selected dataset.")

if not st.session_state.step_features and not st.session_state.step_loaded:
    st.info("Complete Feature Selection first.")
else:
    if st.button("▶️  Run Analysis", key="btn_run_analysis", type="primary"):
        df = st.session_state.df
        sel_cols = st.session_state.selected_cols or df.columns.tolist()
        analysis_df = df[sel_cols] if sel_cols else df
        target = st.session_state.target_col

        progress = st.progress(0, text="Computing descriptive statistics…")
        try:
            report = {}
            report["descriptive"] = descriptive_stats(analysis_df)
            progress.progress(30, text="Computing correlations…")
            report["correlation"] = correlation_matrix(analysis_df)
            progress.progress(60, text="Detecting outliers…")
            report["outliers"] = outlier_summary(analysis_df)
            if target and target in report["correlation"].columns:
                report["top_corr"] = top_correlations(report["correlation"], target)
            else:
                report["top_corr"] = pd.DataFrame()
            progress.progress(90, text="Finalising…")
            st.session_state.analysis_report = report
            st.session_state.step_analysed = True
            progress.progress(100, text="Done.")
            status_ok("Analysis complete.")
        except Exception as exc:
            progress.empty()
            st.error(f"Analysis failed: {exc}")

    if st.session_state.step_analysed and st.session_state.analysis_report:
        report = st.session_state.analysis_report
        a_tab1, a_tab2, a_tab3, a_tab4 = st.tabs([
            "Descriptive Stats", "Correlation Matrix", "Outlier Summary", "Top Correlates"
        ])
        with a_tab1:
            if not report["descriptive"].empty:
                st.dataframe(report["descriptive"].style.background_gradient(subset=["mean"], cmap="YlOrRd"),
                             use_container_width=True)
        with a_tab2:
            if not report["correlation"].empty:
                st.dataframe(report["correlation"].style.background_gradient(cmap="RdYlGn"),
                             use_container_width=True)
        with a_tab3:
            if not report["outliers"].empty:
                st.dataframe(report["outliers"].style.background_gradient(
                    subset=["IQR Outliers", "Z-Score Outliers (|z|>3)"], cmap="YlOrRd"),
                    use_container_width=True, hide_index=True)
        with a_tab4:
            if not report["top_corr"].empty:
                st.dataframe(report["top_corr"], use_container_width=True, hide_index=True)
            else:
                st.info("Set a target variable in Feature Selection to see top correlates.")

        # ── Group Analysis ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.85rem;font-weight:600;">Group-Level Analysis</p>',
                    unsafe_allow_html=True)
        df_analysis = st.session_state.df
        sel = st.session_state.selected_cols or df_analysis.columns.tolist()
        num_in_sel = [c for c in sel if c in detect_numeric_columns(df_analysis)]
        cat_in_sel = [c for c in sel if c in detect_categorical_columns(df_analysis)]

        if num_in_sel and cat_in_sel:
            g1, g2 = st.columns(2)
            grp_cat = g1.selectbox("Group by", cat_in_sel, key="sb_grp_cat")
            grp_val = g2.selectbox("Metric", num_in_sel, key="sb_grp_val")
            if st.button("Compute Group Stats", key="btn_grp"):
                grp_table = group_statistics(df_analysis, grp_cat, grp_val)
                st.dataframe(grp_table.style.background_gradient(subset=["Mean"], cmap="YlOrRd"),
                             use_container_width=True, hide_index=True)
        else:
            st.caption("Need at least one categorical and one numeric column in selection for group analysis.")

divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
section_header(5, "Visualization", "Explore the data visually. Switch engines freely.")

if not st.session_state.step_loaded:
    st.info("Load a dataset first.")
else:
    df = st.session_state.df
    sel = st.session_state.selected_cols or df.columns.tolist()
    num_cols_viz = [c for c in sel if c in detect_numeric_columns(df)]
    cat_cols_viz = [c for c in sel if c in detect_categorical_columns(df)]

    engine_choice = st.radio(
        "Visualization Engine",
        ["Interactive (Plotly)", "Static (Matplotlib / Seaborn)"],
        horizontal=True, key="radio_engine",
    )
    engine: ENGINE = "interactive" if "Plotly" in engine_choice else "static"

    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
        "Distribution", "Box Plot", "Correlation", "Scatter", "Category"
    ])

    with viz_tab1:
        if num_cols_viz:
            dist_col = st.selectbox("Column", num_cols_viz, key="sb_dist_col")
            if st.button("Plot Distribution", key="btn_dist"):
                with st.spinner("Rendering…"):
                    render_distribution(df, dist_col, engine)
        else:
            st.info("No numeric columns available.")

    with viz_tab2:
        if num_cols_viz:
            box_col = st.selectbox("Value Column", num_cols_viz, key="sb_box_col")
            box_grp = st.selectbox("Group By (optional)", ["— None —"] + cat_cols_viz, key="sb_box_grp")
            box_grp = None if box_grp == "— None —" else box_grp
            if st.button("Plot Box Chart", key="btn_box"):
                with st.spinner("Rendering…"):
                    render_boxplot(df, box_col, box_grp, engine)
        else:
            st.info("No numeric columns available.")

    with viz_tab3:
        if len(num_cols_viz) >= 2:
            if st.button("Plot Correlation Heatmap", key="btn_corr_viz"):
                with st.spinner("Rendering…"):
                    corr = correlation_matrix(df, num_cols_viz)
                    render_correlation_heatmap(corr, engine)
        else:
            st.info("Need at least 2 numeric columns.")

    with viz_tab4:
        if len(num_cols_viz) >= 2:
            sc1, sc2, sc3 = st.columns(3)
            x_col = sc1.selectbox("X Axis", num_cols_viz, key="sb_sc_x")
            y_col = sc2.selectbox("Y Axis", [c for c in num_cols_viz if c != x_col], key="sb_sc_y")
            hue_col = sc3.selectbox("Color By", ["— None —"] + cat_cols_viz, key="sb_sc_hue")
            hue_col = None if hue_col == "— None —" else hue_col
            if st.button("Plot Scatter", key="btn_scatter"):
                with st.spinner("Rendering…"):
                    render_scatter(df, x_col, y_col, hue_col, engine)
        else:
            st.info("Need at least 2 numeric columns.")

    with viz_tab5:
        if cat_cols_viz:
            cat_col_viz = st.selectbox("Categorical Column", cat_cols_viz, key="sb_cat_viz")
            top_n_viz = st.slider("Top N categories", 3, 15, 8, key="sl_topn")
            if st.button("Plot Category Distribution", key="btn_cat_viz"):
                with st.spinner("Rendering…"):
                    if engine == "interactive":
                        fig = interactive_pie(df, cat_col_viz, top_n_viz)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        import matplotlib.pyplot as plt
                        vc = df[cat_col_viz].value_counts().head(top_n_viz)
                        from utils import GOLD, SURFACE, BORDER, TEXT_MUTED
                        _colors = ["#C9A84C", "#5C9BE8", "#4CAF7D", "#E85C5C",
                                   "#A85CE8", "#E8A838", "#5CE8D8", "#E85CA8"]
                        plt.rcParams.update({"figure.facecolor": "#242428", "axes.facecolor": "#1C1C1E",
                                             "text.color": "#F0EDE8", "axes.labelcolor": "#F0EDE8",
                                             "axes.edgecolor": "#3A3A42", "xtick.color": "#8A8A9A",
                                             "ytick.color": "#8A8A9A"})
                        fig, ax = plt.subplots(figsize=(9, 4))
                        ax.bar(vc.index.astype(str), vc.values,
                               color=[_colors[i % len(_colors)] for i in range(len(vc))],
                               edgecolor="#3A3A42", linewidth=0.5)
                        ax.set_title(f"Distribution — {cat_col_viz}", fontsize=13)
                        ax.set_xlabel(cat_col_viz)
                        ax.set_ylabel("Count")
                        plt.xticks(rotation=30, ha="right")
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)

            # Value counts table
            if st.checkbox("Show value counts table", key="cb_vc"):
                vc_table = value_counts_table(df, cat_col_viz)
                st.dataframe(vc_table, use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns in current selection.")

divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — RESULTS EXPORT
# ══════════════════════════════════════════════════════════════════════════════
section_header(6, "Export Results", "Download cleaned data, analysis reports, and summary.")

if not st.session_state.step_loaded:
    st.info("Load a dataset first.")
else:
    df = st.session_state.df
    sel = st.session_state.selected_cols or df.columns.tolist()
    export_df = df[[c for c in sel if c in df.columns]]

    exp_c1, exp_c2, exp_c3 = st.columns(3)

    # ── Export: Cleaned CSV ───────────────────────────────────────────────────
    with exp_c1:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
                    padding:1rem;text-align:center;margin-bottom:0.8rem;">
            <p style="font-size:1.6rem;margin:0;">📄</p>
            <p style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{TEXT_MUTED};margin:0.3rem 0;">
                Cleaned Dataset
            </p>
            <p style="font-size:0.75rem;color:{TEXT_MUTED};margin:0;">{len(export_df):,} rows · {export_df.shape[1]} cols</p>
        </div>
        """, unsafe_allow_html=True)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️  Download CSV", data=csv_bytes,
                           file_name="dataforge_cleaned.csv", mime="text/csv",
                           use_container_width=True, key="dl_csv")

    # ── Export: Analysis Report JSON ──────────────────────────────────────────
    with exp_c2:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
                    padding:1rem;text-align:center;margin-bottom:0.8rem;">
            <p style="font-size:1.6rem;margin:0;">📊</p>
            <p style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{TEXT_MUTED};margin:0.3rem 0;">
                Analysis Report
            </p>
            <p style="font-size:0.75rem;color:{TEXT_MUTED};margin:0;">JSON · Descriptive + Outliers</p>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.analysis_report:
            report = st.session_state.analysis_report
            json_out: dict = {}
            for k, v in report.items():
                if isinstance(v, pd.DataFrame) and not v.empty:
                    json_out[k] = v.reset_index().to_dict(orient="records")
            json_bytes = json.dumps(json_out, indent=2, default=str).encode("utf-8")
            st.download_button("⬇️  Download JSON", data=json_bytes,
                               file_name="dataforge_analysis.json", mime="application/json",
                               use_container_width=True, key="dl_json")
        else:
            st.button("⬇️  Download JSON", disabled=True, use_container_width=True, key="dl_json_disabled")
            st.caption("Run Analysis first.")

    # ── Export: Summary CSV ───────────────────────────────────────────────────
    with exp_c3:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
                    padding:1rem;text-align:center;margin-bottom:0.8rem;">
            <p style="font-size:1.6rem;margin:0;">📋</p>
            <p style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{TEXT_MUTED};margin:0.3rem 0;">
                Descriptive Stats
            </p>
            <p style="font-size:0.75rem;color:{TEXT_MUTED};margin:0;">CSV · All numeric metrics</p>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.analysis_report and not st.session_state.analysis_report["descriptive"].empty:
            desc_csv = st.session_state.analysis_report["descriptive"].to_csv().encode("utf-8")
            st.download_button("⬇️  Download Stats CSV", data=desc_csv,
                               file_name="dataforge_descriptive.csv", mime="text/csv",
                               use_container_width=True, key="dl_stats")
        else:
            st.button("⬇️  Download Stats CSV", disabled=True, use_container_width=True, key="dl_stats_disabled")
            st.caption("Run Analysis first.")

    # ── Cleaning Log Export ───────────────────────────────────────────────────
    if st.session_state.clean_log:
        st.markdown("<br>", unsafe_allow_html=True)
        log_content = "\n".join(f"[{i+1}] {entry}" for i, entry in enumerate(st.session_state.clean_log))
        st.download_button(
            "⬇️  Download Cleaning Log (.txt)",
            data=log_content.encode("utf-8"),
            file_name="dataforge_cleaning_log.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_log",
        )

divider()

# Footer
st.markdown(f"""
<div style="text-align:center;padding:1rem 0 0.5rem 0;">
    <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:{TEXT_MUTED};">
        DataForge Employee Analytics · Built with Streamlit · v2.0
    </p>
</div>
""", unsafe_allow_html=True)