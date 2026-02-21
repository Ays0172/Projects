"""
Employee Data Cleaner — Interactive Streamlit App
Run with: streamlit run employee_data_cleaner.py
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Data Cleaner",
    page_icon="🧹",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .main { background-color: #F9F6F0; color: #2C3E50; }
    .stApp { background-color: #F9F6F0; }

    .step-card {
        background: #EFEBE1;
        border: 1px solid #D9D3C7;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(44,62,80,0.04);
    }
    .step-badge {
        display: inline-block;
        background: #D4AF37;
        color: #FFFFFF;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        letter-spacing: 0.04em;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .metric-box {
        background: #F9F6F0;
        border: 1px solid #D9D3C7;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        min-width: 120px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(44,62,80,0.02);
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: #D4AF37;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #5A6B7E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .success-tag { color: #2E7D32; font-weight: 600; }
    .warning-tag { color: #E65100; font-weight: 600; }
    .info-tag    { color: #D4AF37; font-weight: 600; }
    div[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_html(value, label):
    return f"""
    <div class="metric-box">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def show_metrics(*pairs):
    html = '<div class="metric-row">'
    for value, label in pairs:
        html += metric_html(value, label)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def card(step_num, title):
    st.markdown(f"""
    <div class="step-card">
        <span class="step-badge">STEP {step_num}</span>
        <h3 style="margin:0 0 0.3rem 0; color:#2C3E50">{title}</h3>
    </div>""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "steps_done" not in st.session_state:
    st.session_state.steps_done = set()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style="color:#D4AF37; margin-bottom:0.2rem">🧹 Employee Data Cleaner</h1>
<p style="color:#5A6B7E; font-size:0.9rem; margin-bottom:2rem">
    Step-by-step interactive data cleaning pipeline for Indian employee datasets.
</p>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Load Data
# ══════════════════════════════════════════════════════════════════════════════
card(0, "Load Dataset")

col_upload, col_path = st.columns([1, 1])
with col_upload:
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
with col_path:
    st.markdown("<br>", unsafe_allow_html=True)
    manual_path = st.text_input(
        "Or enter local file path",
        placeholder=r"D:\Python\Numpy\indian_employee\employee_data_sample_15000.csv",
    )

if st.button("📂  Load Data", use_container_width=True):
    try:
        if uploaded:
            df = pd.read_csv(uploaded, encoding="utf-8")
        elif manual_path.strip():
            df = pd.read_csv(manual_path.strip(), encoding="utf-8")
        else:
            st.error("Please upload a file or provide a file path.")
            st.stop()

        st.session_state.df = df
        st.session_state.steps_done = set()
        st.success(f"✅  Dataset loaded — **{len(df):,} rows × {df.shape[1]} columns**")
    except Exception as e:
        st.error(f"Error loading file: {e}")

if st.session_state.df is None:
    st.info("⬆️  Load a dataset above to begin.")
    st.stop()

df: pd.DataFrame = st.session_state.df
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Preview Data
# ══════════════════════════════════════════════════════════════════════════════
card(1, "Preview Data")

if st.button("👁️  Preview Data", use_container_width=True):
    st.session_state.steps_done.add(1)

if 1 in st.session_state.steps_done:
    tab1, tab2, tab3 = st.tabs(["First 5 Rows", "Column Names & Types", "Shape"])
    with tab1:
        st.dataframe(df.head(), use_container_width=True)
    with tab2:
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with tab3:
        show_metrics((f"{df.shape[0]:,}", "Rows"), (f"{df.shape[1]}", "Columns"))

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Check Missing Values
# ══════════════════════════════════════════════════════════════════════════════
card(2, "Check Missing Values")

if st.button("🔍  Check Missing Values", use_container_width=True):
    st.session_state.steps_done.add(2)

if 2 in st.session_state.steps_done:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values,
    }).query("`Missing Count` >= 0")

    total_missing = int(missing.sum())
    cols_with_missing = int((missing > 0).sum())

    show_metrics(
        (f"{total_missing:,}", "Total Missing"),
        (f"{cols_with_missing}", "Columns Affected"),
    )
    st.dataframe(
        missing_df.style.background_gradient(subset=["Missing Count"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Convert Numeric Columns
# ══════════════════════════════════════════════════════════════════════════════
card(3, "Convert Numeric Columns")
st.caption("Forces Salary, Performance Rating, Experience, and Age columns to numeric dtype.")

if st.button("🔢  Convert Numeric Columns", use_container_width=True):
    numeric_targets = {
        "Salary (INR)": "Salary (INR)",
        "Performance Rating": "Performance Rating",
        "Experience (Years)": "Experience (Years)",
        "Age": "Age",
    }
    converted, skipped = [], []
    for col in numeric_targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted.append(col)
        else:
            skipped.append(col)

    st.session_state.df = df
    st.session_state.steps_done.add(3)

if 3 in st.session_state.steps_done:
    dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values})
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    st.markdown('<span class="success-tag">✓ Numeric conversion applied.</span>', unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Fill Missing Values
# ══════════════════════════════════════════════════════════════════════════════
card(4, "Fill Missing Values")
st.caption("Salary → Mean · Performance Rating → Median · Experience → Mean · Other numeric → Mean")

if st.button("🩹  Fill Missing Values", use_container_width=True):
    before_missing = df.isnull().sum().sum()

    fill_rules = {
        "Salary (INR)":       lambda c: df[c].mean(),
        "Performance Rating": lambda c: df[c].median(),
        "Experience (Years)": lambda c: df[c].mean(),
    }

    for col, fill_fn in fill_rules.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_fn(col))

    # Remaining numeric columns → mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    after_missing = df.isnull().sum().sum()
    st.session_state.df = df
    st.session_state.steps_done.add(4)

if 4 in st.session_state.steps_done:
    remaining = df.isnull().sum().sum()
    show_metrics((f"{remaining:,}", "Missing Values Left"))
    if remaining == 0:
        st.markdown('<span class="success-tag">✓ No missing values remain.</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="warning-tag">⚠ {remaining} values still missing (likely non-numeric).</span>',
                    unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Remove Duplicates
# ══════════════════════════════════════════════════════════════════════════════
card(5, "Remove Duplicate Records")

if st.button("🗑️  Remove Duplicates", use_container_width=True):
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    st.session_state.df = df
    st.session_state.steps_done.add(5)
    st.session_state["dupes_removed"] = removed

if 5 in st.session_state.steps_done:
    removed = st.session_state.get("dupes_removed", 0)
    show_metrics(
        (f"{removed:,}", "Duplicates Removed"),
        (f"{len(df):,}", "Rows Remaining"),
    )
    st.markdown('<span class="success-tag">✓ Duplicates removed.</span>', unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Fix Negative Salaries
# ══════════════════════════════════════════════════════════════════════════════
card(6, "Fix Negative Salaries")
st.caption("Replaces any negative salary values with the column mean.")

if st.button("💰  Fix Negative Salaries", use_container_width=True):
    if "Salary (INR)" in df.columns:
        neg_count = int((df["Salary (INR)"] < 0).sum())
        mean_salary = df["Salary (INR)"].mean()
        df["Salary (INR)"] = np.where(df["Salary (INR)"] < 0, mean_salary, df["Salary (INR)"])
        st.session_state.df = df
        st.session_state["neg_salary_count"] = neg_count
    st.session_state.steps_done.add(6)

if 6 in st.session_state.steps_done:
    neg_count = st.session_state.get("neg_salary_count", 0)
    show_metrics((f"{neg_count:,}", "Negative Salaries Fixed"))
    st.markdown('<span class="success-tag">✓ Negative salary correction done.</span>', unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Remove Salary Outliers
# ══════════════════════════════════════════════════════════════════════════════
card(7, "Remove Salary Outliers (±3 Std Dev)")
st.caption("Drops rows where Salary falls outside mean ± 3 standard deviations.")

if st.button("📊  Remove Outliers", use_container_width=True):
    if "Salary (INR)" in df.columns:
        before = len(df)
        sal_mean = df["Salary (INR)"].mean()
        sal_std  = df["Salary (INR)"].std()
        lower = sal_mean - 3 * sal_std
        upper = sal_mean + 3 * sal_std
        df = df[(df["Salary (INR)"] >= lower) & (df["Salary (INR)"] <= upper)]
        removed = before - len(df)
        st.session_state.df = df
        st.session_state["outliers_removed"] = removed
        st.session_state["outlier_bounds"] = (round(lower, 2), round(upper, 2))
    st.session_state.steps_done.add(7)

if 7 in st.session_state.steps_done:
    removed = st.session_state.get("outliers_removed", 0)
    lo, hi = st.session_state.get("outlier_bounds", (0, 0))
    show_metrics(
        (f"{removed:,}", "Outliers Removed"),
        (f"₹{lo:,}", "Lower Bound"),
        (f"₹{hi:,}", "Upper Bound"),
        (f"{len(df):,}", "Rows Remaining"),
    )
    st.markdown('<span class="success-tag">✓ Outlier removal complete.</span>', unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save Cleaned Data
# ══════════════════════════════════════════════════════════════════════════════
card(8, "Save Cleaned Dataset")

save_path = st.text_input(
    "Output file path",
    value=r"D:\Python\Numpy\indian_employee\cleaned_employee_data.csv",
    placeholder=r"D:\Python\Numpy\indian_employee\cleaned_employee_data.csv",
)

col_save, col_dl = st.columns([1, 1])

with col_save:
    if st.button("💾  Save to Disk", use_container_width=True):
        try:
            df.to_csv(save_path.strip(), index=False)
            st.success(f"✅  Saved to: `{save_path.strip()}`")
            st.session_state.steps_done.add(8)
        except Exception as e:
            st.error(f"Could not save to that path: {e}")

with col_dl:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download CSV",
        data=csv_bytes,
        file_name="cleaned_employee_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Live Stats
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📋 Pipeline Status")
    steps = {
        0: "Load Data",
        1: "Preview Data",
        2: "Check Missing Values",
        3: "Convert Numeric Cols",
        4: "Fill Missing Values",
        5: "Remove Duplicates",
        6: "Fix Negative Salaries",
        7: "Remove Outliers",
        8: "Save Cleaned Data",
    }
    for n, name in steps.items():
        icon = "✅" if n in st.session_state.steps_done else "⬜"
        # Step 0 always marked if df loaded
        if n == 0 and st.session_state.df is not None:
            icon = "✅"
        st.markdown(f"{icon} **Step {n}** — {name}")

    st.markdown("---")
    st.markdown("### Current Dataset")
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", df.shape[1])
    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    if "Salary (INR)" in df.columns:
        st.metric("Avg Salary (INR)", f"₹{df['Salary (INR)'].mean():,.0f}")