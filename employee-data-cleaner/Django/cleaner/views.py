import uuid
import os
import json
import io
import pandas as pd
import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse

def _get_filepath(request):
    """Get the unique filepath for this user's temp dataset."""
    if "file_id" not in request.session:
        request.session["file_id"] = str(uuid.uuid4())
    filename = f"{request.session['file_id']}.pkl"
    return os.path.join(settings.BASE_DIR, "tmp", filename)

def load_df(request):
    """Load df from the fast temporary pickle file."""
    filepath = _get_filepath(request)
    if not os.path.exists(filepath):
        return None
    try:
        return pd.read_pickle(filepath)
    except Exception:
        return None

def save_df(request, df):
    """Save df to the fast temporary pickle file."""
    filepath = _get_filepath(request)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_pickle(filepath)

def get_summary(df):
    """Generate universal summary object for the frontend."""
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "missing": int(df.isnull().sum().sum())
    }

def index(request):
    ctx = {}
    df = load_df(request)
    if df is not None:
        ctx["summary"] = get_summary(df)
        ctx["columns_json"] = json.dumps(list(df.columns))
        ctx["dtypes_json"] = json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()})
    return render(request, "cleaner/index.html", ctx)

def upload(request):
    if request.method == "POST" and request.FILES.get("file"):
        file = request.FILES["file"]
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file, encoding="utf-8")
            else:
                df = pd.read_excel(file)
            save_df(request, df)
            return JsonResponse({
                "ok": True,
                "summary": get_summary(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            })
        except Exception as e:
            return JsonResponse({"ok": False, "error": str(e)})
    return JsonResponse({"ok": False, "error": "No file uploaded."})

def step_preview(request):
    df = load_df(request)
    if df is None: return JsonResponse({"ok": False, "error": "No data loaded"})
    
    html = df.head(10).to_html(classes="table data-table table-sm", index=False)
    dtypes = df.dtypes
    non_null = df.notnull().sum()
    dtype_rows = [
        {"column": col, "dtype": str(dtypes[col]), "non_null": int(non_null[col])}
        for col in df.columns
    ]
    
    return JsonResponse({
        "ok": True,
        "summary": get_summary(df),
        "head_html": html,
        "dtype_rows": dtype_rows
    })

def step_missing(request):
    df = load_df(request)
    if df is None: return JsonResponse({"ok": False, "error": "No data loaded"})
    
    missing = df.isnull().sum()
    rows = [
        {"column": col, "count": int(missing[col]), "pct": float(missing[col]/len(df)*100), "dtype": str(df[col].dtype)}
        for col in df.columns
    ]
    return JsonResponse({
        "ok": True,
        "summary": get_summary(df),
        "rows": rows
    })

def step_convert(request):
    df = load_df(request)
    if df is None: return JsonResponse({"ok": False, "error": "No data loaded"})
    
    try:
        body = json.loads(request.body)
        conversions = body.get("conversions", {})
    except:
        conversions = {}
        
    converted = []
    
    for col, new_type in conversions.items():
        if col in df.columns:
            before = str(df[col].dtype)
            error = None
            try:
                if new_type == "int":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64") # Allows NaNs
                elif new_type == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                elif new_type == "str":
                    df[col] = df[col].astype(str).replace(["nan", "None", "<NA>"], np.nan)
                elif new_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif new_type == "bool":
                    df[col] = df[col].astype(bool)
            except Exception as e:
                error = str(e)
                
            after = str(df[col].dtype)
            converted.append({
                "column": col, 
                "before": before, 
                "after": after, 
                "error": error
            })
            
    save_df(request, df)
    return JsonResponse({
        "ok": True,
        "summary": get_summary(df),
        "converted": converted,
        "columns": list(df.columns),
        "dtypes": {c: str(dtype) for c, dtype in df.dtypes.items()}
    })

def step_fill(request):
    df = load_df(request)
    if df is None: return JsonResponse({"ok": False, "error": "No data loaded"})

    try:
        body = json.loads(request.body)
        strategies = body.get("strategies", {})
    except Exception:
        strategies = {}

    fill_log = []
    rows_before = len(df)

    # Collect columns to delete rows for (process at end, all at once)
    delete_cols = []

    for col, strat in strategies.items():
        if col not in df.columns:
            continue
        missing_before = int(df[col].isnull().sum())

        if strat == "delete":
            delete_cols.append(col)
            continue

        if missing_before == 0:
            continue

        try:
            numeric_series = pd.to_numeric(df[col], errors="coerce")

            if strat == "mean":
                val = numeric_series.mean()
                val = 0.0 if pd.isna(val) else val
                fill_label = f"{val:.2f}"
            elif strat == "median":
                val = numeric_series.median()
                val = 0.0 if pd.isna(val) else val
                fill_label = f"{val:.2f}"
            elif strat == "mode":
                modes = df[col].mode()
                val = modes.iloc[0] if not modes.empty else np.nan
                if pd.isna(val):
                    val = 0
                fill_label = str(val)
            else:
                continue

            df[col] = df[col].fillna(val)
            fill_log.append({
                "column": col,
                "strategy": strat,
                "fill_val": fill_label,
                "filled": missing_before
            })
        except Exception:
            pass

    # Apply "delete" strategy: drop any row with NaN in flagged columns
    if delete_cols:
        rows_before_delete = len(df)
        df = df.dropna(subset=delete_cols)
        rows_dropped = rows_before_delete - len(df)
        for col in delete_cols:
            fill_log.append({
                "column": col,
                "strategy": "delete",
                "fill_val": "—",
                "filled": rows_dropped
            })

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    save_df(request, df)
    remaining = int(df.isnull().sum().sum())
    return JsonResponse({
        "ok": True,
        "summary": get_summary(df),
        "fill_log": fill_log,
        "remaining": remaining,
        "rows_after": len(df)
    })

def step_duplicates(request):
    df = load_df(request)
    if df is None: return JsonResponse({"ok": False, "error": "No data loaded"})
    
    before = len(df)
    df = df.drop_duplicates()
    save_df(request, df)
    
    return JsonResponse({
        "ok": True,
        "summary": get_summary(df),
        "removed": before - len(df)
    })

def step_negative(request):
    df = load_df(request)
    if df is None: return JsonResponse({"ok": False, "error": "No data loaded"})
    
    try:
        body = json.loads(request.body)
        col = body.get("column", "Salary (INR)")
        mode = body.get("mode", "mean")
        custom = body.get("custom_value", 0)
    except:
        col = "Salary (INR)"
        mode = "mean"
        custom = 0
        
    neg_count = 0
    fill_val = 0
    if col in df.columns:
        neg_count = int((df[col] < 0).sum())
        if neg_count > 0:
            if mode == "mean": fill_val = df[col].mean()
            elif mode == "median": fill_val = df[col].median()
            elif mode == "zero": fill_val = 0
            else: fill_val = float(custom)
                
            df[col] = np.where(df[col] < 0, fill_val, df[col])
            save_df(request, df)
            
    return JsonResponse({
        "ok": True,
        "summary": get_summary(df),
        "neg_count": neg_count,
        "fill_val": float(fill_val)
    })

def step_outliers(request):
    df = load_df(request)
    if df is None: return JsonResponse({"ok": False, "error": "No data loaded"})
    
    try:
        body = json.loads(request.body)
        col = body.get("column", "Salary (INR)")
        method = body.get("method", "zscore")
        n = float(body.get("std_multiplier", 3))
    except:
        col = "Salary (INR)"
        method = "zscore"
        n = 3
        
    removed = 0
    lo = hi = 0
    hist = {}
    
    if col in df.columns:
        before = len(df)
        if method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            lo = mean - n * std
            hi = mean + n * std
        else: # iqr
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - n * iqr
            hi = q3 + n * iqr
            
        df = df[(df[col] >= lo) & (df[col] <= hi)]
        removed = before - len(df)
        save_df(request, df)
        
        hist_data = np.histogram(df[col].dropna(), bins=20)
        hist = {
            "labels": [float(v) for v in hist_data[1][:-1]],
            "data": [int(v) for v in hist_data[0]]
        }
        
    return JsonResponse({
        "ok": True,
        "summary": get_summary(df),
        "removed": removed,
        "lower": float(lo),
        "upper": float(hi),
        "hist": hist
    })

def step_profile(request):
    df = load_df(request)
    if df is None: return JsonResponse({"ok": False, "error": "No data loaded"})
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Ensure describe returns numbers native or convert to dict properly
    desc_df = df[num_cols].describe()
    
    # NaN issues in dict conversion require replacement
    desc_df = desc_df.replace([np.inf, -np.inf, np.nan], None)
    desc = desc_df.to_dict()
    
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    categorical = {}
    for c in cat_cols:
        counts = df[c].value_counts(normalize=True).head(5) * 100
        categorical[c] = [{"value": str(k), "pct": float(v)} for k, v in counts.items()]
        
    hist = {}
    if "Salary (INR)" in df.columns:
        hist_data = np.histogram(df["Salary (INR)"].dropna(), bins=20)
        hist = {
            "labels": [float(v) for v in hist_data[1][:-1]],
            "data": [int(v) for v in hist_data[0]]
        }
        
    return JsonResponse({
        "ok": True,
        "summary": get_summary(df),
        "describe": desc,
        "categorical": categorical,
        "hist": hist
    })

def reset(request):
    filepath = _get_filepath(request)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass
    request.session.pop("file_id", None)
    return JsonResponse({"ok": True})

def download(request):
    df = load_df(request)
    if df is None:
        return HttpResponse("No data available to download.", status=400)

    fmt = request.GET.get("format", "csv")

    if fmt == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        buf.seek(0)
        response = HttpResponse(
            buf.read(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response["Content-Disposition"] = 'attachment; filename="cleaned_employee_data.xlsx"'

    elif fmt == "pdf":
        # Build a styled HTML page with the dataframe table, then return as HTML
        # for the browser to print-to-PDF (no extra server dependencies needed)
        table_html = df.to_html(index=False, border=0, classes="data-tbl")
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cleaned Employee Data</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 11px; padding: 20px; color: #222; }}
  h2 {{ color: #333; margin-bottom: 8px; }}
  .meta {{ color: #666; font-size: 10px; margin-bottom: 16px; }}
  .data-tbl {{ border-collapse: collapse; width: 100%; }}
  .data-tbl th {{ background: #2d3748; color: #fff; padding: 6px 10px; text-align: left; font-size: 10px; }}
  .data-tbl td {{ padding: 5px 10px; border-bottom: 1px solid #e2e8f0; }}
  .data-tbl tr:nth-child(even) td {{ background: #f7fafc; }}
</style>
</head>
<body>
<h2>Cleaned Employee Data</h2>
<p class="meta">Rows: {len(df):,} &nbsp;|&nbsp; Columns: {len(df.columns)} &nbsp;|&nbsp; Generated by Employee Data Cleaner</p>
{table_html}
<script>window.onload = function() {{ window.print(); }}</script>
</body>
</html>"""
        response = HttpResponse(html, content_type="text/html")
        # No attachment — open in browser so user can print/save-as-PDF

    else:  # csv
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        response = HttpResponse(buf.getvalue(), content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="cleaned_employee_data.csv"'

    return response
