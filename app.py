import base64
import io
import itertools
import os
import re

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, MATCH, callback_context, dash_table, dcc, html, no_update

DATA_FILE_PARQUET = os.path.join(os.path.dirname(__file__), "data", "Actuals_Data.parquet")
DATA_FILE_XLSX = os.path.join(os.path.dirname(__file__), "data", "Actuals_Data.xlsx")


def load_data_file():
    """Load Actuals_Data at startup. Prefers the Parquet file (fast); falls
    back to the Excel file if no Parquet file is present yet, so this keeps
    working even before the conversion script has been run once."""
    if os.path.exists(DATA_FILE_PARQUET):
        source_file, reader = DATA_FILE_PARQUET, pd.read_parquet
    else:
        source_file, reader = DATA_FILE_XLSX, pd.read_excel
    try:
        df = reader(source_file)
    except FileNotFoundError:
        return None, f"Data file not found: {source_file}", None
    except Exception as exc:
        return None, f"Could not read data file: {exc}", None
    df.columns = df.columns.str.strip()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}", None
    df["Pick Date"] = pd.to_datetime(df["Pick Date"], errors="coerce")
    valid_dates = df["Pick Date"].dropna()
    if valid_dates.empty:
        return None, "No valid dates found in Pick Date column.", None
    meta = {
        "rows": len(df), "cols": len(df.columns),
        "min_date": valid_dates.min().date().isoformat(),
        "max_date": valid_dates.max().date().isoformat(),
    }
    return (df_to_store(df),
            f"{os.path.basename(source_file)} loaded: {len(df):,} rows, {len(df.columns)} columns",
            meta)


SPEEDS = [200, 400, 600, 800, 1000]
ROW_LENGTH = 3333
FIN_ITEM_FIELDS = [
    "chariot", "trolley", "scales", "hms", "logistics",
    "platform", "tablet", "it", "burro_std", "burro_sw", "burro_maint",
]
REQUIRED_COLUMNS = [
    "Pick Event Number", "Pick Date", "Yield Kg", "Variety Area (ha)",
    "Total Harvest Hours", "Total Harvest Cost", "Distinct Picker Count",
]
FILTER_DEFS = [
    ("Plant", "Plant"), ("Division", "Division"), ("Location", "Location"),
    ("Variety", "Product Variety"), ("Cropping System", "Cropping System"),
]


def fmt_dollar(value):
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def safe_key(value):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_") or "all"


def df_to_store(df):
    return df.to_json(orient="split", date_format="iso")


def df_from_store(data):
    if not data:
        return pd.DataFrame()
    df = pd.read_json(io.StringIO(data), orient="split")
    if "Pick Date" in df.columns:
        df["Pick Date"] = pd.to_datetime(df["Pick Date"], errors="coerce")
    return df


def parse_upload(contents, filename):
    if not contents:
        return None, "Upload a CSV or Excel file to get started.", None
    try:
        _, encoded = contents.split(",", 1)
        decoded = base64.b64decode(encoded)
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(decoded))
        else:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as exc:
        return None, f"Could not read file: {exc}", None
    df.columns = df.columns.str.strip()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}", None
    df["Pick Date"] = pd.to_datetime(df["Pick Date"], errors="coerce")
    valid_dates = df["Pick Date"].dropna()
    if valid_dates.empty:
        return None, "No valid dates found in Pick Date column.", None
    meta = {
        "rows": len(df), "cols": len(df.columns),
        "min_date": valid_dates.min().date().isoformat(),
        "max_date": valid_dates.max().date().isoformat(),
    }
    return df_to_store(df), f"File loaded: {len(df):,} rows, {len(df.columns)} columns", meta


def compute_pick_calcs(df_json, total_pickers, labour_cost, max_pickrate, mid_speeds_tuple):
    df = df_from_store(df_json)
    mid_speeds = list(mid_speeds_tuple)
    out = df.copy()
    for speed in SPEEDS:
        out[f"Hours/block@{speed}"] = out["Variety Area (ha)"] * ROW_LENGTH / speed
        out[f"Required Rate@{speed}"] = np.where(
            (out[f"Hours/block@{speed}"] > 0) & (total_pickers > 0),
            out["Yield Kg"] / out[f"Hours/block@{speed}"] / total_pickers, 0,
        )
        out[f"Benefit@{speed}"] = out["Total Harvest Cost"] - labour_cost * out[f"Hours/block@{speed}"]
        out[f"Pick Achievable@{speed}"] = (out[f"Required Rate@{speed}"] <= max_pickrate).astype(int)
        out[f"Positive Return@{speed}"] = np.where(out[f"Benefit@{speed}"] > 0, 1, np.nan)
        out[f"Pickable Profit@{speed}"] = np.where(
            (out[f"Pick Achievable@{speed}"] == 1) & (out[f"Positive Return@{speed}"] == 1),
            out[f"Benefit@{speed}"], np.nan,
        )
    all_profit_cols = [f"Pickable Profit@{speed}" for speed in SPEEDS]
    mid_profit_cols = [f"Pickable Profit@{speed}" for speed in mid_speeds]
    out["Best Profit"] = out[all_profit_cols].max(axis=1).fillna(0)
    out["Best Profit Mid"] = out[mid_profit_cols].max(axis=1).fillna(0) if mid_profit_cols else 0
    has_any = out[all_profit_cols].notna().any(axis=1)
    def safe_idx(row): return row.idxmax() if row.notna().any() else pd.NA
    optimal = out[all_profit_cols].apply(safe_idx, axis=1)
    out["Optimal Speed"] = (
        optimal.astype(str).str.extract(r"(\d+)", expand=False).where(has_any).astype("Int64")
    )
    out["No of pickers"] = np.where(out["Best Profit"] > 0, out["Distinct Picker Count"], np.nan)
    return out


def rename_calc_cols(df):
    rename_map = {}
    for speed in SPEEDS:
        rename_map[f"Hours/block@{speed}"] = f"Hours per block @ {speed}m/hr"
        rename_map[f"Required Rate@{speed}"] = f"Required kg/hr/ppl @ {speed}m/hr"
        rename_map[f"Benefit@{speed}"] = f"Platform Potential Benefit @ {speed}m/hr"
        rename_map[f"Pick Achievable@{speed}"] = f"Pick achievable @ {speed}m/hr"
        rename_map[f"Positive Return@{speed}"] = f"Positive pick event return @ {speed}m/hr"
        rename_map[f"Pickable Profit@{speed}"] = f"Pickable & Profitable @ {speed}m"
    return df.rename(columns=rename_map)


def build_column_order(df, mid_label):
    base_cols = [
        "Pick Event Number", "Pick Date", "Plant", "Division", "Location", "Product Variety",
        "Cropping System", "Variety Area (ha)", "Yield Kg", "Total Harvest Hours",
        "Total Harvest Cost", "Distinct Picker Count",
        "Pick Rate Kgs/Hr (Picker Only – Derived Kgs)", "Harvest Rate Kgs/Hr (Derived Kgs)",
    ]
    calc_cols = (
        [f"Hours per block @ {speed}m/hr" for speed in SPEEDS]
        + [f"Required kg/hr/ppl @ {speed}m/hr" for speed in SPEEDS]
        + [f"Platform Potential Benefit @ {speed}m/hr" for speed in SPEEDS]
        + [f"Pick achievable @ {speed}m/hr" for speed in SPEEDS]
        + [f"Positive pick event return @ {speed}m/hr" for speed in SPEEDS]
        + [f"Pickable & Profitable @ {speed}m" for speed in SPEEDS]
        + ["Best Profit", mid_label, "Optimal Speed", "No of pickers"]
    )
    base_cols = [col for col in base_cols if col in df.columns]
    calc_cols = [col for col in calc_cols if col in df.columns]
    extra_cols = [col for col in df.columns if col not in set(base_cols + calc_cols)]
    return df[base_cols + calc_cols + extra_cols]


def apply_filters(df, start_date, end_date, selections):
    if start_date and end_date:
        df = df[
            (df["Pick Date"].dt.date >= pd.to_datetime(start_date).date())
            & (df["Pick Date"].dt.date <= pd.to_datetime(end_date).date())
        ].copy()
    for col, values in selections.items():
        if col not in df.columns:
            continue
        if not values:
            return df.iloc[0:0]
        df = df[df[col].astype(str).isin(values)]
    return df


def options_for(df, col):
    if df.empty or col not in df.columns:
        return []
    values = sorted(df[col].dropna().astype(str).unique())
    return [{"label": value, "value": value} for value in values]


def all_values(options):
    return [opt["value"] for opt in options]


def normalise_dates(value):
    if isinstance(value, set): return value
    if isinstance(value, list): return set(value)
    return set()


def greedy_allocate(block_df, n_devices):
    records = []
    for _, row in block_df.iterrows():
        dates = normalise_dates(row.get("Pick Date", set()))
        cropping_system = row.get("Cropping System", "")
        records.append({
            "location": row["Location"], "cropping_system": cropping_system,
            "best_profit": float(row["Best Profit"]), "dates": dates, "allocated": False,
        })
    records.sort(key=lambda item: item["best_profit"], reverse=True)
    allocations = []
    allocated_dates = set()
    for device_num in range(1, n_devices + 1):
        assigned = False
        for rec in records:
            if rec["allocated"] or (rec["dates"] & allocated_dates):
                continue
            rec["allocated"] = True
            allocated_dates |= rec["dates"]
            label = rec["location"] + (f" ({rec['cropping_system']})" if rec["cropping_system"] else "")
            allocations.append({
                "device": device_num, "assigned": True, "label": label,
                "location": rec["location"], "cropping_system": rec["cropping_system"],
                "best_profit": rec["best_profit"], "pick_events": len(rec["dates"]),
            })
            assigned = True
            break
        if not assigned:
            allocations.append({
                "device": device_num, "assigned": False, "label": None,
                "location": None, "cropping_system": None, "best_profit": 0.0, "pick_events": 0,
            })
    return allocations


def find_best_single_block(block_df):
    if block_df.empty or "Best Profit" not in block_df.columns:
        return None
    row = block_df.loc[block_df["Best Profit"].idxmax()]
    cropping_system = row.get("Cropping System", "")
    return {
        "label": row["Location"] + (f" ({cropping_system})" if cropping_system else ""),
        "location": row["Location"], "cropping_system": cropping_system,
        "best_profit": float(row["Best Profit"]),
        "pick_events": len(normalise_dates(row.get("Pick Date", set()))),
    }


def find_optimal_combination(block_df, n_devices, max_combo_size=8):
    if block_df.empty or n_devices <= 0:
        return [], 0.0
    records = []
    for _, row in block_df.iterrows():
        dates = normalise_dates(row.get("Pick Date", set()))
        cropping_system = row.get("Cropping System", "")
        records.append({
            "label": row["Location"] + (f" ({cropping_system})" if cropping_system else ""),
            "location": row["Location"], "cropping_system": cropping_system,
            "best_profit": float(row["Best Profit"]), "dates": dates,
        })
    combo_size = min(n_devices, len(records))
    if len(records) <= max_combo_size:
        best_profit = -1.0
        best_combo = []
        for combo in itertools.combinations(range(len(records)), combo_size):
            used_dates = set()
            valid = True
            for idx in combo:
                if used_dates & records[idx]["dates"]:
                    valid = False
                    break
                used_dates |= records[idx]["dates"]
            if valid:
                total = sum(records[idx]["best_profit"] for idx in combo)
                if total > best_profit:
                    best_profit = total
                    best_combo = [records[idx] for idx in combo]
        return best_combo, max(best_profit, 0.0)
    chosen = []
    allocated_dates = set()
    for rec in sorted(records, key=lambda item: item["best_profit"], reverse=True):
        if len(chosen) >= combo_size: break
        if not (rec["dates"] & allocated_dates):
            chosen.append(rec)
            allocated_dates |= rec["dates"]
    return chosen, sum(item["best_profit"] for item in chosen)


def find_highest_achievable_scenario(block_df):
    if block_df.empty:
        return [], 0.0
    records = []
    for _, row in block_df.iterrows():
        dates = normalise_dates(row.get("Pick Date", set()))
        cropping_system = row.get("Cropping System", "")
        records.append({
            "label": row["Location"] + (f" ({cropping_system})" if cropping_system else ""),
            "location": row["Location"], "cropping_system": cropping_system,
            "best_profit": float(row["Best Profit"]), "dates": dates,
        })
    chosen = []
    allocated_dates = set()
    for rec in sorted(records, key=lambda item: item["best_profit"], reverse=True):
        if rec["best_profit"] <= 0: continue
        if not (rec["dates"] & allocated_dates):
            chosen.append(rec)
            allocated_dates |= rec["dates"]
    return chosen, sum(item["best_profit"] for item in chosen)


def build_block_df(data, mid_label):
    if "Location" not in data.columns or "Best Profit" not in data.columns:
        return pd.DataFrame()
    group_cols = ["Location"]
    if "Cropping System" in data.columns:
        group_cols.append("Cropping System")
    def safe_date_list(values):
        dates = pd.to_datetime(values, errors="coerce")
        return sorted({item.date().isoformat() for item in dates.dropna()})
    agg = {"Best Profit": "sum"}
    if "Pick Date" in data.columns: agg["Pick Date"] = safe_date_list
    if "Distinct Picker Count" in data.columns: agg["Distinct Picker Count"] = "max"
    if "Total Harvest Cost" in data.columns: agg["Total Harvest Cost"] = "sum"
    if mid_label in data.columns: agg[mid_label] = "sum"
    block_df = (
        data.groupby(group_cols, dropna=False).agg(agg).reset_index()
        .sort_values("Best Profit", ascending=False).reset_index(drop=True)
    )
    block_df.insert(0, "Rank", block_df.index + 1)
    return block_df


def records_to_block_df(records):
    df = pd.DataFrame(records)
    if "Pick Date" in df.columns:
        df["Pick Date"] = df["Pick Date"].apply(normalise_dates)
    return df


def build_optimiser_result(data, label, n_devices, mid_label):
    block_df = build_block_df(data, mid_label)
    if block_df.empty:
        return {"label": label, "block_records": [], "allocations": [], "assigned_count": 0, "total_profit": 0.0}
    working_df = records_to_block_df(block_df.to_dict("records"))
    allocations = greedy_allocate(working_df, n_devices)
    assigned = [item for item in allocations if item["assigned"]]
    return {
        "label": label, "block_records": block_df.to_dict("records"),
        "allocations": allocations, "assigned_count": len(assigned),
        "total_profit": sum(item["best_profit"] for item in assigned),
    }


def metric_card(label, value, subtext=None, highlight=False):
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(subtext or "", className="metric-subtext"),
        ],
        className="metric-card" + (" highlight" if highlight else ""),
    )


def alert(message, kind="info"):
    return html.Div(message, className=f"alert alert-{kind}")


def make_table(df, page_size=15):
    if df.empty:
        return alert("No rows to display.", "warning")
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime("%Y-%m-%d")
        if display_df[col].apply(lambda x: isinstance(x, (set, list))).any():
            display_df[col] = display_df[col].apply(
                lambda value: ", ".join(sorted(value)) if isinstance(value, (set, list)) else value
            )
    return dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in display_df.columns],
        page_size=page_size, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "DM Sans, Arial, sans-serif", "fontSize": "12px", "padding": "8px",
                    "minWidth": "110px", "maxWidth": "280px", "whiteSpace": "normal"},
        style_header={"backgroundColor": "#f0f7f0", "fontWeight": "700", "color": "#1a4731"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}],
    )


def summary_group_options(df, include_none=False):
    cols = [col for col in ["Plant", "Division", "Location", "Product Variety", "Cropping System", "Pick Date"]
            if col in df.columns]
    options = [{"label": col, "value": col} for col in cols]
    if include_none:
        return [{"label": "None", "value": "__none__"}] + options
    return options


def build_summary_table(df, mid_label, speed_range, level_1, level_2):
    if df.empty: return pd.DataFrame()
    available = {opt["value"] for opt in summary_group_options(df)}
    level_1 = level_1 if level_1 in available else ("Location" if "Location" in available else next(iter(available), None))
    level_2 = level_2 if level_2 in available else "__none__"
    group_cols = [level_1] if level_1 else []
    if level_2 not in (None, "", "__none__", level_1) and level_2 in available:
        group_cols.append(level_2)
    if not group_cols: return pd.DataFrame()
    agg_map = {}
    if "Pick Event Number" in df.columns: agg_map["Pick Event Number"] = "count"
    if "Best Profit" in df.columns: agg_map["Best Profit"] = "sum"
    if mid_label in df.columns: agg_map[mid_label] = "sum"
    if "Distinct Picker Count" in df.columns: agg_map["Distinct Picker Count"] = "max"
    if "Total Harvest Cost" in df.columns: agg_map["Total Harvest Cost"] = "sum"
    if "Yield Kg" in df.columns: agg_map["Yield Kg"] = "sum"
    if "Total Harvest Hours" in df.columns: agg_map["Total Harvest Hours"] = "sum"
    summary = df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    summary = summary.rename(columns={
        "Pick Event Number": "Pick Events", "Best Profit": "Sum Best Profit",
        mid_label: f"Sum Best Profit ({speed_range[0]}-{speed_range[1]}m)",
        "Distinct Picker Count": "Max Pickers", "Yield Kg": "Total Yield (kg)",
    })
    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_cols] = summary[numeric_cols].round(2)
    return summary


def scenario_card(title, value, body, class_name, badge):
    return html.Div(
        [
            html.Div([html.Span(title), html.Span(badge, className="badge")], className="scenario-title"),
            html.Div(value, className="scenario-profit"),
            html.Div(body, className="scenario-body"),
        ],
        className=f"scenario-card {class_name}",
    )


def render_scenarios(block_df, n_devices):
    best_single = find_best_single_block(block_df)
    opt_combo, opt_profit = find_optimal_combination(block_df, n_devices)
    max_combo, max_profit = find_highest_achievable_scenario(block_df)
    cards = []
    if best_single:
        cards.append(scenario_card("Best Single Block", fmt_dollar(best_single["best_profit"]),
            [html.B(best_single["label"]), html.Div(f'{best_single["pick_events"]} pick event(s)', className="muted")],
            "scenario-best", "1 device"))
    else:
        cards.append(scenario_card("Best Single Block", "$0.00", "No profitable blocks found.", "scenario-best", ""))
    if opt_combo:
        cards.append(scenario_card("Optimal Combination", fmt_dollar(opt_profit),
            [html.Div([html.B(item["label"]), f" - {fmt_dollar(item['best_profit'])}"], className="combo-block")
             for item in opt_combo],
            "scenario-optimal", f"{len(opt_combo)} of {n_devices}"))
    else:
        cards.append(scenario_card("Optimal Combination", "$0.00", "No valid non-overlapping combination.", "scenario-optimal", ""))
    if max_combo:
        gap = max_profit - opt_profit
        body = [html.Div(f"+{fmt_dollar(gap)} vs optimal" if gap > 0.01 else "= optimal", className="muted strong")]
        body.extend([html.Div([html.B(item["label"]), f" - {fmt_dollar(item['best_profit'])}"], className="combo-block") for item in max_combo])
        cards.append(scenario_card("Highest Achievable", fmt_dollar(max_profit), body, "scenario-highest", f"{len(max_combo)} block(s)"))
    else:
        cards.append(scenario_card("Highest Achievable", "$0.00", "No profitable non-overlapping blocks.", "scenario-highest", ""))
    rows = []
    if best_single:
        rows.append({"Scenario": "Best Single Block", "Blocks": 1, "Devices Required": 1,
                     "Total Profit": best_single["best_profit"], "vs Optimal": best_single["best_profit"] - opt_profit,
                     "vs Highest Achievable": best_single["best_profit"] - max_profit})
    if opt_combo:
        rows.append({"Scenario": f"Optimal ({n_devices} device(s))", "Blocks": len(opt_combo),
                     "Devices Required": len(opt_combo), "Total Profit": opt_profit,
                     "vs Optimal": 0.0, "vs Highest Achievable": opt_profit - max_profit})
    if max_combo:
        rows.append({"Scenario": f"Highest Achievable ({len(max_combo)} devices)", "Blocks": len(max_combo),
                     "Devices Required": len(max_combo), "Total Profit": max_profit,
                     "vs Optimal": max_profit - opt_profit, "vs Highest Achievable": 0.0})
    return html.Div([
        html.H3("Scenario Analysis"),
        html.Div(cards, className="scenario-grid"),
        html.Details([
            html.Summary("Scenario Comparison Table"),
            make_table(pd.DataFrame(rows), page_size=10) if rows else alert("No scenario rows.", "warning"),
        ], className="details-panel"),
    ])


def filter_card(label, dropdown_id, all_id, none_id):
    return html.Div(
        [
            html.Div(label, className="filter-label"),
            dcc.Dropdown(id=dropdown_id, options=[], value=None, multi=True,
                         placeholder=f"Select {label}", className="filter-dropdown"),
            html.Div([
                html.Button("All", id=all_id, n_clicks=0, className="mini-button"),
                html.Button("None", id=none_id, n_clicks=0, className="mini-button secondary"),
            ], className="filter-buttons"),
        ],
        className="filter-card",
    )


def numeric_input(label, id_value, value, step, min_value=0):
    return html.Div(
        [
            html.Div(label, className="field-label-text"),
            html.Div([
                html.Button("−", id={"type": "stepper-dec", "target": id_value}, n_clicks=0, className="stepper-btn"),
                dcc.Input(id=id_value, type="text", value=str(value), className="stepper-input", debounce=True,
                          style={"border": "none", "borderLeft": "1px solid #a5d6a7", "borderRight": "1px solid #a5d6a7",
                                 "borderRadius": "0", "textAlign": "center", "color": "#111827",
                                 "WebkitTextFillColor": "#111827", "background": "#ffffff", "fontWeight": "700",
                                 "fontSize": "14px", "height": "38px", "width": "100%", "padding": "0 4px",
                                 "outline": "none", "appearance": "none", "MozAppearance": "textfield"}),
                html.Button("+", id={"type": "stepper-inc", "target": id_value}, n_clicks=0, className="stepper-btn"),
            ], className="stepper-wrap"),
        ],
        className="field-label",
    )


def fin_input(label, label_key, field, value, step=100, min_value=0):
    return dcc.Input(
        id={"type": "fin-input", "label": label_key, "field": field},
        type="text", value=str(value), debounce=True, inputMode="numeric", className="fin-input",
        style={"color": "#111827", "WebkitTextFillColor": "#111827", "background": "#ffffff",
               "border": "1px solid #d1d5db", "borderRadius": "6px", "padding": "7px 10px",
               "fontWeight": "600", "fontSize": "13px", "width": "100%",
               "MozAppearance": "textfield", "appearance": "textfield"},
    )


def parse_fin_number(value, default=0.0):
    try:
        return float(str(value).replace(",", "").strip()) if value not in (None, "", "None") else default
    except Exception:
        return default


def annualised_value(value, life):
    value = parse_fin_number(value, 0.0)
    life = parse_fin_number(life, 1.0)
    if life <= 0: life = 1.0
    return value / life


def financial_multipliers(context):
    if not context: return {field: 0 for field in FIN_ITEM_FIELDS}
    block_df = pd.DataFrame(context.get("block_records", []))
    allocations = context.get("allocations", [])
    assigned_count = int(context.get("assigned_count") or 0)
    total_blocks = max(len(block_df), 1)
    allocated_blocks = [(item["location"], item.get("cropping_system", "")) for item in allocations if item.get("assigned")]
    if block_df.empty or not allocated_blocks:
        max_pickers = 0
    else:
        allocated_rows = block_df[block_df.apply(
            lambda row: (row.get("Location"), row.get("Cropping System", "")) in allocated_blocks, axis=1)]
        max_pickers = allocated_rows["Distinct Picker Count"].sum() if "Distinct Picker Count" in allocated_rows.columns else 0
    trolley_qty = max(1, round(max_pickers * 1.2)) if assigned_count else 0
    partial = assigned_count / total_blocks
    return {"chariot": assigned_count, "trolley": trolley_qty, "scales": trolley_qty, "hms": assigned_count,
            "logistics": partial, "platform": assigned_count, "tablet": assigned_count, "it": assigned_count,
            "burro_std": assigned_count, "burro_sw": assigned_count, "burro_maint": partial}


def fin_step_for_field(field):
    if field.endswith("_life"): return 1
    if field.endswith("_val"): return 100
    return 1


def fin_stepper_input(label, label_key, field, value, step=100, min_value=0):
    return html.Div(
        [
            html.Button("-", id={"type": "fin-stepper-dec", "label": label_key, "field": field},
                        n_clicks=0, className="fin-stepper-btn"),
            fin_input(label, label_key, field, value, step=step, min_value=min_value),
            html.Button("+", id={"type": "fin-stepper-inc", "label": label_key, "field": field},
                        n_clicks=0, className="fin-stepper-btn"),
        ],
        className="fin-stepper-wrap",
    )


def fin_row(name, label_key, prefix, value, life, multiplier=1, note=""):
    annual = annualised_value(value, life)
    total = annual * multiplier
    return html.Div(
        [
            html.Div([html.B(name), html.Span(f" ({note})" if note else "", className="muted")]),
            fin_stepper_input(name, label_key, f"{prefix}_val", value),
            fin_stepper_input(name, label_key, f"{prefix}_life", life, step=1, min_value=0.1),
            html.Div(fmt_dollar(annual), id={"type": "fin-annual", "label": label_key, "field": prefix}, className="fin-calc"),
            html.Div(fmt_dollar(total), id={"type": "fin-total", "label": label_key, "field": prefix}, className="fin-calc strong"),
        ],
        className="fin-row",
    )


def render_financial_controls(label_key, context):
    multipliers = financial_multipliers(context)
    return html.Div(
        [
            dcc.Store(id={"type": "financial-context", "label": label_key}, data=context),
            html.H3("Financial Analysis"),
            html.Div([html.Div("Item", className="fin-head"), html.Div("Value ($)", className="fin-head"),
                      html.Div("Life (yrs)", className="fin-head"), html.Div("Annual Rate", className="fin-head"),
                      html.Div("Total", className="fin-head")], className="fin-row fin-header"),
            html.H4("Equipment Savings"),
            fin_row("Chariot", label_key, "chariot", 35000, 10, multipliers["chariot"]),
            fin_row("Trolleys", label_key, "trolley", 515.22, 10, multipliers["trolley"], "qty calculated"),
            fin_row("Scales", label_key, "scales", 80, 5, multipliers["scales"], "qty calculated"),
            fin_row("HMS Kits", label_key, "hms", 25000, 10, multipliers["hms"]),
            fin_row("Logistics - Trolleys", label_key, "logistics", 30000, 1, multipliers["logistics"]),
            html.Label([
                html.Span("Overhead %"),
                dcc.Slider(id={"type": "fin-input", "label": label_key, "field": "overhead_pct"},
                           min=0, max=50, step=0.1, value=19, marks={0: "0", 19: "19", 50: "50"}),
            ], className="slider-field"),
            html.H4("Equipment Costs"),
            fin_row("Picking Platform", label_key, "platform", 120000, 10, multipliers["platform"]),
            fin_row("Samsung Galaxy Tab", label_key, "tablet", 1543, 5, multipliers["tablet"]),
            fin_row("IT Service Charges", label_key, "it", 160, 1, multipliers["it"]),
            fin_row("Burro Std with tray", label_key, "burro_std", 35000, 10, multipliers["burro_std"]),
            fin_row("Burro Software", label_key, "burro_sw", 3000, 1, multipliers["burro_sw"]),
            fin_row("Burro Maintenance", label_key, "burro_maint", 30000, 10, multipliers["burro_maint"]),
            html.Div(id={"type": "financial-output", "label": label_key}),
        ],
        className="financial-panel",
    )


def render_optimiser_result(result, n_devices, mid_label):
    label = result["label"]
    label_key = safe_key(label)
    block_df = records_to_block_df(result["block_records"])
    allocations = result["allocations"]
    if block_df.empty:
        return alert(f"[{label}] Missing valid Location or Best Profit data.", "warning")
    display_cols = [col for col in ["Rank", "Location", "Cropping System", "Best Profit", mid_label,
                                    "Distinct Picker Count", "Total Harvest Cost"] if col in block_df.columns]
    allocation_cards = []
    for allocation in allocations:
        if allocation["assigned"]:
            allocation_cards.append(html.Div([
                html.B(f'Device {allocation["device"]} -> {allocation["label"]}'),
                html.Div(f'Best Profit: {fmt_dollar(allocation["best_profit"])} | Pick Events: {allocation["pick_events"]}', className="muted"),
            ], className="device-card"))
        else:
            allocation_cards.append(html.Div([
                html.B(f'Device {allocation["device"]} - No valid block'),
                html.Div("All remaining blocks overlap with already allocated dates.", className="muted"),
            ], className="device-card warn"))
    context = {"label": label, "block_records": result["block_records"], "allocations": allocations,
                "assigned_count": result["assigned_count"], "total_profit": result["total_profit"]}
    return html.Div([
        html.H2(label.replace("_", " ")),
        html.H3("Block Tally"),
        make_table(block_df[display_cols], page_size=12),
        render_scenarios(block_df, n_devices),
        html.H3(f"Device Allocation ({n_devices} device(s))"),
        html.Div(allocation_cards),
        html.Div([
            metric_card("Devices Assigned", f'{result["assigned_count"]} / {n_devices}'),
            metric_card("Total Allocated Profit", fmt_dollar(result["total_profit"])),
            metric_card("Blocks Available", str(len(block_df))),
        ], className="metric-grid"),
        render_financial_controls(label_key, context),
    ], className="result-section")


# ── Coverage Planning ─────────────────────────────────────────────────────────

def compute_coverage(df, yield_pct):
    """
    Aggregate by Location (+Cropping System), sort descending Yield Kg,
    walk cumulative sum until yield_pct threshold met.
    Returns (display_df, selected_locs, metrics).
    """
    if df.empty or "Yield Kg" not in df.columns or "Location" not in df.columns:
        return pd.DataFrame(), [], {}

    group_cols = ["Location"]
    if "Cropping System" in df.columns:
        group_cols.append("Cropping System")

    agg = {"Yield Kg": "sum"}
    if "Variety Area (ha)" in df.columns: agg["Variety Area (ha)"] = "sum"
    if "Pick Event Number" in df.columns: agg["Pick Event Number"] = "count"
    if "Product Variety" in df.columns:
        agg["Product Variety"] = lambda x: ", ".join(sorted(x.dropna().astype(str).unique()))
    if "Plant" in df.columns:
        agg["Plant"] = lambda x: ", ".join(sorted(x.dropna().astype(str).unique()))

    agg_df = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    agg_df = agg_df.sort_values("Yield Kg", ascending=False).reset_index(drop=True)

    total_yield = agg_df["Yield Kg"].sum()
    total_ha = agg_df["Variety Area (ha)"].sum() if "Variety Area (ha)" in agg_df.columns else 0

    agg_df["Yield Contribution %"] = (agg_df["Yield Kg"] / total_yield * 100).round(2) if total_yield > 0 else 0
    agg_df["Cumulative Yield %"] = agg_df["Yield Contribution %"].cumsum().round(2)
    if "Variety Area (ha)" in agg_df.columns and total_ha > 0:
        agg_df["Cumulative Ha %"] = (agg_df["Variety Area (ha)"].cumsum() / total_ha * 100).round(2)

    agg_df.insert(0, "Rank", agg_df.index + 1)

    # Include rows where the running total before this row is strictly below the threshold,
    # OR yield_pct is 100 (select everything). The <= ensures the last location is
    # included when cumulative lands exactly on the threshold.
    prev_cumulative = agg_df["Cumulative Yield %"] - agg_df["Yield Contribution %"]
    mask = prev_cumulative < yield_pct
    # Edge case: if yield_pct == 100, include all rows regardless of float rounding
    if yield_pct >= 100:
        mask = pd.Series([True] * len(agg_df), index=agg_df.index)
    selected_df = agg_df[mask]
    selected_locs = selected_df["Location"].dropna().astype(str).unique().tolist()

    covered_yield = selected_df["Yield Kg"].sum()
    covered_ha = selected_df["Variety Area (ha)"].sum() if "Variety Area (ha)" in selected_df.columns else 0
    actual_pct = (covered_yield / total_yield * 100) if total_yield > 0 else 0
    actual_ha_pct = (covered_ha / total_ha * 100) if total_ha > 0 else 0

    rename = {
        "Pick Event Number": "Pick Events", "Variety Area (ha)": "Total Ha",
        "Product Variety": "Varieties", "Plant": "Plants",
    }
    display_df = agg_df.rename(columns=rename)

    metrics = {
        "total_locations": agg_df["Location"].dropna().astype(str).nunique(), "selected_locations": len(selected_locs),
        "total_yield": total_yield, "covered_yield": covered_yield, "actual_pct": actual_pct,
        "total_ha": total_ha, "covered_ha": covered_ha, "actual_ha_pct": actual_ha_pct,
    }
    return display_df, selected_locs, metrics


def compute_dimension_coverage(df, dimension_col, yield_pct):
    """
    Generalised version of compute_coverage's ranking logic for an arbitrary
    dimension column (Plant / Division / Product Variety / Cropping System /
    Location). Ranks distinct values of dimension_col by total Yield Kg
    (descending) and walks the cumulative sum until yield_pct is reached.
    Returns (ranked_df, selected_values, actual_pct).
    """
    if df.empty or dimension_col not in df.columns or "Yield Kg" not in df.columns:
        return pd.DataFrame(), [], 0.0
    agg = df.groupby(dimension_col, dropna=False)["Yield Kg"].sum().reset_index()
    agg = agg.sort_values("Yield Kg", ascending=False).reset_index(drop=True)
    total = agg["Yield Kg"].sum()
    if total <= 0:
        values = agg[dimension_col].dropna().astype(str).tolist()
        return agg, values, 0.0
    agg["Contribution %"] = agg["Yield Kg"] / total * 100
    agg["Cumulative %"] = agg["Contribution %"].cumsum()
    prev_cumulative = agg["Cumulative %"] - agg["Contribution %"]
    if yield_pct >= 100:
        mask = pd.Series([True] * len(agg), index=agg.index)
    else:
        mask = prev_cumulative < yield_pct
    selected = agg[mask]
    selected_values = selected[dimension_col].dropna().astype(str).tolist()
    covered = selected["Yield Kg"].sum()
    actual_pct = (covered / total * 100) if total > 0 else 0.0
    return agg, selected_values, actual_pct


# ── Minimum devices for coverage locations (capacity-aware) ──────────────────

def compute_min_devices(raw_df, selected_locs, machine_hrs_per_day, speed):
    """
    Capacity-aware machine allocation for coverage-selected locations.

    Logic:
      1. Aggregate raw pick-event data by Location+CroppingSystem to get:
           - Variety Area (ha) per block
           - Set of pick dates per block
           - Total Yield Kg per block (for sorting)
      2. Calculate hours_needed per block:
           hours_needed = (ha × ROW_LENGTH) / speed
      3. Calculate machines_needed per block (parallel machines for large blocks):
           machines_needed = ceil(hours_needed / machine_hrs_per_day)
           hours_per_machine = hours_needed / machines_needed
      4. Maintain a pool of machines. Each machine tracks:
           - remaining_hours: dict of {date: remaining_hrs} (resets to
             machine_hrs_per_day for dates not yet seen)
      5. For each block (highest Yield Kg first):
           - Need machines_needed machines that ALL have enough remaining hours
             on ALL of this block's pick dates
           - Find machines_needed existing machines that satisfy this
           - For any shortfall, open new machines
           - Deduct hours_per_machine from each assigned machine on each pick date
      6. Peak machines = max machines open at any point = len(machine_pool)
         (since machines are never closed — they may be reused across blocks
          on different dates)

    Returns (peak_machines, breakdown_list).
    """
    import math

    if raw_df.empty or not selected_locs or "Location" not in raw_df.columns:
        return None, []
    if machine_hrs_per_day <= 0 or speed <= 0:
        return None, []

    group_cols = ["Location"]
    if "Cropping System" in raw_df.columns:
        group_cols.append("Cropping System")

    subset = raw_df[raw_df["Location"].astype(str).isin([str(l) for l in selected_locs])]
    if subset.empty:
        return None, []

    def safe_date_set(values):
        dates = pd.to_datetime(values, errors="coerce")
        return {d.date().isoformat() for d in dates.dropna()}

    agg = {"Yield Kg": "sum"}
    if "Variety Area (ha)" in subset.columns:
        agg["Variety Area (ha)"] = "sum"
    if "Pick Date" in subset.columns:
        agg["Pick Date"] = safe_date_set

    block_df = (
        subset.groupby(group_cols, dropna=False)
        .agg(agg)
        .reset_index()
        .sort_values("Yield Kg", ascending=False)
        .reset_index(drop=True)
    )

    # Build block records with hours_needed and machines_needed
    records = []
    for _, row in block_df.iterrows():
        raw_dates = row.get("Pick Date", set())
        if isinstance(raw_dates, set): dates = raw_dates
        elif isinstance(raw_dates, (list, tuple)): dates = set(raw_dates)
        else: dates = set()

        ha = float(row.get("Variety Area (ha)", 0) or 0)
        hours_total = (ha * ROW_LENGTH) / speed if speed > 0 else 0
        n_dates = max(len(dates), 1)  # avoid divide by zero
        hours_per_visit = hours_total / n_dates
        machines_needed = max(1, math.ceil(hours_per_visit / machine_hrs_per_day)) if hours_per_visit > 0 else 1
        hours_per_machine = hours_per_visit / machines_needed if machines_needed > 0 else 0

        label = str(row["Location"])
        cs = str(row.get("Cropping System", "") or "")
        if cs: label += f" ({cs})"

        records.append({
            "label": label,
            "location": str(row["Location"]),
            "cropping_system": cs,
            "yield_kg": float(row.get("Yield Kg", 0) or 0),
            "ha": ha,
            "hours_total": round(hours_total, 2),
            "hours_per_visit": round(hours_per_visit, 2),
            "machines_needed": machines_needed,
            "hours_per_machine": round(hours_per_machine, 2),
            "dates": dates,
            "n_dates": n_dates,
        })

    # Machine pool: each machine is a dict of {date: remaining_hours}
    # A machine has full capacity on any date not yet in its dict
    machine_pool = []  # list of {date: remaining_hours}
    breakdown = []     # one entry per block showing assignment

    for rec in records:
        dates = rec["dates"]
        needed = rec["machines_needed"]
        hpm = rec["hours_per_machine"]  # hours each assigned machine must contribute

        # Find existing machines that can handle this block:
        # must have >= hpm remaining on ALL of this block's pick dates
        def machine_can_take(m):
            for d in dates:
                remaining = m.get(d, machine_hrs_per_day)
                if remaining < hpm:
                    return False
            return True

        eligible = [m for m in machine_pool if machine_can_take(m)]

        # Take up to needed from eligible; open new ones for shortfall
        assigned_machines = eligible[:needed]
        shortfall = needed - len(assigned_machines)
        for _ in range(shortfall):
            new_machine = {}
            machine_pool.append(new_machine)
            assigned_machines.append(new_machine)

        # Deduct hours from each assigned machine on each pick date
        for m in assigned_machines:
            for d in dates:
                current = m.get(d, machine_hrs_per_day)
                m[d] = round(current - hpm, 4)

        machine_idx = [machine_pool.index(m) + 1 for m in assigned_machines]
        breakdown.append({
            "label": rec["label"],
            "yield_kg": rec["yield_kg"],
            "ha": rec["ha"],
            "hours_total": rec["hours_total"],
            "hours_per_visit": rec["hours_per_visit"],
            "machines_needed": rec["machines_needed"],
            "hours_per_machine": rec["hours_per_machine"],
            "n_dates": rec["n_dates"],
            "assigned_machine_ids": machine_idx,
        })

    peak_machines = len(machine_pool)
    return peak_machines, breakdown


def render_results_panel():
    return html.Div([
        html.H2("Harvest Results"),
        alert("Change sidebar filters freely. Calculations run only when you click Apply Filters & Run.", "info"),
        html.Button("Apply Filters & Run", id="apply-filters", n_clicks=0, className="primary-button"),
        html.Div(id="run-status"),
        html.Div(id="harvest-output"),
    ])


def render_optimiser_panel():
    return html.Div([
        html.H2("Block Optimiser & Financial Analysis"),
        html.Div([
            numeric_input("Number of Devices", "n-devices", 1, 1, 1),
            html.Label([
                html.Span("Group analysis by"),
                dcc.Dropdown(id="optimiser-group-dim",
                             options=[{"label": "None (all data)", "value": "None (all data)"},
                                      {"label": "Plant", "value": "Plant"}],
                             value="None (all data)", clearable=False),
            ], className="field-label light"),
            html.Label([
                html.Span("Selected groups"),
                dcc.Dropdown(id="optimiser-selected-groups", options=[], value=[], multi=True,
                             placeholder="Used when grouping is selected"),
            ], className="field-label light"),
            html.Button("Run Optimiser", id="run-optimiser", n_clicks=0, className="primary-button"),
        ], className="control-panel"),
        html.Div(id="optimiser-status"),
        html.Div(id="optimiser-output"),
    ])


def render_coverage_panel():
    return html.Div([
        html.H2("Coverage Planning"),
        alert(
            "Coverage is calculated within your current date range. Move the slider to rank Plant, "
            "Division, Variety, Cropping System and Location by yield and select the minimum set needed "
            "to reach your threshold — every filter above updates live. You can also edit any of those "
            "filters directly: the slider will snap to show the actual % of yield your manual selection "
            "covers.",
            "info",
        ),
        html.Div([
            html.Div([
                html.Span("Yield Coverage Target", className="coverage-slider-label"),
                html.Span([
                    html.Span("Target: ", className="coverage-pct-sublabel"),
                    html.Span(id="coverage-pct-label", children="100%"),
                ], className="coverage-pct-badge"),
                html.Span([
                    html.Span("Actual: ", className="coverage-pct-sublabel"),
                    html.Span(id="coverage-actual-pct-label", children="—%"),
                ], className="coverage-pct-badge actual"),
            ], className="coverage-slider-row"),
            dcc.Slider(id="coverage-yield-slider", min=10, max=100, step=5, value=100,
                       marks={v: f"{v}%" for v in [10, 25, 50, 75, 90, 100]},
                       tooltip={"always_visible": False}),
            html.Div([
                html.Button("✕  Reset to All Locations", id="coverage-reset-btn",
                            n_clicks=0, className="coverage-reset-btn"),
                html.Div(id="coverage-applied-status"),
            ], className="coverage-apply-row"),
        ], className="coverage-controls"),
        html.Div(id="coverage-metrics"),
        html.Div(id="coverage-table-output"),
    ])


# ─────────────────────────────────────────────────────────────────────────────

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&display=swap"
]

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
app.title = "Rubus Picking KPI"

app.layout = html.Div([
    dcc.Store(id="startup-trigger", data=1),
    dcc.Store(id="raw-data-store"),
    dcc.Store(id="raw-meta-store"),
    dcc.Store(id="calc-result-store"),
    dcc.Store(id="optimiser-result-store"),
    html.Div([
        html.Div([
            html.H2("Rubus KPI"),
            html.Hr(),
            html.H3("Data"),
            html.Div(id="upload-status", className="sidebar-status"),
            html.Hr(),
            html.H3("Harvest Inputs"),
            numeric_input("Pickers", "pickers", 5, 1, 0),
            numeric_input("Supervisors", "supervisors", 1, 1, 0),
            numeric_input("Picker Cost ($/hr)", "picker-cost", 32.0, 0.5, 0),
            numeric_input("Supervisor Cost ($/hr)", "supervisor-cost", 36.0, 0.5, 0),
            numeric_input("Max Pick Rate (kg/hr/ppl)", "max-pickrate", 8.0, 0.5, 0.1),
            html.Label([
                html.Span("Speed Range (m/hr)"),
                dcc.RangeSlider(id="speed-range", min=min(SPEEDS), max=max(SPEEDS), step=None,
                                marks={speed: str(speed) for speed in SPEEDS}, value=[400, 600]),
            ], className="slider-field"),
            numeric_input("Machine capacity (hrs/day)", "machine-hours", 8.0, 0.5, 0.5),
            html.Hr(),
            html.H3("Date Filter"),
            dcc.DatePickerRange(id="date-range", display_format="YYYY/MM/DD", disabled=True, className="date-picker"),
            html.Div(id="date-display", style={"marginTop": "8px", "background": "#245c3f", "borderRadius": "8px",
                                                "padding": "8px 12px", "fontSize": "13px", "color": "#a5d6a7",
                                                "lineHeight": "1.6", "display": "none"}),
            html.Hr(),
            html.H3("Filters"),
            filter_card("Plant", "plant-filter", "plant-all", "plant-none"),
            filter_card("Division", "division-filter", "division-all", "division-none"),
            filter_card("Location", "location-filter", "location-all", "location-none"),
            filter_card("Variety", "variety-filter", "variety-all", "variety-none"),
            filter_card("Cropping System", "cropping-filter", "cropping-all", "cropping-none"),
        ], className="sidebar"),
        html.Div([
            html.H1("Rubus Picking KPI - Gen 1 Calculator"),
            dcc.Tabs(id="tabs", value="tab-results", children=[
                dcc.Tab(label="Harvest Results", value="tab-results"),
                dcc.Tab(label="Block Optimiser & Financial Analysis", value="tab-optimiser"),
                dcc.Tab(label="Coverage Planning", value="tab-coverage"),
            ]),
            # All three panels are mounted permanently and toggled via CSS
            # display, rather than being created/destroyed on tab switch.
            # This keeps every component ID (coverage-yield-slider,
            # optimiser-group-dim, etc.) present in the DOM at all times, so
            # callbacks that reference them never hit a "nonexistent object"
            # error just because a different tab happens to be active.
            html.Div(id="panel-results", className="content-card", children=render_results_panel()),
            html.Div(id="panel-optimiser", className="content-card", style={"display": "none"},
                     children=render_optimiser_panel()),
            html.Div(id="panel-coverage", className="content-card", style={"display": "none"},
                     children=render_coverage_panel()),
        ], className="main"),
    ], className="app-shell"),
])

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body { margin: 0; font-family: 'DM Sans', Arial, sans-serif; background: #f8faf8; color: #1f2937; }
            h1, h2, h3 { font-family: 'DM Serif Display', Georgia, serif; letter-spacing: 0; color: #1a4731; }
            h1 { margin: 0 0 18px; font-size: 34px; }
            h2 { margin: 20px 0 12px; }
            h3 { margin: 18px 0 10px; }

            .app-shell { display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: 100vh; }
            .sidebar { background: #1a4731; color: #e8f5e9; padding: 22px; overflow-y: auto; max-height: 100vh; position: sticky; top: 0; }
            .sidebar h2 { color: #ffffff; }
            .sidebar h3 { font-family: 'DM Sans', Arial, sans-serif; font-size: 14px; text-transform: uppercase; letter-spacing: .08em; color: #a5d6a7; margin: 18px 0 10px; }
            .sidebar hr { border: 0; border-top: 1px solid #2d7a52; margin: 18px 0; }
            .main { padding: 28px; min-width: 0; }
            .content-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-top: 12px; }

            .upload-box { border: 2px dashed #a5d6a7; border-radius: 8px; padding: 18px; text-align: center; cursor: pointer; background: #245c3f; color: #fff; }
            .sidebar-status { margin-top: 10px; font-size: 13px; line-height: 1.4; color: #a5d6a7; }

            .field-label { display: block; margin: 12px 0; }
            .field-label-text { display: block; margin-bottom: 6px; font-weight: 600; color: #c8e6c9; font-size: 13px; }
            .stepper-wrap { display: grid; grid-template-columns: 36px 1fr 36px; border: 1px solid #a5d6a7; border-radius: 8px; overflow: hidden; background: #fff; }
            .stepper-btn { background: #245c3f !important; color: #fff !important; border: none !important; font-size: 20px !important; font-weight: 700 !important; height: 38px !important; cursor: pointer !important; line-height: 1 !important; display: flex !important; align-items: center !important; justify-content: center !important; padding: 0 !important; border-radius: 0 !important; }
            .stepper-btn:hover { background: #1a4731 !important; }
            .stepper-input { border: none !important; border-left: 1px solid #a5d6a7 !important; border-right: 1px solid #a5d6a7 !important; text-align: center !important; color: #111827 !important; -webkit-text-fill-color: #111827 !important; background: #fff !important; font-weight: 700 !important; font-size: 14px !important; height: 38px !important; width: 100% !important; padding: 0 4px !important; -moz-appearance: textfield !important; appearance: textfield !important; outline: none !important; }
            .stepper-input::-webkit-inner-spin-button, .stepper-input::-webkit-outer-spin-button { -webkit-appearance: none !important; display: none !important; }
            .slider-field { display: block; margin: 14px 0; }
            .slider-field span { display: block; margin-bottom: 6px; font-weight: 600; color: #c8e6c9; font-size: 13px; }

            .filter-card { margin-bottom: 14px; }
            .filter-label { font-size: 13px; font-weight: 700; margin-bottom: 6px; color: #c8e6c9; }
            .filter-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }

            .sidebar .Select-control { background: #ffffff !important; border: 1px solid #a5d6a7 !important; border-radius: 6px !important; min-height: 38px !important; cursor: pointer !important; }
            .sidebar .Select-placeholder { color: #9ca3af !important; -webkit-text-fill-color: #9ca3af !important; line-height: 36px !important; padding: 0 10px !important; }
            .sidebar .Select--multi .Select-value { background: #d1fae5 !important; border: 1px solid #6ee7b7 !important; border-radius: 4px !important; margin: 3px 2px !important; }
            .sidebar .Select--multi .Select-value-label { color: #065f46 !important; -webkit-text-fill-color: #065f46 !important; font-weight: 600 !important; font-size: 12px !important; padding: 2px 5px !important; }
            .sidebar .Select--multi .Select-value-icon { color: #065f46 !important; border-right: 1px solid #6ee7b7 !important; padding: 2px 5px !important; }
            .sidebar .Select--multi .Select-value-icon:hover { background: #a7f3d0 !important; }
            .sidebar .Select-input { background: transparent !important; }
            .sidebar .Select-input > input { color: #111827 !important; -webkit-text-fill-color: #111827 !important; background: transparent !important; }
            .sidebar .Select-arrow-zone { color: #a5d6a7 !important; padding-right: 8px !important; }
            .sidebar .Select-arrow { border-top-color: #a5d6a7 !important; }
            .sidebar .Select-clear-zone { color: #a5d6a7 !important; }
            .sidebar .Select-menu-outer { background: #ffffff !important; border: 1px solid #a5d6a7 !important; border-radius: 0 0 6px 6px !important; box-shadow: 0 6px 16px rgba(0,0,0,0.18) !important; z-index: 9999 !important; }
            .sidebar .Select-option { background: #ffffff !important; color: #111827 !important; -webkit-text-fill-color: #111827 !important; font-size: 13px !important; padding: 8px 12px !important; cursor: pointer !important; }
            .sidebar .Select-option.is-focused { background: #f0fdf4 !important; }
            .sidebar .Select-option.is-selected { background: #d1fae5 !important; color: #065f46 !important; -webkit-text-fill-color: #065f46 !important; font-weight: 700 !important; }

            .sidebar .DateRangePickerInput, .sidebar .DateRangePickerInput_withBorder { background: #ffffff !important; border: 1px solid #a5d6a7 !important; border-radius: 6px !important; width: 100% !important; display: flex !important; align-items: center !important; }
            .sidebar .DateInput { background: #ffffff !important; flex: 1 !important; }
            .sidebar .DateInput_input { color: #111827 !important; -webkit-text-fill-color: #111827 !important; background: #ffffff !important; font-weight: 600 !important; font-size: 13px !important; padding: 6px 8px !important; border: none !important; width: 100% !important; }
            .sidebar .DateInput_input::placeholder { color: #9ca3af !important; }
            .sidebar .DateRangePickerInput_arrow { color: #6b7280 !important; padding: 0 4px !important; }

            [class*="DateRangePicker_picker"] { background: #ffffff !important; z-index: 10000 !important; border-radius: 8px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.2) !important; border: 1px solid #e5e7eb !important; }
            [class*="DayPicker"], [class*="DayPicker_transitionContainer"], [class*="CalendarMonthGrid"], [class*="CalendarMonth_table"], [class*="CalendarMonth"]:not([class*="caption"]) { background: #ffffff !important; }
            [class*="CalendarMonth_caption"] strong, [class*="CalendarMonth_caption"] { color: #111827 !important; -webkit-text-fill-color: #111827 !important; font-size: 15px !important; }
            [class*="DayPicker_weekHeader_li"], [class*="DayPicker_weekHeader_li"] small { color: #6b7280 !important; -webkit-text-fill-color: #6b7280 !important; }
            [class*="CalendarDay__default"] { background: #ffffff !important; color: #111827 !important; -webkit-text-fill-color: #111827 !important; border: 1px solid #f3f4f6 !important; font-size: 13px !important; }
            [class*="CalendarDay__default"]:hover { background: #f0fdf4 !important; color: #1a4731 !important; -webkit-text-fill-color: #1a4731 !important; }
            [class*="CalendarDay__outside"] { color: #d1d5db !important; -webkit-text-fill-color: #d1d5db !important; }
            [class*="CalendarDay__selected"]:not([class*="span"]) { background: #7c3aed !important; border-color: #7c3aed !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; font-weight: 700 !important; }
            [class*="CalendarDay__selected_span"] { background: #ede9fe !important; border-color: #c4b5fd !important; color: #4c1d95 !important; -webkit-text-fill-color: #4c1d95 !important; }
            [class*="CalendarDay__hovered_span"] { background: #f5f3ff !important; color: #111827 !important; -webkit-text-fill-color: #111827 !important; }
            [class*="DayPickerNavigation_button"] { background: #ffffff !important; border: 1px solid #e5e7eb !important; border-radius: 50% !important; cursor: pointer !important; }
            [class*="DayPickerNavigation_button"]:hover { background: #f0fdf4 !important; }
            [class*="DayPickerNavigation_svg"] { fill: #374151 !important; }

            button, .button { border: 0; border-radius: 8px; padding: 9px 14px; background: #1a4731; color: #fff; font-weight: 700; cursor: pointer; font-family: inherit; }
            button:hover { background: #245c3f; }
            .mini-button { background: #2d7a52 !important; color: #ffffff !important; border-radius: 6px !important; padding: 6px 10px !important; font-size: 12px !important; }
            .mini-button.secondary { background: #4b5563 !important; color: #ffffff !important; }
            .primary-button { width: 100%; margin: 10px 0 12px; font-size: 16px; padding: 12px; }

            .alert { padding: 12px 14px; border-radius: 8px; margin: 10px 0; border: 1px solid transparent; font-size: 14px; }
            .alert-info { background: #eff6ff; border-color: #bfdbfe; color: #1e3a8a; }
            .alert-success { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
            .alert-warning { background: #fffbeb; border-color: #fde68a; color: #92400e; }

            .metric-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 14px 0; }
            .metric-grid.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
            .metric-grid.five { grid-template-columns: repeat(5, minmax(0, 1fr)); }
            .metric-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }
            .metric-card.highlight { background: #f0f7f0; border-color: #1a4731; }
            .metric-label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; font-weight: 700; }
            .metric-value { font-family: 'DM Serif Display', Georgia, serif; color: #1a4731; font-size: 26px; margin-top: 4px; }
            .metric-subtext, .muted { color: #6b7280; font-size: 13px; }
            .strong { font-weight: 700; }

            .summary-controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin: 8px 0 16px; }
            .field-label.light { display: block; margin: 0; }
            .field-label.light span { display: block; color: #111827; font-weight: 600; font-size: 13px; margin-bottom: 6px; }

            .scenario-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
            .scenario-card { border-radius: 12px; padding: 18px; border: 2px solid #d1d5db; }
            .scenario-best { background: #f0f7f0; border-color: #1a4731; }
            .scenario-optimal { background: #eff6ff; border-color: #1565c0; }
            .scenario-highest { background: #fffbeb; border-color: #f59e0b; }
            .scenario-title { display: flex; justify-content: space-between; gap: 8px; font-family: 'DM Serif Display', Georgia, serif; font-size: 18px; color: #111827; }
            .scenario-profit { font-size: 28px; color: #1a4731; font-weight: 700; margin: 8px 0; }
            .combo-block { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; margin: 6px 0; font-size: 13px; }
            .badge { padding: 4px 8px; border-radius: 999px; background: #d1fae5; color: #065f46; font: 700 11px 'DM Sans'; white-space: nowrap; }

            .device-card { background: #f0f7f0; border-left: 4px solid #1a4731; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; border-top: 1px solid #a5d6a7; border-right: 1px solid #a5d6a7; border-bottom: 1px solid #a5d6a7; }
            .device-card.warn { background: #fffbeb; border-color: #f59e0b; }

            .details-panel { margin: 14px 0; }
            .financial-panel { border-top: 1px solid #e5e7eb; margin-top: 18px; padding-top: 10px; }
            .fin-row { display: grid; grid-template-columns: 2.5fr 1.2fr 1fr 1.2fr 1.2fr; gap: 10px; align-items: center; padding: 7px 0; border-bottom: 1px solid #f3f4f6; }
            .fin-header { font-weight: 700; color: #1a4731; border-bottom: 2px solid #d1d5db; }
            .fin-input { width: 100%; border: 1px solid #d1d5db; border-radius: 6px; padding: 8px; font: inherit; color: #111827; background: #fff; }
            .fin-stepper-wrap { display: grid; grid-template-columns: 30px minmax(0, 1fr) 30px; align-items: stretch; }
            .fin-stepper-wrap .fin-input { border-radius: 0 !important; text-align: center; min-width: 0; }
            .fin-stepper-btn { background: #1a4731 !important; color: #ffffff !important; border: 1px solid #1a4731 !important; border-radius: 0 !important; padding: 0 !important; font-size: 16px !important; font-weight: 800 !important; height: 35px !important; line-height: 1 !important; }
            .fin-stepper-btn:first-child { border-radius: 6px 0 0 6px !important; }
            .fin-stepper-btn:last-child { border-radius: 0 6px 6px 0 !important; }
            .fin-calc { color: #111827; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px 10px; font-size: 13px; min-height: 35px; display: flex; align-items: center; }
            .fin-total { background: #f0f7f0; color: #1a4731; font-weight: 700; padding: 12px; border-radius: 8px; margin: 12px 0; }
            .result-section { padding-bottom: 26px; margin-bottom: 26px; border-bottom: 1px solid #e5e7eb; }

            /* ── COVERAGE PLANNING ── */
            .coverage-controls { background: #f0f7f0; border: 1px solid #c6e8d0; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px; }
            .coverage-slider-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 10px; flex-wrap: wrap; }
            .coverage-pct-sublabel { font-size: 11px; font-weight: 600; opacity: 0.85; text-transform: uppercase; letter-spacing: .04em; }
            .coverage-pct-badge.actual { background: #245c3f; border: 1px solid #6ee7b7; }
            .coverage-slider-label { font-size: 15px; font-weight: 700; color: #1a4731; }
            .coverage-pct-badge { background: #1a4731; color: #fff; padding: 5px 18px; border-radius: 999px; font-size: 20px; font-weight: 800; letter-spacing: 0.02em; }
            .coverage-apply-row { display: flex; gap: 12px; margin-top: 18px; align-items: center; flex-wrap: wrap; }
            .coverage-apply-btn { background: #1a4731 !important; color: #fff !important; padding: 10px 28px !important; font-size: 15px !important; border-radius: 8px !important; border: none !important; cursor: pointer !important; font-weight: 700 !important; }
            .coverage-apply-btn:hover { background: #245c3f !important; }
            .coverage-reset-btn { background: #6b7280 !important; color: #fff !important; padding: 10px 20px !important; font-size: 14px !important; border-radius: 8px !important; border: none !important; cursor: pointer !important; font-weight: 600 !important; }
            .coverage-applied-banner { background: #ecfdf5; border: 1px solid #6ee7b7; border-radius: 8px; padding: 10px 16px; color: #065f46; font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 10px; }
            .coverage-tag { display: inline-block; background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; border-radius: 4px; padding: 3px 9px; font-size: 12px; font-weight: 600; margin: 2px 3px; }
            .coverage-tags-wrap { margin-top: 14px; margin-bottom: 4px; }
            .coverage-tags-label { font-weight: 700; margin-bottom: 6px; font-size: 13px; color: #374151; }

            @media (max-width: 1100px) {
                .app-shell { grid-template-columns: 1fr; }
                .sidebar { position: static; max-height: none; }
                .scenario-grid, .metric-grid, .metric-grid.four, .metric-grid.five, .summary-controls { grid-template-columns: 1fr; }
                .fin-row { grid-template-columns: 1fr; }
                .coverage-apply-row { flex-direction: column; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

from dash import ALL

def _safe_float(val, default, min_val):
    try: return max(min_val, float(str(val).strip()))
    except (ValueError, TypeError): return default


@app.callback(Output("pickers", "value"),
    Input({"type": "stepper-inc", "target": "pickers"}, "n_clicks"),
    Input({"type": "stepper-dec", "target": "pickers"}, "n_clicks"),
    State("pickers", "value"), prevent_initial_call=True)
def _step_pickers(inc, dec, val):
    v = _safe_float(val, 5, 0)
    return str(int(v + 1 if callback_context.triggered_id.get("type") == "stepper-inc" else max(0, v - 1)))

@app.callback(Output("supervisors", "value"),
    Input({"type": "stepper-inc", "target": "supervisors"}, "n_clicks"),
    Input({"type": "stepper-dec", "target": "supervisors"}, "n_clicks"),
    State("supervisors", "value"), prevent_initial_call=True)
def _step_supervisors(inc, dec, val):
    v = _safe_float(val, 1, 0)
    return str(int(v + 1 if callback_context.triggered_id.get("type") == "stepper-inc" else max(0, v - 1)))

@app.callback(Output("picker-cost", "value"),
    Input({"type": "stepper-inc", "target": "picker-cost"}, "n_clicks"),
    Input({"type": "stepper-dec", "target": "picker-cost"}, "n_clicks"),
    State("picker-cost", "value"), prevent_initial_call=True)
def _step_picker_cost(inc, dec, val):
    v = _safe_float(val, 32.0, 0)
    return str(round(v + 0.5 if callback_context.triggered_id.get("type") == "stepper-inc" else max(0, v - 0.5), 2))

@app.callback(Output("supervisor-cost", "value"),
    Input({"type": "stepper-inc", "target": "supervisor-cost"}, "n_clicks"),
    Input({"type": "stepper-dec", "target": "supervisor-cost"}, "n_clicks"),
    State("supervisor-cost", "value"), prevent_initial_call=True)
def _step_supervisor_cost(inc, dec, val):
    v = _safe_float(val, 36.0, 0)
    return str(round(v + 0.5 if callback_context.triggered_id.get("type") == "stepper-inc" else max(0, v - 0.5), 2))

@app.callback(Output("max-pickrate", "value"),
    Input({"type": "stepper-inc", "target": "max-pickrate"}, "n_clicks"),
    Input({"type": "stepper-dec", "target": "max-pickrate"}, "n_clicks"),
    State("max-pickrate", "value"), prevent_initial_call=True)
def _step_max_pickrate(inc, dec, val):
    v = _safe_float(val, 8.0, 0.1)
    return str(round(v + 0.5 if callback_context.triggered_id.get("type") == "stepper-inc" else max(0.1, v - 0.5), 2))

@app.callback(Output("machine-hours", "value"),
    Input({"type": "stepper-inc", "target": "machine-hours"}, "n_clicks"),
    Input({"type": "stepper-dec", "target": "machine-hours"}, "n_clicks"),
    State("machine-hours", "value"), prevent_initial_call=True)
def _step_machine_hours(inc, dec, val):
    v = _safe_float(val, 8.0, 0.5)
    return str(round(v + 0.5 if callback_context.triggered_id.get("type") == "stepper-inc" else max(0.5, v - 0.5), 2))

@app.callback(Output("n-devices", "value"),
    Input({"type": "stepper-inc", "target": "n-devices"}, "n_clicks"),
    Input({"type": "stepper-dec", "target": "n-devices"}, "n_clicks"),
    State("n-devices", "value"), prevent_initial_call=True)
def _step_n_devices(inc, dec, val):
    v = _safe_float(val, 1, 1)
    return str(int(v + 1 if callback_context.triggered_id.get("type") == "stepper-inc" else max(1, v - 1)))

@app.callback(
    Output({"type": "fin-input", "label": MATCH, "field": MATCH}, "value"),
    Input({"type": "fin-stepper-inc", "label": MATCH, "field": MATCH}, "n_clicks"),
    Input({"type": "fin-stepper-dec", "label": MATCH, "field": MATCH}, "n_clicks"),
    State({"type": "fin-input", "label": MATCH, "field": MATCH}, "value"),
    prevent_initial_call=True,
)
def _step_financial_input(inc, dec, val):
    trigger = callback_context.triggered_id or {}
    field = trigger.get("field", "")
    step = fin_step_for_field(field)
    min_value = 0.1 if field.endswith("_life") else 0
    value = _safe_float(val, min_value, min_value)
    if trigger.get("type") == "fin-stepper-inc": value += step
    else: value = max(min_value, value - step)
    return str(int(round(value))) if field.endswith("_life") else str(round(value, 2))


@app.callback(
    Output("raw-data-store", "data"), Output("raw-meta-store", "data"),
    Output("upload-status", "children"), Output("date-range", "min_date_allowed"),
    Output("date-range", "max_date_allowed"), Output("date-range", "start_date"),
    Output("date-range", "end_date"), Output("date-range", "disabled"),
    Input("startup-trigger", "data"),
)
def load_file(trigger):
    data, message, meta = load_data_file()
    if not data:
        return None, None, alert(message, "warning"), None, None, None, None, True
    return (data, meta, alert(message, "success"),
            meta["min_date"], meta["max_date"], meta["min_date"], meta["max_date"], False)


def cascade_options(raw_data, start_date, end_date, parent_values):
    df = df_from_store(raw_data)
    if df.empty: return pd.DataFrame()
    return apply_filters(df, start_date, end_date, parent_values)


def resolve_filter_value(options, current_value, all_clicks, none_clicks):
    triggered = callback_context.triggered or []
    triggered_ids = {t["prop_id"].split(".")[0] for t in triggered}
    available = all_values(options)
    if any(tid.endswith("-none") for tid in triggered_ids): return []
    if any(tid.endswith("-all") for tid in triggered_ids): return available
    if "raw-data-store" in triggered_ids: return available
    if current_value is None: return available
    if current_value == []: return []
    kept = [v for v in current_value if v in available]
    return kept if kept else (available if available else [])


# ── Single engine owning OPTIONS + VALUE of the slider and all 5 filters ─────
# This has to be one callback: Dash rejects a callback graph where the slider
# and a filter each write to something the other reads (a real cross-callback
# cycle). Within a SINGLE callback, the same component/prop can be both an
# Input and an Output (a "self-loop"), which is the only Dash-legal way to
# get true two-way sync between the slider and the filters.
#
# Options and value are ALSO combined into this one callback (rather than
# options living in their own separate callbacks) because splitting them
# causes a real race: when options and value for the same dropdown are set
# by two different callbacks, their responses can arrive at the browser out
# of order. A dropdown that receives its new `value` before its matching
# `options` will treat those values as invalid, silently clear itself to
# `[]`, and report that clearing back to the server as a genuine user edit --
# which then looks identical to a deliberate "None" click and sticks
# permanently. Returning options and value together in one response makes
# them atomic, so this can't happen.

@app.callback(
    Output("coverage-yield-slider", "value"),
    Output("plant-filter", "options"), Output("plant-filter", "value"),
    Output("division-filter", "options"), Output("division-filter", "value"),
    Output("variety-filter", "options"), Output("variety-filter", "value"),
    Output("cropping-filter", "options"), Output("cropping-filter", "value"),
    Output("location-filter", "options"), Output("location-filter", "value"),
    Output("coverage-applied-status", "children"),
    Input("coverage-yield-slider", "value"),
    Input("plant-filter", "value"),
    Input("division-filter", "value"),
    Input("variety-filter", "value"),
    Input("cropping-filter", "value"),
    Input("location-filter", "value"),
    Input("plant-all", "n_clicks"), Input("plant-none", "n_clicks"),
    Input("division-all", "n_clicks"), Input("division-none", "n_clicks"),
    Input("variety-all", "n_clicks"), Input("variety-none", "n_clicks"),
    Input("cropping-all", "n_clicks"), Input("cropping-none", "n_clicks"),
    Input("location-all", "n_clicks"), Input("location-none", "n_clicks"),
    Input("coverage-reset-btn", "n_clicks"),
    Input("raw-data-store", "data"),
    Input("date-range", "start_date"), Input("date-range", "end_date"),
    State("tabs", "value"),
    prevent_initial_call=True,
)
def filter_and_coverage_engine(slider_val, plant_val, division_val, variety_val, cropping_val, location_val,
                               plant_all_c, plant_none_c, division_all_c, division_none_c,
                               variety_all_c, variety_none_c, cropping_all_c, cropping_none_c,
                               location_all_c, location_none_c, reset_clicks,
                               raw_data, start_date, end_date, active_tab):
    triggered_ids = {t["prop_id"].split(".")[0] for t in (callback_context.triggered or [])}
    NU = no_update

    def opts(col, restrict):
        return options_for(cascade_options(raw_data, start_date, end_date, restrict), col)

    if "coverage-reset-btn" in triggered_ids:
        p_opts = opts("Plant", {}); p_val = all_values(p_opts)
        d_opts = opts("Division", {}); d_val = all_values(d_opts)
        v_opts = opts("Product Variety", {}); v_val = all_values(v_opts)
        c_opts = opts("Cropping System", {}); c_val = all_values(c_opts)
        l_opts = opts("Location", {}); l_val = all_values(l_opts)
        banner = alert("Reset — slider and all filters restored to full coverage.", "info")
        return (100, p_opts, p_val, d_opts, d_val, v_opts, v_val, c_opts, c_val, l_opts, l_val, banner)

    if not raw_data:
        return (NU, NU, NU, NU, NU, NU, NU, NU, NU, NU, NU, NU)

    # ---- Ordinary cascade / All / None / kept-selection behaviour, exactly
    # as before, applied on every tab (Harvest Results / Optimiser both rely
    # on this too, since the sidebar filters are shared across tabs). ----
    plant_opts = opts("Plant", {})
    plant_resolved = resolve_filter_value(plant_opts, plant_val, plant_all_c, plant_none_c)

    division_opts = opts("Division", {"Plant": plant_resolved})
    division_resolved = resolve_filter_value(division_opts, division_val, division_all_c, division_none_c)

    variety_opts = opts("Product Variety", {"Plant": plant_resolved, "Division": division_resolved})
    variety_resolved = resolve_filter_value(variety_opts, variety_val, variety_all_c, variety_none_c)

    cropping_opts = opts("Cropping System",
                         {"Plant": plant_resolved, "Division": division_resolved, "Product Variety": variety_resolved})
    cropping_resolved = resolve_filter_value(cropping_opts, cropping_val, cropping_all_c, cropping_none_c)

    location_opts = opts("Location", {"Plant": plant_resolved, "Division": division_resolved})
    location_resolved = resolve_filter_value(location_opts, location_val, location_all_c, location_none_c)

    if active_tab != "tab-coverage":
        return (NU, plant_opts, plant_resolved, division_opts, division_resolved,
                variety_opts, variety_resolved, cropping_opts, cropping_resolved,
                location_opts, location_resolved, NU)

    # ---- On the Coverage tab: layer the yield-based engine on top ----
    df = df_from_store(raw_data)
    if df.empty:
        return (NU, plant_opts, plant_resolved, division_opts, division_resolved,
                variety_opts, variety_resolved, cropping_opts, cropping_resolved,
                location_opts, location_resolved, alert("Upload a dataset to use Coverage Planning.", "info"))
    if start_date and end_date:
        df = df[(df["Pick Date"].dt.date >= pd.to_datetime(start_date).date())
                & (df["Pick Date"].dt.date <= pd.to_datetime(end_date).date())]

    # 1) Slider dragged -> rank Locations against the FULL universe, then
    #    derive which Plant/Division/Variety/Cropping values are present
    #    among the winning locations. (Ranking each dimension separately at
    #    the same % compounds -- 60% at every stage nets ~50% overall -- so
    #    we deliberately don't do that.)
    if "coverage-yield-slider" in triggered_ids:
        target = slider_val if slider_val is not None else 100
        _, selected_locs, actual = compute_dimension_coverage(df, "Location", target)
        subset = df[df["Location"].astype(str).isin(selected_locs)]
        new_plants = sorted(subset["Plant"].dropna().astype(str).unique().tolist())
        new_divisions = sorted(subset["Division"].dropna().astype(str).unique().tolist())
        new_varieties = sorted(subset["Product Variety"].dropna().astype(str).unique().tolist())
        new_croppings = sorted(subset["Cropping System"].dropna().astype(str).unique().tolist())

        p_opts2 = opts("Plant", {})
        d_opts2 = opts("Division", {"Plant": new_plants})
        v_opts2 = opts("Product Variety", {"Plant": new_plants, "Division": new_divisions})
        c_opts2 = opts("Cropping System",
                       {"Plant": new_plants, "Division": new_divisions, "Product Variety": new_varieties})
        l_opts2 = opts("Location", {"Plant": new_plants, "Division": new_divisions})

        banner = html.Div(f"✓  {len(selected_locs)} location(s) covering {actual:.1f}% of yield — "
                          f"Plant / Division / Variety / Cropping filters updated to match.",
                          className="coverage-applied-banner")
        return (NU, p_opts2, new_plants, d_opts2, new_divisions, v_opts2, new_varieties,
                c_opts2, new_croppings, l_opts2, selected_locs, banner)

    # 2) Location edited manually -> respect the exact pick, just snap the
    #    slider to whatever % it actually achieves.
    if triggered_ids & {"location-filter", "location-all", "location-none"}:
        universe_selections = {"Plant": plant_resolved, "Division": division_resolved,
                               "Product Variety": variety_resolved, "Cropping System": cropping_resolved}
        universe = apply_filters(df, None, None, universe_selections)
        total = universe["Yield Kg"].sum() if "Yield Kg" in universe.columns else 0
        covered = universe[universe["Location"].astype(str).isin(location_resolved)]["Yield Kg"].sum() \
            if "Yield Kg" in universe.columns else 0
        new_target = round((covered / total * 100) if total else 0, 1)
        banner = html.Div(f"✓  Slider synced to {new_target:.1f}% based on your Location selection.",
                          className="coverage-applied-banner")
        return (new_target, plant_opts, plant_resolved, division_opts, division_resolved,
                variety_opts, variety_resolved, cropping_opts, cropping_resolved,
                location_opts, location_resolved, banner)

    # 3) Plant / Division / Variety / Cropping edited manually (or raw data /
    #    date range changed) -> rank Locations within that narrower universe
    #    at the current slider target (as before), then snap the slider to
    #    the actual % achieved.
    universe_selections = {"Plant": plant_resolved, "Division": division_resolved,
                           "Product Variety": variety_resolved, "Cropping System": cropping_resolved}
    universe = apply_filters(df, None, None, universe_selections)
    if universe.empty:
        banner = alert("No data matches current filters (excluding Location).", "warning")
        return (NU, plant_opts, plant_resolved, division_opts, division_resolved,
                variety_opts, variety_resolved, cropping_opts, cropping_resolved,
                location_opts, location_resolved, banner)

    target = slider_val if slider_val is not None else 100
    _, ranked_locs, metrics = compute_coverage(universe, target)
    if not ranked_locs:
        banner = alert("No locations qualify at this threshold.", "warning")
        return (NU, plant_opts, plant_resolved, division_opts, division_resolved,
                variety_opts, variety_resolved, cropping_opts, cropping_resolved,
                location_opts, location_resolved, banner)

    actual_pct = round(metrics["actual_pct"], 1)
    banner = html.Div(f"✓  {metrics['selected_locations']} location(s) covering {actual_pct:.1f}% of yield — "
                      f"Location filter and slider updated.", className="coverage-applied-banner")
    return (actual_pct, plant_opts, plant_resolved, division_opts, division_resolved,
            variety_opts, variety_resolved, cropping_opts, cropping_resolved,
            location_opts, ranked_locs, banner)



@app.callback(
    Output("panel-results", "style"),
    Output("panel-optimiser", "style"),
    Output("panel-coverage", "style"),
    Input("tabs", "value"),
)
def toggle_tab_panels(tab):
    shown = {"display": "block"}
    hidden = {"display": "none"}
    return (
        shown if tab == "tab-results" else hidden,
        shown if tab == "tab-optimiser" else hidden,
        shown if tab == "tab-coverage" else hidden,
    )


# ── Coverage: live badge ──────────────────────────────────────────────────────
@app.callback(Output("coverage-pct-label", "children"), Input("coverage-yield-slider", "value"))
def update_coverage_label(val):
    return f"{val}%"


# ── Coverage: metrics + table — reads the Location filter back ("vice versa") ─
@app.callback(
    Output("coverage-metrics", "children"),
    Output("coverage-table-output", "children"),
    Output("coverage-actual-pct-label", "children"),
    Input("coverage-yield-slider", "value"),
    Input("raw-data-store", "data"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("plant-filter", "value"),
    Input("division-filter", "value"),
    Input("variety-filter", "value"),
    Input("cropping-filter", "value"),
    Input("location-filter", "value"),
    Input("calc-result-store", "data"),
    Input("machine-hours", "value"),
    Input("speed-range", "value"),
)
def update_coverage_table(yield_pct, raw_data, start_date, end_date,
                          plants, divisions, varieties, cropping_systems, locations,
                          calc_payload, machine_hours, speed_range):
    if not raw_data:
        return alert("Upload a dataset to use Coverage Planning.", "info"), html.Div(), "—%"

    df = df_from_store(raw_data)
    # Same universe as the sync callback: everything except Location. None
    # means "not resolved yet" (no restriction); [] means the user
    # deliberately picked nothing (zero rows) -- these must not be conflated.
    universe_selections = {}
    if plants is not None: universe_selections["Plant"] = plants
    if divisions is not None: universe_selections["Division"] = divisions
    if varieties is not None: universe_selections["Product Variety"] = varieties
    if cropping_systems is not None: universe_selections["Cropping System"] = cropping_systems
    df_filtered = apply_filters(df, start_date, end_date, universe_selections)

    if df_filtered.empty:
        return alert("No data matches current filters (excluding Location).", "warning"), html.Div(), "—%"

    display_df, ranked_locs, metrics = compute_coverage(df_filtered, yield_pct)
    if display_df.empty:
        return alert("Could not compute coverage — check Location and Yield Kg columns exist.", "warning"), html.Div(), "—%"

    # "vice versa": use whatever's actually selected in the Location filter
    # right now, intersected with what's available in the current universe.
    available_locs = set(display_df["Location"].astype(str))
    if locations:
        selected_locs = [l for l in locations if str(l) in available_locs]
        if not selected_locs:
            # Location filter cleared everything within this universe — fall
            # back to the slider's own ranking so the page isn't just empty.
            selected_locs = ranked_locs
    else:
        selected_locs = ranked_locs

    covered_yield = display_df[display_df["Location"].astype(str).isin(selected_locs)]["Yield Kg"].sum()
    total_yield = metrics["total_yield"]
    covered_ha = display_df[display_df["Location"].astype(str).isin(selected_locs)]["Total Ha"].sum() \
        if "Total Ha" in display_df.columns else 0
    total_ha = metrics["total_ha"]
    actual_pct = (covered_yield / total_yield * 100) if total_yield > 0 else 0
    actual_ha_pct = (covered_ha / total_ha * 100) if total_ha > 0 else 0
    n_sel, n_tot = len(selected_locs), metrics["total_locations"]

    # How many locations exist in the FULL dataset (date-range only, before
    # Plant/Division/Variety/Cropping narrow things down) -- this is what
    # lets us say "14 of 160 total locations match your current filters"
    # instead of leaving people to wonder why a small coverage set is being
    # measured against a small denominator.
    dataset_df = df
    if start_date and end_date:
        dataset_df = dataset_df[(dataset_df["Pick Date"].dt.date >= pd.to_datetime(start_date).date())
                                 & (dataset_df["Pick Date"].dt.date <= pd.to_datetime(end_date).date())]
    total_dataset_locations = dataset_df["Location"].dropna().astype(str).nunique() \
        if "Location" in dataset_df.columns else n_tot
    scope_banner = None
    if total_dataset_locations > n_tot:
        scope_banner = html.Div(
            f"ℹ️  Your Plant / Division / Variety / Cropping filters narrow the dataset to {n_tot} of "
            f"{total_dataset_locations} total locations — the numbers below are all relative to that {n_tot}.",
            className="alert alert-info", style={"marginBottom": "12px"},
        )

    # ── Machines required (capacity-aware) — against the actual selection ────
    try:
        mhpd = float(str(machine_hours).strip()) if machine_hours not in (None, "", "None") else 8.0
    except Exception:
        mhpd = 8.0
    speed_range = speed_range or [400, 600]
    mid_speed = (speed_range[0] + speed_range[1]) / 2

    if "Variety Area (ha)" in df_filtered.columns:
        min_devices, device_breakdown = compute_min_devices(df_filtered, selected_locs, mhpd, mid_speed)
    elif calc_payload:
        calc_df = df_from_store(calc_payload["df"])
        min_devices, device_breakdown = compute_min_devices(calc_df, selected_locs, mhpd, mid_speed)
    else:
        min_devices, device_breakdown = None, []

    if min_devices is not None:
        machines_value = str(min_devices)
        machines_sub = f"peak concurrent — {mhpd}hrs/day @ {mid_speed:.0f}m/hr"
        machines_highlight = True
    else:
        machines_value = "—"
        machines_sub = "upload dataset with Variety Area (ha)"
        machines_highlight = False

    metrics_row = html.Div([
        metric_card("Locations Selected", f"{n_sel} / {n_tot}",
                    f"{yield_pct}% target of the {n_tot}-location filtered universe", highlight=True),
        metric_card("Yield Covered", f"{actual_pct:.1f}%",
                    f"{covered_yield:,.0f} of {total_yield:,.0f} kg", highlight=True),
        metric_card("Ha Covered", f"{actual_ha_pct:.1f}%",
                    f"{covered_ha:,.1f} of {total_ha:,.1f} ha"),
        metric_card("Locations Excluded", str(n_tot - n_sel), "below threshold or manually deselected"),
        metric_card("Min. Machines Required", machines_value, machines_sub, highlight=machines_highlight),
    ], className="metric-grid five", style={"marginBottom": "16px"})

    # Build table with In Coverage marker (reflects actual Location filter selection)
    display_df2 = display_df.copy()
    in_mask = display_df2["Location"].astype(str).isin(selected_locs)
    display_df2["In Coverage"] = in_mask.map({True: "✓ Selected", False: "— Excluded"})

    cols = list(display_df2.columns)
    ordered = ["Rank", "In Coverage", "Location"]
    if "Cropping System" in cols: ordered.append("Cropping System")
    if "Plants" in cols: ordered.append("Plants")
    if "Varieties" in cols: ordered.append("Varieties")
    ordered += [c for c in ["Yield Kg", "Total Ha", "Pick Events",
                             "Yield Contribution %", "Cumulative Yield %", "Cumulative Ha %"] if c in cols]
    ordered += [c for c in cols if c not in ordered]
    display_df2 = display_df2[[c for c in ordered if c in display_df2.columns]]

    style_data_conditional = [{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}]
    for i, (_, row) in enumerate(display_df2.iterrows()):
        if row.get("In Coverage") == "✓ Selected":
            style_data_conditional.append({"if": {"row_index": i},
                                            "backgroundColor": "#f0fdf4", "borderLeft": "4px solid #1a4731"})
        else:
            style_data_conditional.append({"if": {"row_index": i},
                                            "backgroundColor": "#fff7ed", "color": "#9ca3af"})

    table = dash_table.DataTable(
        data=display_df2.to_dict("records"),
        columns=[{"name": col, "id": col} for col in display_df2.columns],
        page_size=25, sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "DM Sans, Arial, sans-serif", "fontSize": "12px", "padding": "8px",
                    "minWidth": "100px", "maxWidth": "260px", "whiteSpace": "normal"},
        style_header={"backgroundColor": "#f0f7f0", "fontWeight": "700", "color": "#1a4731"},
        style_data_conditional=style_data_conditional,
    )

    tags_section = html.Div([
        html.Div("Selected locations:", className="coverage-tags-label"),
        html.Div([html.Span(loc, className="coverage-tag") for loc in selected_locs]),
    ], className="coverage-tags-wrap") if selected_locs else html.Div()

    # Device breakdown collapsible
    if device_breakdown:
        breakdown_rows = []
        for d in device_breakdown:
            machine_ids = ", ".join(f"M{m}" for m in d["assigned_machine_ids"])
            breakdown_rows.append(html.Div([
                html.B(d["label"]),
                html.Span(f" — {d['ha']:.1f} ha, {d['hours_total']:.1f} hrs total, "
                          f"{d['hours_per_visit']:.1f} hrs/visit × {d['n_dates']} dates, "
                          f"{d['machines_needed']} machine(s) × {d['hours_per_machine']:.1f} hrs/visit",
                          className="muted"),
                html.Span(f" → {machine_ids}",
                          style={"fontSize": "12px", "color": "#1a4731", "fontWeight": "600"}),
            ], style={"padding": "6px 0", "borderBottom": "1px solid #f3f4f6"}))
        breakdown_section = html.Details([
            html.Summary(
                f"Machine breakdown ({min_devices} machine(s) required — "
                f"{mhpd}hrs/day @ {mid_speed:.0f}m/hr)",
                style={"fontWeight": "700", "cursor": "pointer", "color": "#1a4731",
                       "fontSize": "14px", "margin": "14px 0 8px"}),
            html.Div(breakdown_rows, style={"fontSize": "13px", "padding": "0 8px"}),
        ], style={"marginBottom": "16px"})
    else:
        breakdown_section = html.Div()

    return (html.Div([scope_banner, metrics_row]) if scope_banner else metrics_row,
            html.Div([tags_section, breakdown_section, html.H3("Location Yield Ranking"), table]),
            f"{actual_pct:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("calc-result-store", "data"), Output("run-status", "children"),
    Input("apply-filters", "n_clicks"),
    State("raw-data-store", "data"), State("date-range", "start_date"),
    State("date-range", "end_date"), State("plant-filter", "value"),
    State("division-filter", "value"), State("location-filter", "value"),
    State("variety-filter", "value"), State("cropping-filter", "value"),
    State("pickers", "value"), State("supervisors", "value"),
    State("picker-cost", "value"), State("supervisor-cost", "value"),
    State("max-pickrate", "value"), State("speed-range", "value"),
    prevent_initial_call=True,
)
def run_harvest(n_clicks, raw_data, start_date, end_date, plants, divisions, locations,
                varieties, cropping_systems, pickers, supervisors, picker_cost,
                supervisor_cost, max_pickrate, speed_range):
    if not raw_data: return None, alert("Upload a dataset first.", "warning")
    df = df_from_store(raw_data)
    selections = {"Plant": plants or [], "Division": divisions or [], "Location": locations or [],
                  "Product Variety": varieties or [], "Cropping System": cropping_systems or []}
    df_filtered = apply_filters(df, start_date, end_date, selections)
    if df_filtered.empty: return None, alert("No rows match the selected filters.", "warning")
    speed_range = speed_range or [400, 600]
    mid_speeds = [s for s in SPEEDS if speed_range[0] <= s <= speed_range[1]]
    mid_label = f"Best Profit ({speed_range[0]}-{speed_range[1]}m)"
    def _f(v, d=0.0):
        try: return float(str(v).strip()) if v not in (None, "", "None") else d
        except: return d
    total_pickers = int(_f(pickers, 5)) + int(_f(supervisors, 1))
    labour_cost = _f(pickers, 5) * _f(picker_cost, 32) + _f(supervisors, 1) * _f(supervisor_cost, 36)
    raw_calc_df = compute_pick_calcs(df_to_store(df_filtered), total_pickers, labour_cost,
                                     _f(max_pickrate, 8), tuple(mid_speeds))
    optimiser_df = raw_calc_df.copy()
    optimiser_df.rename(columns={"Best Profit Mid": mid_label}, inplace=True)
    calc_df = raw_calc_df.copy()
    calc_df.rename(columns={"Best Profit Mid": mid_label}, inplace=True)
    calc_df = rename_calc_cols(calc_df)
    calc_df = build_column_order(calc_df, mid_label)
    payload = {"df": df_to_store(optimiser_df), "df_display": df_to_store(calc_df),
               "mid_label": mid_label, "speed_range": speed_range, "run_id": n_clicks}
    return payload, alert(f"{len(calc_df):,} pick events loaded.", "success")


@app.callback(Output("harvest-output", "children"), Input("calc-result-store", "data"))
def render_harvest_output(calc_payload):
    if not calc_payload: return alert("Select filters and click Apply Filters & Run.", "info")
    df = df_from_store(calc_payload["df"])
    level_1_options = summary_group_options(df)
    level_2_options = summary_group_options(df, include_none=True)
    default_level_1 = "Location" if "Location" in [o["value"] for o in level_1_options] else (
        level_1_options[0]["value"] if level_1_options else None)
    level_2_values = [o["value"] for o in level_2_options]
    default_level_2 = "Product Variety" if "Product Variety" in level_2_values else "__none__"
    if default_level_2 == default_level_1: default_level_2 = "__none__"
    return html.Div([
        html.H3("Pick Event Results"),
        make_table(df, page_size=20),
        html.H3("Summary"),
        html.Div([
            html.Label([html.Span("Group by (Level 1)"),
                        dcc.Dropdown(id="summary-level-1", options=level_1_options,
                                     value=default_level_1, clearable=False)], className="field-label light"),
            html.Label([html.Span("Group by (Level 2)"),
                        dcc.Dropdown(id="summary-level-2", options=level_2_options,
                                     value=default_level_2, clearable=False)], className="field-label light"),
        ], className="summary-controls"),
        html.Div(id="summary-output"),
    ])


@app.callback(Output("summary-output", "children"),
    Input("calc-result-store", "data"), Input("summary-level-1", "value"), Input("summary-level-2", "value"))
def render_summary_output(calc_payload, level_1, level_2):
    if not calc_payload: return alert("Run Harvest Results first.", "info")
    df = df_from_store(calc_payload["df"])
    summary = build_summary_table(df, calc_payload["mid_label"], calc_payload["speed_range"], level_1, level_2)
    return make_table(summary, page_size=20)


@app.callback(
    Output("optimiser-selected-groups", "options"), Output("optimiser-selected-groups", "value"),
    Output("optimiser-selected-groups", "disabled"),
    Input("optimiser-group-dim", "value"), Input("calc-result-store", "data"),
)
def update_optimiser_groups(group_dim, calc_payload):
    if not calc_payload or group_dim == "None (all data)": return [], [], True
    df = df_from_store(calc_payload["df"])
    if group_dim not in df.columns: return [], [], True
    values = sorted(df[group_dim].dropna().astype(str).unique())
    return [{"label": v, "value": v} for v in values], values, False


@app.callback(
    Output("optimiser-result-store", "data"), Output("optimiser-status", "children"),
    Input("run-optimiser", "n_clicks"),
    State("calc-result-store", "data"), State("raw-data-store", "data"),
    State("n-devices", "value"), State("optimiser-group-dim", "value"),
    State("optimiser-selected-groups", "value"), State("date-range", "start_date"),
    State("date-range", "end_date"), State("plant-filter", "value"),
    State("division-filter", "value"), State("location-filter", "value"),
    State("variety-filter", "value"), State("cropping-filter", "value"),
    State("pickers", "value"), State("supervisors", "value"),
    State("picker-cost", "value"), State("supervisor-cost", "value"),
    State("max-pickrate", "value"), State("speed-range", "value"),
    prevent_initial_call=True,
)
def run_optimiser(n_clicks, calc_payload, raw_data, n_devices, group_dim, selected_groups,
                  start_date, end_date, plants, divisions, locations, varieties, cropping_systems,
                  pickers, supervisors, picker_cost, supervisor_cost, max_pickrate, speed_range):
    if not raw_data or not calc_payload: return None, alert("Upload a dataset and run Harvest Results first.", "warning")
    def _f(v, d=0.0):
        try: return float(str(v).strip()) if v not in (None, "", "None") else d
        except: return d
    speed_range = speed_range or [400, 600]
    mid_speeds = [s for s in [200, 400, 600, 800, 1000] if speed_range[0] <= s <= speed_range[1]]
    mid_label = f"Best Profit ({speed_range[0]}-{speed_range[1]}m)"
    total_pickers = int(_f(pickers, 5)) + int(_f(supervisors, 1))
    labour_cost = _f(pickers, 5) * _f(picker_cost, 32) + _f(supervisors, 1) * _f(supervisor_cost, 36)
    raw_df = df_from_store(raw_data)
    selections = {"Plant": plants or [], "Division": divisions or [], "Location": locations or [],
                  "Product Variety": varieties or [], "Cropping System": cropping_systems or []}
    df_filtered = apply_filters(raw_df, start_date, end_date, selections)
    if df_filtered.empty: return None, alert("No rows match the current filters.", "warning")
    raw_calc = compute_pick_calcs(df_to_store(df_filtered), total_pickers, labour_cost,
                                  _f(max_pickrate, 8), tuple(mid_speeds))
    raw_calc.rename(columns={"Best Profit Mid": mid_label}, inplace=True)
    df = raw_calc
    n_devices = max(1, int(_f(n_devices, 1)))
    results = []
    if group_dim == "None (all data)":
        results.append(build_optimiser_result(df, "All_Filtered_Data", n_devices, mid_label))
    else:
        if not selected_groups: return None, alert(f"Select at least one {group_dim} to analyse.", "warning")
        for gv in selected_groups:
            results.append(build_optimiser_result(df[df[group_dim].astype(str) == str(gv)],
                                                   f"{group_dim}_{gv}", n_devices, mid_label))
    return ({"results": results, "n_devices": n_devices, "mid_label": mid_label,
             "calc_run_id": calc_payload.get("run_id")},
            alert("Optimiser results updated.", "success"))


@app.callback(Output("optimiser-output", "children"),
    Input("optimiser-result-store", "data"), State("calc-result-store", "data"))
def render_optimiser_output(optimiser_payload, calc_payload):
    if not calc_payload: return alert("Run Harvest Results first.", "info")
    if not optimiser_payload: return alert("Choose optimiser settings and click Run Optimiser.", "info")
    if optimiser_payload.get("calc_run_id") != calc_payload.get("run_id"):
        return alert("Harvest results have changed. Click Run Optimiser to refresh Tab 2.", "warning")
    return html.Div([render_optimiser_result(r, optimiser_payload["n_devices"], optimiser_payload["mid_label"])
                     for r in optimiser_payload["results"]])


@app.callback(
    [Output({"type": "financial-output", "label": MATCH}, "children")]
    + [Output({"type": "fin-annual", "label": MATCH, "field": f}, "children") for f in FIN_ITEM_FIELDS]
    + [Output({"type": "fin-total", "label": MATCH, "field": f}, "children") for f in FIN_ITEM_FIELDS],
    Input({"type": "fin-input", "label": MATCH, "field": "chariot_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "chariot_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "trolley_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "trolley_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "scales_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "scales_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "hms_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "hms_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "logistics_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "logistics_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "overhead_pct"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "platform_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "platform_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "tablet_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "tablet_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "it_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "it_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "burro_std_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "burro_std_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "burro_sw_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "burro_sw_life"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "burro_maint_val"}, "value"),
    Input({"type": "fin-input", "label": MATCH, "field": "burro_maint_life"}, "value"),
    State({"type": "financial-context", "label": MATCH}, "data"),
)
def update_financial_output(
    chariot_val, chariot_life, trolley_val, trolley_life, scales_val, scales_life,
    hms_val, hms_life, logistics_val, logistics_life, overhead_pct,
    platform_val, platform_life, tablet_val, tablet_life, it_val, it_life,
    burro_std_val, burro_std_life, burro_sw_val, burro_sw_life, burro_maint_val, burro_maint_life, context,
):
    output_count = 1 + (len(FIN_ITEM_FIELDS) * 2)
    if not context: return tuple([no_update] * output_count)
    block_df = pd.DataFrame(context["block_records"])
    allocations = context["allocations"]
    assigned_count = context["assigned_count"]
    total_profit = context["total_profit"]
    if assigned_count == 0 or block_df.empty:
        zero_rows = [fmt_dollar(0) for _ in FIN_ITEM_FIELDS]
        return (alert("No devices assigned, so financial analysis is not available.", "warning"), *zero_rows, *zero_rows)
    allocated_blocks = [(item["location"], item["cropping_system"]) for item in allocations if item["assigned"]]
    allocated_rows = block_df[block_df.apply(
        lambda row: (row["Location"], row.get("Cropping System", "")) in allocated_blocks, axis=1)]
    max_pickers = allocated_rows["Distinct Picker Count"].sum() if "Distinct Picker Count" in allocated_rows.columns else 0
    harvest_cost = allocated_rows["Total Harvest Cost"].sum() if "Total Harvest Cost" in allocated_rows.columns else 0
    total_blocks = len(block_df)
    trolley_qty = max(1, round(max_pickers * 1.2))
    multipliers = {"chariot": assigned_count, "trolley": trolley_qty, "scales": trolley_qty, "hms": assigned_count,
                   "logistics": assigned_count / max(total_blocks, 1), "platform": assigned_count,
                   "tablet": assigned_count, "it": assigned_count, "burro_std": assigned_count,
                   "burro_sw": assigned_count, "burro_maint": assigned_count / max(total_blocks, 1)}
    financial_values = {"chariot": (chariot_val, chariot_life), "trolley": (trolley_val, trolley_life),
                        "scales": (scales_val, scales_life), "hms": (hms_val, hms_life),
                        "logistics": (logistics_val, logistics_life), "platform": (platform_val, platform_life),
                        "tablet": (tablet_val, tablet_life), "it": (it_val, it_life),
                        "burro_std": (burro_std_val, burro_std_life), "burro_sw": (burro_sw_val, burro_sw_life),
                        "burro_maint": (burro_maint_val, burro_maint_life)}
    row_details = {f: {"annual": annualised_value(v, l), "total": annualised_value(v, l) * multipliers[f]}
                   for f, (v, l) in financial_values.items()}
    savings = sum(row_details[f]["total"] for f in ["chariot", "trolley", "scales", "hms", "logistics"])
    overhead_total = harvest_cost * (parse_fin_number(overhead_pct, 0.0) / 100)
    total_savings = savings + overhead_total
    total_costs = sum(row_details[f]["total"] for f in ["platform", "tablet", "it", "burro_std", "burro_sw", "burro_maint"])
    total_benefit = total_profit + total_savings - total_costs
    status = (f"Deploying {assigned_count} device(s) generates {fmt_dollar(total_benefit)} total annual benefit."
              if total_benefit >= 0
              else f"Deploying {assigned_count} device(s) results in a net annual cost of {fmt_dollar(abs(total_benefit))}.")
    annual_outputs = [fmt_dollar(row_details[f]["annual"]) for f in FIN_ITEM_FIELDS]
    total_outputs = [fmt_dollar(row_details[f]["total"]) for f in FIN_ITEM_FIELDS]
    return (html.Div([
        html.Div(f"Total Equipment Savings: {fmt_dollar(total_savings)}", className="fin-total"),
        html.Div(f"Total Equipment Costs: {fmt_dollar(total_costs)}", className="fin-total"),
        html.Div([metric_card("Harvest Profit", fmt_dollar(total_profit)),
                  metric_card("Equipment Savings", fmt_dollar(total_savings)),
                  metric_card("Equipment Costs", fmt_dollar(total_costs)),
                  metric_card("Total Annual Benefit", fmt_dollar(total_benefit))], className="metric-grid four"),
        alert(status, "success" if total_benefit >= 0 else "warning"),
    ]), *annual_outputs, *total_outputs)


@app.callback(
    Output("date-display", "children"), Output("date-display", "style"),
    Input("date-range", "start_date"), Input("date-range", "end_date"),
)
def update_date_display(start_date, end_date):
    base_style = {"marginTop": "8px", "background": "#245c3f", "borderRadius": "8px",
                  "padding": "8px 12px", "fontSize": "13px", "color": "#a5d6a7", "lineHeight": "1.6"}
    if not start_date or not end_date:
        return "", {**base_style, "display": "none"}
    try:
        s = pd.to_datetime(start_date).strftime("%d %b %Y")
        e = pd.to_datetime(end_date).strftime("%d %b %Y")
        text = [html.Div("Selected Date Range", style={"fontWeight": "700", "color": "#e8f5e9"}),
                html.Div([html.Span("Start: "), html.B(s, style={"color": "#e8f5e9"})]),
                html.Div([html.Span("End: "), html.B(e, style={"color": "#e8f5e9"})])]
        return text, {**base_style, "display": "block"}
    except Exception:
        return "", {**base_style, "display": "none"}


if __name__ == "__main__":
    app.run(debug=True)
