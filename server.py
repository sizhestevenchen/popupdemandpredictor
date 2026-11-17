"""Local web application for Vendor Predictor. Run: python server.py."""
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO, BytesIO
from pathlib import Path
import json
import base64
import math
import mimetypes
import os
import webbrowser
from threading import Timer

import pandas as pd
import numpy as np
from predictor import PopupDemandModel, fetch_weather, prepare, estimate_attendance, similar_events_forecast
from friendly import build_template, friendly_columns
from signals import is_holiday_window
from signals import (CATEGORY_PLACE_TYPES, SocialStats, IntentSignals, social_score,
                     competitor_index, learn_hourly_profiles, guess_category)

ROOT = Path(__file__).resolve().parent
SAMPLE = (ROOT / "sample_history.csv").read_text()
DATA_FILE = ROOT / "my_events.csv"
REQUIRED = {"date", "product", "units_sold"}
LOG_COLUMNS = ["date", "event_type", "product", "category", "units_brought", "units_sold",
               "price", "unit_cost", "expected_attendance", "start_hour", "end_hour",
               "temp_c", "precip_mm", "social_index", "competitor_index"]
NUMERIC = set(LOG_COLUMNS) - {"date", "event_type", "product", "category"}
CATEGORIES = sorted(set(CATEGORY_PLACE_TYPES) | {"default", "food"})


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def normalize(df):
    df = friendly_columns(df)   # plain headings ("Sold", "Start time", "Temp (°F)") -> model columns
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization.")
    df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
    for c in LOG_COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[LOG_COLUMNS + [c for c in df.columns if c not in LOG_COLUMNS]]
    if df.empty:
        return df
    if df[list(REQUIRED)].isna().any().any():
        n_bad = int(df[list(REQUIRED)].isna().any(axis=1).sum())
        raise ValueError(f"{n_bad} row(s) are missing a Date, Product or Sold number. "
                         "Fill those in (or delete the rows) and import again.")
    df["date"] = pd.to_datetime(df["date"], errors="raise", format="mixed").dt.strftime("%Y-%m-%d")
    for c in ["product", "event_type", "category"]:
        df[c] = df[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c].dropna()).all():
            raise ValueError(f"{c} must be a finite number.")
        if c != "temp_c" and (df[c].dropna() < 0).any():
            raise ValueError(f"{c} cannot be negative.")
    for c, low, high in [("start_hour", 0, 23), ("end_hour", 1, 24)]:
        if not df[c].dropna().between(low, high).all():
            raise ValueError(f"{c} is outside the 24-hour clock.")
    if (df["end_hour"] <= df["start_hour"]).any():
        raise ValueError("Closing hour must be after opening hour.")
    return df


def read_csv(csv):
    return normalize(pd.read_csv(StringIO(csv))) if csv.strip() else pd.DataFrame(columns=LOG_COLUMNS)


def read_upload(file):
    raw = base64.b64decode(file["data"], validate=True)
    name = file["name"].lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(raw))
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(raw))
    raise ValueError("Import a CSV or Excel file.")


def imported_history(files, rows):
    frames = [pd.DataFrame(rows)] if rows else []
    for file in files:
        df = friendly_columns(read_upload(file)).dropna(how="all")
        if missing := REQUIRED - set(df.columns):
            names = {"date": "Date", "product": "Product", "units_sold": "Sold"}
            raise ValueError(f"{file['name']} needs columns: " + ", ".join(names[m] for m in sorted(missing))
                             + ". Download the Excel template to see the layout.")
        frames.append(df)
    return normalize(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=LOG_COLUMNS)


def saved_history():
    return read_csv(DATA_FILE.read_text(encoding="utf-8")) if DATA_FILE.exists() else pd.DataFrame(columns=LOG_COLUMNS)


def save_history(df):
    temp = DATA_FILE.with_suffix(".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, DATA_FILE)


def history_model(df, transactions=""):
    df = normalize(df)
    if missing := REQUIRED - set(df.columns):
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if df.empty or df[list(REQUIRED)].isna().any().any():
        raise ValueError("History must contain at least one complete sale record.")
    df = df.dropna(axis=1, how="all")
    if {"start_hour", "end_hour"} <= set(df.columns):
        duration = df["end_hour"] - df["start_hour"]
        df["hours"] = df["hours"].fillna(duration) if "hours" in df else duration
    df = prepare(df)
    learned = None
    if transactions:
        tx = pd.read_csv(StringIO(transactions))
        if not {"timestamp", "product", "qty"} <= set(tx.columns):
            raise ValueError("Hourly sales need timestamp, product and qty columns.")
        cats = df.groupby("product")["category"].first().to_dict() if "category" in df else {}
        learned = learn_hourly_profiles(tx, cats)
    return df, PopupDemandModel(learned_hourly=learned).fit(df)


def load_history(df, transactions=""):
    records = df.to_dict("records")
    if df.empty:
        return {"events": 0, "rows": 0, "units": 0, "sellouts": 0, "products": [],
                "event_types": [], "categories": CATEGORIES, "signals": {}, "columns": df.columns.tolist(),
                "records": records, "trend": []}
    df, model = history_model(df, transactions)
    def number(v):
        return float(v) if pd.notna(v) else None
    products = []
    for p in model.spec.products:
        group = df[df["product"] == p]
        pc = {key: group[key].median() for key in ("price", "unit_cost") if key in group}
        products.append({"product": p, "category": model.spec.category(p),
                         "price": number(pc.get("price")), "unit_cost": number(pc.get("unit_cost")),
                         "leftover_value": None, "same_stalls": None, "competitor_index": None})
    trend = df.groupby("date", as_index=False)["units_sold"].sum().sort_values("date").tail(12)
    return {"events": df.event_id.nunique(), "rows": len(df), "units": float(df.units_sold.sum()),
            "sellouts": model.n_censored, "products": products, "event_types": model.spec.event_types,
            "categories": CATEGORIES,
            "signals": model.has_signal, "columns": list(records[0]), 
            "records": records, "trend": trend.to_dict("records")}


def social_details(payload, history):
    inputs = payload.get("social", {})
    fields = {k: inputs.get(k) for k in SocialStats.__dataclass_fields__ if k != "platform"}
    custom = {}
    for row in payload.get("custom", []):
        if not row.get("name"):
            continue
        if row.get("value") is None or row.get("weight") is None:
            raise ValueError("Custom signals need a count and buyer weight.")
        custom[row["name"]] = (float(row["value"]), float(row["weight"]))
    if not any(v is not None for v in inputs.values()) and not custom:
        return {"social": None, "attendance": None}
    accounts = [SocialStats(**fields)] if any(v is not None for v in fields.values()) else []
    social = social_score(accounts, IntentSignals(
        rsvps=inputs.get("rsvps") or 0, preorders=inputs.get("preorders") or 0,
        promo_redemptions_last=inputs.get("promo_redemptions_last") or 0,
        email_sms_clicks=inputs.get("email_sms_clicks") or 0, custom=custom))
    prepared = prepare(history.dropna(axis=1, how="all")) if not history.empty else None
    est = estimate_attendance(social["social_index"], prepared) if social["social_index"] > 0 else None
    return {"social": social, "attendance": est}


def forecast(payload):
    df, model = history_model(pd.DataFrame(payload["rows"]), payload.get("transactions", ""))
    event = {k: v for k, v in payload["event"].items() if v is not None and v != ""}
    date.fromisoformat(event["date"])
    start, end = event.get("start_hour"), event.get("end_hour")
    if start is not None and end is not None:
        if not 0 <= start < end <= 24:
            raise ValueError("Closing time must be after opening time (same day).")
        event["hours"] = end-start
    if event.get("expected_attendance", 0) < 0 or event.get("precip_mm", 0) < 0:
        raise ValueError("Attendance and rainfall must be non-negative.")
    details = social_details(payload, df)
    if details["social"] is not None:
        event["social_index"] = details["social"]["social_index"]
    attendance_est = None
    if "expected_attendance" not in event and details["attendance"]:
        attendance_est = details["attendance"]
        event["expected_attendance"] = attendance_est["attendance"]
    products = payload["products"]
    names = [p["product"].strip() for p in products]
    if not names or any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Add at least one product, with unique, non-blank names.")
    for p in products:
        if any(p.get(k) is not None and p[k] < 0 for k in ("price", "unit_cost", "leftover_value", "same_stalls", "competitor_index")):
            raise ValueError("Product costs, prices, leftover values and stalls cannot be negative.")
    def mapping(key): return {n: p[key] for n, p in zip(names, products) if p.get(key) is not None}
    event["competitor_index"] = {n: p["competitor_index"] if p["competitor_index"] is not None else p["same_stalls"]
                                 for n, p in zip(names, products)
                                 if p["competitor_index"] is not None or p["same_stalls"] is not None}
    categories = {n: p.get("category") or guess_category(n) for n, p in zip(names, products)}
    # Explicitly pass unknown prices as None so clearing a field does not
    # silently fall back to an old historical price.
    prices = {n: p.get("price") for n, p in zip(names, products)}
    unit_costs = {n: p.get("unit_cost") for n, p in zip(names, products)}
    recs = model.recommend(event, products=names, price=prices, unit_cost=unit_costs,
                           salvage=mapping("leftover_value"), categories=categories)
    rows = [r.as_dict() for r in recs]
    def schedule(r):
        # Allocate integer units exactly, using the model's hourly schedule as weights.
        result = model.schedule(r.product, event, r.bring).copy()
        delta = r.bring - int(result["units"].sum())
        while delta:
            index = result["units"].idxmax()
            result.loc[index, "units"] += 1 if delta > 0 else -1
            delta += -1 if delta > 0 else 1
        result["cumulative"] = result["units"].cumsum()
        return result.to_dict("records")
    costs = mapping("unit_cost")
    similar = []
    for r in recs:
        match = similar_events_forecast(df, event, r.product, critical_ratio=r.critical_ratio)
        if match:
            match["matches"] = match["matches"].to_dict("records")
            similar.append(match)
    return {"recommendations": rows,
            "prep": {r.product: schedule(r) for r in recs} if start is not None and end is not None else {},
            "social": details["social"], "attendance_est": attendance_est, "event": event,
            "holiday": bool(is_holiday_window(event["date"])),
            "similar": similar, "signals": model.has_signal, "explain": model.explain().to_dict(),
            "missing_costs": [n for n in names if n not in costs or n not in mapping("price")],
            "stock_cost": sum(r.bring * costs[r.product] for r in recs) if all(n in costs for n in names) else None}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, status=200):
        data = json.dumps(safe_json(obj), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_HEAD(self):
        # Hosting health checks (Render etc.) send HEAD requests.
        self.send_response(200 if self.path in ("/", "/healthz") else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/api/sample", "/api/start"):
            try:
                self.send_json({"history": load_history(read_csv(SAMPLE) if self.path == "/api/sample" else saved_history())})
            except Exception as e:
                self.send_json({"error": str(e)}, 400)
            return
        if self.path == "/api/template.xlsx":
            data = build_template()
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", 'attachment; filename="vendor_sales_template.xlsx"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        files = {"/": "index.html", "/style.css": "style.css", "/log.css": "log.css", "/app.js": "app.js"}
        if self.path not in files:
            self.send_error(404)
            return
        path = ROOT / "web" / files[self.path]
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 40_000_000:
                raise ValueError("Please upload smaller files (under 25 MB combined).")
            p = json.loads(self.rfile.read(length))
            if self.path == "/api/history":
                raw = normalize(pd.DataFrame(p.get("rows", [])))
                result = load_history(raw, p.get("transactions", ""))
                if p.get("save"):
                    save_history(raw)
            elif self.path == "/api/import":
                raw = imported_history(p["files"], p.get("rows", []))
                result = load_history(raw, p.get("transactions", ""))
                if p.get("save"):
                    save_history(raw)
            elif self.path == "/api/transactions":
                tx = read_upload(p["file"])
                tx.columns = [str(c).strip().lower().replace(" ", "_") for c in tx.columns]
                if not {"timestamp", "product", "qty"} <= set(tx.columns):
                    raise ValueError("Hourly sales need timestamp, product and qty columns.")
                result = {"csv": tx.to_csv(index=False)}
            elif self.path == "/api/social":
                result = social_details(p, pd.DataFrame(p.get("rows", [])))
            elif self.path == "/api/forecast":
                result = forecast(p)
            elif self.path == "/api/weather":
                result = fetch_weather(float(p["lat"]), float(p["lon"]), p["date"],
                                       int(p.get("start_hour") if p.get("start_hour") is not None else 10),
                                       int(p.get("end_hour") if p.get("end_hour") is not None else 18))
            elif self.path == "/api/competitors":
                result = competitor_index(float(p["lat"]), float(p["lon"]), p.get("category") or guess_category(p.get("product", "")),
                                          p["key"], int(p["radius"]), int(p.get("same_stalls") or 0))
            else:
                self.send_error(404)
                return
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)}, 400)


if __name__ == "__main__":
    # Hosting services (Render, Railway, etc.) set PORT and need 0.0.0.0; locally we stay on 127.0.0.1.
    hosted = "PORT" in os.environ
    port = int(os.environ.get("PORT") or os.environ.get("VENDOR_PORT", "8501"))
    host = os.environ.get("VENDOR_HOST", "0.0.0.0" if hosted else "127.0.0.1")
    url = f"http://127.0.0.1:{port}"
    print(f"Vendor Predictor is ready at {url}")
    if not hosted and os.environ.get("VENDOR_NO_BROWSER") != "1":
        Timer(0.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer((host, port), Handler).serve_forever()
