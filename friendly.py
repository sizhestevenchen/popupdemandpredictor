"""
Plain-English spreadsheet support:

* build_template()   -> a blank Excel template with simple columns, dropdowns and a
                        "How to fill this in" sheet. Opens fine in Google Sheets too.
* friendly_columns() -> turns plain column names and values ("Sold", "Start time" = "11am",
                        "Temp (°F)" = 75, "Rain" = "light") into the columns the model uses.
"""
from __future__ import annotations

import datetime as dt
import re
from io import BytesIO

import numpy as np
import pandas as pd

# Plain column headings used in the template, in order: (heading, width, note)
TEMPLATE_COLUMNS = [
    ("Date", 13, "Required. The day of the event, e.g. 6/14/2026"),
    ("Product", 20, "Required. One row per product per event"),
    ("Sold", 9, "Required. How many you sold"),
    ("Brought", 10, "How many you brought. If Sold = Brought, you sold out (important!)"),
    ("Event type", 16, "Pick from the list or type your own"),
    ("Price", 9, "What you sell one for, in $"),
    ("Cost", 9, "What one costs you to make or buy, in $"),
    ("Attendance", 12, "Rough crowd size, if you know it"),
    ("Start time", 11, "When you opened, e.g. 11am"),
    ("End time", 11, "When you closed, e.g. 5pm"),
    ("Temp (°F)", 10, "Rough temperature during the event"),
    ("Rain", 9, "none, light or heavy"),
    ("Notes", 30, "Anything else. Ignored by the model"),
]
EVENT_TYPES = ["farmers market", "festival", "night market", "brand pop-up", "craft fair",
               "concert", "sports game", "other"]
RAIN_CHOICES = ["none", "light", "heavy"]
RAIN_MM = {"none": 0.0, "light": 2.0, "heavy": 10.0}

# Squashed heading (lowercase letters/digits only) -> model column
ALIASES = {
    "date": "date", "eventdate": "date", "day": "date",
    "product": "product", "item": "product", "productname": "product", "itemname": "product",
    "sold": "units_sold", "unitssold": "units_sold", "qtysold": "units_sold", "quantitysold": "units_sold",
    "itemssold": "units_sold", "qty": "units_sold", "quantity": "units_sold",
    "brought": "units_brought", "unitsbrought": "units_brought", "prepared": "units_brought",
    "made": "units_brought", "stock": "units_brought", "inventory": "units_brought",
    "eventtype": "event_type", "event": "event_type", "type": "event_type",
    "price": "price", "sellprice": "price", "sellingprice": "price", "priceeach": "price",
    "cost": "unit_cost", "unitcost": "unit_cost", "costeach": "unit_cost",
    "attendance": "expected_attendance", "expectedattendance": "expected_attendance",
    "crowd": "expected_attendance", "crowdsize": "expected_attendance", "people": "expected_attendance",
    "starttime": "start_time", "start": "start_time", "opened": "start_time", "opentime": "start_time",
    "starthour": "start_hour",
    "endtime": "end_time", "end": "end_time", "closed": "end_time", "closetime": "end_time",
    "endhour": "end_hour",
    "tempf": "temp_f", "temperaturef": "temp_f", "temp": "temp_f", "temperature": "temp_f",
    "tempc": "temp_c", "temperaturec": "temp_c",
    "rain": "rain", "weather": "rain", "precipmm": "precip_mm", "rainmm": "precip_mm",
    "category": "category", "eventid": "event_id", "eventname": "event_id",
    "socialindex": "social_index", "competitorindex": "competitor_index",
    "notes": "notes", "note": "notes",
}


def _squash(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_hour(v):
    """'11am', '5:30 PM', '17:00', 'noon', 11, Excel time -> hours as a number (11.0, 17.5)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return np.nan
    if isinstance(v, (dt.time, dt.datetime, pd.Timestamp)):
        return v.hour + v.minute / 60
    if isinstance(v, (int, float, np.number)):
        x = float(v)
        return x * 24 if 0 < x < 1 else x      # Excel stores times as a fraction of a day
    s = str(v).strip().lower().replace(".", "")
    if not s:
        return np.nan
    if s in ("noon", "midday"):
        return 12.0
    if s == "midnight":
        return 24.0
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(?::\d{2})?\s*(am|pm|a|p)?", s)
    if not m:
        raise ValueError(f"Couldn't read the time '{v}'. Use something like 11am or 17:00.")
    h, mins, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap and ap.startswith("p") and h != 12:
        h += 12
    if ap and ap.startswith("a") and h == 12:
        h = 0
    return h + mins / 60


def parse_rain(v):
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().lower()
    if not s:
        return np.nan
    if any(w in s for w in ("heavy", "storm", "pour", "thunder")):
        return RAIN_MM["heavy"]
    if any(w in s for w in ("light", "drizzle", "little", "some", "shower")):
        return RAIN_MM["light"]
    if s in ("none", "no", "dry", "sunny", "clear", "cloudy", "0"):
        return 0.0
    if any(w in s for w in ("yes", "rain", "wet", "moderate")):
        return 5.0
    raise ValueError(f"Couldn't read the rain value '{v}'. Use none, light or heavy.")


def friendly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename plain headings and convert plain values to the model's columns."""
    df = df.copy()
    renamed, seen = {}, set()
    for c in df.columns:
        target = ALIASES.get(_squash(c), str(c).strip().lower().replace(" ", "_"))
        if target in seen:          # e.g. both "Qty" and "Sold": keep the first
            target = f"{target}_dup"
        seen.add(target)
        renamed[c] = target
    df = df.rename(columns=renamed)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_dup") or c == "notes"])

    if "start_time" in df:
        hours = df["start_time"].map(parse_hour)
        df["start_hour"] = df["start_hour"].fillna(hours) if "start_hour" in df else hours
        df = df.drop(columns="start_time")
    if "end_time" in df:
        hours = df["end_time"].map(parse_hour)
        df["end_hour"] = df["end_hour"].fillna(hours) if "end_hour" in df else hours
        df = df.drop(columns="end_time")
    if "temp_f" in df:
        c = (pd.to_numeric(df["temp_f"], errors="coerce") - 32) * 5 / 9
        df["temp_c"] = df["temp_c"].fillna(c.round(1)) if "temp_c" in df else c.round(1)
        df = df.drop(columns="temp_f")
    if "rain" in df:
        mm = df["rain"].map(parse_rain)
        df["precip_mm"] = df["precip_mm"].fillna(mm) if "precip_mm" in df else mm
        df = df.drop(columns="rain")
    if "event_type" in df:   # "Farmers Market" and "farmers_market" are the same event type
        df["event_type"] = df["event_type"].map(
            lambda v: re.sub(r"[\s\-]+", "_", str(v).strip().lower()) if pd.notna(v) and str(v).strip() else np.nan)
    return df


def build_template() -> bytes:
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    head_fill = PatternFill("solid", fgColor="277758")
    req_fill = PatternFill("solid", fgColor="1D6046")
    for i, (name, width, note) in enumerate(TEMPLATE_COLUMNS, start=1):
        cell = ws.cell(row=1, column=i, value=name + (" *" if i <= 3 else ""))
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = req_fill if i <= 3 else head_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.comment = Comment(note, "Vendor")
        ws.column_dimensions[cell.column_letter].width = width
    ws.row_dimensions[1].height = 24
    ws.freeze_panes = "A2"
    last = 1000
    col = {name: i for i, (name, *_ ) in enumerate(TEMPLATE_COLUMNS, start=1)}
    letter = lambda n: ws.cell(row=1, column=col[n]).column_letter  # noqa: E731

    for r in range(2, last + 1):
        ws.cell(row=r, column=col["Date"]).number_format = "m/d/yyyy"
        for n in ("Price", "Cost"):
            ws.cell(row=r, column=col[n]).number_format = '"$"#,##0.00'

    ev = DataValidation(type="list", formula1='"' + ",".join(EVENT_TYPES) + '"', allow_blank=True,
                        showErrorMessage=False)   # list is a suggestion; own types allowed
    rain = DataValidation(type="list", formula1='"' + ",".join(RAIN_CHOICES) + '"', allow_blank=True)
    whole = DataValidation(type="whole", operator="greaterThanOrEqual", formula1="0", allow_blank=True,
                           error="Enter a whole number, 0 or more.", showErrorMessage=True)
    money = DataValidation(type="decimal", operator="greaterThanOrEqual", formula1="0", allow_blank=True,
                           error="Enter an amount, 0 or more.", showErrorMessage=True)
    date_v = DataValidation(type="date", operator="greaterThan", formula1="36526", allow_blank=True,
                            error="Enter a date, e.g. 6/14/2026.", showErrorMessage=True)
    for v in (ev, rain, whole, money, date_v):
        ws.add_data_validation(v)
    ev.add(f"{letter('Event type')}2:{letter('Event type')}{last}")
    rain.add(f"{letter('Rain')}2:{letter('Rain')}{last}")
    for n in ("Sold", "Brought", "Attendance"):
        whole.add(f"{letter(n)}2:{letter(n)}{last}")
    for n in ("Price", "Cost"):
        money.add(f"{letter(n)}2:{letter(n)}{last}")
    date_v.add(f"{letter('Date')}2:{letter('Date')}{last}")

    # --- Instructions sheet
    help_ws = wb.create_sheet("How to fill this in")
    help_ws.column_dimensions["A"].width = 16
    help_ws.column_dimensions["B"].width = 70
    lines = [
        ("Vendor sales template", None),
        ("", None),
        ("The basics", "One row for each product at each event. Selling 3 products at a market = 3 rows "
                       "with the same date."),
        ("Required", "Date, Product and Sold. Everything else is optional, but the more you fill in, "
                     "the better the forecast."),
        ("Most useful extra", "Brought. If you sold everything you brought, the app knows demand was "
                              "higher than your sales and won't tell you to under-order again."),
        ("Times", "Type them however you like: 11am, 5:30pm, 17:00."),
        ("Rain", "none, light or heavy."),
        ("Leave blanks", "Blank cells are fine. Don't type 0 unless it really was 0."),
        ("", None),
        ("Example rows", "(for reference only, don't copy these into the Sales sheet)"),
        ("", "6/14/2026 | Iced latte | Sold 48 | Brought 60 | farmers market | $6 | $1.50 | 1200 | 8am | 1pm | 78 | none"),
        ("", "6/14/2026 | Croissant | Sold 30 | Brought 30 | farmers market | $4 | $1.20 | 1200 | 8am | 1pm | 78 | none"),
        ("", None),
        ("Google Sheets", "Upload this file to Google Drive and open it with Google Sheets. "
                          "When you're done, File > Download > Microsoft Excel (.xlsx) or CSV, then import it in the app."),
    ]
    for i, (a, b) in enumerate(lines, start=1):
        help_ws.cell(row=i, column=1, value=a).font = Font(bold=True, size=14 if i == 1 else 11)
        if b:
            c = help_ws.cell(row=i, column=2, value=b)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
