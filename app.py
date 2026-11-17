"""
Browser app for the pop-up demand predictor.

Run:  streamlit run app.py
Then open http://localhost:8501 (it usually opens automatically).

Your events are saved to my_events.csv in this folder and loaded automatically.

Optional: put your Google API key in a .streamlit/secrets.toml file as
    GOOGLE_API_KEY = "..."
so you don't have to paste it each time.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor import (PopupDemandModel, estimate_attendance, fetch_weather, prepare,
                       similar_events_forecast)
from signals import (CATEGORY_PLACE_TYPES, DEFAULT_RADIUS_M, IntentSignals, SocialStats,
                     competitor_index, guess_category, learn_hourly_profiles, social_score)

DATA_FILE = Path(__file__).parent / "my_events.csv"

# Columns of the event log. Only date, product and units_sold are required.
LOG_COLUMNS = {
    "date": "datetime64[ns]", "event_type": "string", "product": "string", "category": "string",
    "units_brought": "float", "units_sold": "float", "price": "float", "unit_cost": "float",
    "expected_attendance": "float", "start_hour": "float", "end_hour": "float",
    "temp_c": "float", "precip_mm": "float", "social_index": "float", "competitor_index": "float",
}
CATEGORIES = sorted(CATEGORY_PLACE_TYPES) + ["default"]

st.set_page_config(page_title="Pop-up Demand Predictor", page_icon="📦", layout="wide")
st.title("📦 Pop-up Demand Predictor")
st.caption("How many units should you bring to your next event?")


def empty_log() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in LOG_COLUMNS.items()})


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an imported file to the log's columns and types; keep any extra columns."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for c, t in LOG_COLUMNS.items():
        if c not in df:
            df[c] = pd.Series(dtype=t)
        elif t == "float":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif t.startswith("datetime"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = df[c].astype("string")
    return df[list(LOG_COLUMNS) + [c for c in df.columns if c not in LOG_COLUMNS]]


def read_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def save(df: pd.DataFrame):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out.to_csv(DATA_FILE, index=False)


if "log" not in st.session_state:
    st.session_state.log = normalise(pd.read_csv(DATA_FILE)) if DATA_FILE.exists() else empty_log()

# ---------------------------------------------------------------- 1. History
st.header("1. Your past events")
st.caption("One row per product per event. Import a file, type rows in the table, or both. "
           "Required: date, product, units sold. Units brought matters a lot: it tells a sellout apart from low demand.")

imp1, imp2 = st.columns([3, 1])
files = imp1.file_uploader("Import sales / event files (CSV or Excel, e.g. a Square or Shopify report)",
                           type=["csv", "xlsx", "xls"], accept_multiple_files=True)
if files and imp2.button("Add to my events", type="primary"):
    new = [normalise(read_any(f)) for f in files]
    st.session_state.log = pd.concat([st.session_state.log, *new], ignore_index=True)
    save(st.session_state.log)
    st.success(f"Imported {sum(len(n) for n in new)} rows.")

edited = st.data_editor(
    st.session_state.log, num_rows="dynamic", width="stretch", hide_index=True, key="log_editor",
    column_config={
        "date": st.column_config.DateColumn(required=True),
        "category": st.column_config.SelectboxColumn(options=CATEGORIES),
        "start_hour": st.column_config.NumberColumn(min_value=0, max_value=23, help="24h clock"),
        "end_hour": st.column_config.NumberColumn(min_value=1, max_value=24, help="24h clock"),
    })
b1, b2, _ = st.columns([1, 1, 4])
if b1.button("💾 Save changes"):
    st.session_state.log = edited
    save(edited)
    st.success(f"Saved to {DATA_FILE.name}")
if b2.button("🗑️ Clear all"):
    st.session_state.log = empty_log()
    DATA_FILE.unlink(missing_ok=True)
    st.rerun()

history = edited.dropna(subset=["date", "product", "units_sold"])
history = history[history["product"].astype(str).str.strip() != ""]
if history.empty:
    st.info("Add at least one past event (date, product, units sold) to get a recommendation.")
    st.stop()
history = history.assign(date=pd.to_datetime(history["date"]).dt.date.astype(str))
history = history.dropna(axis=1, how="all")   # columns you haven't filled are ignored

tx_file = st.file_uploader(
    "Optional: import timestamped sales (columns: timestamp, product, qty) to learn when during the day each product sells.",
    type=["csv", "xlsx", "xls"], key="tx")


@st.cache_resource(show_spinner="Training model...")
def train(df: pd.DataFrame, tx: pd.DataFrame | None) -> PopupDemandModel:
    learned = None
    if tx is not None:
        cats = df.groupby("product")["category"].first().dropna().to_dict() if "category" in df else {}
        learned = learn_hourly_profiles(tx, cats)
    return PopupDemandModel(learned_hourly=learned).fit(df)


tx = read_any(tx_file) if tx_file is not None else None
model = train(history, tx)
n_events = prepare(history)["event_id"].nunique()
st.caption(f"Trained on {n_events} events, {history['product'].nunique()} products.")
if n_events < 10:
    st.warning("With fewer than ~10 events the numbers lean heavily on attendance-based guesses. "
               "They'll sharpen as you log more.")
if model.n_censored:
    st.caption(f"{model.n_censored} rows were sellouts; treated as 'demand was at least this much'.")
for name, ok in model.has_signal.items():
    if not ok:
        st.caption(f"No {name} data in your history yet, so that input won't move the numbers until you log it.")

# ---------------------------------------------------------------- 2. Next event
st.header("2. Your next event")
c1, c2, c3 = st.columns(3)
event_date = c1.date_input("Date", value=None)
event_type = c2.selectbox("Event type", [t for t in model.spec.event_types if t != "unknown"] + ["other"],
                          index=None, placeholder="Choose...", accept_new_options=True)
attendance = c3.number_input("Expected attendance", min_value=0, value=None, step=100)
c4, c5, c6, c7 = st.columns(4)
start_hour = c4.number_input("Opens (hour, 24h)", 0, 23, value=None)
end_hour = c5.number_input("Closes (hour, 24h)", 1, 24, value=None)
lat = c6.number_input("Latitude", value=None, format="%.4f", help="Right-click a spot in Google Maps to copy it")
lon = c7.number_input("Longitude", value=None, format="%.4f")
if start_hour is not None and end_hour is not None and end_hour <= start_hour:
    st.error("Closing hour must be after opening hour.")
    st.stop()

st.subheader("Weather")
w1, w2, w3 = st.columns([1, 1, 1])
temp_c = w1.number_input("Temperature (°C)", value=None, step=1.0)
precip_mm = w2.number_input("Rain (mm)", min_value=0.0, value=None, step=0.5)
if w3.button("Fill from forecast", disabled=not (event_date and lat is not None and lon is not None),
             help="Needs date, latitude and longitude"):
    try:
        wx = fetch_weather(lat, lon, event_date.isoformat(), start_hour or 10, end_hour or 18)
        st.session_state["wx"] = wx
    except Exception as e:  # noqa: BLE001
        st.warning(f"Couldn't fetch the forecast ({e}).")
if temp_c is None and precip_mm is None and "wx" in st.session_state:
    temp_c, precip_mm = st.session_state["wx"]["temp_c"], st.session_state["wx"]["precip_mm"]
    st.success(f"Forecast: {temp_c} °C, {precip_mm} mm rain")

# ---------------------------------------------------------------- 3. Social
st.header("3. Social media & buzz (optional)")
st.caption("Leave blank whatever you don't track. Real intent (RSVPs, clicks, DMs, preorders) counts most; "
           "reach is discounted when the account's numbers look bot-inflated.")
s1, s2, s3, s4 = st.columns(4)
followers = s1.number_input("Followers", min_value=0, value=None, step=100)
avg_likes = s2.number_input("Avg likes / post", min_value=0, value=None)
avg_comments = s3.number_input("Avg comments / post", min_value=0, value=None)
reach = s4.number_input("Reach of event posts", min_value=0, value=None, step=100)
s5, s6, s7, s8 = st.columns(4)
local_pct = s5.number_input("% of followers in your city", min_value=0, max_value=100, value=None)
growth = s6.number_input("Follower growth last 30 days (%)", value=None)
shares = s7.number_input("Shares + saves of event posts", min_value=0, value=None)
dms = s8.number_input("DMs asking about the event", min_value=0, value=None)
i1, i2, i3, i4 = st.columns(4)
link_clicks = i1.number_input("Link clicks", min_value=0, value=None)
rsvps = i2.number_input("RSVPs / interested", min_value=0, value=None)
preorders = i3.number_input("Preorders", min_value=0, value=None)
promo = i4.number_input("Promo code uses last event", min_value=0, value=None)

st.markdown("**Anything else you track** (TikTok views, flyers, newsletter opens...). "
            "Weight = roughly how many extra buyers one unit of it brings.")
custom = st.data_editor(
    pd.DataFrame({"signal": pd.Series(dtype="string"), "value": pd.Series(dtype="float"),
                  "weight": pd.Series(dtype="float")}),
    num_rows="dynamic", hide_index=True, width="stretch", key="custom")
custom = custom.dropna()

acct_vals = dict(followers=followers, avg_likes=avg_likes, avg_comments=avg_comments, event_post_reach=reach,
                 local_audience_pct=local_pct, follower_growth_30d_pct=growth, shares=shares, dms=dms,
                 link_clicks=link_clicks)
intent_vals = dict(rsvps=rsvps, preorders=preorders, promo_redemptions_last=promo)
social_index = None
if any(v is not None for v in [*acct_vals.values(), *intent_vals.values()]) or len(custom):
    accounts = [SocialStats(**acct_vals)] if any(v is not None for v in acct_vals.values()) else []
    social = social_score(accounts, IntentSignals(
        **{k: v or 0 for k, v in intent_vals.items()},
        custom={r.signal: (r.value, r.weight) for r in custom.itertuples() if r.signal}))
    social_index = social["social_index"]
    m1, m2 = st.columns(2)
    m1.metric("Social index (≈ extra buyers)", f"{social_index:.0f}")
    if accounts:
        m2.metric("Audience authenticity", f"{social['authenticity']:.0%}")
    for n in social["notes"]:
        st.warning(n)

attendance_note = None
if attendance is None and social_index:
    est = estimate_attendance(social_index, prepare(history))
    attendance = est["attendance"]
    attendance_note = f"No attendance entered, so using an estimate from your social numbers: ~{attendance:,} people, via {est['method']}."
    st.info(attendance_note)

# ---------------------------------------------------------------- 4. Products & competitors
st.header("4. Products, costs & nearby competition")
st.caption("Products come from your history; add new ones as rows. Leftover value: 0 for food you throw out, "
           "about the unit cost for things you can sell next time.")
pc = model.price_cost
prods = model.spec.products
known_cat = history.groupby("product")["category"].first().to_dict() if "category" in history else {}
products = st.data_editor(
    pd.DataFrame({
        "product": pd.Series(prods, dtype="string"),
        "category": pd.Series([known_cat.get(p) for p in prods], dtype="string"),
        "price": pd.Series([pc.get(p, {}).get("price") for p in prods], dtype="float"),
        "unit_cost": pd.Series([pc.get(p, {}).get("unit_cost") for p in prods], dtype="float"),
        "leftover_value": pd.Series([None] * len(prods), dtype="float"),
        "same_stalls_at_event": pd.Series([None] * len(prods), dtype="float"),
        "competitor_index": pd.Series([None] * len(prods), dtype="float"),
    }),
    num_rows="dynamic", width="stretch", hide_index=True, key="products",
    column_config={
        "category": st.column_config.SelectboxColumn(options=CATEGORIES,
                                                     help="Blank = guessed from the product name"),
        "same_stalls_at_event": st.column_config.NumberColumn(help="Other stalls at the event selling the same thing"),
        "competitor_index": st.column_config.NumberColumn(help="Filled by the Google scan, or type your own"),
    })
products = products.dropna(subset=["product"])
products = products[products["product"].astype(str).str.strip() != ""]
products["category"] = [c if isinstance(c, str) and c else guess_category(p)
                        for p, c in zip(products["product"], products["category"])]


def _secret_key() -> str:
    try:
        return st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:  # noqa: BLE001  (no secrets.toml file)
        return ""


g1, g2 = st.columns([2, 1])
api_key = g1.text_input("Google Places API key (optional)", type="password", value=_secret_key())
radius = g2.slider("Search radius (m)", 100, 2000, DEFAULT_RADIUS_M, step=50,
                   help="500 m ≈ 6-minute walk. Use ~200 m in dense downtowns, 1000 m+ in suburbs.")
can_scan = bool(api_key) and lat is not None and lon is not None
if st.button("Scan nearby competitors", disabled=not can_scan, help="Needs an API key, latitude and longitude"):
    vals, places = {}, {}
    for r in products.itertuples():
        stalls = 0 if pd.isna(r.same_stalls_at_event) else int(r.same_stalls_at_event)
        try:
            res = competitor_index(lat, lon, r.category, api_key, radius, stalls)
            vals[r.product], places[r.product] = res["competitor_index"], res["places"]
        except Exception as e:  # noqa: BLE001
            st.error(f"Google scan failed for {r.product}: {e}")
    st.session_state["comp"] = (vals, places)
comp_values, comp_places = st.session_state.get("comp", ({}, {}))
for p, places in comp_places.items():
    with st.expander(f"{p}: competitor index {comp_values[p]} ({len(places)} places)"):
        st.dataframe(pd.DataFrame(places), width="stretch", hide_index=True)

competitors = {}
for r in products.itertuples():
    if pd.notna(r.competitor_index):
        competitors[r.product] = float(r.competitor_index)
    elif r.product in comp_values:
        competitors[r.product] = comp_values[r.product]
    elif pd.notna(r.same_stalls_at_event):
        competitors[r.product] = float(r.same_stalls_at_event)

# ---------------------------------------------------------------- 5. Results
st.header("5. Recommendation")
if event_date is None or products.empty:
    st.info("Enter the event date (and anything else you know) above to see how much to bring.")
    st.stop()

event = {"date": event_date.isoformat(), "event_type": event_type or "unknown"}
for k, v in {"expected_attendance": attendance, "temp_c": temp_c, "precip_mm": precip_mm,
             "start_hour": start_hour, "end_hour": end_hour, "social_index": social_index}.items():
    if v is not None:
        event[k] = v
if start_hour is not None and end_hour is not None:
    event["hours"] = end_hour - start_hour
if competitors:
    event["competitor_index"] = competitors


def per_product(col):
    return {p: float(v) for p, v in zip(products["product"], products[col]) if pd.notna(v)}


recs = model.recommend(
    event, products=products["product"].tolist(),
    price=per_product("price"), unit_cost=per_product("unit_cost"), salvage=per_product("leftover_value"),
    categories=dict(zip(products["product"], products["category"])),
)
out = pd.DataFrame([r.as_dict() for r in recs])
no_cost = [p for p in products["product"] if p not in per_product("price") or p not in per_product("unit_cost")]
if no_cost:
    st.caption(f"No price/cost for {', '.join(no_cost)}: using a middle-of-the-road safety margin.")

cols = st.columns(max(len(out), 1))
for col, r in zip(cols, out.itertuples()):
    col.metric(r.product, f"Bring {r.bring}", f"demand ~{r.expected_demand:.0f} ({r.low}-{r.high})",
               delta_color="off")
    if r.hours_fit < 0.5:
        col.caption(f"🕐 Your hours catch only {r.hours_fit:.0%} of this item's usual demand.")
    if r.history_events < 4:
        col.caption("⚠️ Little history: leaning on an attendance-based estimate.")

st.dataframe(out.rename(columns={
    "bring": "Bring", "expected_demand": "Expected demand", "low": "Low (10%)", "high": "High (90%)",
    "sellout_prob": "Chance of selling out", "expected_leftover": "Expected leftovers",
    "critical_ratio": "Service level", "hours_fit": "Hours fit", "history_events": "Past events",
}), width="stretch", hide_index=True)

# ---- Second opinion: similar past events
st.subheader("Second opinion: your most similar past events")
st.caption("Looks up the past events most like this one (type, crowd, weather, hours, weekday, buzz) and "
           "averages what you sold there. Simpler than the main model and easy to sanity-check. "
           "If the two disagree a lot, trust the one whose reasoning matches what you know about the event.")
cr_by_product = dict(zip(out["product"], out["critical_ratio"]))
sim = [similar_events_forecast(history, event, p, critical_ratio=cr_by_product.get(p, 0.6))
       for p in products["product"]]
sim = [s for s in sim if s]
if not sim:
    st.caption("None of these products have past events yet.")
else:
    cmp = pd.DataFrame([{
        "product": s["product"],
        "Main model: bring": int(out.loc[out["product"] == s["product"], "bring"].iloc[0]),
        "Similar events: bring": s["bring"],
        "Similar events: avg demand": s["expected_demand"],
        "Range at those events": f"{s['low']}-{s['high']}",
        "Events used": s["events_used"],
    } for s in sim])
    st.dataframe(cmp, width="stretch", hide_index=True)
    for s in sim:
        with st.expander(f"Which events were used for {s['product']}"):
            st.dataframe(s["matches"], width="stretch", hide_index=True)

if start_hour is not None and end_hour is not None:
    st.subheader("Prep schedule")
    pick = st.selectbox("Product", out["product"].tolist())
    units = int(out.loc[out["product"] == pick, "bring"].iloc[0])
    st.caption("How many to have ready by each hour, based on when this item usually sells.")
    st.bar_chart(model.schedule(pick, event, units).set_index("hour")["units"])

with st.expander("What the model learned"):
    st.caption("Multiplier on demand: 1.0 = no effect, 0.66 = 34% fewer sales, 1.2 = 20% more.")
    st.bar_chart(model.explain())
