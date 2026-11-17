"""
Pop-up shop demand predictor.

Given a log of past events (what you sold, what you brought, weather, attendance,
social buzz, nearby competition, hours), predict how many units of each product
to bring to the next event.

How it works
------------
1. Demand model: a Poisson regression (log link), one row per (event, product).
   Features: product, event type, log attendance, rain, temperature (with a
   per-product heat effect), weekend, holiday, season, social index (bot-adjusted),
   nearby competitor index, and event length. Category-like effects are shrunk
   toward the average (ridge penalty) so thin data doesn't overfit.
   Opening hours enter as a fixed offset: if you're open 6-10pm, a coffee stall
   only sees the share of a day's coffee demand that falls in those hours.
2. Sellouts: if you sold everything you brought, true demand was *at least*
   that much. Those rows are treated as censored and re-estimated (EM).
3. Uncertainty: negative binomial (real demand is noisier than Poisson).
4. Decision: the "newsvendor" rule. Bring the demand quantile at
   critical_ratio = (price - cost) / (price - salvage + holding cost).
5. Cold start: with little history for a product, blend with
   attendance x conversion rate x units per buyer x product share.

Dependencies: numpy, pandas, scipy.

Input columns (CSV / DataFrame), one row per product per event:
    required: event_id, date, product, units_sold
    optional: units_brought, event_type, category, expected_attendance, temp_c,
              precip_mm, hours, start_hour, end_hour, price, unit_cost,
              social_index (or the raw columns in signals.SOCIAL_RAW_COLUMNS),
              competitor_index (per product; see signals.competitor_index)
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from signals import (SOCIAL_RAW_COLUMNS, guess_category, hourly_profile, hours_coverage,
                     is_holiday_window, prep_schedule, social_index_from_row)


# ----------------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Fill derived columns: category, social_index from raw social stats."""
    df = df.copy()
    if "event_type" not in df:
        df["event_type"] = "unknown"
    df["event_type"] = df["event_type"].replace("", np.nan).fillna("unknown")
    if "event_id" not in df or df["event_id"].isna().any():
        # no event id: treat each date + event type as one event
        auto = df["date"].astype(str) + " " + df["event_type"].astype(str)
        df["event_id"] = df["event_id"].fillna(auto) if "event_id" in df else auto
    if "category" not in df:
        df["category"] = df["product"].map(guess_category)
    df["category"] = df["category"].fillna(df["product"].map(guess_category))
    if "social_index" not in df and any(c in df for c in SOCIAL_RAW_COLUMNS):
        df["social_index"] = df.apply(social_index_from_row, axis=1)
    return df


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------

NUMERIC_FEATURES = ["log_att", "att_missing", "rain", "temp", "temp_sq", "weekend",
                    "holiday", "season_sin", "season_cos", "log_social", "social_missing",
                    "log_comp", "log_hours"]


@dataclass
class _FeatureSpec:
    products: list
    event_types: list
    categories: dict
    att_median: float
    hours_median: float
    social_median: float
    comp_median: float
    learned_hourly: dict | None = None

    @classmethod
    def from_frame(cls, df: pd.DataFrame, learned_hourly=None) -> "_FeatureSpec":
        def med(c, default):
            return float(df[c].median()) if c in df and df[c].notna().any() else default
        return cls(
            products=sorted(df["product"].unique().tolist()),
            event_types=sorted(df["event_type"].fillna("unknown").unique().tolist()),
            categories=df.groupby("product")["category"].first().to_dict(),
            att_median=med("expected_attendance", 1.0),
            hours_median=med("hours", 1.0),
            social_median=med("social_index", 0.0),
            comp_median=med("competitor_index", 0.0),
            learned_hourly=learned_hourly,
        )

    def columns(self) -> list[str]:
        return (["intercept"] + NUMERIC_FEATURES
                + [f"p:{p}" for p in self.products]
                + [f"e:{e}" for e in self.event_types]
                + [f"temp x {p}" for p in self.products])

    def penalties(self, strong: float, weak: float) -> np.ndarray:
        out = []
        for c in self.columns():
            if c == "intercept":
                out.append(0.0)
            elif c.startswith(("p:", "e:", "temp x")):
                out.append(strong)
            elif c in ("season_sin", "season_cos", "holiday"):
                out.append(strong)   # needs a lot of data across the year to trust
            else:
                out.append(weak)
        return np.array(out)

    def category(self, product: str) -> str:
        return self.categories.get(product, guess_category(product))

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (X, offset)."""
        n = len(df)
        def col(name, default=np.nan):
            return df[name].astype(float).to_numpy() if name in df else np.full(n, default)

        att = col("expected_attendance")
        att_missing = np.isnan(att).astype(float)
        log_att = np.where(att_missing == 1, 0.0, np.log(np.maximum(att, 1) / self.att_median))

        precip = np.nan_to_num(col("precip_mm", 0.0), nan=0.0)
        t = (np.nan_to_num(col("temp_c", 20.0), nan=20.0) - 20.0) / 10.0

        hours = np.nan_to_num(col("hours", self.hours_median), nan=self.hours_median)
        log_hours = np.log(np.maximum(hours, 0.5) / self.hours_median)

        dates = pd.to_datetime(df["date"])
        weekend = (dates.dt.dayofweek >= 5).astype(float).to_numpy()
        holiday = np.array([is_holiday_window(d) for d in dates])
        doy = dates.dt.dayofyear.to_numpy()
        season_sin, season_cos = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

        soc = col("social_index")
        social_missing = np.isnan(soc).astype(float)
        log_social = np.where(social_missing == 1, 0.0, np.log1p(np.maximum(soc, 0)) - np.log1p(self.social_median))

        comp = col("competitor_index")
        log_comp = np.where(np.isnan(comp), 0.0, np.log1p(np.maximum(comp, 0)) - np.log1p(self.comp_median))

        X = [np.ones(n), log_att, att_missing, np.log1p(precip), t, t ** 2, weekend, holiday,
             season_sin, season_cos, log_social, social_missing, log_comp, log_hours]
        prod = df["product"].to_numpy()
        for p in self.products:
            X.append((prod == p).astype(float))
        et = df["event_type"].fillna("unknown").to_numpy()
        for e in self.event_types:
            X.append((et == e).astype(float))
        for p in self.products:
            X.append((prod == p).astype(float) * t)

        # Opening-hours offset: share of the product's daily demand inside open hours.
        start, end = col("start_hour"), col("end_hour")
        offset = np.zeros(n)
        for i in range(n):
            if not (np.isnan(start[i]) or np.isnan(end[i])):
                prof = hourly_profile(self.category(prod[i]), self.learned_hourly, prod[i])
                offset[i] = np.log(hours_coverage(prof, start[i], end[i]))
        return np.column_stack(X), offset


# ----------------------------------------------------------------------------
# Poisson regression with ridge penalty (IRLS) + censoring (EM)
# ----------------------------------------------------------------------------

def _fit_poisson_ridge(X, y, pen, offset, iters=100, tol=1e-8):
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-3))
    P = np.diag(pen)
    for _ in range(iters):
        lin = np.clip(X @ beta, -20, 20)
        mu = np.exp(lin + offset)
        z = lin + (y - mu) / mu
        XtW = X.T * mu
        new = np.linalg.solve(XtW @ X + P, XtW @ z)
        if np.max(np.abs(new - beta)) < tol:
            return new
        beta = new
    return beta


def _censored_expectation(mu, c):
    """E[Y | Y >= c] for Y ~ Poisson(mu). Uses E[Y; Y>=c] = mu * P(Y >= c-1)."""
    num = mu * stats.poisson.sf(c - 2, mu)
    den = stats.poisson.sf(c - 1, mu)
    out = np.where(den > 1e-12, num / np.maximum(den, 1e-300), c)
    return np.maximum(out, c)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

@dataclass
class ColdStartPrior:
    """Used when a product has little or no history. Tune these per business."""
    conversion_rate: float = 0.05     # share of attendees who buy anything
    units_per_buyer: float = 1.3      # avg units per purchasing customer
    product_share: dict = field(default_factory=dict)  # product -> share of units (defaults to equal)
    blend_events: float = 4.0         # history needed before model gets ~50% weight


@dataclass
class Recommendation:
    product: str
    bring: int                 # the number to prepare
    expected_demand: float
    low: int                   # 10th percentile of demand
    high: int                  # 90th percentile of demand
    sellout_prob: float        # chance demand >= bring
    expected_leftover: float
    critical_ratio: float
    hours_fit: float           # demand in your open hours vs a 10am-4pm market
    history_events: int

    def as_dict(self):
        return self.__dict__.copy()


class PopupDemandModel:
    def __init__(self, category_shrink: float = 2.0, feature_shrink: float = 0.5,
                 prior: ColdStartPrior | None = None, learned_hourly: dict | None = None):
        self.category_shrink = category_shrink
        self.feature_shrink = feature_shrink
        self.prior = prior or ColdStartPrior()
        self.learned_hourly = learned_hourly   # from signals.learn_hourly_profiles()

    # -- training ------------------------------------------------------------
    def fit(self, history: pd.DataFrame, em_iters: int = 10) -> "PopupDemandModel":
        df = prepare(history)
        self.spec = _FeatureSpec.from_frame(df, self.learned_hourly)
        X, off = self.spec.transform(df)
        y_obs = df["units_sold"].astype(float).to_numpy()

        brought = df["units_brought"].astype(float).to_numpy() if "units_brought" in df else np.full(len(df), np.nan)
        censored = ~np.isnan(brought) & (y_obs >= brought) & (y_obs > 0)

        pen = self.spec.penalties(self.category_shrink, self.feature_shrink)
        y = y_obs.copy()
        beta = _fit_poisson_ridge(X, y, pen, off)
        for _ in range(em_iters if censored.any() else 0):
            mu = np.exp(X @ beta + off)
            y[censored] = _censored_expectation(mu[censored], y_obs[censored])
            beta = _fit_poisson_ridge(X, y, pen, off, iters=30)
        self.beta = beta

        mu = np.exp(X @ beta + off)
        m = ~censored
        if m.sum() > 5:
            a = np.mean(((y_obs[m] - mu[m]) ** 2 - y_obs[m]) / mu[m] ** 2)
            self.alpha = float(np.clip(a, 0.01, 2.0))
        else:
            self.alpha = 0.1

        self.history_counts = df.groupby("product")["event_id"].nunique().to_dict()
        self.n_censored = int(censored.sum())
        self.has_signal = {
            "social": "social_index" in df and df["social_index"].notna().sum() >= 5,
            "competitors": "competitor_index" in df and df["competitor_index"].notna().sum() >= 5,
            "hours": {"start_hour", "end_hour"} <= set(df.columns),
        }
        self.price_cost = (
            df.groupby("product")[["price", "unit_cost"]].median().to_dict("index")
            if {"price", "unit_cost"} <= set(df.columns) else {}
        )
        self.coef_ = pd.Series(beta, index=self.spec.columns())
        return self

    # -- prediction ----------------------------------------------------------
    def _row(self, event: dict, product: str) -> pd.DataFrame:
        e = dict(event)
        comp = e.get("competitor_index")
        if isinstance(comp, dict):          # per-product competition
            e["competitor_index"] = comp.get(product, np.nan)
        e.setdefault("event_type", "unknown")
        e.setdefault("start_hour", np.nan)
        e.setdefault("end_hour", np.nan)
        return pd.DataFrame([{**e, "product": product}])

    def expected_demand(self, event: dict, product: str) -> float:
        X, off = self.spec.transform(self._row(event, product))
        mu_model = float(np.exp(X @ self.beta + off)[0])

        n_hist = self.history_counts.get(product, 0)
        att = event.get("expected_attendance")
        if att is None:
            return mu_model
        pr = self.prior
        share = pr.product_share.get(product, 1.0 / max(len(self.spec.products), 1))
        mu_prior = att * pr.conversion_rate * pr.units_per_buyer * share * float(np.exp(off[0]))
        w = n_hist / (n_hist + pr.blend_events)
        return w * mu_model + (1 - w) * mu_prior

    def recommend(self, event: dict, products: list[str] | None = None,
                  price: dict | None = None, unit_cost: dict | None = None,
                  salvage: dict | float = 0.0, holding_frac: float = 0.1,
                  max_service: float = 0.97, categories: dict | None = None) -> list[Recommendation]:
        """
        event: dict with date, event_type, expected_attendance, temp_c, precip_mm, hours,
               start_hour, end_hour, social_index, competitor_index (float or {product: float})
        price / unit_cost: per-product dicts (falls back to history medians)
        salvage: what a leftover unit is worth (0 for perishable food, ~unit_cost for merch)
        holding_frac: extra cost of a leftover as a share of unit_cost (hauling, storage)
        max_service: cap on the critical ratio, so we never plan for a 1-in-1000 day
        categories: product -> category for products the model hasn't seen
        """
        if categories:
            self.spec.categories.update({k: v for k, v in categories.items() if k not in self.spec.categories})
        products = products or self.spec.products
        out = []
        for p in products:
            mu = max(self.expected_demand(event, p), 1e-6)
            pc = self.price_cost.get(p, {})
            pr_ = (price or {}).get(p, pc.get("price"))
            co_ = (unit_cost or {}).get(p, pc.get("unit_cost"))
            sv = salvage.get(p, 0.0) if isinstance(salvage, dict) else salvage
            if pr_ is None or co_ is None or pd.isna(pr_) or pd.isna(co_):
                cr = 0.6   # unknown margins: aim a little above the median
            else:
                under = max(pr_ - co_, 1e-6)
                over = max(co_ - sv, 0.0) + holding_frac * co_ + 1e-6
                cr = min(under / (under + over), max_service)

            n_param = 1.0 / self.alpha
            dist = stats.nbinom(n_param, n_param / (n_param + mu))
            q = int(dist.ppf(cr))
            ks = np.arange(0, q + 1)
            leftover = float(np.sum((q - ks) * dist.pmf(ks)))
            _, off = self.spec.transform(self._row(event, p))
            out.append(Recommendation(
                product=p, bring=q, expected_demand=round(mu, 1),
                low=int(dist.ppf(0.10)), high=int(dist.ppf(0.90)),
                sellout_prob=round(float(dist.sf(q - 1)), 3),
                expected_leftover=round(leftover, 1),
                critical_ratio=round(cr, 3),
                hours_fit=round(float(np.exp(off[0])), 2),
                history_events=self.history_counts.get(p, 0),
            ))
        return out

    def schedule(self, product: str, event: dict, units: int) -> pd.DataFrame:
        """How many of `units` to have ready by each hour of the event."""
        prof = hourly_profile(self.spec.category(product), self.learned_hourly, product)
        return prep_schedule(prof, event.get("start_hour", 10), event.get("end_hour", 16), units)

    def explain(self) -> pd.Series:
        """Multiplicative effect of each feature on demand (1.0 = no effect)."""
        return np.exp(self.coef_.drop("intercept")).round(3)


# ----------------------------------------------------------------------------
# Attendance estimate from social media
# ----------------------------------------------------------------------------

def estimate_attendance(social_index: float, history: pd.DataFrame | None = None,
                        conversion_rate: float = 0.05) -> dict:
    """Rough expected attendance when you don't have the organizer's number.

    * If your past events have both attendance and social_index logged, use your own
      typical ratio (median attendance per point of social index).
    * Otherwise fall back to: social_index is ~extra buyers, and ~5% of attendees buy,
      so attendance ~ social_index / 0.05. This only counts people YOUR buzz pulls in,
      so for big events it will undershoot; treat it as a floor.
    """
    if history is not None and {"expected_attendance", "social_index"} <= set(history.columns):
        h = history.dropna(subset=["expected_attendance", "social_index"])
        h = h[h["social_index"] > 0].drop_duplicates(subset=["date", "expected_attendance"])
        if len(h) >= 3:
            ratio = float((h["expected_attendance"] / h["social_index"]).median())
            return {"attendance": int(round(ratio * social_index)),
                    "method": f"your past events (≈{ratio:.0f} attendees per social point, {len(h)} events)"}
    floor = int(round(social_index / conversion_rate))
    if history is not None and "expected_attendance" in history and history["expected_attendance"].notna().any():
        typical = int(history.drop_duplicates(subset=["date"])["expected_attendance"].median())
        if typical > floor:   # buzz alone undershoots; don't drag below your usual crowd
            return {"attendance": typical,
                    "method": f"your typical past crowd ({typical:,}); social buzz alone suggests at least {floor:,}"}
    return {"attendance": floor,
            "method": f"rule of thumb (social index ÷ {conversion_rate:.0%} buy rate); a rough floor"}


# ----------------------------------------------------------------------------
# Second predictor: similar past events
# ----------------------------------------------------------------------------

def similar_events_forecast(history: pd.DataFrame, event: dict, product: str, k: int = 5,
                            critical_ratio: float = 0.6) -> dict | None:
    """'What did I sell at events most like this one?'

    Finds the k past events closest to the new one (same event type, similar
    attendance, weather, hours, weekday, buzz, recency) and takes a weighted average
    of what you sold there. Sellouts get a 15% bump since demand was higher than sales.
    Simple and easy to sanity-check: you can see exactly which events it used."""
    df = prepare(history)
    rows = df[df["product"] == product].copy()
    if rows.empty:
        return None

    def num(col, default=np.nan):
        return pd.to_numeric(rows[col], errors="coerce") if col in rows else pd.Series(default, index=rows.index)

    d = pd.Series(0.0, index=rows.index)
    if event.get("event_type") not in (None, "unknown"):
        d += (rows["event_type"] != event["event_type"]).astype(float) * 1.0
    if event.get("expected_attendance"):
        d += np.abs(np.log(num("expected_attendance").clip(lower=1) / max(event["expected_attendance"], 1))).fillna(0.7)
    if event.get("temp_c") is not None:
        d += (np.abs(num("temp_c") - event["temp_c"]) / 10).fillna(0.5)
    if event.get("precip_mm") is not None:
        d += np.abs(np.log1p(num("precip_mm").fillna(0)) - np.log1p(event["precip_mm"])) * 0.7
    if event.get("start_hour") is not None and event.get("end_hour") is not None:
        mid = (event["start_hour"] + event["end_hour"]) / 2
        d += (np.abs((num("start_hour") + num("end_hour")) / 2 - mid) / 4).fillna(0.5)
    if event.get("social_index") is not None:
        d += (np.abs(np.log1p(num("social_index")) - np.log1p(event["social_index"])) * 0.5).fillna(0.3)
    dates = pd.to_datetime(rows["date"])
    ev_date = pd.Timestamp(event["date"])
    d += ((dates.dt.dayofweek >= 5) != (ev_date.dayofweek >= 5)).astype(float) * 0.5
    d += (np.abs((ev_date - dates).dt.days) / 365) * 0.3          # older events count a bit less

    sold = rows["units_sold"].astype(float)
    if "units_brought" in rows:
        sold_out = sold >= pd.to_numeric(rows["units_brought"], errors="coerce")
        sold = np.where(sold_out, sold * 1.15, sold)
    rows["est_demand"] = sold
    rows["distance"] = d
    top = rows.nsmallest(min(k, len(rows)), "distance")
    w = 1.0 / (0.25 + top["distance"])
    mean = float(np.average(top["est_demand"], weights=w))

    # weighted quantile at the service level for the bring number
    order = np.argsort(top["est_demand"].to_numpy())
    vals, ws = top["est_demand"].to_numpy()[order], w.to_numpy()[order]
    cdf = np.cumsum(ws) / ws.sum()
    q = float(vals[np.searchsorted(cdf, critical_ratio)]) if len(vals) > 1 else mean
    cols = [c for c in ["date", "event_type", "expected_attendance", "temp_c", "units_brought", "units_sold"] if c in top]
    return {
        "product": product,
        "bring": int(np.ceil(0.5 * mean + 0.5 * max(q, mean))),
        "expected_demand": round(mean, 1),
        "low": int(top["est_demand"].min()),
        "high": int(np.ceil(top["est_demand"].max())),
        "events_used": len(top),
        "matches": top[cols + ["distance"]].round(2).sort_values("distance"),
    }


def predict_units(history: pd.DataFrame, event: dict, **kwargs) -> pd.DataFrame:
    """One-call convenience wrapper: fit on history, return a table of recommendations."""
    model = PopupDemandModel().fit(history)
    return pd.DataFrame([r.as_dict() for r in model.recommend(event, **kwargs)])


# ----------------------------------------------------------------------------
# Weather helper (free, no API key): https://open-meteo.com
# ----------------------------------------------------------------------------

def fetch_weather(lat: float, lon: float, date: str, start_hour: int = 10, end_hour: int = 18) -> dict:
    """Mean temperature and total precipitation over the event hours.
    Uses the forecast API for dates up to ~16 days ahead, the archive for the past."""
    is_past = pd.Timestamp(date) < pd.Timestamp.today().normalize()
    base = "https://archive-api.open-meteo.com/v1/archive" if is_past else "https://api.open-meteo.com/v1/forecast"
    url = (f"{base}?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation"
           f"&start_date={date}&end_date={date}&timezone=auto")
    with urllib.request.urlopen(url, timeout=15) as r:
        h = json.load(r)["hourly"]
    temps = h["temperature_2m"][int(start_hour):int(end_hour)]
    rain = h["precipitation"][int(start_hour):int(end_hour)]
    return {"temp_c": round(float(np.mean(temps)), 1), "precip_mm": round(float(np.sum(rain)), 1)}
