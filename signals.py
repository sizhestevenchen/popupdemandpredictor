"""
Extra demand signals for the pop-up predictor:

1. social_score()       social media -> one "intent" number, discounted for bots
2. competitor_index()   nearby same-category businesses via Google Places API
3. time-of-day profiles + yearly seasonality / holidays

Each one boils down to a single number per event (or per product per event)
that the model in predictor.py learns a weight for from your sales history.
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. Social media
# =============================================================================

@dataclass
class SocialStats:
    """Numbers for ONE account on ONE platform, measured in the days before an event.
    Leave anything you don't have as None. All of it is in Instagram / TikTok insights."""
    platform: str = "instagram"
    followers: float | None = None
    avg_likes: float | None = None            # per recent post
    avg_comments: float | None = None         # per recent post
    event_post_reach: float | None = None     # unique accounts reached by posts about THIS event
    local_audience_pct: float | None = None   # % of followers in your city (Insights > Audience)
    follower_growth_30d_pct: float | None = None
    shares: float | None = None               # shares + saves of the event posts
    dms: float | None = None                  # DMs asking about the event
    link_clicks: float | None = None          # clicks on a tracked link (Linktree, UTM)


@dataclass
class IntentSignals:
    """Actions that mean someone actually plans to come. Worth far more than likes."""
    rsvps: float = 0                 # "going" / "interested" on the event page
    preorders: float = 0             # units already paid for
    promo_redemptions_last: float = 0  # promo code uses at your last event (repeat-customer signal)
    email_sms_clicks: float = 0
    # "Account for anything": any other signal you track, as name -> (value, weight).
    # weight = roughly how many extra buyers one unit of it is worth, e.g.
    #   {"tiktok_views": (12000, 0.001), "flyers_handed_out": (300, 0.02)}
    custom: dict = field(default_factory=dict)


# Rough "extra buyers per unit of signal" priors. The model learns a correction on top.
INTENT_WEIGHTS = {
    "reach": 0.01,        # 1 in 100 genuinely-reached LOCAL people shows up and buys
    "shares": 0.10,
    "dms": 0.40,
    "link_clicks": 0.15,
    "rsvps": 0.30,        # RSVPs are flaky, about 30% come and buy
    "preorders": 1.00,
    "promo_redemptions_last": 0.50,
    "email_sms_clicks": 0.20,
}


def authenticity(s: SocialStats) -> tuple[float, list[str]]:
    """0.2-1.0 score for how much of the audience is real, active people.
    Heuristics used by influencer-audit tools:
      * engagement rate (likes+comments / followers): real small accounts sit ~1-8%.
        Way below means fake/dead followers; way above with few comments means bought likes.
      * comment-to-like ratio: real ~1-5%; bought likes rarely come with comments.
      * sudden follower jumps with no engagement jump = bought followers.
    """
    score, notes = 1.0, []
    f, likes, com = s.followers, s.avg_likes, s.avg_comments
    if f and likes is not None:
        er = (likes + (com or 0)) / max(f, 1)
        if er < 0.005:
            score *= 0.4; notes.append(f"very low engagement rate ({er:.1%}): many followers look fake or inactive")
        elif er < 0.01:
            score *= 0.7; notes.append(f"low engagement rate ({er:.1%})")
        elif er > 0.20 and f > 2000:
            score *= 0.6; notes.append(f"suspiciously high engagement rate ({er:.1%})")
    if likes and com is not None and likes > 50:
        cl = com / likes
        if cl < 0.003:
            score *= 0.6; notes.append(f"almost no comments per like ({cl:.2%}): likes may be bought")
        elif cl < 0.01:
            score *= 0.85; notes.append(f"few comments per like ({cl:.2%})")
    g = s.follower_growth_30d_pct
    if g is not None and g > 30:
        score *= 0.7; notes.append(f"follower spike (+{g:.0f}% in 30 days)")
    return float(np.clip(score, 0.2, 1.0)), notes


def social_score(accounts: list[SocialStats] | None = None,
                 intent: IntentSignals | None = None) -> dict:
    """Combine every social/intent signal into 'expected extra buyers'-ish units.
    Returns {"social_index": float, "authenticity": float, "notes": [...], "parts": {...}}"""
    accounts = accounts or []
    intent = intent or IntentSignals()
    parts, notes, auths = {}, [], []

    for a in accounts:
        auth, n = authenticity(a)
        auths.append(auth)
        notes += [f"{a.platform}: {x}" for x in n]
        local = (a.local_audience_pct / 100) if a.local_audience_pct is not None else 0.5
        reach = a.event_post_reach
        if reach is None and a.followers:     # assume a post reaches ~25% of followers
            reach = 0.25 * a.followers
        k = a.platform
        parts[f"{k}_reach"] = (reach or 0) * auth * local * INTENT_WEIGHTS["reach"]
        parts[f"{k}_shares"] = (a.shares or 0) * auth * INTENT_WEIGHTS["shares"]
        parts[f"{k}_dms"] = (a.dms or 0) * INTENT_WEIGHTS["dms"]                 # DMs are hard to fake
        parts[f"{k}_link_clicks"] = (a.link_clicks or 0) * auth * INTENT_WEIGHTS["link_clicks"]

    for name in ("rsvps", "preorders", "promo_redemptions_last", "email_sms_clicks"):
        parts[name] = getattr(intent, name) * INTENT_WEIGHTS[name]
    for name, (value, weight) in intent.custom.items():
        parts[f"custom:{name}"] = float(value) * float(weight)

    return {
        "social_index": round(float(sum(parts.values())), 2),
        "authenticity": round(float(np.mean(auths)) if auths else 1.0, 2),
        "notes": notes,
        "parts": {k: round(v, 2) for k, v in parts.items() if v},
    }


SOCIAL_RAW_COLUMNS = ["followers", "avg_likes", "avg_comments", "event_post_reach",
                      "local_audience_pct", "follower_growth_30d_pct", "shares", "dms",
                      "link_clicks", "rsvps", "preorders", "promo_redemptions_last",
                      "email_sms_clicks"]


def social_index_from_row(row: pd.Series) -> float:
    """Build social_index from raw columns in a history CSV (single account)."""
    def g(k):
        v = row.get(k)
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    acct = SocialStats(**{k: g(k) for k in SocialStats.__dataclass_fields__ if k != "platform"})
    intent = IntentSignals(**{k: g(k) or 0 for k in ("rsvps", "preorders", "promo_redemptions_last", "email_sms_clicks")})
    has_acct = any(getattr(acct, k) is not None for k in acct.__dataclass_fields__ if k != "platform")
    return social_score([acct] if has_acct else [], intent)["social_index"]


# =============================================================================
# 2. Nearby competitors (Google Places API, "New" version)
# =============================================================================

# Which Google place types compete with each product category.
# Same category = direct competitor; "near" types count at 40%.
CATEGORY_PLACE_TYPES = {
    "coffee":    {"direct": ["coffee_shop", "cafe"], "near": ["tea_house", "bakery", "donut_shop"]},
    "tea":       {"direct": ["tea_house"], "near": ["cafe", "coffee_shop", "juice_shop"]},
    "bakery":    {"direct": ["bakery", "donut_shop"], "near": ["cafe", "dessert_shop"]},
    "dessert":   {"direct": ["dessert_shop", "ice_cream_shop", "confectionery"], "near": ["bakery", "donut_shop"]},
    "ice_cream": {"direct": ["ice_cream_shop"], "near": ["dessert_shop"]},
    "drinks":    {"direct": ["juice_shop"], "near": ["cafe", "coffee_shop", "tea_house"]},
    "alcohol":   {"direct": ["bar", "pub", "wine_bar"], "near": []},
    "burger":    {"direct": ["hamburger_restaurant"], "near": ["fast_food_restaurant", "american_restaurant"]},
    "pizza":     {"direct": ["pizza_restaurant"], "near": ["fast_food_restaurant", "italian_restaurant"]},
    "tacos":     {"direct": ["mexican_restaurant"], "near": ["fast_food_restaurant"]},
    "asian":     {"direct": ["chinese_restaurant", "japanese_restaurant", "thai_restaurant",
                             "korean_restaurant", "ramen_restaurant", "vietnamese_restaurant"], "near": []},
    "sandwich":  {"direct": ["sandwich_shop"], "near": ["fast_food_restaurant", "cafe"]},
    "food":      {"direct": ["fast_food_restaurant", "restaurant"], "near": []},
    "clothing":  {"direct": ["clothing_store"], "near": ["shoe_store"]},
    "merch":     {"direct": ["gift_shop"], "near": ["clothing_store"]},
    "jewelry":   {"direct": ["jewelry_store"], "near": ["gift_shop"]},
    "books":     {"direct": ["book_store"], "near": []},
    "plants":    {"direct": ["florist", "garden_center"], "near": []},
}

DEFAULT_RADIUS_M = 500     # ~6-minute walk. People at a pop-up rarely walk further for a substitute.
DISTANCE_DECAY_M = 200     # a shop 200 m away counts ~37% as much as one next door

_CACHE_FILE = Path(__file__).parent / ".places_cache.json"


def guess_category(product: str) -> str:
    p = product.lower()
    rules = [("coffee", "coffee"), ("latte", "coffee"), ("espresso", "coffee"), ("matcha", "tea"),
             ("tea", "tea"), ("boba", "tea"), ("pastry", "bakery"), ("croissant", "bakery"),
             ("bread", "bakery"), ("cookie", "dessert"), ("cake", "dessert"), ("ice cream", "ice_cream"),
             ("gelato", "ice_cream"), ("lemonade", "drinks"), ("juice", "drinks"), ("beer", "alcohol"),
             ("wine", "alcohol"), ("burger", "burger"), ("pizza", "pizza"), ("taco", "tacos"),
             ("burrito", "tacos"), ("ramen", "asian"), ("dumpling", "asian"), ("sandwich", "sandwich"),
             ("shirt", "clothing"), ("hoodie", "clothing"), ("tee", "clothing"), ("tote", "merch"),
             ("sticker", "merch"), ("necklace", "jewelry"), ("ring", "jewelry"), ("book", "books"),
             ("plant", "plants")]
    for kw, cat in rules:
        if kw in p:
            return cat
    return "food"


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_nearby_places(lat: float, lon: float, place_types: list[str], api_key: str,
                        radius_m: int = DEFAULT_RADIUS_M) -> list[dict]:
    """Google Places API (New) Nearby Search. Max 20 results per call, closest first.
    Needs an API key with "Places API (New)" enabled. Results are cached on disk."""
    if not place_types:
        return []
    key = f"{lat:.4f},{lon:.4f},{radius_m},{','.join(sorted(place_types))}"
    cache = json.loads(_CACHE_FILE.read_text()) if _CACHE_FILE.exists() else {}
    if key in cache:
        return cache[key]

    body = {
        "includedTypes": place_types,
        "maxResultCount": 20,
        "rankPreference": "DISTANCE",
        "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": float(radius_m)}},
    }
    req = urllib.request.Request(
        "https://places.googleapis.com/v1/places:searchNearby",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.location,places.types,"
                                "places.userRatingCount,places.rating,places.businessStatus",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        places = json.load(r).get("places", [])
    out = [{
        "name": p.get("displayName", {}).get("text", "?"),
        "lat": p["location"]["latitude"], "lon": p["location"]["longitude"],
        "types": p.get("types", []),
        "reviews": p.get("userRatingCount", 0),
        "rating": p.get("rating"),
        "open": p.get("businessStatus", "OPERATIONAL") == "OPERATIONAL",
    } for p in places]
    cache[key] = out
    try:
        _CACHE_FILE.write_text(json.dumps(cache))
    except OSError:
        pass
    return out


def competitor_index(lat: float, lon: float, category: str, api_key: str,
                     radius_m: int = DEFAULT_RADIUS_M, other_vendors_same_category: int = 0) -> dict:
    """Distance- and popularity-weighted count of competitors for one product category.
    other_vendors_same_category: stalls at the same market selling the same thing.
    Google can't see those, and inside a market they're usually the biggest competition."""
    spec = CATEGORY_PLACE_TYPES.get(category, CATEGORY_PLACE_TYPES["food"])
    places = fetch_nearby_places(lat, lon, spec["direct"] + spec["near"], api_key, radius_m)
    idx, listed = 0.0, []
    for p in places:
        if not p["open"]:
            continue
        d = _haversine_m(lat, lon, p["lat"], p["lon"])
        direct = any(t in spec["direct"] for t in p["types"])
        popularity = 0.5 + 0.5 * min(1.0, math.log1p(p["reviews"]) / math.log1p(500))
        w = math.exp(-d / DISTANCE_DECAY_M) * popularity * (1.0 if direct else 0.4)
        idx += w
        listed.append({"name": p["name"], "distance_m": round(d), "direct": direct, "weight": round(w, 2)})
    idx += 1.0 * other_vendors_same_category   # a stall 20 m away = full weight
    return {"competitor_index": round(idx, 2), "places": sorted(listed, key=lambda x: x["distance_m"])}


# =============================================================================
# 3. Time of day + time of year
# =============================================================================

def _bumps(*peaks):
    """24-hour curve from (hour, width, height) bumps, normalised to sum to 1."""
    h = np.arange(24)
    y = np.full(24, 0.02)
    for c, w, a in peaks:
        y += a * np.exp(-0.5 * ((h - c) / w) ** 2)
    y[:6] *= 0.1
    return y / y.sum()


# When during the day each category sells. Defaults from typical retail/food patterns;
# replaced by YOUR pattern once you load timestamped sales (Square/Shopify export).
DEFAULT_HOURLY = {
    "coffee":    _bumps((8.5, 1.5, 1.0), (14, 1.5, 0.4)),
    "tea":       _bumps((11, 2, 0.6), (15, 2, 0.8)),
    "bakery":    _bumps((9.5, 1.5, 1.0), (14, 2, 0.3)),
    "dessert":   _bumps((15, 2, 0.8), (20, 1.5, 0.7)),
    "ice_cream": _bumps((15, 2, 1.0), (19.5, 1.5, 0.6)),
    "drinks":    _bumps((13, 2.5, 1.0), (17, 2, 0.5)),
    "alcohol":   _bumps((19, 1.5, 0.8), (21.5, 1.5, 1.0)),
    "burger":    _bumps((12.5, 1.2, 1.0), (18.5, 1.5, 0.9)),
    "pizza":     _bumps((12.5, 1.2, 0.8), (19, 1.8, 1.0)),
    "tacos":     _bumps((12.5, 1.2, 1.0), (19, 2, 0.9)),
    "asian":     _bumps((12.5, 1.2, 0.9), (18.5, 1.5, 1.0)),
    "sandwich":  _bumps((12.5, 1.3, 1.0), (17, 1.5, 0.3)),
    "food":      _bumps((12.5, 1.3, 1.0), (18.5, 1.6, 0.9)),
    "default":   _bumps((13, 3.0, 1.0), (17, 2.0, 0.6)),   # merch, clothing, etc.
}
REFERENCE_HOURS = (10, 16)   # "normal" market hours; coverage is relative to this


def hourly_profile(category: str, learned: dict | None = None, product: str | None = None) -> np.ndarray:
    if learned and product in learned:
        return learned[product]
    return DEFAULT_HOURLY.get(category, DEFAULT_HOURLY["default"])


def learn_hourly_profiles(transactions: pd.DataFrame, categories: dict,
                          prior_strength: float = 30.0) -> dict:
    """transactions: columns timestamp, product, qty (Square/Shopify item export).
    Returns product -> 24-hour profile. Blends your data with the category default,
    as if the default were worth `prior_strength` sales, so a few sales don't overfit."""
    t = transactions.copy()
    t["hour"] = pd.to_datetime(t["timestamp"]).dt.hour
    out = {}
    for p, g in t.groupby("product"):
        counts = np.bincount(g["hour"], weights=g.get("qty", pd.Series(1, index=g.index)), minlength=24)[:24]
        prior = DEFAULT_HOURLY.get(categories.get(p, guess_category(p)), DEFAULT_HOURLY["default"])
        prof = counts + prior_strength * prior
        out[p] = prof / prof.sum()
    return out


def hours_coverage(profile: np.ndarray, start_hour: float, end_hour: float) -> float:
    """Share of a normal day's demand that falls inside your open hours, relative to
    a 10am-4pm market. 1.0 = same as 10-4; 0.5 = half as much demand."""
    def share(a, b):
        hrs = np.arange(24)
        overlap = np.clip(np.minimum(hrs + 1, b) - np.maximum(hrs, a), 0, 1)
        return float((profile * overlap).sum())
    return max(share(start_hour, end_hour), 1e-3) / share(*REFERENCE_HOURS)


def prep_schedule(profile: np.ndarray, start_hour: int, end_hour: int, total_units: int) -> pd.DataFrame:
    """Split the recommended units across open hours: how much to have ready by when."""
    hrs = list(range(int(start_hour), int(math.ceil(end_hour))))
    w = np.array([profile[h] for h in hrs])
    w = w / w.sum() if w.sum() else np.full(len(hrs), 1 / len(hrs))
    units = np.round(w * total_units).astype(int)
    return pd.DataFrame({"hour": [f"{h:02d}:00" for h in hrs], "units": units,
                         "cumulative": np.cumsum(units)})


def us_holidays(year: int) -> set[date]:
    """Major US holidays that change pop-up traffic (no extra package needed)."""
    def nth_weekday(month, weekday, n):  # n=-1 for last
        if n > 0:
            d = date(year, month, 1)
            d += timedelta(days=(weekday - d.weekday()) % 7)
            return d + timedelta(weeks=n - 1)
        d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        return d - timedelta(days=(d.weekday() - weekday) % 7)
    return {
        date(year, 1, 1), nth_weekday(1, 0, 3), nth_weekday(2, 0, 3), date(year, 2, 14),
        nth_weekday(5, 0, -1), date(year, 6, 19), date(year, 7, 4), nth_weekday(9, 0, 1),
        date(year, 10, 31), nth_weekday(11, 3, 4), nth_weekday(11, 3, 4) + timedelta(days=1),
        date(year, 12, 24), date(year, 12, 25), date(year, 12, 31), nth_weekday(5, 6, 2),  # Mother's Day
    }


def is_holiday_window(d) -> float:
    """1.0 on a holiday or the day either side (holiday weekends)."""
    d = pd.Timestamp(d).date()
    hs = us_holidays(d.year) | us_holidays(d.year + 1) | us_holidays(d.year - 1)
    return float(any(abs((d - h).days) <= 1 for h in hs))
