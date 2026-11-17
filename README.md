# Pop-up demand predictor (v0)

Tells a pop-up vendor how many units of each product to bring to the next event.

```bash
pip install -r requirements.txt
streamlit run app.py    # browser app at http://localhost:8501
```

```python
import pandas as pd
from predictor import PopupDemandModel, fetch_weather

history = pd.read_csv("my_events.csv")   # one row per product per event
model = PopupDemandModel().fit(history)

event = {"date": "2026-11-21", "event_type": "farmers_market",
         "expected_attendance": 1200, "hours": 5,
         **fetch_weather(40.71, -74.00, "2026-11-21")}  # free Open-Meteo, no key

for r in model.recommend(event, salvage={"tote_bag": 8.0}):   # 0 for perishables
    print(r.product, r.bring, f"(demand {r.low}-{r.high})")

print(model.explain())   # what the model learned, e.g. rain = 0.66x demand
```

## Using the app
- Starts completely empty. Import CSV/Excel files (Square, Shopify or your own sheet) and/or type rows into the event table. Column names are matched loosely ("Units Sold" = units_sold).
- Your log is saved to `my_events.csv` in the app folder and reloaded next time.
- Every input is blank until you fill it; blank inputs are simply ignored.
- **Attendance from social:** if you leave attendance blank but enter social numbers, it estimates attendance. Once 3+ past events have both logged it uses your own ratio; before that it uses social index ÷ 5% buy rate, which is a rough floor.
- **Two predictors:** the main model, plus a "similar past events" second opinion that averages what you sold at the 5 most similar past events and shows exactly which ones.

## Input columns
| column | required | notes |
|---|---|---|
| event_id, date, product, units_sold | yes | |
| units_brought | strongly recommended | lets the model tell a sellout from low demand |
| event_type | optional | farmers_market, festival, night_market, brand_popup, ... |
| category | optional | coffee, burger, merch... (guessed from the product name if missing) |
| expected_attendance | optional | organizer's estimate or your own tally count |
| temp_c, precip_mm | optional | `fetch_weather()` fills these for past and future dates |
| start_hour, end_hour | optional | opening hours, 24h. Drives the time-of-day effect |
| price, unit_cost | optional | used for the bring-more-or-less decision |
| social_index *or* followers, avg_likes, avg_comments, event_post_reach, local_audience_pct, follower_growth_30d_pct, shares, dms, link_clicks, rsvps, preorders | optional | see Social below |
| competitor_index | optional | per product; from `signals.competitor_index()` |

## How it works
1. **Demand model:** Poisson regression on product, event type, log attendance, rain, temperature (with a per-product heat effect), weekend, holidays, season, social index, competitor index and event length. Ridge shrinkage lets thin data borrow from the rest.
2. **Sellouts:** if you sold everything you brought, demand was *at least* that. Those rows are re-estimated (EM).
3. **Uncertainty:** negative binomial gives a 10th–90th percentile range.
4. **Decision (newsvendor):** bring the demand quantile at `(price − cost) / (price − salvage + holding)`.
5. **Cold start:** products with little history are blended with `attendance × conversion × units/buyer × share`.

## Social media (`signals.social_score`)
Turns everything into one "≈ extra buyers" number:
- **Intent counts most:** preorders (1.0 buyer each), DMs (0.4), RSVPs (0.3), link clicks (0.15), shares/saves (0.1). Reach counts at 1%, and only the local share of your audience.
- **Bot discount (authenticity 20–100%):** flags engagement rate under 1% (fake/dead followers) or over 20% (bought likes), comments under ~1% of likes, and follower spikes over 30%/month. Reach, shares and clicks get multiplied down. DMs and preorders aren't, since they're hard to fake.
- **Anything else:** `IntentSignals(custom={"tiktok_views": (12000, 0.001), "flyers": (300, 0.02)})`, value and buyers-per-unit weight. Multiple accounts/platforms can be passed as a list.
The weights are starting guesses; the model learns how much social_index actually matters for you once it's logged with ~15+ events.

## Nearby competitors (`signals.competitor_index`)
- Google **Places API (New)** Nearby Search. Enable it in Google Cloud and create an API key. Results are cached in `.places_cache.json`.
- Each product category maps to Google place types: burger → `hamburger_restaurant` (direct) + `fast_food_restaurant` (40% weight), coffee → `coffee_shop`, `cafe` (+ bakeries at 40%), etc. So a burger joint next door hurts your burgers but not your coffee.
- Default radius **500 m** (~6-minute walk). Each place is weighted by distance (one 200 m away counts ~37%) and popularity (review count). Closed places are skipped.
- Add `other_vendors_same_category` for stalls at the same market; Google can't see those, and they're often the main competition.

## Time of day and year
- **Time of day:** every category has a default hourly demand curve (coffee peaks ~8–9am, burgers at lunch and ~6:30pm, alcohol late evening). Opening hours apply as a share of the day's demand vs a 10am–4pm market, so iced coffee at a 5–10pm night market is cut to a fraction. Upload timestamped sales (timestamp, product, qty) and `learn_hourly_profiles()` replaces the defaults with your real pattern.
- **Prep schedule:** `model.schedule(product, event, units)` splits the recommendation by hour.
- **Time of year:** US holiday windows (±1 day) and a smooth seasonal curve. Both are shrunk hard, since you need close to a full year of events before seasonality is trustworthy (weather already explains a lot of it).

## Demo result (synthetic)
80 events, 3 products (iced coffee, smash burger, tote bag). Trained on 60 events, tested on 20:
gut rule (1.2× average past sales) **$7,314** vs model **$8,987** profit (**+23%**).
Synthetic data flatters any model, so treat this as "the plumbing works", not a real-world number.

## Next steps
- Square / Shopify sales import
- Simple event-logging form (date, location, type, attendance, brought, sold)
- Real-data backtest once someone has 15–20 logged events
