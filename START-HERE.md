# Vendor Demand Studio v3

## Quick start (Windows)

1. Extract the ZIP into a folder.
2. Install Python 3.10 or later if you don't already have it.
3. Open the extracted Vendor Predictor folder.
4. Double-click **Start Vendor.bat**. On the first launch it installs the Python dependencies.
5. Your browser opens at http://127.0.0.1:8501.

Keep the terminal window open while using the app. Press Ctrl+C in the terminal to stop it.

## macOS, Linux or manual setup

Open a terminal in the Vendor Predictor folder and run:

```bash
python3 -m pip install -r requirements-web.txt
python3 server.py
```

On Windows, use `py` instead of `python3`.

## How to use the dashboard

- **Overview:** See saved event counts, total units, sellout records and recent sales.
- **Sales history:** Start with an empty log. Import multiple CSV, XLSX or XLS files or manually log, edit and remove sales records. Each row needs date, product and units sold; an event ID is optional. Imports append to your existing history. Column names ignore case and spaces. Download the blank template for the expected layout (exported POS reports may need column mapping first). Optional timestamped sales teach the model when products sell best.
- **Event planner:** Only the event date is required. Add attendance, opening hours, weather and marketing signals if known. If attendance is blank and social activity produces a positive score, the new engine estimates attendance and explains the method. Weather lookup requires date and coordinates and depends on Open-Meteo availability.
- **Products:** Review price, unit cost, leftover value and competition. Leave unknown fields blank; the new engine uses a 60% service target when margins are unknown. Add new products or remove products that won't be on your menu. A Google Places API key is only needed for the optional scan.
- **Inventory forecast:** See the main model recommendation, estimated demand, sellout probability, and a second opinion based on up to five similar events with their actual sales and similarity distances. Enter both opening and closing hours to see prep schedules. Export a comparison CSV.

The included sample history is synthetic and only loaded when you enter demo mode. It never replaces your saved records. Forecasts are estimates, not guarantees. Fewer than ten past events means the model has limited evidence.

The dashboard serves locally on your computer. Your actual event log is automatically saved in my_events.csv when you import, add, edit or remove records. It reloads at startup. Export your log before clearing it. Event planner settings, temporary demo data, product overrides and hourly transactions are session-only. The optional weather and competition lookups require internet access. The newly uploaded predictor.py, signals.py and original Streamlit app.py are preserved.

If port 8501 is occupied, close the other application or set the VENDOR_PORT environment variable to another port before starting server.py.

## Simple Excel template

A blank Excel template with plain columns (Date, Product, Sold, Brought, Event type, Price, Cost,
Attendance, Start time, End time, Temp (°F), Rain) is available at
http://127.0.0.1:8501/api/template.xlsx while the app is running (it also opens in Google Sheets).
Imports understand plain headings and values: times like "11am" or "5:30 PM", temperature in °F,
and rain as none / light / heavy. The original column names still work too.
