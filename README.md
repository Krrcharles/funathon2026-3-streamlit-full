# Funathon 2026 — Streamlit full acquisition pipeline

This Streamlit app implements the **API-based full pipeline** from `1-acquisition.qmd`:

1. Input an address/city.
2. Geocode via Nominatim API.
3. Query CDSE OData catalog for the most recent Sentinel-2 L2A product intersecting the point.
4. Call Sentinel Hub Process API (CDSE) to fetch an RGB tile (B04/B03/B02).
5. Call Copernicus CLC+ ImageServer API for the exact same area.
6. Overlay RGB + CLC+ label on an interactive map.

## Credentials

Set environment variables:

- `CDSE_CLIENT_ID`
- `CDSE_CLIENT_SECRET`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown by Streamlit.
