import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import folium
import numpy as np
import requests
import streamlit as st
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from rasterio.io import MemoryFile
from streamlit_folium import st_folium

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SENTINEL_HUB_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
ODATA_COLLECTION_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

MLFLOW_MODEL_S3_RUN_PATH = "projet-funathon/mlflow-artifacts/1/a2cc538d334b474389efcc54115ef13d/artifacts/"
MLFLOW_ENDPOINT = "https://minio.lab.sspcloud.fr"

CLC_CLASSES = [
    ("Sealed (1)", "#FF0100"),
    ("Woody -- needle leaved trees (2)", "#238B23"),
    ("Woody -- Broadleaved deciduous trees (3)", "#80FF00"),
    ("Woody -- Broadleaved evergreen trees (4)", "#00FF00"),
    ("Low-growing woody plants (5)", "#804000"),
    ("Permanent herbaceous (6)", "#CCF24E"),
    ("Periodically herbaceous (7)", "#FEFF80"),
    ("Lichens and mosses (8)", "#FF81FF"),
    ("Non- and sparsely-vegetated (9)", "#BFBFBF"),
    ("Water (10)", "#0080FF"),
]


def geocode_address(address: str) -> tuple[float, float, str]:
    params = {"q": address, "format": "jsonv2", "limit": 1}
    headers = {"User-Agent": "funathon-streamlit-pipeline/1.0"}
    resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError("No geocoding result for this address.")
    return float(data[0]["lon"]), float(data[0]["lat"]), data[0]["display_name"]


def cdse_access_token(client_id: str, client_secret: str) -> str:
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    resp = requests.post(CDSE_TOKEN_URL, data=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def query_best_l2a_product(lon: float, lat: float, start_date: date, end_date: date) -> dict:
    footprint = f"POINT({lon} {lat})"
    filt = (
        "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and "
        f"ContentDate/Start ge {start_date.isoformat()}T00:00:00.000Z and "
        f"ContentDate/Start le {end_date.isoformat()}T23:59:59.999Z"
    )
    params = {
        "$filter": filt,
        "$orderby": "ContentDate/Start desc",
        "$top": 1,
    }
    resp = requests.get(ODATA_COLLECTION_URL, params=params, timeout=45)
    resp.raise_for_status()
    vals = resp.json().get("value", [])
    if not vals:
        raise ValueError("No Sentinel-2 L2A products found for this area/date range.")
    return vals[0]


def sentinelhub_rgb_ndvi(token: str, lon: float, lat: float, half_size_deg: float, size_px: int):
    bbox = [lon - half_size_deg, lat - half_size_deg, lon + half_size_deg, lat + half_size_deg]
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"mosaickingOrder": "mostRecent"}}],
        },
        "output": {"width": size_px, "height": size_px, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": """
//VERSION=3
function setup() {
  return {input: ["B04", "B03", "B02", "B08"], output: {bands: 4, sampleType: "FLOAT32"}};
}
function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02, sample.B08];
}
""",
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(SENTINEL_HUB_PROCESS_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()

    with MemoryFile(resp.content) as memfile:
        with memfile.open() as src:
            bands = src.read([1, 2, 3, 4]).astype(np.float32)
            bounds_4326 = src.bounds

    rgb = np.transpose(bands[:3], (1, 2, 0))
    p98 = np.percentile(rgb, 98)
    rgb = np.clip(rgb / (p98 if p98 > 0 else 1), 0, 1)

    red = bands[0]
    nir = bands[3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1, 1)

    ndvi_norm = (ndvi + 1) / 2.0
    ndvi_rgba = colormaps["RdYlGn"](ndvi_norm).astype(np.float32)
    ndvi_rgba[..., 3] = 0.65
    return rgb, ndvi_rgba, bands, bounds_4326


@st.cache_resource(show_spinner=False)
def load_inference_model():
    import mlflow
    import s3fs

    fs = s3fs.S3FileSystem(anon=True, endpoint_url=MLFLOW_ENDPOINT)
    local_model_dir = Path(tempfile.mkdtemp()) / "model"
    fs.get(MLFLOW_MODEL_S3_RUN_PATH + "model", str(local_model_dir), recursive=True)
    try:
        model = mlflow.pyfunc.load_model(str(local_model_dir))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the pretrained MLflow model. "
            "This is usually caused by an incompatible transformers version; "
            "use transformers==4.46.3 and tokenizers==0.20.3."
        ) from exc

    run_params = requests.get(MLFLOW_ENDPOINT + "/" + MLFLOW_MODEL_S3_RUN_PATH + "params.json", timeout=30).json()
    return model, run_params


def predict_clc_like_rgba(bands: np.ndarray, alpha: float = 0.55):
    model, run_params = load_inference_model()
    n_bands = int(run_params["n_bands"])
    norm_mean = np.array(run_params["normalization_mean"][:n_bands], dtype=np.float32)
    norm_std = np.array(run_params["normalization_std"][:n_bands], dtype=np.float32)

    chw = bands[:n_bands].astype(np.float32)
    hwc = np.transpose(chw, (1, 2, 0))
    hwc = (hwc - norm_mean) / (norm_std + 1e-6)
    batch = np.expand_dims(np.transpose(hwc, (2, 0, 1)), 0)

    logits = model.predict(batch)
    pred = np.argmax(logits[0], axis=0).astype(np.uint8) + 1

    color_lut = np.zeros((11, 4), dtype=np.float32)
    for idx, (_, hex_color) in enumerate(CLC_CLASSES, start=1):
        color_lut[idx] = list(to_rgba(hex_color, alpha=alpha))
    return color_lut[pred], np.unique(pred).tolist()


def clc_label_rgba(bounds_4326, year: int, out_width: int, out_height: int, alpha: float = 0.6):
    xmin, ymin, xmax, ymax = bounds_4326.left, bounds_4326.bottom, bounds_4326.right, bounds_4326.top
    export_url = (
        f"https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/CLC_plus/"
        f"CLMS_CLCplus_RASTER_{year}_010m_eu/ImageServer/exportImage"
    )
    params = {
        "f": "image", "bbox": f"{xmin},{ymin},{xmax},{ymax}", "bboxSR": "4326", "imageSR": "4326",
        "size": f"{int(out_width)},{int(out_height)}", "format": "tiff", "interpolation": "RSP_NearestNeighbor",
    }
    resp = requests.get(export_url, params=params, timeout=90)
    resp.raise_for_status()
    with MemoryFile(resp.content) as memfile:
        with memfile.open() as src:
            label = src.read(1)
    label[(label == 254) | (label == 255)] = 0
    color_lut = np.zeros((11, 4), dtype=np.float32)
    color_lut[0] = [0, 0, 0, 0]
    for idx, (_, hex_color) in enumerate(CLC_CLASSES, start=1):
        color_lut[idx] = list(to_rgba(hex_color, alpha=alpha))
    label = np.clip(label, 0, 10).astype(np.uint8)
    return color_lut[label], np.unique(label).tolist()


def main():
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    st.set_page_config(page_title="Funathon Full Acquisition Pipeline", layout="wide")
    st.title("Funathon — Full API Pipeline (Address → Sentinel-2 RGB + CLC+)")

    with st.sidebar:
        with st.form("pipeline_form"):
            address = st.text_input("Address / city", value="Luxembourg City, Luxembourg")
            year = st.number_input("CLC+ year", min_value=2021, max_value=2021, value=2021)
            half_size_deg = st.slider("Tile half-size (degrees)", 0.005, 0.03, 0.0125, 0.0025)
            size_px = st.selectbox("Output image size", [256, 384, 512], index=2)
            end_d = st.date_input("Latest acquisition date", value=date.today())
            days_back = st.slider("Lookback window (days)", 5, 365, 60)
            run_btn = st.form_submit_button("Run full pipeline")

    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        st.error("Missing CDSE credentials in env vars: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET")
        st.stop()

    if run_btn:
        try:
            lon, lat, resolved = geocode_address(address)
            product = query_best_l2a_product(lon, lat, end_d - timedelta(days=days_back), end_d)
            token = cdse_access_token(client_id, client_secret)
            rgb, ndvi_rgba, bands, bounds_4326 = sentinelhub_rgb_ndvi(token, lon, lat, half_size_deg, size_px)
            out_h, out_w = rgb.shape[0], rgb.shape[1]
            label_rgba, classes_present = clc_label_rgba(bounds_4326, int(year), out_w, out_h)
            pred_rgba, pred_classes = predict_clc_like_rgba(bands)

            st.session_state.pipeline_result = {
                "resolved": resolved, "lon": lon, "lat": lat,
                "product_name": product.get("Name"), "product_date": product.get("ContentDate", {}).get("Start"),
                "bounds": bounds_4326, "rgb": rgb, "ndvi_rgba": ndvi_rgba,
                "label_rgba": label_rgba, "classes_present": classes_present,
                "pred_rgba": pred_rgba, "pred_classes": pred_classes,
            }
        except Exception as exc:
            st.session_state.pipeline_result = {"error": str(exc)}

    result = st.session_state.pipeline_result
    if not result:
        st.info("Configure inputs and click **Run full pipeline**.")
        return
    if "error" in result:
        st.error(result["error"])
        return

    b = result["bounds"]
    west, south, east, north = b.left, b.bottom, b.right, b.top
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=13)
    folium.raster_layers.ImageOverlay(image=result["rgb"], bounds=[[south, west], [north, east]], name="Sentinel-2 RGB").add_to(m)
    folium.raster_layers.ImageOverlay(image=result["label_rgba"], bounds=[[south, west], [north, east]], name="CLC+ label", opacity=0.8).add_to(m)
    folium.raster_layers.ImageOverlay(image=result["pred_rgba"], bounds=[[south, west], [north, east]], name="Predicted CLC+ like", opacity=0.8).add_to(m)
    folium.raster_layers.ImageOverlay(image=result["ndvi_rgba"], bounds=[[south, west], [north, east]], name="NDVI", opacity=0.8).add_to(m)
    folium.LayerControl(position="topleft").add_to(m)
    st.write(f"Predicted classes: {result['pred_classes']}")
    st_folium(m, width=1100, height=650)


if __name__ == "__main__":
    main()
