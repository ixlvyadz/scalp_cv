import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so "src" can be imported

import requests
import streamlit as st
from PIL import Image

from src.predict import predict

API_URL = os.environ.get("API_URL")     # unset: the model runs inside the app


def get_result(raw):
    if API_URL:
        r = requests.post(f"{API_URL}/predict", files={"file": ("photo.jpg", raw)}, timeout=30)
        r.raise_for_status()
        return r.json()
    return predict(Image.open(io.BytesIO(raw)))


st.set_page_config(page_title="Hair and Scalp Condition Classifier")
st.title("Hair and Scalp Condition Classifier")
st.warning("Educational demo only. This is not a medical diagnosis. "
           "Please see a dermatologist or other health professional about any concern.")

f = st.file_uploader("Upload a photo of the hair or scalp", type=["jpg", "jpeg", "png"])
if f:
    raw = f.getvalue()
    img = Image.open(io.BytesIO(raw))
    st.image(img, caption="Your photo")
    r = get_result(raw)
    if r["uncertain"]:
        st.error("Uncertain. The model is not confident about this photo, "
                 "so please consult a professional or try a clearer photo.")
        st.caption("Closest matches (low confidence):")
    else:
        st.success(f"Most likely: {r['top'][0][0]}")
    for name, p in r["top"]:
        st.write(f"**{name}**: {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
    st.caption("Confidence is calibrated on photos from the training dataset and may be less "
               "reliable for photos taken in different conditions.")