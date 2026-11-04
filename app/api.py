import io, os
import requests

API_URL = os.environ.get("API_URL")     # unset: run the model inside the app


def get_result(raw):
    if API_URL:
        r = requests.post(f"{API_URL}/predict", files={"file": ("photo.jpg", raw)}, timeout=30)
        r.raise_for_status()
        return r.json()
    return predict(Image.open(io.BytesIO(raw)))