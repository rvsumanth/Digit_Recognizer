"""
FastAPI backend for the Handwritten Digit Recognizer.

Serves a single-page web UI (static/index.html) and exposes a POST /predict
endpoint that accepts a base64-encoded canvas image and returns the model's
predicted digit plus the full probability distribution.

Run with:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000 in your browser.
"""

import base64
import io
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "saved_models", "mnist_cnn_model.h5")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

app = FastAPI(title="Digit Recognizer API")

model = None  # loaded on startup


@app.on_event("startup")
def load_model() -> None:
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file '{MODEL_PATH}' not found. Run model.py first to train "
            "and save it, then place it next to main.py (or set MODEL_PATH)."
        )
    model = tf.keras.models.load_model(MODEL_PATH)


class ImagePayload(BaseModel):
    image: str  # base64 data URL, e.g. "data:image/png;base64,...."


def preprocess(data_url: str) -> np.ndarray:
    """Convert a base64 data URL from the canvas into a (1, 28, 28, 1) array,
    using the same centering/scaling convention MNIST digits were built with.

    A raw resize (320x320 -> 28x28) does NOT match how MNIST digits look:
    real MNIST digits are cropped to their bounding box, scaled to fit inside
    a 20x20 box, and centered in the 28x28 frame by center of mass. Skipping
    this step is the #1 reason browser-drawn digits get misclassified (often
    collapsing to a single "favorite" class like 8) - the input distribution
    just doesn't match what the model was trained on.
    """
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    try:
        img_bytes = base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    arr = np.array(img).astype("float32")

    # Find the bounding box of the drawn strokes (non-background pixels).
    # Canvas is black background (0) with white strokes (255).
    threshold = 20
    rows = np.any(arr > threshold, axis=1)
    cols = np.any(arr > threshold, axis=0)

    if not rows.any() or not cols.any():
        # Blank canvas - fall back to a plain resize, model will just be unsure.
        img = img.resize((28, 28), Image.LANCZOS)
        out = np.array(img).astype("float32") / 255.0
        return out.reshape(1, 28, 28, 1)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    digit = Image.fromarray(arr[rmin:rmax + 1, cmin:cmax + 1].astype("uint8"))

    # Scale the cropped digit to fit inside a 20x20 box, preserving aspect ratio.
    w, h = digit.size
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = digit.resize((new_w, new_h), Image.LANCZOS)

    # Paste onto a 28x28 black canvas, centered by center of mass.
    canvas28 = Image.new("L", (28, 28), color=0)
    digit_arr = np.array(digit).astype("float32")
    total = digit_arr.sum()
    if total > 0:
        yy, xx = np.indices(digit_arr.shape)
        com_x = (xx * digit_arr).sum() / total
        com_y = (yy * digit_arr).sum() / total
    else:
        com_x, com_y = new_w / 2.0, new_h / 2.0

    paste_x = round(14 - com_x)
    paste_y = round(14 - com_y)
    canvas28.paste(digit, (paste_x, paste_y))

    out = np.array(canvas28).astype("float32") / 255.0
    return out.reshape(1, 28, 28, 1)


@app.post("/predict")
def predict(payload: ImagePayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    arr = preprocess(payload.image)
    preds = model.predict(arr, verbose=0)[0]

    return {
        "digit": int(np.argmax(preds)),
        "confidence": float(np.max(preds)),
        "probabilities": [float(p) for p in preds],
    }


# Serve the single-page UI. Mounted last so it doesn't shadow /predict.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")