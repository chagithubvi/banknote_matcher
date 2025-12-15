import os, re, pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==================== Paths ====================
DATASET_DIR = r"D:\New folder\banknote_app\syrcurr"
ENCODER_PATH = r"D:\New folder\banknote-net\models\banknote_net_encoder.h5"
OUTPUT_EMBEDDINGS = r"D:\New folder\banknote_app\banknote2_embeddings.pkl"

# ==================== Patch DepthwiseConv2D ====================
class DepthwiseConv2DPatched(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

# ==================== Load Model ====================
encoder = load_model(
    ENCODER_PATH,
    custom_objects={"DepthwiseConv2D": DepthwiseConv2DPatched}
)

# ==================== Helpers ====================
UNIT_MAPPING = {
    "L": "Livres",
    "LIV": "Livres",
    "PIAS": "Piastres",
    "P": "Pounds"
}

def parse_filename(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")

    currency = parts[0]
    denom_raw = parts[1]
    year = parts[2]

    # side: always last token (F, F2, B, B2)
    side_token = parts[-1].upper()
    side = "front" if side_token.startswith("F") else "back"

    # version: optional token before side (V1, V2, ...)
    version = "V1"
    if len(parts) >= 5 and re.fullmatch(r"V\d+", parts[-2], re.IGNORECASE):
        version = parts[-2].upper()

    m = re.match(r"(\d+)([A-Za-z]+)", denom_raw)
    denom_val, denom_unit = m.groups() if m else (denom_raw, "")
    denom_unit = UNIT_MAPPING.get(denom_unit.upper(), denom_unit)

    return currency, year, denom_val, denom_unit, side, version

# ==================== Generate Embeddings ====================
gallery = []

for fname in tqdm(os.listdir(DATASET_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = load_img(os.path.join(DATASET_DIR, fname), target_size=(224, 224))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    emb = encoder.predict(x, verbose=0)
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)

    currency, year, dval, dunit, side, version = parse_filename(fname)

    gallery.append({
        "filename": fname,
        "currency": currency,
        "year": year,
        "denomination": f"{dval} {dunit}",
        "version": version,
        "side": side,
        "embedding": emb[0]
    })

# ==================== Save ====================
with open(OUTPUT_EMBEDDINGS, "wb") as f:
    pickle.dump(gallery, f)

print(f"Saved {len(gallery)} embeddings → {OUTPUT_EMBEDDINGS}")
