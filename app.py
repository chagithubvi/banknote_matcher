import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import ImageFile

# ==================== Fix truncated images ====================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==================== Paths ====================
ENCODER_PATH = "banknote_net_encoder.h5"
EMBEDDINGS_PATH = "banknote2_embeddings.pkl"

# ==================== Patch DepthwiseConv2D ====================
class DepthwiseConv2DPatched(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

# ==================== Country Mapping ====================
COUNTRY_MAPPING = {
    "SYR": "Syria",
    "USA": "United States",
    "IND": "India",
    "EUR": "Eurozone",
}

# ==================== Load Model + Embeddings ====================
@st.cache_resource
def load_resources():
    encoder = load_model(
        ENCODER_PATH,
        custom_objects={"DepthwiseConv2D": DepthwiseConv2DPatched}
    )

    with open(EMBEDDINGS_PATH, "rb") as f:
        gallery = pickle.load(f)

    gallery_embeddings = np.array([
        np.array(item["embedding"]).reshape(-1)
        for item in gallery
    ])

    return encoder, gallery_embeddings, gallery

# ==================== Extract Embedding ====================
def extract_embedding(encoder, image):
    x = img_to_array(image) / 255.0
    x = np.expand_dims(x, axis=0)

    emb = encoder.predict(x, verbose=0)[0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb

# ==================== Find Matches ====================
def find_top_matches(query_emb, gallery_embeddings, gallery_metadata, top_k=3):
    sims = gallery_embeddings @ query_emb
    idxs = sims.argsort()[::-1][:top_k]

    return [{
        "similarity": float(sims[i]),
        **gallery_metadata[i]
    } for i in idxs]

# ==================== Confidence Settings ====================
CONF_THRESHOLD = 0.85
MARGIN = 0.05

# ==================== UI ====================
st.set_page_config("Banknote Identifier", layout="wide")
st.title("🏦 Banknote Identification System")

try:
    encoder, gallery_embeddings, gallery_metadata = load_resources()
    st.success(f"Loaded banknote embeddings")
except Exception as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Banknote Image")
    uploaded = st.file_uploader("Upload image", ["jpg", "jpeg", "png"])

    image = None
    if uploaded:
        image = load_img(uploaded, target_size=(224, 224))
        st.image(image, use_column_width=True)

with col2:
    st.subheader("Results")

    if uploaded and image:
        with st.spinner("Analyzing..."):
            query_emb = extract_embedding(encoder, image)
            matches = find_top_matches(
                query_emb, gallery_embeddings, gallery_metadata
            )

        # confidence check
        confident = True
        if len(matches) >= 2:
            top1, top2 = matches[0], matches[1]
            confident = (
                top1["similarity"] >= CONF_THRESHOLD and
                (top1["similarity"] - top2["similarity"]) >= MARGIN
            )

        if confident:
            st.success("✅ High confidence match")
        else:
            st.warning("⚠️ Multiple possible matches")

        for i, m in enumerate(matches, 1):
            sim = max(0.0, min(1.0, m["similarity"])) * 100
            country = COUNTRY_MAPPING.get(m["currency"], m["currency"])

            version_line = ""
            if m.get("version") and m["version"] != "V1":
                version_line = f"- Version: {m['version']}"

            st.markdown(f"""
**Match #{i}**
- Country: {country}
- Denomination: {m['denomination']}
- Year: {m['year']}
{version_line}
- Side: {m['side'].title()}
- Similarity: `{sim:.1f}%`
""")
    else:
        st.info("Upload an image to start")


