import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import os

# ==================== Paths ====================
ENCODER_PATH = r"D:\New folder\banknote-net\models\banknote_net_encoder.h5"
EMBEDDINGS_PATH = r"D:\New folder\banknote_embeddings.pkl"

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
    # Add other mappings as needed
}

# ==================== Load Model and Embeddings ====================
@st.cache_resource
def load_resources():
    encoder = load_model(
        ENCODER_PATH,
        custom_objects={"DepthwiseConv2D": DepthwiseConv2DPatched}
    )

    with open(EMBEDDINGS_PATH, "rb") as f:
        gallery = pickle.load(f)

    # Flatten embeddings to correct shape (256,)
    gallery_embeddings = np.array([
        np.array(item["embedding"]).reshape(-1)
        for item in gallery
    ])

    return encoder, gallery_embeddings, gallery

# ==================== Extract Query Embedding ====================
def extract_embedding(encoder, image):
    x = img_to_array(image) / 255.0
    x = np.expand_dims(x, axis=0)

    emb = encoder.predict(x, verbose=0)[0]  # shape (256,)
    emb = emb / np.linalg.norm(emb)        # normalize properly

    return emb

# ==================== Find Top Matches ====================
def find_top_matches(query_emb, gallery_embeddings, gallery_metadata, top_k=3):
    similarities = gallery_embeddings @ query_emb  # (N,) dot (256,) → (N,)
    top_indices = similarities.argsort()[::-1][:top_k]

    matches = []
    for idx in top_indices:
        matches.append({
            "similarity": float(similarities[idx]),
            **gallery_metadata[idx]
        })
    return matches

# ==================== Streamlit UI ====================
st.set_page_config(page_title="Banknote Matcher", layout="wide")
st.title("Banknote Identification 🏦")

# Load model + embeddings
try:
    encoder, gallery_embeddings, gallery_metadata = load_resources()
    st.success(f"✅ Loaded model and {len(gallery_metadata)} reference banknotes")
except Exception as e:
    st.error(f"❌ Failed to load resources: {e}")
    st.stop()

# Layout: 50/50 columns

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Banknote Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    image = None
    if uploaded_file is not None:
        image = load_img(uploaded_file, target_size=(224, 224))
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("Top 3 Matches")
    if uploaded_file is not None and image is not None:
        with st.spinner("Analyzing banknote..."):
            query_emb = extract_embedding(encoder, image)
            matches = find_top_matches(query_emb, gallery_embeddings, gallery_metadata)

        for i, match in enumerate(matches, 1):
            similarity_pct = match["similarity"] * 100
            country_full = COUNTRY_MAPPING.get(match["currency"], match["currency"])
            st.markdown(f"""
            **Match #{i}**  
            - Country: {country_full}  
            - Denomination: {match['denomination']}  
            - Year: {match['year']}  
            - Side: {match['side'].title()}  
            - Similarity: `{similarity_pct:.1f}%`  
            """)
    else:
        st.info("👆 Upload an image to see the top matches")

st.markdown("---")
st.caption("Powered by Banknote-Net Encoder")
