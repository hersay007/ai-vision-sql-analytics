import streamlit as st
import tensorflow as tf
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
from PIL import Image

# --- 1. DATABASE ARCHITECTURE ---
def get_connection():
    return sqlite3.connect('analytics.db', check_same_thread=False)

def init_db():
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS product_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            product_name TEXT,
            confidence REAL
        )
    ''')
    conn.commit()

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_ai_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_ai_model()

def run_inference(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

    return decoded[1], float(decoded[2])

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="AI SQL Analytics Pipeline", layout="wide")
init_db()

st.title("🎥 AI-Vision to SQL Analytics Pipeline")
st.markdown("""
**Skills Demonstrated:**
TensorFlow/Keras (CNN), SQL Window Functions, SQLite, Streamlit, Pandas
""")

col1, col2 = st.columns([1, 1])

# --- INPUT SECTION ---
with col1:
    st.subheader("📥 Input Source")
    mode = st.radio("Choose Input Method:", ["Webcam", "Upload Image"])

    img_file = None

    if mode == "Webcam":
        img_file = st.camera_input("Capture an object")
    else:
        img_file = st.file_uploader("Upload a JPG/PNG", type=["jpg", "png", "jpeg"])

    if img_file:
        img = Image.open(img_file)

        # Run AI
        label, score = run_inference(img)

        # Save to DB
        conn = get_connection()
        conn.execute(
            "INSERT INTO product_logs (timestamp, product_name, confidence) VALUES (?, ?, ?)",
            (datetime.now().strftime("%H:%M:%S"), label, score)
        )
        conn.commit()

        st.success(f"AI Detected: {label} | Confidence: {score:.2%}")

# --- DATABASE VIEW ---
with col2:
    st.subheader("💾 Recent Database Entries")
    df_raw = pd.read_sql_query(
        "SELECT * FROM product_logs ORDER BY id DESC LIMIT 5",
        get_connection()
    )
    st.dataframe(df_raw, use_container_width=True)

# --- ANALYTICS SECTION ---
st.divider()
st.header("📊 Advanced SQL Analytics (Window Functions)")

window_query = """
SELECT
    product_name,
    confidence,

    RANK() OVER (PARTITION BY product_name ORDER BY confidence DESC) AS rank_by_label,

    AVG(confidence) OVER (
        ORDER BY id
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_conf,

    LAG(product_name) OVER (ORDER BY id) AS previous_item

FROM product_logs
ORDER BY id DESC
"""

try:
    df_analytics = pd.read_sql_query(window_query, get_connection())
    st.dataframe(df_analytics, use_container_width=True)

    with st.expander("Explain SQL Logic"):
        st.code(window_query, language="sql")
        st.markdown("""
        - **RANK()** → Ranks detections within each object category
        - **AVG OVER** → Rolling average of last 3 confidence values
        - **LAG()** → Shows previous detected object
        """)
except Exception:
    st.info("No data yet. Capture or upload images to generate analytics.")

# --- CLEAR DATA BUTTON ---
if st.button("Clear Data"):
    conn = get_connection()
    conn.execute("DELETE FROM product_logs")
    conn.commit()
    st.rerun()
