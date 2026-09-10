"""
SequenceLab — Versi 1 (rilis awal)
Deploy 2 model Week 3: LSTM suhu (Jena) dan GRU sentimen (IMDB).
Versi ini sengaja sederhana: tanpa tema khusus, tanpa grafik, satu input per model.

Cara pakai untuk versioning: deploy file ini dulu sebagai v1 (rename jadi app.py di repo/branch v1),
screenshot, lalu deploy app.py sebagai v2.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "models"
FEATURES = [
    "p (mbar)", "T (degC)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
    "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)",
    "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)",
]
TARGET_IDX = 1
SEQ_TEMP, SEQ_TEXT, VOCAB = 72, 400, 20000

st.set_page_config(page_title="SequenceLab v1", page_icon="🌡️")
st.title("SequenceLab v1.0")
st.write("Deployment model LSTM (prediksi suhu) dan GRU (sentimen ulasan film) — Akmal Nugraha Saputra (2609)")


@st.cache_resource
def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def clean_text(t):
    t = re.sub(r"<.*?>", " ", t).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


menu = st.selectbox("Pilih model", ["Prediksi suhu (LSTM)", "Sentimen ulasan (GRU)"])

if menu.startswith("Prediksi"):
    st.subheader("Prediksi suhu 1 jam ke depan")
    up = st.file_uploader("Unggah CSV Jena per jam (minimal 72 baris, 13 kolom fitur)", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        for c in ["wv (m/s)", "max. wv (m/s)"]:
            if c in df.columns:
                df.loc[df[c] == -9999.0, c] = 0.0
        if any(c not in df.columns for c in FEATURES):
            st.error("Kolom fitur tidak lengkap.")
        elif len(df) < SEQ_TEMP:
            st.error("Butuh minimal 72 baris.")
        elif st.button("Prediksi"):
            model = load_model(os.path.join(MODEL_DIR, "LSTM_A_seq72.h5"))
            with open(os.path.join(MODEL_DIR, "scaler.json")) as f:
                sc = json.load(f)
            mean, scale = np.array(sc["mean"]), np.array(sc["scale"])
            x = (df[FEATURES].tail(SEQ_TEMP).to_numpy() - mean) / scale
            y = model.predict(x[np.newaxis].astype(np.float32), verbose=0)[0][0]
            st.success(f"Prediksi suhu 1 jam berikutnya: {y * scale[TARGET_IDX] + mean[TARGET_IDX]:.2f} °C")

else:
    st.subheader("Klasifikasi sentimen ulasan film")
    teks = st.text_area("Ulasan (bahasa Inggris)")
    if st.button("Analisis") and teks.strip():
        model = load_model(os.path.join(MODEL_DIR, "GRU_A_seq400.h5"))
        with open(os.path.join(MODEL_DIR, "tokenizer.json"), encoding="utf-8") as f:
            data = json.load(f)
        wi = data["config"]["word_index"] if "config" in data else data
        wi = json.loads(wi) if isinstance(wi, str) else wi
        idx = [wi.get(w, 1) if wi.get(w, 1) < VOCAB else 1 for w in clean_text(teks).split()]
        seq = np.zeros(SEQ_TEXT, dtype=np.int32)
        if idx:
            idx = idx[-SEQ_TEXT:]
            seq[-len(idx):] = idx
        p = float(model.predict(seq[np.newaxis], verbose=0)[0][0])
        st.write(f"Probabilitas positif: {p:.3f}")
        st.success("Sentimen: POSITIF" if p > 0.5 else "Sentimen: NEGATIF")
