"""Versi 1 - Sentimen ulasan IMDB (GRU). Tampilan dasar Streamlit, satu model, input teks saja.
Main file path di Streamlit Cloud: imdb/v1/app.py"""
import importlib.util
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("runtun_app", ROOT / "app.py")
core = importlib.util.module_from_spec(spec)
sys.modules["runtun_app"] = core
spec.loader.exec_module(core)

st.set_page_config(page_title="Sentimen Ulasan v1")
st.title("Analisis Sentimen Ulasan Film (v1)")
st.write("Model GRU mengklasifikasikan ulasan film berbahasa Inggris menjadi positif atau negatif.")

models, tok = core.find_models()["imdb"], core.load_tokenizer()
if not models or tok is None:
    st.error("File model atau tokenizer.json tidak ditemukan di folder models/.")
    st.stop()

model = models[0]
text = st.text_area("Masukkan ulasan (bahasa Inggris)")
if st.button("Prediksi"):
    ids = core.to_ids(core.clean_text(text).split(), tok)
    p, _ = core.run_model(str(model["path"]), core.pad_pre(ids, model["seq"])[None, :], "keras")
    p = float(p[0])
    st.write("Sentimen:", "Positif" if p >= 0.5 else "Negatif")
    st.write(f"Probabilitas positif: {p:.3f}")
