"""Versi 1 - Prakiraan suhu Jena (LSTM). Tampilan dasar Streamlit, satu model, input sederhana.
Main file path di Streamlit Cloud: jena/v1/app.py"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("runtun_app", ROOT / "app.py")
core = importlib.util.module_from_spec(spec)
sys.modules["runtun_app"] = core
spec.loader.exec_module(core)  # hanya memakai fungsi pemuat data dan model

st.set_page_config(page_title="Prakiraan Suhu v1")
st.title("Prakiraan Suhu Jena (v1)")
st.write("Model LSTM memprediksi suhu 1 jam ke depan dari 72 jam data cuaca sebelumnya.")

models, scaler, df = core.find_models()["jena"], core.load_scaler(), core.load_jena_sample()
if not models or scaler is None or df is None:
    st.error("File model, scaler.json, atau sample_jena.csv tidak ditemukan di folder models/.")
    st.stop()

model = models[0]
seq = model["seq"]
df = df.dropna(subset=scaler["features"])
scaled = core.scale_frame(df, scaler)
t_mean, t_std = core.target_stats(scaler)

idx = st.slider("Pilih indeks jam pada data uji", seq, len(df) - 1, seq + 500)
if st.button("Prediksi"):
    pred, _ = core.run_model(str(model["path"]), scaled[idx - seq:idx][None, ...], "keras")
    pred = float(pred[0] * t_std + t_mean)
    st.write("Waktu:", df.index[idx])
    st.write(f"Prediksi suhu: {pred:.2f} °C")
    st.write(f"Suhu sebenarnya: {df[core.TARGET].iloc[idx]:.2f} °C")
    st.line_chart(pd.DataFrame({"suhu": df[core.TARGET].iloc[idx - seq:idx].values}))
