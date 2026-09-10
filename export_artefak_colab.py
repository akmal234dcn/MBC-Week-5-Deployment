# =============================================================================
# Jalankan sel ini di NOTEBOOK WEEK 3 (setelah semua sel training selesai)
# untuk mengekspor artefak yang dibutuhkan aplikasi Streamlit.
# Hasilnya ada di folder  <base_dir>/deploy_artefak/  di Google Drive.
# Salin semua file ke folder  models/  di repo GitHub.
# =============================================================================
import json, os, shutil

OUT = os.path.join(base_dir, "deploy_artefak")
os.makedirs(OUT, exist_ok=True)

# 1. Model terbaik (.h5)
shutil.copy(os.path.join(MODEL_DIR, "LSTM_A_seq72.h5"), OUT)        # suhu
shutil.copy(os.path.join(MODEL_DIR_TXT, "GRU_A_seq400.h5"), OUT)    # sentimen

# 2. Scaler -> JSON (tidak bergantung versi sklearn)
with open(os.path.join(OUT, "scaler.json"), "w") as f:
    json.dump({
        "columns": list(features.columns),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }, f)

# 3. Tokenizer -> JSON
with open(os.path.join(OUT, "tokenizer.json"), "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

# 4. Data contoh: 400 jam terakhir dari data uji (per jam, dengan kolom Date Time)
sample = df_hourly.iloc[-400:].drop(columns=["Tpot (K)"]).copy()
sample["Date Time"] = sample["Date Time"].dt.strftime("%d.%m.%Y %H:%M:%S")
sample.to_csv(os.path.join(OUT, "sample_jena.csv"), index=False)

print("Selesai. Isi folder:", os.listdir(OUT))
