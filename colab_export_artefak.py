# =====================================================================
# CADANGAN: jalankan di Google Colab HANYA jika scaler.json / tokenizer.json /
# sample_jena.csv yang lama tidak terbaca oleh aplikasi.
# Tidak perlu training ulang: preprocessing di bawah sama persis dengan notebook Week 3.
# Hasil: folder "artefak_deploy" di Google Drive, lalu upload isinya ke folder models/ di GitHub.
# =====================================================================
import os, glob, zipfile, json, re, shutil
import numpy as np, pandas as pd
from google.colab import drive
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

drive.mount('/content/drive')
base_dir = "/content/drive/MyDrive/504002_AKMAL NUGRAHA SAPUTRA/MBC LAS/Week3"
OUT = os.path.join(base_dir, "artefak_deploy")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. Jena: scaler.json + sample_jena.csv ----------
os.makedirs("/content/jena", exist_ok=True)
with zipfile.ZipFile(os.path.join(base_dir, "Jena Climate Dataset.zip")) as z:
    z.extractall("/content/jena")
df = pd.read_csv(glob.glob("/content/jena/**/*.csv", recursive=True)[0])
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
for c in ['wv (m/s)', 'max. wv (m/s)']:
    df.loc[df[c] == -9999.0, c] = 0.0
df_hourly = df.iloc[5::6].reset_index(drop=True)
features = df_hourly.drop(columns=['Date Time', 'Tpot (K)'])
n = len(features); train_end, val_end = int(n * 0.70), int(n * 0.85)
scaler = StandardScaler().fit(features.iloc[:train_end])
json.dump({"features": list(features.columns), "mean": scaler.mean_.tolist(),
           "scale": scaler.scale_.tolist(), "target": "T (degC)"},
          open(os.path.join(OUT, "scaler.json"), "w"), indent=1)
sample = df_hourly.iloc[val_end:].copy()            # periode test: Okt 2015 - Des 2016
sample['Date Time'] = sample['Date Time'].dt.strftime('%d.%m.%Y %H:%M:%S')
sample.drop(columns=['Tpot (K)']).to_csv(os.path.join(OUT, "sample_jena.csv"), index=False)
print("Jena selesai:", sample.shape)

# ---------- 2. IMDB: tokenizer.json + sample_ulasan.csv ----------
os.makedirs("/content/imdb", exist_ok=True)
with zipfile.ZipFile(os.path.join(base_dir, "IMDB Dataset of 50K Movie Reviews.zip")) as z:
    z.extractall("/content/imdb")
imdb = pd.read_csv(glob.glob("/content/imdb/**/*.csv", recursive=True)[0])
imdb = imdb.drop_duplicates(subset='review').reset_index(drop=True)
def clean_text(t):
    t = re.sub(r'<.*?>', ' ', t).lower()
    t = re.sub(r"[^a-z\s]", ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
imdb['clean'] = imdb['review'].apply(clean_text)
imdb['label'] = (imdb['sentiment'] == 'positive').astype(int)
X_tr, X_tmp, y_tr, y_tmp, r_tr, r_tmp = train_test_split(
    imdb['clean'].values, imdb['label'].values, imdb['review'].values,
    test_size=0.30, stratify=imdb['label'].values, random_state=42)
_, X_te, _, y_te, _, r_te = train_test_split(X_tmp, y_tmp, r_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
tok = Tokenizer(num_words=20000, oov_token='<OOV>'); tok.fit_on_texts(X_tr)
open(os.path.join(OUT, "tokenizer.json"), "w").write(tok.to_json())
pd.DataFrame({"review": r_te[:200], "sentiment": np.where(y_te[:200] == 1, "positive", "negative")}) \
  .to_csv(os.path.join(OUT, "sample_ulasan.csv"), index=False)
print("IMDB selesai. Kosakata:", len(tok.word_index))

# ---------- 3. Salin model terbaik ----------
shutil.copy(os.path.join(base_dir, "models_jena", "LSTM_A_seq72.h5"), OUT)
shutil.copy(os.path.join(base_dir, "models_imdb", "GRU_A_seq400.h5"), OUT)
# (opsional) model pembanding, akan otomatis muncul di aplikasi:
# shutil.copy(os.path.join(base_dir, "models_jena", "GRU_A_seq72.h5"), OUT)
print("Selesai. Isi folder:", os.listdir(OUT))
