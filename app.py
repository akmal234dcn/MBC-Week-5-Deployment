"""
SequenceLab — Deployment Model LSTM & GRU (Week 3 Big Data)
Versi 2 (final)

Model yang dideploy:
  1. LSTM (1 layer, 64 unit, seq 72 jam) -> prediksi suhu 1 jam ke depan, Jena Climate
  2. GRU  (1 layer, 64 unit, seq 400 token) -> klasifikasi sentimen ulasan film, IMDB 50K

Nama : AKMAL NUGRAHA SAPUTRA | Kode CaAs : 2609 | NIM : 103052500014
"""

import json
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Konstanta
# --------------------------------------------------------------------------- #
APP_VERSION = "v2.0"
MODEL_DIR = "models"

TEMP_MODEL_PATH = os.path.join(MODEL_DIR, "LSTM_A_seq72.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.json")
SAMPLE_JENA_PATH = os.path.join(MODEL_DIR, "sample_jena.csv")

SENT_MODEL_PATH = os.path.join(MODEL_DIR, "GRU_A_seq400.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

SEQ_LEN_TEMP = 72
SEQ_LEN_TEXT = 400
VOCAB_SIZE = 20000
OOV_INDEX = 1

FEATURES = [
    "p (mbar)", "T (degC)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
    "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)",
    "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)",
]
TARGET = "T (degC)"
TARGET_IDX = FEATURES.index(TARGET)

CONTOH_ULASAN = {
    "Ulasan positif": (
        "One of the best films I have seen this year. The story is gripping, the "
        "acting is superb and the soundtrack fits every scene perfectly. I was on "
        "the edge of my seat until the very end and would happily watch it again."
    ),
    "Ulasan negatif": (
        "What a waste of two hours. The plot made no sense, the dialogue was "
        "painfully cheesy and the ending felt rushed. Even the lead actor looked "
        "bored. I honestly cannot recommend this to anyone."
    ),
    "Ulasan campuran": (
        "The visuals were stunning and the cast clearly gave their best, but the "
        "script let everyone down. Some scenes dragged on forever and the humour "
        "rarely landed. Not terrible, just disappointing considering the budget."
    ),
}

# --------------------------------------------------------------------------- #
# Tampilan
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="SequenceLab · LSTM & GRU",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, textarea {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
h1, h2, h3, h4 { font-family: 'Sora', system-ui, sans-serif; letter-spacing: -0.01em; }

/* sidebar gelap */
[data-testid="stSidebar"] {
    background: #0F1B2D;
    border-right: 1px solid #1D2B45;
}
[data-testid="stSidebar"] * { color: #E4E9F2 !important; }
[data-testid="stSidebar"] .stRadio label { padding: 4px 0; }
[data-testid="stSidebar"] hr { border-color: #24354F; }

.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }

/* hero */
.hero {
    display: flex; align-items: flex-end; justify-content: space-between; gap: 24px;
    padding: 28px 32px; border-radius: 18px;
    background: linear-gradient(120deg, #0F1B2D 0%, #1C3A6B 55%, #2F5BEA 100%);
    color: #F4F6FA; margin-bottom: 22px;
}
.hero h1 { font-size: 2.1rem; margin: 0 0 8px 0; color: #FFFFFF; }
.hero p  { margin: 0; color: #C9D4EA; max-width: 640px; line-height: 1.55; }
.hero .tag {
    font-family: 'Sora', sans-serif; font-size: 0.8rem; padding: 6px 12px; border-radius: 999px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
    white-space: nowrap;
}

/* kartu */
.card {
    background: #FFFFFF; border: 1px solid #E3E8F0; border-radius: 16px;
    padding: 22px 24px; height: 100%;
}
.card h3 { margin: 0 0 6px 0; font-size: 1.15rem; }
.card .sub { color: #5B6778; margin: 0 0 14px 0; font-size: 0.93rem; }
.card.warm { border-top: 5px solid #FF7A45; }
.card.cool { border-top: 5px solid #17A2A0; }

.metric-row { display: flex; gap: 12px; flex-wrap: wrap; }
.metric {
    flex: 1 1 120px; background: #F4F6FA; border-radius: 12px; padding: 12px 14px;
}
.metric .v { font-family: 'Sora', 'IBM Plex Sans', sans-serif; font-size: 1.35rem; font-weight: 600; color: #172033; }
.metric .l { font-size: 0.78rem; color: #5B6778; }

/* hasil */
.result {
    border-radius: 16px; padding: 22px 26px; margin: 8px 0 16px 0; color: #FFFFFF;
}
.result .big { font-family: 'Sora', 'IBM Plex Sans', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1.1; }
.result .lbl { opacity: 0.85; font-size: 0.95rem; }
.result.temp { background: linear-gradient(120deg, #FF7A45, #FF4D6D); }
.result.pos  { background: linear-gradient(120deg, #17A2A0, #2ECC8A); }
.result.neg  { background: linear-gradient(120deg, #E0475B, #B0306E); }
.result.mid  { background: linear-gradient(120deg, #6B7A90, #4A5568); }

.steps { counter-reset: s; padding-left: 0; list-style: none; }
.steps li { counter-increment: s; position: relative; padding-left: 38px; margin: 10px 0; line-height: 1.5; }
.steps li::before {
    content: counter(s); position: absolute; left: 0; top: 1px;
    width: 26px; height: 26px; border-radius: 50%; background: #2F5BEA; color: white;
    font-family: 'Sora', sans-serif; font-size: 0.8rem; font-weight: 600;
    display: flex; align-items: center; justify-content: center;
}
.note { background: #FFF6EC; border: 1px solid #FFD9B8; border-radius: 12px; padding: 12px 16px; color: #7A3E00; }
.footer { color: #8A94A6; font-size: 0.85rem; margin-top: 40px; text-align: center; }

div.stButton > button[kind="primary"] {
    background: #2F5BEA; border: none; border-radius: 10px; padding: 10px 22px;
    font-family: 'Sora', sans-serif; font-weight: 600;
}
div.stButton > button[kind="primary"]:hover { background: #1F45C8; }
div.stButton > button { border-radius: 10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Pemuatan artefak (di-cache supaya hanya dimuat sekali per server)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Memuat model…")
def load_keras_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource
def load_scaler(path: str):
    with open(path) as f:
        sc = json.load(f)
    return np.array(sc["mean"], dtype=np.float32), np.array(sc["scale"], dtype=np.float32)


@st.cache_resource
def load_word_index(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # format keluaran tokenizer.to_json() dari Keras
    if isinstance(data, dict) and "config" in data:
        wi = data["config"]["word_index"]
        return json.loads(wi) if isinstance(wi, str) else wi
    return data  # sudah berupa {kata: indeks}


@st.cache_data
def load_sample_jena(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def artefak_ada(*paths) -> bool:
    return all(os.path.exists(p) for p in paths)


# --------------------------------------------------------------------------- #
# Preprocessing (identik dengan notebook Week 3)
# --------------------------------------------------------------------------- #
def siapkan_jena(df: pd.DataFrame) -> pd.DataFrame:
    """Terima CSV Jena mentah (10 menit) atau data per jam; kembalikan 13 fitur per jam."""
    df = df.copy()
    for c in ["wv (m/s)", "max. wv (m/s)"]:
        if c in df.columns:
            df.loc[df[c] == -9999.0, c] = 0.0

    # kalau masih 10 menitan (interval pertama < 1 jam), downsample ke per jam
    if "Date Time" in df.columns:
        try:
            dt = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
            if dt.isna().all():
                dt = pd.to_datetime(df["Date Time"], errors="coerce")
            df["Date Time"] = dt
            if len(dt) > 1 and (dt.iloc[1] - dt.iloc[0]) < pd.Timedelta(hours=1):
                df = df.iloc[5::6].reset_index(drop=True)
        except Exception:
            pass

    hilang = [c for c in FEATURES if c not in df.columns]
    if hilang:
        raise ValueError(f"Kolom berikut tidak ditemukan: {', '.join(hilang)}")
    return df


def prediksi_suhu(model, mean, scale, window: np.ndarray) -> float:
    """window: array (72, 13) dalam satuan asli. Kembalikan suhu (°C) 1 jam berikutnya."""
    x = (window - mean) / scale
    y = model.predict(x[np.newaxis, ...].astype(np.float32), verbose=0).flatten()[0]
    return float(y * scale[TARGET_IDX] + mean[TARGET_IDX])


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def teks_ke_sequence(text: str, word_index: dict):
    kata = clean_text(text).split()
    idx, dikenal = [], 0
    for w in kata:
        i = word_index.get(w, OOV_INDEX)
        if i >= VOCAB_SIZE:
            i = OOV_INDEX
        if i != OOV_INDEX:
            dikenal += 1
        idx.append(i)
    seq = np.zeros(SEQ_LEN_TEXT, dtype=np.int32)
    if idx:
        potong = idx[-SEQ_LEN_TEXT:]          # truncating='pre'
        seq[-len(potong):] = potong           # padding='pre'
    return seq, len(kata), dikenal


def prediksi_sentimen(model, word_index, text: str):
    seq, n_kata, dikenal = teks_ke_sequence(text, word_index)
    p = float(model.predict(seq[np.newaxis, :], verbose=0).flatten()[0])
    return p, n_kata, dikenal


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🌡️ SequenceLab")
    st.caption("LSTM & GRU · Deployment Week 5")
    halaman = st.radio(
        "Menu",
        ["Beranda", "Prediksi Suhu (LSTM)", "Analisis Sentimen (GRU)", "Versioning & Info"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Status model**")
    st.markdown(("🟢" if artefak_ada(TEMP_MODEL_PATH, SCALER_PATH) else "🔴") + " LSTM suhu")
    st.markdown(("🟢" if artefak_ada(SENT_MODEL_PATH, TOKENIZER_PATH) else "🔴") + " GRU sentimen")
    st.divider()
    st.markdown(
        f"**Akmal Nugraha Saputra**  \nCaAs 2609 · NIM 103052500014  \nVersi aplikasi: {APP_VERSION}"
    )


# --------------------------------------------------------------------------- #
# Halaman: Beranda
# --------------------------------------------------------------------------- #
def halaman_beranda():
    st.markdown(
        """
        <div class="hero">
          <div>
            <h1>Dua model sequential, satu aplikasi</h1>
            <p>Coba langsung model terbaik dari Tugas Week 3: LSTM untuk meramal suhu udara
            satu jam ke depan dari data cuaca Jena, dan GRU untuk menebak sentimen ulasan film IMDB.
            Pilih model di menu sebelah kiri.</p>
          </div>
          <div class="tag">Week 3 → Week 5 Deployment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="card warm">
              <h3>🌡️ Prediksi suhu · LSTM</h3>
              <p class="sub">Jena Climate 2009–2016 · 13 variabel cuaca · jendela 72 jam</p>
              <div class="metric-row">
                <div class="metric"><div class="v">0,477 °C</div><div class="l">MAE data uji</div></div>
                <div class="metric"><div class="v">0,669 °C</div><div class="l">RMSE data uji</div></div>
                <div class="metric"><div class="v">20.033</div><div class="l">parameter</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card cool">
              <h3>🎬 Analisis sentimen · GRU</h3>
              <p class="sub">IMDB 50K Movie Reviews · vocabulary 20.000 kata · 400 token</p>
              <div class="metric-row">
                <div class="metric"><div class="v">89,35 %</div><div class="l">akurasi data uji</div></div>
                <div class="metric"><div class="v">0,895</div><div class="l">F1-score</div></div>
                <div class="metric"><div class="v">1,31 jt</div><div class="l">parameter</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Cara memakai")
    st.markdown(
        """
        <ol class="steps">
          <li><b>Prediksi Suhu:</b> pakai data contoh yang sudah disediakan, atau unggah CSV Jena
              (format asli 10 menit maupun per jam). Geser jendela waktu, lalu klik <i>Prediksi</i>.
              Kalau data punya jam berikutnya, aplikasi ikut menampilkan nilai aktual sebagai pembanding.</li>
          <li><b>Analisis Sentimen:</b> tulis ulasan film berbahasa Inggris atau pilih contoh, lalu klik
              <i>Analisis</i>. Ada juga mode batch untuk banyak ulasan sekaligus dari file CSV.</li>
          <li><b>Versioning & Info:</b> catatan perubahan tiap versi deployment beserta ringkasan
              eksperimen 8 skenario dari notebook.</li>
        </ol>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Kenapa dua model ini yang dipilih?"):
        st.markdown(
            """
            Dari 16 skenario yang dilatih di Week 3 (LSTM/GRU × shallow/deep × 2 sequence length untuk
            masing-masing dataset), dua skenario ini punya skor uji terbaik:

            | Tugas | Model terbaik | Skor uji |
            |---|---|---|
            | Jena Climate (regresi) | LSTM, 1 layer 64 unit, seq 72 | MAE 0,4772 °C · RMSE 0,6685 °C |
            | IMDB (klasifikasi) | GRU, 1 layer 64 unit, seq 400 | Accuracy 0,8935 · F1 0,8949 |

            Konfigurasi 1 layer mengungguli 2 layer + dropout pada kedua dataset, jadi model yang lebih
            kecil justru dipakai untuk deployment: lebih ringan dan lebih cepat merespons.
            """
        )


# --------------------------------------------------------------------------- #
# Halaman: Prediksi Suhu
# --------------------------------------------------------------------------- #
def halaman_suhu():
    st.markdown("## 🌡️ Prediksi suhu 1 jam ke depan")
    st.caption("LSTM · 1 layer 64 unit · input 72 jam terakhir × 13 variabel cuaca (Jena, Jerman)")

    if not artefak_ada(TEMP_MODEL_PATH, SCALER_PATH):
        st.error(
            f"File model belum lengkap. Pastikan `{TEMP_MODEL_PATH}` dan `{SCALER_PATH}` ada di repo."
        )
        return

    model = load_keras_model(TEMP_MODEL_PATH)
    mean, scale = load_scaler(SCALER_PATH)

    kiri, kanan = st.columns([1, 1.6], gap="large")
    with kiri:
        st.markdown("#### 1. Sumber data")
        opsi = ["Data contoh (bawaan)", "Unggah CSV sendiri"]
        if not os.path.exists(SAMPLE_JENA_PATH):
            opsi = opsi[1:]
        sumber = st.radio("Sumber data", opsi, label_visibility="collapsed")

        df = None
        if sumber.startswith("Data contoh"):
            df = load_sample_jena(SAMPLE_JENA_PATH)
            st.info(f"Data contoh: {len(df)} jam terakhir dari data uji (Okt 2015 – Des 2016).")
        else:
            up = st.file_uploader("CSV Jena Climate (10 menit atau per jam)", type=["csv"])
            if up is not None:
                df = pd.read_csv(up)

        if df is None:
            st.markdown(
                '<div class="note">Unggah CSV dengan 13 kolom cuaca Jena, minimal 72 baris per jam '
                '(atau 432 baris data 10 menit).</div>',
                unsafe_allow_html=True,
            )
            return

        try:
            df = siapkan_jena(df)
        except ValueError as e:
            st.error(str(e))
            return

        if len(df) < SEQ_LEN_TEMP:
            st.error(f"Data hanya {len(df)} jam. Model butuh minimal {SEQ_LEN_TEMP} jam.")
            return

        st.markdown("#### 2. Pilih jendela 72 jam")
        akhir_max = len(df)
        akhir = st.slider(
            "Baris terakhir yang dipakai sebagai 'sekarang'",
            min_value=SEQ_LEN_TEMP, max_value=akhir_max, value=akhir_max,
            help="Geser ke kiri supaya jam berikutnya masih ada di data, jadi bisa dibandingkan dengan nilai aktual.",
        )
        n_langkah = st.slider(
            "Ramalan lanjutan (jam)", 1, 12, 1,
            help="Lebih dari 1 jam = prediksi berantai; nilai fitur lain diasumsikan tetap, jadi hanya estimasi kasar.",
        )
        tombol = st.button("Prediksi suhu", type="primary", use_container_width=True)

    window_df = df.iloc[akhir - SEQ_LEN_TEMP:akhir]
    window = window_df[FEATURES].to_numpy(dtype=np.float32)
    waktu = window_df["Date Time"] if "Date Time" in window_df.columns else pd.RangeIndex(akhir - SEQ_LEN_TEMP, akhir)

    with kanan:
        st.markdown("#### 3. Hasil")
        if not tombol:
            st.markdown(
                '<div class="note">Atur jendela di kiri, lalu klik <b>Prediksi suhu</b>.</div>',
                unsafe_allow_html=True,
            )
            _plot_suhu(waktu, window[:, TARGET_IDX], None, None, None)
            return

        with st.spinner("Menghitung…"):
            hasil = []
            w = window.copy()
            for _ in range(n_langkah):
                y = prediksi_suhu(model, mean, scale, w)
                hasil.append(y)
                baris_baru = w[-1].copy()
                baris_baru[TARGET_IDX] = y
                w = np.vstack([w[1:], baris_baru])

        suhu_terakhir = float(window[-1, TARGET_IDX])
        pred1 = hasil[0]
        delta = pred1 - suhu_terakhir
        aktual = float(df.iloc[akhir][TARGET]) if akhir < len(df) else None

        st.markdown(
            f"""
            <div class="result temp">
              <div class="lbl">Suhu 1 jam berikutnya (prediksi)</div>
              <div class="big">{pred1:.2f} °C</div>
              <div class="lbl">Suhu terakhir {suhu_terakhir:.2f} °C · perubahan {delta:+.2f} °C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediksi 1 jam", f"{pred1:.2f} °C", f"{delta:+.2f} °C")
        if aktual is not None:
            m2.metric("Aktual di data", f"{aktual:.2f} °C")
            m3.metric("Selisih", f"{abs(pred1 - aktual):.2f} °C")
        else:
            m2.metric("Rata-rata 72 jam", f"{window[:, TARGET_IDX].mean():.2f} °C")
            m3.metric("Min / maks 72 jam", f"{window[:, TARGET_IDX].min():.0f} / {window[:, TARGET_IDX].max():.0f} °C")

        _plot_suhu(waktu, window[:, TARGET_IDX], hasil, aktual, df if akhir < len(df) else None)

        if n_langkah > 1:
            tabel = pd.DataFrame({"Jam ke-": range(1, n_langkah + 1), "Prediksi (°C)": np.round(hasil, 2)})
            st.dataframe(tabel, hide_index=True, use_container_width=True)

    with st.expander("Lihat 72 jam data yang dipakai"):
        st.dataframe(window_df, use_container_width=True, height=260)

    with st.expander("Tentang model ini"):
        st.markdown(
            """
            - Arsitektur: `Input(72, 13) → LSTM(64) → Dense(1)`, 20.033 parameter, Adam lr 0,001, loss MSE.
            - Data dinormalisasi dengan StandardScaler yang di-fit pada data training (2009–Agu 2014).
            - Evaluasi data uji (Okt 2015 – Des 2016): **MAE 0,4772 °C**, **RMSE 0,6685 °C**.
            - Kolom `Tpot (K)` dibuang karena hampir identik dengan target; nilai angin -9999 dianggap 0.
            """
        )


def _plot_suhu(waktu, suhu, hasil, aktual, df_lanjut):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(waktu), y=suhu, mode="lines", name="72 jam terakhir",
        line=dict(color="#2F5BEA", width=2.2),
    ))
    if hasil:
        if isinstance(waktu, pd.Series) and pd.api.types.is_datetime64_any_dtype(waktu):
            t_akhir = waktu.iloc[-1]
            x_pred = [t_akhir + pd.Timedelta(hours=i) for i in range(1, len(hasil) + 1)]
            x_link = [t_akhir] + x_pred
        else:
            akhir = list(waktu)[-1]
            x_pred = [akhir + i for i in range(1, len(hasil) + 1)]
            x_link = [akhir] + x_pred
        fig.add_trace(go.Scatter(
            x=x_link, y=[suhu[-1]] + list(hasil), mode="lines+markers", name="Prediksi",
            line=dict(color="#FF7A45", width=2.5, dash="dot"), marker=dict(size=9),
        ))
        if aktual is not None:
            fig.add_trace(go.Scatter(
                x=[x_pred[0]], y=[aktual], mode="markers", name="Aktual",
                marker=dict(color="#17A2A0", size=12, symbol="diamond"),
            ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", y=1.08, x=0),
        yaxis_title="T (°C)", xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#EEF1F6"),
        font=dict(family="IBM Plex Sans, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Halaman: Analisis Sentimen
# --------------------------------------------------------------------------- #
def halaman_sentimen():
    st.markdown("## 🎬 Analisis sentimen ulasan film")
    st.caption("GRU · Embedding 64 dim · 1 layer 64 unit · 400 token · IMDB 50K Reviews (bahasa Inggris)")

    if not artefak_ada(SENT_MODEL_PATH, TOKENIZER_PATH):
        st.error(
            f"File model belum lengkap. Pastikan `{SENT_MODEL_PATH}` dan `{TOKENIZER_PATH}` ada di repo."
        )
        return

    model = load_keras_model(SENT_MODEL_PATH)
    word_index = load_word_index(TOKENIZER_PATH)

    tab1, tab2 = st.tabs(["Satu ulasan", "Banyak ulasan (batch)"])

    with tab1:
        if "ulasan" not in st.session_state:
            st.session_state["ulasan"] = ""

        st.markdown("Coba contoh:")
        kolom = st.columns(len(CONTOH_ULASAN))
        for (nama, teks), kol in zip(CONTOH_ULASAN.items(), kolom):
            if kol.button(nama, use_container_width=True):
                st.session_state["ulasan"] = teks

        teks = st.text_area(
            "Tulis ulasan film (bahasa Inggris)", key="ulasan", height=170,
            placeholder="Contoh: The movie was surprisingly good, the pacing never dropped and…",
        )
        tombol = st.button("Analisis sentimen", type="primary")

        if tombol:
            if not teks.strip():
                st.warning("Ulasannya masih kosong.")
            else:
                with st.spinner("Menganalisis…"):
                    p, n_kata, dikenal = prediksi_sentimen(model, word_index, teks)
                _tampilkan_hasil_sentimen(p, n_kata, dikenal)

    with tab2:
        st.markdown(
            "Unggah CSV dengan kolom `review` (atau file `.txt`, satu ulasan per baris). "
            "Hasilnya bisa diunduh kembali sebagai CSV."
        )
        up = st.file_uploader("File ulasan", type=["csv", "txt"])
        if up is not None:
            if up.name.endswith(".txt"):
                baris = [b.strip() for b in up.read().decode("utf-8").splitlines() if b.strip()]
                df = pd.DataFrame({"review": baris})
            else:
                df = pd.read_csv(up)
                if "review" not in df.columns:
                    st.error("CSV harus punya kolom bernama `review`.")
                    df = None
            if df is not None and len(df):
                df = df.head(500)
                if st.button("Analisis semua", type="primary"):
                    with st.spinner(f"Menganalisis {len(df)} ulasan…"):
                        seqs = np.stack([teks_ke_sequence(str(t), word_index)[0] for t in df["review"]])
                        prob = model.predict(seqs, verbose=0).flatten()
                    df["prob_positif"] = np.round(prob, 4)
                    df["sentimen"] = np.where(prob > 0.5, "positive", "negative")
                    pos = int((prob > 0.5).sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total ulasan", len(df))
                    c2.metric("Positif", pos)
                    c3.metric("Negatif", len(df) - pos)
                    st.dataframe(df, use_container_width=True, height=320)
                    st.download_button(
                        "Unduh hasil (CSV)", df.to_csv(index=False).encode("utf-8"),
                        "hasil_sentimen.csv", "text/csv",
                    )

    with st.expander("Tentang model ini"):
        st.markdown(
            """
            - Arsitektur: `Input(400) → Embedding(20000, 64) → GRU(64) → Dense(1, sigmoid)`, 1.305.025 parameter.
            - Teks dibersihkan persis seperti saat training: tag HTML dihapus, huruf kecil, hanya huruf a–z.
              Kata di luar 20.000 kosakata teratas jadi `<OOV>`; ulasan > 400 kata dipotong dari depan
              supaya bagian akhir (biasanya kesimpulan penulis) tetap dipakai.
            - Evaluasi data uji (7.438 ulasan): **Accuracy 0,8935 · Precision 0,8865 · Recall 0,9036 · F1 0,8949**.
            - Ambang keputusan 0,5. Nilai di sekitar 0,4–0,6 berarti model ragu, biasanya pada ulasan campuran atau sarkastik.
            """
        )


def _tampilkan_hasil_sentimen(p: float, n_kata: int, dikenal: int):
    if p >= 0.6:
        kelas, label, emoji = "pos", "Positif", "👍"
    elif p <= 0.4:
        kelas, label, emoji = "neg", "Negatif", "👎"
    else:
        kelas, label, emoji = "mid", "Positif" if p > 0.5 else "Negatif", "🤔"
    keyakinan = p if p > 0.5 else 1 - p

    st.markdown(
        f"""
        <div class="result {kelas}">
          <div class="lbl">Sentimen ulasan</div>
          <div class="big">{emoji} {label}</div>
          <div class="lbl">Keyakinan {keyakinan*100:.1f} % · probabilitas positif {p:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if kelas == "mid":
        st.info("Model agak ragu di ulasan ini. Biasanya terjadi pada ulasan campuran atau sarkasme.")

    fig = go.Figure(go.Bar(
        x=[1 - p, p], y=["Negatif", "Positif"], orientation="h",
        marker_color=["#E0475B", "#17A2A0"], text=[f"{(1-p)*100:.1f}%", f"{p*100:.1f}%"],
        textposition="outside",
    ))
    fig.update_layout(
        height=150, margin=dict(l=10, r=40, t=10, b=10), xaxis=dict(range=[0, 1.15], visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Sans, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah kata", n_kata)
    c2.metric("Kata dikenal model", f"{(dikenal / n_kata * 100) if n_kata else 0:.0f} %")
    c3.metric("Token dipakai", min(n_kata, SEQ_LEN_TEXT))
    if n_kata > SEQ_LEN_TEXT:
        st.caption(f"Ulasan lebih dari {SEQ_LEN_TEXT} kata, jadi {n_kata - SEQ_LEN_TEXT} kata pertama tidak ikut dibaca model.")


# --------------------------------------------------------------------------- #
# Halaman: Versioning & Info
# --------------------------------------------------------------------------- #
def halaman_versioning():
    st.markdown("## 🗂️ Versioning & info eksperimen")

    st.markdown("### Riwayat versi deployment")
    versi = pd.DataFrame([
        {
            "Tanggal": "10-09-2026", "Aplikasi": "SequenceLab", "Versi": "v1.0",
            "Permasalahan": "Belum ada cara mudah mengecek prediksi vs nilai aktual; input suhu hanya lewat unggah CSV; "
                            "sentimen hanya satu ulasan; tampilan bawaan Streamlit polos.",
            "Pemecahan": "Deploy versi dasar dulu: 2 model dimuat, 1 input per model, output angka.",
            "Pembaharuan fitur": "Rilis awal: prediksi suhu 1 jam (unggah CSV) dan klasifikasi sentimen 1 ulasan.",
            "Lanjut ke versi berikutnya": "Ya → v2.0",
            "Link": "ISI_LINK_STREAMLIT_V1",
            "Dokumentasi": "screenshot/v1_*.png",
        },
        {
            "Tanggal": "11-09-2026", "Aplikasi": "SequenceLab", "Versi": "v2.0 (final)",
            "Permasalahan": "Pengguna lain sulit memahami cara pakai; belum ada visualisasi; model dimuat ulang tiap klik.",
            "Pemecahan": "Desain ulang: tema warna & tipografi, halaman beranda dengan panduan, kartu hasil, grafik Plotly, "
                         "st.cache_resource agar model hanya dimuat sekali (optimasi).",
            "Pembaharuan fitur": "Data contoh bawaan + slider jendela waktu + pembanding nilai aktual; ramalan berantai s/d 12 jam; "
                                 "contoh ulasan sekali klik; mode batch CSV/TXT + unduh hasil; indikator status model; halaman versioning.",
            "Lanjut ke versi berikutnya": "Tidak (versi final)",
            "Link": "ISI_LINK_STREAMLIT_V2",
            "Dokumentasi": "screenshot/v2_*.png",
        },
    ])
    st.dataframe(versi, hide_index=True, use_container_width=True)
    st.caption("Ganti kolom Link dengan URL Streamlit tiap versi, dan simpan screenshot tiap versi di folder `screenshot/`.")

    st.markdown("### Ringkasan eksperimen Week 3")
    t1, t2 = st.tabs(["Jena Climate (regresi)", "IMDB (klasifikasi)"])
    with t1:
        jena = pd.DataFrame({
            "Model": ["LSTM", "GRU", "LSTM", "GRU", "LSTM", "GRU", "LSTM", "GRU"],
            "Konfigurasi": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "Seq": [24, 24, 24, 24, 72, 72, 72, 72],
            "Params": [20033, 15233, 32417, 24609, 20033, 15233, 32417, 24609],
            "Val Loss": [0.00781, 0.00720, 0.00833, 0.00791, 0.00679, 0.00695, 0.00763, 0.00775],
            "MAE (°C)": [0.4981, 0.4804, 0.5127, 0.5102, 0.4772, 0.4782, 0.4989, 0.5063],
            "RMSE (°C)": [0.7178, 0.6871, 0.7269, 0.7170, 0.6685, 0.6738, 0.7029, 0.7090],
            "Waktu (s)": [129.9, 85.2, 72.9, 68.2, 120.0, 130.6, 96.5, 86.8],
        })
        st.dataframe(
            jena.style.format({"Val Loss": "{:.5f}", "MAE (°C)": "{:.4f}", "RMSE (°C)": "{:.4f}", "Waktu (s)": "{:.1f}"})
                .highlight_min(subset=["MAE (°C)", "RMSE (°C)"], color="#FFE3D6"),
            hide_index=True, use_container_width=True,
        )
        st.caption("Dideploy: LSTM A seq 72 (baris tersorot).")
    with t2:
        imdb = pd.DataFrame({
            "Model": ["LSTM", "GRU", "LSTM", "GRU", "LSTM", "GRU", "LSTM", "GRU"],
            "Konfigurasi": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "Seq": [200, 200, 200, 200, 400, 400, 400, 400],
            "Accuracy": [0.8735, 0.8736, 0.8754, 0.8720, 0.8686, 0.8935, 0.8766, 0.8834],
            "Precision": [0.8522, 0.8885, 0.8661, 0.9093, 0.8489, 0.8865, 0.8649, 0.8558],
            "Recall": [0.9049, 0.8556, 0.8891, 0.8275, 0.8982, 0.9036, 0.8937, 0.9234],
            "F1": [0.8777, 0.8717, 0.8775, 0.8665, 0.8728, 0.8949, 0.8791, 0.8883],
            "Waktu (s)": [32.9, 27.3, 35.9, 55.5, 35.1, 46.2, 94.8, 76.7],
        })
        st.dataframe(
            imdb.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1"]} | {"Waktu (s)": "{:.1f}"})
                .highlight_max(subset=["Accuracy", "F1"], color="#D6F3EF"),
            hide_index=True, use_container_width=True,
        )
        st.caption("Dideploy: GRU A seq 400 (baris tersorot).")

    st.markdown("### Struktur repo")
    st.code(
        """app.py                     # aplikasi ini
requirements.txt
.streamlit/config.toml     # tema warna
models/
  LSTM_A_seq72.h5          # model suhu
  scaler.json              # mean & scale StandardScaler (13 fitur)
  sample_jena.csv          # data contoh per jam (opsional, dari data uji)
  GRU_A_seq400.h5          # model sentimen
  tokenizer.json           # word_index tokenizer (vocab 20.000)
screenshot/                # dokumentasi tiap versi""",
        language="text",
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
if halaman == "Beranda":
    halaman_beranda()
elif halaman.startswith("Prediksi Suhu"):
    halaman_suhu()
elif halaman.startswith("Analisis Sentimen"):
    halaman_sentimen()
else:
    halaman_versioning()

st.markdown(
    f'<div class="footer">SequenceLab {APP_VERSION} · Tugas Week 5 Deployment & MLOps · '
    f"Akmal Nugraha Saputra (2609)</div>",
    unsafe_allow_html=True,
)
