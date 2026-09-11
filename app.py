"""
Runtun: aplikasi deploy model sekuensial (LSTM/GRU) - Tugas MBC LAS Week 5
Akmal Nugraha Saputra (2609)

Model yang dipakai (hasil terbaik Week 3):
  1. Prakiraan suhu Jena Climate  -> models/LSTM_A_seq72.h5 (atau model *_seq72 / *_seq24 lain)
  2. Sentimen ulasan film IMDB    -> models/GRU_A_seq400.h5 (atau model *_seq200 / *_seq400 lain)

Artefak pendukung di folder models/:
  scaler.json, tokenizer.json, sample_jena.csv, (opsional) sample_ulasan.csv
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
APP_VERSION = "v3.0"

# ---------------------------------------------------------------------------
# Token desain
# ---------------------------------------------------------------------------
INK = "#172033"
MUTED = "#586179"
PAPER = "#EEF1F5"
SURFACE = "#FFFFFF"
LINE = "#D6DCE6"
COLD = "#2C4FB8"
WARM = "#C8283E"

# Skala divergen dingin -> hangat (dipakai untuk suhu DAN sentimen)
THERMAL = ["#14306E", "#2C4FB8", "#6E8FDB", "#B8C8EE", "#EDEAE4",
           "#F4B9AB", "#E0705C", "#C8283E", "#7F1128"]

JENA_FEATURES_DEFAULT = [
    "p (mbar)", "T (degC)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
    "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)",
    "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)",
]
TARGET = "T (degC)"

# Hasil evaluasi test set dari notebook Week 3 (dipakai untuk kartu model)
JENA_METRICS = {
    ("LSTM", "A", 24): (0.4981, 0.7178), ("GRU", "A", 24): (0.4804, 0.6871),
    ("LSTM", "B", 24): (0.5127, 0.7269), ("GRU", "B", 24): (0.5102, 0.7170),
    ("LSTM", "A", 72): (0.4772, 0.6685), ("GRU", "A", 72): (0.4782, 0.6738),
    ("LSTM", "B", 72): (0.4989, 0.7029), ("GRU", "B", 72): (0.5063, 0.7090),
}
IMDB_METRICS = {  # accuracy, precision, recall, f1
    ("LSTM", "A", 200): (0.8735, 0.8522, 0.9049, 0.8777),
    ("GRU", "A", 200): (0.8736, 0.8885, 0.8556, 0.8717),
    ("LSTM", "B", 200): (0.8754, 0.8661, 0.8891, 0.8775),
    ("GRU", "B", 200): (0.8720, 0.9093, 0.8275, 0.8665),
    ("LSTM", "A", 400): (0.8686, 0.8489, 0.8982, 0.8728),
    ("GRU", "A", 400): (0.8935, 0.8865, 0.9036, 0.8949),
    ("LSTM", "B", 400): (0.8766, 0.8649, 0.8937, 0.8791),
    ("GRU", "B", 400): (0.8834, 0.8558, 0.9234, 0.8883),
}
PARAMS = {
    ("jena", "LSTM", "A"): 20033, ("jena", "GRU", "A"): 15233,
    ("jena", "LSTM", "B"): 32417, ("jena", "GRU", "B"): 24609,
    ("imdb", "LSTM", "A"): 1313089, ("imdb", "GRU", "A"): 1305025,
    ("imdb", "LSTM", "B"): 1325473, ("imdb", "GRU", "B"): 1314401,
}
CONFIG_TEXT = {"A": "1 layer, 64 unit", "B": "2 layer (64 dan 32 unit) + dropout"}

EXAMPLES = {
    "Hangat": ("I went in with low expectations and walked out grinning. The cast has "
               "real chemistry, the jokes land, and the final twenty minutes are "
               "genuinely moving. I would happily watch it again with friends."),
    "Dingin": ("Two hours I will never get back. The plot makes no sense, the dialogue "
               "is wooden, and even the talented lead cannot save a script this lazy. "
               "Skip it."),
    "Campur": ("The actors were great and the soundtrack is lovely, but the story drags "
               "so badly that I kept checking my watch. What a waste of a good cast."),
    "Negasi": "Not good. Not funny. Not worth your time or your money.",
}


# ---------------------------------------------------------------------------
# Gaya
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&display=swap');

.stMarkdown p, .stMarkdown li, textarea, input {{
  font-family: 'Atkinson Hyperlegible', system-ui, sans-serif;
}}
h1, h2, h3, h4 {{
  font-family: 'Bricolage Grotesque', 'Atkinson Hyperlegible', sans-serif !important;
  color: {INK};
  letter-spacing: -0.015em;
}}
.block-container {{ max-width: 1120px; padding-top: 4.5rem; padding-bottom: 4rem; }}
p, li {{ line-height: 1.6; }}

/* Hero */
.stMarkdown .hero-title {{
  font-family: 'Bricolage Grotesque', sans-serif !important; font-weight: 800 !important;
  font-size: clamp(3.4rem, 10vw, 7rem) !important; line-height: 1 !important; margin: 0 0 .6rem -0.04em !important;
  letter-spacing: -0.04em; color: {INK}; padding: 0 !important;
}}
.stMarkdown .hero-lede {{ font-size: 1.25rem; color: {INK}; max-width: 38ch; margin: 0 0 1.6rem 0; }}
.hero-note {{ color: {MUTED}; font-size: .95rem; max-width: 60ch; }}

/* Garis suhu (warming stripes) */
.stripes {{ display: flex; height: 112px; width: 100%; border-radius: 3px; overflow: hidden; }}
.stripes span {{ flex: 1 1 0; }}
.stripes.small {{ height: 18px; }}
.stripe-legend {{ display: flex; justify-content: space-between; color: {MUTED};
  font-size: .85rem; margin-top: .45rem; }}

/* Panel hasil ("bacaan") */
.reading {{ background: {SURFACE}; border: 1px solid {LINE}; border-left: 8px solid var(--tone, {COLD});
  border-radius: 6px; padding: 1.25rem 1.5rem; }}
.reading .label {{ color: {MUTED}; font-size: .95rem; margin: 0; }}
.reading .value {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: clamp(2.6rem, 6vw, 3.6rem); line-height: 1.05; color: {INK}; margin: .15rem 0 .35rem 0; }}
.reading .sub {{ color: {INK}; margin: 0; }}
.facts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .75rem 1.5rem; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid {LINE}; }}
.facts div b {{ display: block; font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 1.35rem; color: {INK}; }}
.facts div span {{ color: {MUTED}; font-size: .9rem; }}

/* Termometer sentimen */
.gauge {{ position: relative; height: 16px; border-radius: 999px; margin: 1.1rem 0 .4rem 0;
  background: linear-gradient(90deg, {", ".join(THERMAL)}); }}
.gauge .pin {{ position: absolute; top: -7px; width: 6px; height: 30px; border-radius: 3px;
  background: {INK}; box-shadow: 0 0 0 3px {SURFACE}; transform: translateX(-3px); }}
.gauge-legend {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: .85rem; }}

/* Teks dengan sorotan kata */
.marked {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px;
  padding: 1rem 1.2rem; line-height: 2.05; font-size: 1.02rem; max-height: 340px; overflow-y: auto; }}
.marked span {{ padding: .12rem .22rem; border-radius: 3px; }}

/* Entri beranda */
.entry h3 {{ margin-top: .2rem; }}
.entry .num {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: 2.4rem; color: {INK}; line-height: 1; }}
.entry .num-label {{ color: {MUTED}; font-size: .92rem; }}
.steps {{ counter-reset: s; list-style: none; padding: 0; margin: 0; }}
.steps li {{ counter-increment: s; position: relative; padding: 0 0 .9rem 2.6rem; }}
.steps li::before {{ content: counter(s); position: absolute; left: 0; top: -.1rem;
  width: 1.8rem; height: 1.8rem; border-radius: 50%; background: {INK}; color: {SURFACE};
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; display: grid; place-items: center; }}

.quiet, .stMarkdown p.quiet, .stMarkdown .quiet {{ color: {MUTED}; font-size: .95rem; }}
.pill {{ display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .85rem;
  border: 1px solid {LINE}; background: {SURFACE}; color: {INK}; }}

/* Fokus keyboard yang jelas */
button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible,
[role="tab"]:focus-visible, [role="radio"]:focus-visible {{
  outline: 3px solid {COLD} !important; outline-offset: 2px;
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""",
        unsafe_allow_html=True,
    )


def thermal_color(x: float, lo: float, hi: float) -> str:
    """Warna dari skala dingin-hangat untuk nilai x di rentang [lo, hi]."""
    t = 0.0 if hi == lo else float(np.clip((x - lo) / (hi - lo), 0, 1))
    pos = t * (len(THERMAL) - 1)
    i = int(np.floor(pos))
    j = min(i + 1, len(THERMAL) - 1)
    f = pos - i
    a = np.array([int(THERMAL[i][k:k + 2], 16) for k in (1, 3, 5)])
    b = np.array([int(THERMAL[j][k:k + 2], 16) for k in (1, 3, 5)])
    c = (a + (b - a) * f).round().astype(int)
    return "#%02x%02x%02x" % tuple(c)


HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]


def tanggal_id(t: pd.Timestamp) -> str:
    return f"{HARI[t.weekday()]}, {t.day} {BULAN[t.month - 1]} {t.year} pukul {t.hour:02d}.00"


def tgl_pendek(t: pd.Timestamp) -> str:
    return f"{t.day} {BULAN[t.month - 1][:3]} {t.year}"


def temp_color(t_c: float) -> str:
    return thermal_color(t_c, -10, 30)


def stripes_html(values, lo, hi, small=False) -> str:
    spans = "".join(f'<span style="background:{thermal_color(v, lo, hi)}"></span>' for v in values)
    cls = "stripes small" if small else "stripes"
    return f'<div class="{cls}" role="img" aria-label="Garis warna suhu harian">{spans}</div>'


def altair_theme():
    return {
        "config": {
            "font": "Atkinson Hyperlegible, system-ui, sans-serif",
            "background": "transparent",
            "view": {"stroke": None},
            "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": "#E3E7EE",
                     "domainColor": LINE, "tickColor": LINE, "labelFontSize": 12,
                     "titleFontSize": 12, "titleFontWeight": "normal"},
            "legend": {"labelColor": INK, "titleColor": MUTED, "orient": "top",
                       "labelFontSize": 12, "titleFontSize": 12},
        }
    }


# ---------------------------------------------------------------------------
# Pencarian model dan artefak
# ---------------------------------------------------------------------------
NAME_RE = re.compile(r"(LSTM|GRU)_([AB])_seq(\d+)", re.I)


def describe_model(path: Path) -> dict:
    m = NAME_RE.search(path.stem)
    if not m:
        return {"path": path, "cell": path.stem, "cfg": "?", "seq": None, "task": None}
    cell, cfg, seq = m.group(1).upper(), m.group(2).upper(), int(m.group(3))
    task = "jena" if seq <= 168 else "imdb"
    return {"path": path, "cell": cell, "cfg": cfg, "seq": seq, "task": task}


def model_label(d: dict) -> str:
    unit = "jam" if d["task"] == "jena" else "kata"
    return f'{d["cell"]} {CONFIG_TEXT.get(d["cfg"], "")}, {d["seq"]} {unit}'


@st.cache_data(show_spinner=False)
def find_models() -> dict:
    out = {"jena": [], "imdb": []}
    if MODEL_DIR.exists():
        for p in sorted(MODEL_DIR.rglob("*.h5")):
            d = describe_model(p)
            if d["task"]:
                out[d["task"]].append(d)

    def rank(d):  # model terbaik menurut notebook ditaruh paling depan
        if d["task"] == "jena":
            return JENA_METRICS.get((d["cell"], d["cfg"], d["seq"]), (9, 9))[0]
        return -IMDB_METRICS.get((d["cell"], d["cfg"], d["seq"]), (0, 0, 0, 0))[3]

    for k in out:
        out[k].sort(key=rank)
    return out


def find_file(*names: str) -> Path | None:
    for n in names:
        for base in (MODEL_DIR, ROOT, ROOT / "jena", ROOT / "imdb",
                     ROOT / "jena" / "models", ROOT / "imdb" / "models"):
            p = base / n
            if p.exists():
                return p
    return None


@st.cache_resource(show_spinner=False)
def get_tf():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # impor lambat: halaman beranda tidak perlu menunggu TensorFlow
    return tf


@st.cache_resource(show_spinner=False)
def load_keras(path: str):
    tf = get_tf()
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner=False)
def build_tflite(path: str) -> bytes:
    """Optimasi: konversi .h5 ke TFLite dengan dynamic-range quantization (bobot int8).
    Jika sudah ada file .tflite di samping .h5, file itu yang dipakai."""
    ready = Path(path).with_suffix(".tflite")
    if ready.exists():
        return ready.read_bytes()
    tf = get_tf()
    model = load_keras(path)
    shape = [1] + [int(s) for s in model.input_shape[1:]]
    dtype = tf.int32 if len(shape) == 2 else tf.float32

    @tf.function(input_signature=[tf.TensorSpec(shape, dtype)])
    def serve(x):
        return model(x, training=False)

    conv = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()], model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        return conv.convert()
    except Exception:
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        return conv.convert()


TFLITE_FAILED: set = set()  # model yang tidak bisa dijalankan TFLite di server ini


def _run_tflite(path: str, X: np.ndarray) -> tuple[np.ndarray, float]:
    tf = get_tf()
    interp = tf.lite.Interpreter(model_content=build_tflite(path))
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    preds = np.empty(len(X), dtype=np.float32)
    t0 = time.perf_counter()
    for k in range(len(X)):
        interp.set_tensor(inp["index"], X[k:k + 1].astype(inp["dtype"]))
        interp.invoke()
        preds[k] = interp.get_tensor(outp["index"]).ravel()[0]
    return preds, (time.perf_counter() - t0) * 1000 / max(len(X), 1)


def run_model(path: str, X: np.ndarray, engine: str) -> tuple[np.ndarray, float]:
    """Mengembalikan (prediksi, milidetik per sampel).
    Jika mode TFLite gagal untuk model tertentu, otomatis kembali ke Keras supaya aplikasi tidak berhenti."""
    if engine == "tflite" and path not in TFLITE_FAILED:
        try:
            return _run_tflite(path, X)
        except Exception:
            TFLITE_FAILED.add(path)
    if engine == "tflite":
        st.caption(":material/info: Mode TFLite belum didukung untuk model ini di server, "
                   "jadi prediksi memakai Keras (.h5). Hasilnya tetap sama.")
    model = load_keras(path)
    t0 = time.perf_counter()
    parts = [model(X[i:i + 256], training=False).numpy().ravel() for i in range(0, len(X), 256)]
    preds = np.concatenate(parts) if parts else np.array([])
    return preds, (time.perf_counter() - t0) * 1000 / max(len(X), 1)


# ---------------------------------------------------------------------------
# Data Jena
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_scaler() -> dict | None:
    p = find_file("scaler.json", "scaler_jena.json", "jena_scaler.json")
    if p is None:
        return None
    raw = json.loads(p.read_text())

    def pick(*keys):
        for k in keys:
            if k in raw:
                return raw[k]
        return None

    mean = pick("mean", "mean_", "means", "feature_mean")
    scale = pick("scale", "scale_", "std", "stds", "feature_std")
    feats = pick("features", "feature_names", "columns", "feature_cols", "feature_names_in_")
    if mean is None or scale is None:
        return None
    feats = list(feats) if feats else JENA_FEATURES_DEFAULT[: len(mean)]
    return {"features": feats, "mean": np.asarray(mean, float), "scale": np.asarray(scale, float)}


def prepare_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan data mentah dengan preprocessing di notebook: bersihkan -9999,
    ubah ke per jam bila datanya per 10 menit, lalu jadikan waktu sebagai indeks."""
    df = df.copy()
    tcol = next((c for c in df.columns if c.strip().lower() in ("date time", "datetime", "date_time", "time", "waktu")), None)
    if tcol is None:
        tcol = df.columns[0]
    t = pd.to_datetime(df[tcol], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    if t.isna().mean() > 0.5:
        t = pd.to_datetime(df[tcol], errors="coerce")
    df.index = t
    df = df[~df.index.isna()].drop(columns=[tcol]).sort_index()
    for c in ("wv (m/s)", "max. wv (m/s)"):
        if c in df.columns:
            df.loc[df[c] == -9999.0, c] = 0.0
    if len(df) > 3:
        step = pd.Series(df.index).diff().median()
        if pd.notna(step) and step < pd.Timedelta(minutes=30):
            df = df.iloc[5::6]
    return df


@st.cache_data(show_spinner=False)
def load_jena_sample() -> pd.DataFrame | None:
    p = find_file("sample_jena.csv", "jena_sample.csv", "sample_test_jena.csv")
    if p is None:
        return None
    return prepare_hourly(pd.read_csv(p))


@st.cache_data(show_spinner=False)
def read_uploaded_csv(data: bytes) -> pd.DataFrame:
    import io
    return prepare_hourly(pd.read_csv(io.BytesIO(data)))


def scale_frame(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    return ((df[scaler["features"]].to_numpy(float) - scaler["mean"]) / scaler["scale"]).astype(np.float32)


def target_stats(scaler: dict) -> tuple[float, float]:
    i = scaler["features"].index(TARGET)
    return float(scaler["mean"][i]), float(scaler["scale"][i])


# ---------------------------------------------------------------------------
# Teks IMDB
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tokenizer() -> dict | None:
    p = find_file("tokenizer.json", "tokenizer_imdb.json", "imdb_tokenizer.json")
    if p is None:
        return None
    raw = json.loads(p.read_text())
    cfg = raw.get("config", raw)
    wi = cfg.get("word_index", raw.get("word_index"))
    if isinstance(wi, str):
        wi = json.loads(wi)
    if wi is None and all(isinstance(v, int) for v in list(raw.values())[:50]):
        wi = raw
    if not wi:
        return None
    num_words = cfg.get("num_words") or raw.get("num_words") or raw.get("vocab_size") or 20000
    oov_tok = cfg.get("oov_token") or raw.get("oov_token") or "<OOV>"
    return {"word_index": {k: int(v) for k, v in wi.items()}, "num_words": int(num_words),
            "oov": int(wi.get(oov_tok, 1))}


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", str(text))
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_ids(words: list[str], tok: dict) -> list[int]:
    wi, n, oov = tok["word_index"], tok["num_words"], tok["oov"]
    ids = []
    for w in words:
        i = wi.get(w)
        ids.append(oov if i is None or i >= n else i)
    return ids


def pad_pre(ids: list[int], L: int) -> np.ndarray:
    ids = ids[-L:]
    return np.array([0] * (L - len(ids)) + ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# Komponen kecil
# ---------------------------------------------------------------------------
def reading(label: str, value: str, sub: str, tone: str, facts: list[tuple[str, str]] | None = None) -> None:
    facts_html = ""
    if facts:
        facts_html = '<div class="facts">' + "".join(
            f"<div><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>" for k, v in facts) + "</div>"
    st.markdown(
        f'<div class="reading" style="--tone:{tone}"><p class="label">{html.escape(label)}</p>'
        f'<p class="value">{html.escape(value)}</p><p class="sub">{sub}</p>{facts_html}</div>',
        unsafe_allow_html=True,
    )


def gauge(pos01: float, left: str, right: str, middle: str = "", aria: str = "") -> None:
    mid = f"<span>{middle}</span>" if middle else ""
    st.markdown(f'<div class="gauge" role="img" aria-label="{html.escape(aria)}">'
                f'<div class="pin" style="left:{np.clip(pos01, 0, 1) * 100:.1f}%"></div></div>'
                f'<div class="gauge-legend"><span>{left}</span>{mid}<span>{right}</span></div>',
                unsafe_allow_html=True)


def legend_chips(items: list[tuple[str, str, str]]) -> None:
    """items: (label, warna, bentuk) bentuk = 'line' | 'dot' | 'diamond' | 'dash'"""
    parts = []
    for label, color, shape in items:
        if shape == "line":
            mark = f'<i style="display:inline-block;width:18px;height:3px;background:{color};vertical-align:middle"></i>'
        elif shape == "dash":
            mark = f'<i style="display:inline-block;width:18px;border-top:3px dashed {color};vertical-align:middle"></i>'
        elif shape == "diamond":
            mark = f'<i style="display:inline-block;width:9px;height:9px;background:{color};transform:rotate(45deg);vertical-align:middle"></i>'
        else:
            mark = f'<i style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{color};vertical-align:middle"></i>'
        parts.append(f'<span style="margin-right:1.1rem;white-space:nowrap">{mark}&nbsp;&nbsp;{html.escape(label)}</span>')
    st.markdown(f'<div class="quiet">{"".join(parts)}</div>', unsafe_allow_html=True)


def missing_box(what: str, files: str) -> None:
    st.warning(
        f"{what} belum bisa dijalankan karena file berikut tidak ditemukan di folder `models/`: "
        f"{files}. Unggah file tersebut ke repositori GitHub, lalu muat ulang halaman ini."
    )


# ---------------------------------------------------------------------------
# Halaman: Beranda
# ---------------------------------------------------------------------------
def page_home() -> None:
    st.markdown('<div class="hero-title" role="heading" aria-level="1">Runtun</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-lede">Dua model yang membaca urutan: suhu udara jam demi jam, '
        'dan kata demi kata dalam ulasan film.</p>', unsafe_allow_html=True)

    sample, scaler = load_jena_sample(), load_scaler()
    if sample is not None and TARGET in sample.columns:
        daily = sample[TARGET].resample("D").mean().dropna()
        st.markdown(stripes_html(daily.values, -10, 25), unsafe_allow_html=True)
        st.markdown(
            f'<div class="stripe-legend"><span>{tgl_pendek(daily.index[0])}</span>'
            f'<span>Setiap garis adalah rata-rata suhu satu hari di Jena, Jerman. Biru lebih dingin, merah lebih hangat.</span>'
            f'<span>{tgl_pendek(daily.index[-1])}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(stripes_html(np.sin(np.linspace(0, 2 * np.pi, 120)) * 15 + 7, -10, 25), unsafe_allow_html=True)

    st.write("")
    models = find_models()
    left, right = st.columns(2, gap="large")
    with left:
        best = models["jena"][0] if models["jena"] else None
        mae = JENA_METRICS.get((best["cell"], best["cfg"], best["seq"]), (None,))[0] if best else None
        st.markdown(
            f"""<div class="entry"><h3>Prakiraan suhu</h3>
<p>Berikan 72 jam data cuaca terakhir, model menebak suhu satu jam berikutnya.</p>
<div class="num">±{mae:.2f} °C</div><div class="num-label">rata-rata meleset pada data uji 2015–2016</div></div>"""
            if mae else '<div class="entry"><h3>Prakiraan suhu</h3><p>Model belum diunggah.</p></div>',
            unsafe_allow_html=True)
        st.write("")
        st.page_link(PAGES["suhu"], label="Buka prakiraan suhu", icon=":material/thermostat:")
    with right:
        best = models["imdb"][0] if models["imdb"] else None
        acc = IMDB_METRICS.get((best["cell"], best["cfg"], best["seq"]), (None,))[0] if best else None
        st.markdown(
            f"""<div class="entry"><h3>Pembaca ulasan film</h3>
<p>Tempel ulasan film berbahasa Inggris, model menilai apakah sambutannya hangat atau dingin.</p>
<div class="num">{acc * 100:.1f}%</div><div class="num-label">ulasan uji yang ditebak benar</div></div>"""
            if acc else '<div class="entry"><h3>Pembaca ulasan film</h3><p>Model belum diunggah.</p></div>',
            unsafe_allow_html=True)
        st.write("")
        st.page_link(PAGES["sentimen"], label="Buka pembaca ulasan", icon=":material/movie:")

    st.divider()
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        st.subheader("Cara memakai")
        st.markdown(
            """<ol class="steps">
<li>Pilih halaman di menu atas: prakiraan suhu atau pembaca ulasan.</li>
<li>Pakai data contoh yang sudah tersedia, atau unggah data sendiri.</li>
<li>Tekan tombol prediksi. Hasil muncul di panel berwarna, lengkap dengan penjelasannya.</li>
</ol>""", unsafe_allow_html=True)
    with c2:
        st.subheader("Kenapa satu warna")
        st.markdown(
            "Kedua model memakai skala yang sama dari biru ke merah. Untuk suhu, biru berarti dingin. "
            "Untuk ulasan, biru berarti penonton kecewa dan merah berarti penonton puas.")
        st.markdown(f'<div class="gauge" aria-hidden="true"></div>'
                    f'<div class="gauge-legend"><span>Dingin</span><span>Hangat</span></div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Halaman: Prakiraan suhu
# ---------------------------------------------------------------------------
def page_forecast() -> None:
    st.title("Prakiraan suhu satu jam ke depan")
    st.markdown('<p class="quiet">Model membaca 13 variabel cuaca (tekanan, kelembapan, angin, dan lainnya) '
                'selama beberapa jam terakhir, lalu menebak suhu udara pada jam berikutnya.</p>',
                unsafe_allow_html=True)

    models, scaler = find_models()["jena"], load_scaler()
    if not models or scaler is None:
        missing_box("Prakiraan suhu", "`LSTM_A_seq72.h5` (atau model Jena lain) dan `scaler.json`")
        return

    chosen, engine = models[0], "keras"  # model terbaik (MAE terkecil di notebook)
    source = st.radio("Sumber data", ["Data cuaca Jena 2015–2016", "Unggah data sendiri (CSV)"], horizontal=True,
                      help="Data cuaca Jena adalah catatan asli Okt 2015 sampai Des 2016 yang tidak dipakai saat melatih model.")

    df = None
    if source.startswith("Unggah"):
        up = st.file_uploader("Unggah CSV dengan kolom seperti dataset Jena Climate (termasuk `Date Time`)",
                              type="csv", help="Data per 10 menit otomatis diubah menjadi per jam, sama seperti di notebook.")
        if up is None:
            st.info("Belum ada file. Kolom yang dibutuhkan: `Date Time` dan " + ", ".join(f"`{f}`" for f in scaler["features"]) + ".")
            return
        try:
            df = read_uploaded_csv(up.getvalue())
        except Exception as e:  # pesan singkat yang bisa ditindaklanjuti
            st.error(f"File tidak bisa dibaca sebagai CSV cuaca: {e}")
            return
    else:
        df = load_jena_sample()
        if df is None:
            missing_box("Data contoh", "`sample_jena.csv`")
            return

    missing = [f for f in scaler["features"] if f not in df.columns]
    if missing:
        st.error("Kolom berikut tidak ada di data: " + ", ".join(f"`{m}`" for m in missing))
        return
    df = df.dropna(subset=scaler["features"])
    seq = chosen["seq"]
    if len(df) <= seq:
        st.error(f"Data terlalu pendek. Model ini butuh minimal {seq + 1} jam data, file ini hanya {len(df)} jam.")
        return

    scaled = scale_frame(df, scaler)
    t_mean, t_std = target_stats(scaler)
    temps = df[TARGET].to_numpy(float)
    times = df.index

    forecast_single(models=models, chosen=chosen, engine=engine, scaled=scaled, temps=temps, times=times,
                    t_mean=t_mean, t_std=t_std, seq=seq)


def forecast_single(models, chosen, engine, scaled, temps, times, t_mean, t_std, seq) -> None:
    first, last = times[seq].to_pydatetime(), times[-1].to_pydatetime() + pd.Timedelta(hours=1)
    default = pd.Timestamp("2016-07-20 15:00")
    if not (times[seq] <= default <= times[-1]):
        default = times[-1]
    d1, d2 = st.columns([1, 2])
    with d1:
        day = st.date_input("Tanggal", value=default.date(), min_value=first.date(), max_value=last.date(), format="DD/MM/YYYY")
    with d2:
        hour = st.select_slider("Jam yang ingin ditebak", options=list(range(24)), value=int(default.hour),
                                format_func=lambda h: f"{h:02d}.00")
    target_time = pd.Timestamp(day) + pd.Timedelta(hours=hour)

    # posisi jam target di data; jika melewati akhir data, prediksi masa depan
    pos = int(times.searchsorted(target_time))
    future = pos >= len(times)
    exact = (not future) and times[pos] == target_time
    if not future and not exact:
        target_time = times[pos]
    if pos < seq:
        st.warning(f"Jam ini terlalu awal. Pilih waktu setelah {times[seq]:%d/%m/%Y %H.00} agar ada {seq} jam riwayat.")
        return

    window = scaled[pos - seq:pos][None, ...]
    with st.spinner("Model sedang membaca riwayat cuaca..."):
        pred_s, ms = run_model(str(chosen["path"]), window, engine)
    pred = float(pred_s[0] * t_std + t_mean)
    actual = None if future else float(temps[pos])
    prev = float(temps[pos - 1])

    left, right = st.columns([1, 1.6], gap="large")
    with left:
        when = tanggal_id(target_time)
        if actual is None:
            sub = f"Ini tebakan untuk jam setelah data berakhir. Satu jam sebelumnya suhunya {prev:.1f} °C."
            facts = [("jam sebelumnya", f"{prev:.1f} °C"), ("waktu hitung", f"{ms:.1f} ms")]
        else:
            err = pred - actual
            sub = f"Suhu yang tercatat sebenarnya {actual:.1f} °C, jadi tebakan meleset {abs(err):.2f} °C."
            facts = [("suhu sebenarnya", f"{actual:.1f} °C"), ("selisih", f"{err:+.2f} °C"),
                     ("jam sebelumnya", f"{prev:.1f} °C"), ("waktu hitung", f"{ms:.1f} ms")]
        reading(f"Tebakan suhu, {when}", f"{pred:.1f} °C", sub, temp_color(pred) if abs(pred - 10) > 6 else COLD, facts)
        gauge((pred + 10) / 40, "−10 °C", "30 °C", "10 °C", f"Posisi suhu {pred:.1f} derajat pada skala")
    with right:
        hist = pd.DataFrame({"waktu": times[pos - seq:pos], "suhu": temps[pos - seq:pos], "jenis": "Riwayat"})
        pts = [{"waktu": target_time, "suhu": pred, "jenis": "Tebakan model"}]
        if actual is not None:
            pts.append({"waktu": target_time, "suhu": actual, "jenis": "Suhu sebenarnya"})
        pts = pd.DataFrame(pts)
        color = alt.Scale(domain=["Riwayat", "Tebakan model", "Suhu sebenarnya"], range=[INK, WARM, COLD])
        base = alt.Chart(hist).mark_line(strokeWidth=2).encode(
            x=alt.X("waktu:T", title=None, axis=alt.Axis(format="%d/%m %H.00", labelAngle=0, tickCount=6)),
            y=alt.Y("suhu:Q", title="Suhu (°C)", scale=alt.Scale(zero=False)),
            color=alt.Color("jenis:N", scale=color, legend=None),
            tooltip=[alt.Tooltip("waktu:T", format="%d/%m/%Y %H.00"), alt.Tooltip("suhu:Q", format=".1f")])
        dots = alt.Chart(pts).mark_point(size=160, filled=True, opacity=1).encode(
            x="waktu:T", y="suhu:Q", color=alt.Color("jenis:N", scale=color, legend=None),
            shape=alt.Shape("jenis:N", scale=alt.Scale(domain=["Tebakan model", "Suhu sebenarnya"],
                                                       range=["circle", "diamond"]), legend=None),
            tooltip=["jenis:N", alt.Tooltip("suhu:Q", format=".2f")])
        st.markdown(f"**{seq} jam yang dibaca model**")
        legend_chips([("Riwayat suhu", INK, "line"), ("Tebakan model", WARM, "dot")]
                     + ([("Suhu sebenarnya", COLD, "diamond")] if actual is not None else []))
        st.altair_chart((base + dots).properties(height=300), width="stretch")


# ---------------------------------------------------------------------------
# Halaman: Pembaca ulasan
# ---------------------------------------------------------------------------
def verdict_text(p: float) -> tuple[str, str]:
    if p >= 0.5:
        strength = "sangat yakin" if p >= 0.85 else "cukup yakin" if p >= 0.65 else "ragu-ragu"
        return "Sambutan hangat", f"Model {strength} ulasan ini positif."
    strength = "sangat yakin" if p <= 0.15 else "cukup yakin" if p <= 0.35 else "ragu-ragu"
    return "Sambutan dingin", f"Model {strength} ulasan ini negatif."


def page_sentiment() -> None:
    st.title("Pembaca ulasan film")
    st.markdown('<p class="quiet">Model dilatih dari 50.000 ulasan IMDB berbahasa Inggris. '
                'Tulis atau tempel ulasan dalam bahasa Inggris agar hasilnya bermakna.</p>', unsafe_allow_html=True)

    models, tok = find_models()["imdb"], load_tokenizer()
    if not models or tok is None:
        missing_box("Pembaca ulasan", "`GRU_A_seq400.h5` (atau model IMDB lain) dan `tokenizer.json`")
        return

    chosen, engine = models[0], "keras"  # model terbaik (F1 tertinggi di notebook)
    L = chosen["seq"]

    tab1, tab2 = st.tabs(["Satu ulasan", "Banyak ulasan sekaligus"])
    with tab1:
        sentiment_single(chosen, engine, tok, L)
    with tab2:
        sentiment_batch(chosen, tok, L)


def sentiment_single(chosen, engine, tok, L) -> None:
    if "review" not in st.session_state:
        st.session_state.review = EXAMPLES["Campur"]
    ex = st.pills("Coba contoh", list(EXAMPLES), key="ex_pick", help="Klik salah satu untuk mengisi kotak ulasan.")
    if ex and st.session_state.get("_last_ex") != ex:
        st.session_state.review = EXAMPLES[ex]
        st.session_state._last_ex = ex
    text = st.text_area("Ulasan film (bahasa Inggris)", key="review", height=150,
                        placeholder="Contoh: The story was touching and the acting was superb...")
    explain = st.toggle("Tunjukkan kata yang paling berpengaruh", value=True,
                        help="Setiap kata dihapus satu per satu untuk melihat seberapa jauh keyakinan model berubah.")
    go = st.button("Baca ulasan", type="primary")

    if go:
        words = clean_text(text).split()
        if len(words) < 3:
            st.warning("Ulasan terlalu pendek. Tulis minimal tiga kata dalam bahasa Inggris.")
            return
        ids = to_ids(words, tok)
        known = sum(1 for i in ids if i != tok["oov"])
        if known / len(ids) < 0.4:
            st.warning("Sebagian besar kata tidak dikenali model. Pastikan ulasannya ditulis dalam bahasa Inggris.")
        x = pad_pre(ids, L)[None, :]
        with st.spinner("Model sedang membaca ulasan..."):
            p, ms = run_model(str(chosen["path"]), x, engine)
        p = float(p[0])
        title, sub = verdict_text(p)
        tone = WARM if p >= 0.5 else COLD

        left, right = st.columns([1.1, 1.1], gap="large")
        with left:
            reading("Hasil bacaan", title, f"Peluang positif {p * 100:.0f}%. {sub}", tone,
                    [("jumlah kata", f"{len(words)}"), ("kata dikenal", f"{known / len(words) * 100:.0f}%"),
                     ("waktu hitung", f"{ms:.1f} ms")])
            gauge(p, "Dingin (negatif)", "Hangat (positif)", "Ragu", f"Keyakinan {p * 100:.0f} persen positif")
            if len(words) > L:
                st.caption(f"Ulasan lebih dari {L} kata, jadi model hanya membaca {L} kata terakhir, sama seperti saat training.")
        with right:
            if explain:
                kept = words[-L:]
                uniq = list(dict.fromkeys(kept))[:300]
                base_ids = to_ids(kept, tok)
                variants = np.stack([pad_pre([i for w, i in zip(kept, base_ids) if w != u], L) for u in uniq])
                with st.spinner("Mengukur pengaruh tiap kata..."):
                    q, _ = run_model(str(chosen["path"]), variants, "keras")
                infl = dict(zip(uniq, p - q))  # >0 berarti kata itu mendorong ke positif
                lim = max(max(abs(v) for v in infl.values()), 0.05)
                spans = []
                for w in kept:
                    v = infl.get(w, 0.0)
                    a = min(abs(v) / lim, 1.0)
                    rgb = (200, 40, 62) if v > 0 else (44, 79, 184)
                    style = f"background: rgba({rgb[0]},{rgb[1]},{rgb[2]},{0.08 + 0.55 * a:.2f})" if a > 0.08 else ""
                    spans.append(f'<span style="{style}" title="{v:+.3f}">{html.escape(w)}</span>')
                st.markdown("**Kata yang menggeser keputusan**")
                st.markdown('<div class="marked">' + " ".join(spans) + "</div>", unsafe_allow_html=True)
                top_pos = [w for w, v in sorted(infl.items(), key=lambda kv: -kv[1]) if v > 0.005][:5]
                top_neg = [w for w, v in sorted(infl.items(), key=lambda kv: kv[1]) if v < -0.005][:5]
                st.markdown(
                    f'<p class="quiet" style="margin-top:.6rem">Merah mendorong ke positif: '
                    f'{", ".join(top_pos) or "tidak ada"}. Biru mendorong ke negatif: {", ".join(top_neg) or "tidak ada"}.</p>',
                    unsafe_allow_html=True)
            else:
                st.markdown("**Teks yang dibaca model**")
                st.markdown('<div class="marked">' + html.escape(" ".join(words[-L:])) + "</div>", unsafe_allow_html=True)
                st.caption("Huruf kecil semua, tanpa angka dan tanda baca: sama seperti pembersihan teks saat training.")


def sentiment_batch(chosen, tok, L) -> None:
    st.markdown("Unggah CSV berisi kolom `review`. Jika ada kolom `sentiment` (positive/negative), "
                "aplikasi juga menghitung akurasinya.")
    up = st.file_uploader("File CSV ulasan", type="csv", key="batch_csv")
    sample_path = find_file("sample_ulasan.csv", "sample_imdb.csv")
    use_sample = False
    if up is None and sample_path is not None:
        use_sample = st.checkbox("Pakai contoh ulasan dari data uji", value=False)
    if up is None and not use_sample:
        st.info("Belum ada file. Maksimal 2.000 baris per unggahan agar tetap cepat.")
        return
    data = pd.read_csv(up) if up is not None else pd.read_csv(sample_path)
    tcol = next((c for c in data.columns if c.lower() in ("review", "text", "ulasan")), None)
    if tcol is None:
        obj = [c for c in data.columns if data[c].dtype == object]
        tcol = obj[0] if obj else None
    if tcol is None:
        st.error("Tidak ada kolom teks. Beri nama kolom ulasan `review`.")
        return
    data = data.head(2000).copy()
    if st.button("Baca semua ulasan", type="primary"):
        X = np.stack([pad_pre(to_ids(clean_text(t).split(), tok), L) for t in data[tcol].astype(str)])
        with st.spinner(f"Membaca {len(X)} ulasan..."):
            probs, ms = run_model(str(chosen["path"]), X, "keras")
        data["prob_positif"] = probs.round(4)
        data["prediksi"] = np.where(probs >= 0.5, "positive", "negative")
        lcol = next((c for c in data.columns if c.lower() in ("sentiment", "label")), None)
        facts = [("ulasan dibaca", f"{len(data)}"), ("positif", f"{(probs >= .5).mean() * 100:.0f}%"),
                 ("waktu per ulasan", f"{ms:.1f} ms")]
        if lcol:
            truth = data[lcol].astype(str).str.lower().map({"positive": "positive", "1": "positive",
                                                           "negative": "negative", "0": "negative"})
            acc = (truth == data["prediksi"]).mean()
            reading("Akurasi pada file ini", f"{acc * 100:.1f}%", "Dibandingkan dengan kolom label di file.", COLD, facts)
        else:
            reading("Ulasan positif", f"{(probs >= .5).mean() * 100:.0f}%", "Dari seluruh ulasan di file.", thermal_color((probs >= .5).mean(), 0, 1), facts)
        st.write("")
        view = data.copy()
        view[tcol] = view[tcol].astype(str).map(lambda t: t if len(t) <= 140 else t[:140] + "...")
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"prob_positif": st.column_config.ProgressColumn(
                         "Keyakinan positif", min_value=0.0, max_value=1.0, format="%.2f")})
        st.download_button("Unduh hasil (CSV)", data.to_csv(index=False).encode(), "hasil_sentimen.csv", "text/csv")


# ---------------------------------------------------------------------------
# Halaman: Tentang model dan versi
# ---------------------------------------------------------------------------
def file_size(p: Path) -> str:
    kb = p.stat().st_size / 1024
    return f"{kb / 1024:.2f} MB" if kb > 1024 else f"{kb:.0f} KB"


def page_about() -> None:
    st.title("Tentang model dan versi")
    models = find_models()

    st.subheader("Model yang dipakai")
    st.markdown('<p class="quiet">Semua angka di bawah berasal dari evaluasi test set di notebook Week 3 '
                '(LSTM vs GRU). Aplikasi ini tidak melatih ulang model; yang dimuat adalah file .h5 hasil training di Colab.</p>',
                unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    with cols[0]:
        st.markdown("#### Prakiraan suhu (Jena Climate)")
        for d in models["jena"]:
            mae, rmse = JENA_METRICS.get((d["cell"], d["cfg"], d["seq"]), (float("nan"),) * 2)
            n = PARAMS.get(("jena", d["cell"], d["cfg"]), 0)
            st.markdown(f"**{model_label(d)}** <span class='pill'>{d['path'].name}</span>", unsafe_allow_html=True)
            st.markdown(f"MAE {mae:.4f} °C, RMSE {rmse:.4f} °C, {n:,} parameter.".replace(f"{n:,}", f"{n:,}".replace(",", ".")))
        st.markdown("Input 72 jam × 13 fitur yang dinormalisasi dengan StandardScaler dari data training. "
                    "Output suhu satu jam berikutnya. Data dibagi kronologis 70/15/15.")
    with cols[1]:
        st.markdown("#### Sentimen ulasan (IMDB 50K)")
        for d in models["imdb"]:
            acc, pr, rc, f1 = IMDB_METRICS.get((d["cell"], d["cfg"], d["seq"]), (float("nan"),) * 4)
            n = PARAMS.get(("imdb", d["cell"], d["cfg"]), 0)
            st.markdown(f"**{model_label(d)}** <span class='pill'>{d['path'].name}</span>", unsafe_allow_html=True)
            st.markdown(f"Akurasi {acc:.4f}, precision {pr:.4f}, recall {rc:.4f}, F1 {f1:.4f}, {n:,} parameter.".replace(f"{n:,}", f"{n:,}".replace(",", ".")))
        st.markdown("Embedding 64 dimensi dengan kosakata 20.000 kata, lalu satu layer recurrent dan Dense sigmoid. "
                    "Ulasan dipotong atau diberi padding di depan agar bagian akhir ulasan tetap terbaca.")

    st.divider()
    st.subheader("Optimasi")
    st.markdown(
        "Tiga optimasi dipakai supaya aplikasi tetap ringan di Streamlit Cloud: "
        "model dan data dimuat sekali lalu disimpan di cache (`st.cache_resource` dan `st.cache_data`), "
        "TensorFlow baru diimpor saat halaman model dibuka, dan setiap model bisa dijalankan sebagai "
        "TFLite dengan kuantisasi dinamis (bobot float32 menjadi int8).")
    if st.button("Ukur ukuran dan kecepatan model"):
        rows = []
        prog = st.progress(0.0)
        allm = models["jena"] + models["imdb"]
        for n, d in enumerate(allm):
            model = load_keras(str(d["path"]))
            shape = (1,) + tuple(int(s) for s in model.input_shape[1:])
            x = (np.random.randint(1, 20000, shape).astype(np.int32) if len(shape) == 2
                 else np.random.randn(*shape).astype(np.float32))
            xb = np.repeat(x, 20, axis=0)
            run_model(str(d["path"]), x, "keras")
            _, ms_k = run_model(str(d["path"]), xb, "keras")
            _, ms_k1 = run_model(str(d["path"]), x, "keras")
            try:
                tfl = build_tflite(str(d["path"]))
                _, ms_t = _run_tflite(str(d["path"]), xb)
                size_t = f"{len(tfl) / 1024 / 1024:.2f} MB" if len(tfl) > 1048576 else f"{len(tfl) / 1024:.0f} KB"
                ms_t = round(ms_t, 2)
            except Exception:
                size_t, ms_t = "tidak didukung", None
            rows.append({"Model": d["path"].name, "Ukuran .h5": file_size(d["path"]), "Ukuran TFLite": size_t,
                         "Keras, 1 tebakan (ms)": round(ms_k1, 2), "TFLite, 1 tebakan (ms)": ms_t})
            prog.progress((n + 1) / len(allm))
        prog.empty()
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Ukuran .h5 termasuk state optimizer dari training. Kecepatan diukur di server saat ini dan bisa berbeda tiap kali.")

    st.divider()
    st.subheader("Riwayat versi")
    vp = find_file("versioning.csv")
    if vp is not None:
        try:
            st.dataframe(pd.read_csv(vp), hide_index=True, width="stretch")
        except Exception:
            st.caption("versioning.csv ada tetapi tidak bisa dibaca.")
    else:
        st.markdown(
            "| Versi | Isi |\n|---|---|\n"
            "| v1 | Satu model per aplikasi, input dasar, tampilan bawaan Streamlit. |\n"
            "| v2 | Desain baru, dua model dalam satu aplikasi, uji rentang waktu, perbandingan model, "
            "sorotan kata berpengaruh, unggah CSV, dan mode TFLite terkuantisasi. |")
    st.caption(f"Runtun {APP_VERSION}. Akmal Nugraha Saputra, 2609, MBC LAS 2026.")


# ---------------------------------------------------------------------------
# Navigasi
# ---------------------------------------------------------------------------
PAGES: dict = {}


def run(only: str | None = None) -> None:
    """only=None -> aplikasi lengkap; only='suhu' / 'sentimen' -> satu model saja (untuk link terpisah)."""
    st.set_page_config(page_title="Runtun: suhu dan ulasan", page_icon=":material/stacked_line_chart:",
                       layout="wide", initial_sidebar_state="collapsed")
    try:
        alt.theme.register("runtun", enable=True)(altair_theme)
    except Exception:
        alt.themes.register("runtun", altair_theme)
        alt.themes.enable("runtun")
    inject_css()

    PAGES["home"] = st.Page(page_home, title="Beranda", icon=":material/home:", url_path="beranda", default=only is None)
    PAGES["suhu"] = st.Page(page_forecast, title="Prakiraan suhu", icon=":material/thermostat:",
                            url_path="prakiraan-suhu", default=only == "suhu")
    PAGES["sentimen"] = st.Page(page_sentiment, title="Pembaca ulasan", icon=":material/movie:",
                                url_path="pembaca-ulasan", default=only == "sentimen")
    PAGES["about"] = st.Page(page_about, title="Tentang model", icon=":material/info:", url_path="tentang")

    if only == "suhu":
        pages = [PAGES["suhu"], PAGES["about"]]
    elif only == "sentimen":
        pages = [PAGES["sentimen"], PAGES["about"]]
    else:
        pages = [PAGES["home"], PAGES["suhu"], PAGES["sentimen"], PAGES["about"]]
    st.navigation(pages, position="top").run()


if __name__ == "__main__":
    run()
