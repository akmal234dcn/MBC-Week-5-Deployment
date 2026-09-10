# Runtun: deploy LSTM/GRU (MBC LAS Week 5)

Akmal Nugraha Saputra (2609). Deploy dua model terbaik dari Week 3:

- Prakiraan suhu Jena Climate: `LSTM_A_seq72.h5` (MAE 0,4772 °C)
- Sentimen ulasan IMDB: `GRU_A_seq400.h5` (akurasi 89,35%, F1 0,8949)

## Struktur

```
app.py                  aplikasi final (v2): beranda, prakiraan suhu, pembaca ulasan, tentang model
.streamlit/config.toml  tema warna dan font
requirements.txt
models/                 LSTM_A_seq72.h5, GRU_A_seq400.h5, scaler.json, tokenizer.json,
                        sample_jena.csv, (opsional) sample_ulasan.csv, model pembanding lain
jena/v1/app.py          versi 1 prakiraan suhu
jena/v2/app.py          versi 2 prakiraan suhu (halaman suhu dari app.py)
imdb/v1/app.py          versi 1 sentimen
imdb/v2/app.py          versi 2 sentimen (halaman ulasan dari app.py)
colab_export_artefak.py cadangan untuk membuat ulang scaler/tokenizer/sampel di Colab
versioning.csv          tabel versioning (juga tampil di halaman Tentang model)
```

## Menjalankan lokal

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy di Streamlit Community Cloud

Buat app baru dari repo ini, pilih Python 3.12, lalu isi *Main file path* sesuai link yang ingin dibuat:
`jena/v1/app.py`, `imdb/v1/app.py`, `jena/v2/app.py`, `imdb/v2/app.py`, atau `app.py` (gabungan).

## Optimasi

- Model dan data dimuat sekali lalu disimpan di cache (`st.cache_resource`, `st.cache_data`).
- TensorFlow baru diimpor saat halaman model dibuka.
- Mode "TFLite ringan": model .h5 dikonversi otomatis ke TFLite dengan dynamic-range quantization
  (bobot int8), sekitar 4x lebih kecil dan jauh lebih cepat untuk satu prediksi.
