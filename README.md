# SequenceLab — Deployment LSTM & GRU (Week 5)

Akmal Nugraha Saputra · CaAs 2609 · NIM 103052500014

Dua model terbaik dari Tugas Week 3 dideploy ke Streamlit:

| Model | Tugas | File | Skor uji |
|---|---|---|---|
| LSTM (1 layer 64 unit, seq 72) | Prediksi suhu 1 jam ke depan, Jena Climate | `models/LSTM_A_seq72.h5` | MAE 0,4772 °C |
| GRU (1 layer 64 unit, seq 400) | Sentimen ulasan film, IMDB 50K | `models/GRU_A_seq400.h5` | Accuracy 0,8935 |

## Langkah deploy (urut)

1. **Ekspor artefak dari Colab.** Buka notebook Week 3 (semua sel sudah dijalankan), tempel isi
   `export_artefak_colab.py` ke sel baru, jalankan. Hasilnya di Drive: folder `deploy_artefak/`
   berisi 2 file `.h5`, `scaler.json`, `tokenizer.json`, `sample_jena.csv`.
2. **Buat repo GitHub** dengan struktur:
   ```
   app.py
   requirements.txt
   .streamlit/config.toml
   models/  (isi dari deploy_artefak/)
   screenshot/
   ```
3. **Deploy v1 dulu** untuk dokumentasi versioning: di repo, ganti `app.py` dengan isi `app_v1.py`
   (atau buat branch `v1`), deploy di share.streamlit.io, screenshot, catat link.
4. **Deploy v2** (file `app.py` di folder ini), screenshot, catat link.
5. Isi tabel versioning (`versioning.csv` sudah disiapkan, tinggal ganti kolom Link).

Di Streamlit Cloud pilih Python 3.11 atau 3.12 di Advanced settings. Model `.h5` ukurannya
kecil (< 6 MB), jadi aman langsung di-commit tanpa Git LFS.

## Menjalankan lokal

```
pip install -r requirements.txt
streamlit run app.py
```

## Fitur v2 (nilai bonus)
- `st.cache_resource`: model dan tokenizer hanya dimuat sekali per server, klik berikutnya instan.
- Tokenisasi ditulis ulang dengan numpy murni (tanpa `keras.preprocessing`), lebih ringan dan
  tidak tergantung versi Keras.
- Data contoh bawaan + slider jendela waktu, jadi pengguna bisa langsung membandingkan prediksi
  dengan nilai aktual tanpa mengunggah apa pun.
