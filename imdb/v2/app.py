"""Runtun v2 - hanya halaman sentimen (dipakai untuk link deploy terpisah).
Main file path di Streamlit Cloud: imdb/v2/app.py"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("runtun_app", ROOT / "app.py")
runtun = importlib.util.module_from_spec(spec)
sys.modules["runtun_app"] = runtun
spec.loader.exec_module(runtun)

runtun.run(only="sentimen")
