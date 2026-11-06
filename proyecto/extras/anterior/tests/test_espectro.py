import mne
import numpy as np

from extras.anterior import espectro


def raw_sintetico(red_uv=20.0, deriva_uv=0.0, fs=1000.0, dur=60.0):
    rng = np.random.default_rng(1)
    t = np.arange(int(dur * fs)) / fs
    x = rng.normal(0, 5e-6, (4, len(t)))
    x += red_uv * 1e-6 * np.sin(2 * np.pi * 50 * t)
    x += deriva_uv * 1e-6 * np.sin(2 * np.pi * 0.2 * t)
    info = mne.create_info(["Fz", "Cz", "Pz", "Fp1"], fs, "eeg")
    return mne.io.RawArray(x, info, verbose="ERROR")


def test_detecta_ruido_de_red():
    con = espectro.diagnostico(raw_sintetico(red_uv=20))
    sin = espectro.diagnostico(raw_sintetico(red_uv=0))
    assert con["red_50hz_db"] > 15
    assert abs(sin["red_50hz_db"]) < 5


def test_detecta_deriva_lenta():
    con = espectro.diagnostico(raw_sintetico(red_uv=0, deriva_uv=100))
    sin = espectro.diagnostico(raw_sintetico(red_uv=0, deriva_uv=0))
    assert con["deriva_pct_menor_1hz"] > 50 > sin["deriva_pct_menor_1hz"]


def test_bandas_suman_cerca_de_100():
    d = espectro.diagnostico(raw_sintetico(red_uv=0))
    assert 80 < sum(d["bandas_pct_1_100hz"].values()) < 120


def test_grafico_se_guarda(tmp_path):
    ruta = tmp_path / "e.png"
    espectro.graficar({"ses-1": raw_sintetico()}, ["Cz", "Fz"], ruta)
    assert ruta.stat().st_size > 5000
