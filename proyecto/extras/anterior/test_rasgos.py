"""Tests de rasgos.py con epocas sinteticas de rasgos conocidos.

Correr desde la carpeta que contiene `procesamiento/` y `tests/`:
    python -m pytest tests/test_rasgos.py -q
"""
import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")

from extras.anterior import rasgos  # noqa: E402

SFREQ = 1000.0
CANALES = ["Fz", "Cz", "Pz", "C3", "C4", "VEO"]
N_EP = 40


def epocas_sinteticas(alfa_post_uv=0.0, lep=True, semilla=0):
    """Epocas de -1 a 2 s con ruido de 1 uV. En Cz: N2 de -10 uV a 200 ms y P2 de +15 uV a 380 ms
    (si lep) y un seno de 10 Hz solo en 0-1 s (si alfa_post_uv > 0)."""
    rng = np.random.default_rng(semilla)
    t = np.arange(int(3 * SFREQ)) / SFREQ - 1.0
    x = rng.normal(0, 1e-6, (N_EP, len(CANALES), len(t)))
    cz = CANALES.index("Cz")
    if lep:
        x[:, cz] += (-10e-6 * np.exp(-((t - 0.2) / 0.03) ** 2) + 15e-6 * np.exp(-((t - 0.38) / 0.05) ** 2))
    if alfa_post_uv:
        x[:, cz] += alfa_post_uv * 1e-6 * np.sin(2 * np.pi * 10 * t) * ((t >= 0) & (t < 1))
    info = mne.create_info(CANALES, SFREQ, ["eeg"] * 5 + ["eog"])
    niveles = np.tile([1, 2, 3, 4], N_EP // 4)
    meta = pd.DataFrame({"sujeto": "sub-x", "sesion": "ses-1", "nivel": niveles,
                         "condicion": [f"E{k}" for k in niveles], "ensayo": np.arange(N_EP)})
    return mne.EpochsArray(x, info, tmin=-1.0, verbose="ERROR",
                           events=np.c_[np.arange(N_EP) * 5000, np.zeros(N_EP, int), niveles],
                           event_id={"E1": 1, "E2": 2, "E3": 3, "E4": 4}, metadata=meta)


def test_n2p2_recupera_amplitudes_y_latencias():
    r = rasgos.n2p2_por_ensayo(epocas_sinteticas())
    assert len(r) == N_EP
    assert r["N2_uV"].mean() == pytest.approx(-10, abs=2.5)
    assert r["P2_uV"].mean() == pytest.approx(15, abs=2.5)
    assert r["N2P2_uV"].mean() == pytest.approx(25, abs=4)
    assert r["N2_lat_ms"].between(160, 240).all()
    assert r["P2_lat_ms"].between(330, 430).all()


def test_potencia_detecta_el_alfa_solo_en_la_ventana_post():
    r = rasgos.rasgos_de_potencia(epocas_sinteticas(alfa_post_uv=20.0, lep=False))
    assert r["Cz_alfa_cambio_dB"].mean() > 20                  # 20 uV de seno contra 1 uV de ruido
    assert abs(r["Cz_delta_cambio_dB"].mean()) < 2             # sin cambio donde no se puso nada
    assert abs(r["Cz_gamma_cambio_dB"].mean()) < 2
    assert abs(r["Fz_alfa_cambio_dB"].mean()) < 2              # el otro canal no cambia


def test_tabla_tiene_metadata_y_columnas():
    t = rasgos.tabla_rasgos(epocas_sinteticas())
    assert len(t) == N_EP
    for col in ("sujeto", "condicion", "N2P2_uV", "Cz_alfa_post_dB", "Fz_gamma_cambio_dB"):
        assert col in t.columns
    assert "VEO" not in " ".join(t.columns)                    # el canal de ojo no es un rasgo
    assert t.isna().sum().sum() == 0


def test_sin_metadata_da_error_claro():
    ep = epocas_sinteticas()
    ep.metadata = None
    with pytest.raises(ValueError, match="metadata"):
        rasgos.tabla_rasgos(ep)


def test_reunir_junta_los_sujetos(tmp_path):
    for s in ("sub-001", "sub-002"):
        (tmp_path / s).mkdir()
        pd.DataFrame({"sujeto": [s] * 3, "N2P2_uV": [1.0, 2.0, 3.0]}).to_csv(tmp_path / s / f"{s}_rasgos.csv", index=False)
    todos = rasgos.reunir_rasgos(tmp_path)
    assert len(todos) == 6 and set(todos["sujeto"]) == {"sub-001", "sub-002"}
    assert (tmp_path / "rasgos_todos.csv").exists()
    assert rasgos.reunir_rasgos(tmp_path / "vacia") is None
