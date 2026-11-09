"""Tests de diagnostico.py con epocas sinteticas."""
import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")

from extras.anterior import diagnostico  # noqa: E402

FS = 250
CANALES = ["Fp1", "Fz", "Cz", "Pz", "C3", "T8", "VEO"]


def epocas():
    rng = np.random.default_rng(0)
    t = np.arange(int(-1 * FS), int(2 * FS)) / FS
    x = rng.normal(0, 2e-6, (6, len(CANALES), len(t)))
    # 0: LEP enorme en Cz entre 0.2 y 0.4 s; 1: salto tardio (1.5 s) en T8; 2: salto previo (-0.5 s) en T8
    x[0, CANALES.index("Cz")] += 100e-6 * (-np.exp(-((t - 0.22) / 0.03) ** 2) + np.exp(-((t - 0.38) / 0.04) ** 2))
    x[1, CANALES.index("T8")] += 300e-6 * np.exp(-((t - 1.5) / 0.1) ** 2)
    x[2, CANALES.index("T8")] += 300e-6 * np.exp(-((t + 0.5) / 0.1) ** 2)
    x[3, CANALES.index("Fp1")] += 150e-6 * np.exp(-((t - 0.3) / 0.1) ** 2)       # frontal: no cuenta
    info = mne.create_info(CANALES, FS, ["eeg"] * 6 + ["eog"])
    ep = mne.EpochsArray(x, info, tmin=-1.0, baseline=(-1, 0), verbose="ERROR")
    ep.metadata = pd.DataFrame({"condicion": ["E4", "E4", "E3", "E1", "E1", "E2"]})
    return ep


def test_clasifica_lep_tarde_y_pre():
    t = diagnostico.epocas_rechazadas_por_eeg(epocas(), umbral_uV=60).set_index("epoca")
    assert t.loc[0, "tipo"] == "LEP" and t.loc[0, "canal"] in diagnostico.CANALES_CENTRALES
    assert t.loc[1, "tipo"] == "tarde" and t.loc[1, "canal"] == "T8"
    assert t.loc[2, "tipo"] == "pre"


def test_frontales_no_cuentan():
    t = diagnostico.epocas_rechazadas_por_eeg(epocas(), umbral_uV=60)
    assert 3 not in set(t["epoca"])


def test_resumen_por_nivel():
    t = diagnostico.epocas_rechazadas_por_eeg(epocas(), umbral_uV=60)
    r = diagnostico.resumen_por_nivel(t)
    assert r.loc["E4", "LEP"] == 1 and r.loc["E4", "tarde"] == 1 and r.loc["E3", "pre"] == 1
    assert list(r.index) == ["E1", "E2", "E3", "E4"]
