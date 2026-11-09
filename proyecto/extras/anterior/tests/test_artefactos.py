"""Tests de artefactos.py (sin ICA) con datos sinteticos.

Correr desde proyecto/:   python -m pytest tests/test_artefactos.py -q
"""
import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")

from extras.anterior import artefactos  # noqa: E402

SFREQ = 250.0
CANALES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC1", "FC2", "C3", "Cz", "C4",
           "CP1", "CP2", "P3", "Pz", "P4", "O1", "O2", "VEO"]
N_EP = 40
PESO = {"VEO": 1.0, "Fp1": 0.8, "Fp2": 0.8, "F7": 0.4, "F8": 0.4, "F3": 0.35, "Fz": 0.35, "F4": 0.35,
        "FC1": 0.15, "FC2": 0.15}


def epocas_sinteticas(semilla=0):
    """Epocas de -1 a 2 s con parpadeos en lugares conocidos:
        0-9    parpadeo a +0.3 s  (dentro de la ventana critica -> rechazo por VEO)
        10-19  parpadeo a +1.5 s  (fuera de la ventana -> se conserva)
        20-29  parpadeo a -0.5 s  (antes de la ventana -> se conserva)
        30     salto enorme en Pz (rechazo por EEG)
        31-39  limpias
    """
    rng = np.random.default_rng(semilla)
    t = np.arange(int(3 * SFREQ)) / SFREQ - 1.0
    mezcla = np.array([PESO.get(c, 0.03) for c in CANALES])
    x = rng.normal(0, 3e-6, (N_EP, len(CANALES), len(t)))
    for i in range(N_EP):
        t0 = 0.3 if i < 10 else 1.5 if i < 20 else -0.5 if i < 30 else None
        if t0 is not None:
            x[i] += mezcla[:, None] * 150e-6 * np.exp(-((t - t0) / 0.08) ** 2)
    x[30, CANALES.index("Pz"), 500:520] += 400e-6
    info = mne.create_info(CANALES, SFREQ, ["eeg"] * (len(CANALES) - 1) + ["eog"])
    niveles = np.tile([1, 2, 3, 4], N_EP // 4)
    ep = mne.EpochsArray(x, info, tmin=-1.0,
                         events=np.c_[np.arange(N_EP) * 1000, np.zeros(N_EP, int), niveles],
                         event_id={"E1": 1, "E2": 2, "E3": 3, "E4": 4},
                         metadata=pd.DataFrame({"condicion": [f"E{k}" for k in niveles]}), verbose="ERROR")
    ep.set_montage("standard_1005", verbose="ERROR")
    return artefactos.con_referencia_promedio(ep)


def test_rechazo_por_veo_solo_en_la_ventana_y_por_eeg():
    ep = epocas_sinteticas()
    limpio, tabla, medidas = artefactos.rechazar(ep, umbral_veo_uV=100, ventana_veo=(-0.2, 0.8),
                                                 umbral_eeg_uV=200)
    assert set(np.flatnonzero(medidas["por_veo"])) == set(range(10))      # solo parpadeos en la ventana
    assert set(np.flatnonzero(medidas["por_eeg"])) == {30}
    assert len(limpio) == N_EP - 11
    assert tabla["rechazadas"].sum() == 11
    assert len(limpio.metadata) == len(limpio)


def test_parpadeo_fuera_de_la_ventana_no_se_rechaza_por_eeg():
    """Un parpadeo grande a +1.5 s (fuera de la ventana de VEO) supera 100 uV en Fp1,
    pero no debe descartar la epoca: los frontopolares no cuentan para el criterio EEG."""
    ep = epocas_sinteticas()
    limpio, tabla, medidas = artefactos.rechazar(ep, umbral_veo_uV=100, ventana_veo=(-0.2, 0.8),
                                                 umbral_eeg_uV=60)
    fp1 = artefactos.pico_a_pico_uV(ep, ["Fp1"])
    assert (fp1[10:20] > 60).all()                         # en Fp1 el parpadeo supera el umbral...
    assert not medidas["por_eeg"].iloc[10:20].any()        # ...pero no cuenta para el criterio EEG