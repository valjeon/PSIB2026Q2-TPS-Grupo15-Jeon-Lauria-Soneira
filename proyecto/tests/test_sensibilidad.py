"""Tests de sensibilidad.py con epocas sinteticas en memoria.

    python -m pytest tests/test_sensibilidad.py -q
"""
import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")

from procesamiento import config, sensibilidad  # noqa: E402

FS = 250
CANALES = ["Fp1", "Fz", "Cz", "Pz", "C3", "T8", "VEO"]


def hacer_epocas(semilla=0):
    rng = np.random.default_rng(semilla)
    niveles = np.repeat([1, 2, 3, 4], 20)
    n, t = len(niveles), np.arange(int(-1 * FS), int(2 * FS)) / FS
    x = rng.normal(0, 3e-6, (n, len(CANALES), len(t)))
    cz = CANALES.index("Cz")
    for i, nv in enumerate(niveles):
        x[i, cz] += nv * 4e-6 * (-np.exp(-((t - 0.22) / 0.03) ** 2) + 1.5 * np.exp(-((t - 0.38) / 0.05) ** 2))
    # epoca 0: parpadeo grande en VEO dentro de la ventana; epoca 1: canal T8 ruidoso
    x[0, CANALES.index("VEO")] += 400e-6 * np.exp(-((t - 0.3) / 0.05) ** 2)
    x[1, CANALES.index("T8")] += 400e-6 * np.exp(-((t - 0.3) / 0.05) ** 2)
    x[2, CANALES.index("Fp1")] += 400e-6 * np.exp(-((t - 0.3) / 0.05) ** 2)   # frontal ruidoso
    info = mne.create_info(CANALES, FS, ["eeg"] * 6 + ["eog"])
    ep = mne.EpochsArray(x, info, tmin=-1.0, baseline=(-1, 0), verbose="ERROR")
    ep.metadata = pd.DataFrame({"condicion": [f"E{v}" for v in niveles]})
    return ep


@pytest.fixture(scope="module")
def ep():
    return hacer_epocas()


def test_variantes_distinguen_veo_y_canal_malo(ep):
    t, _ = sensibilidad.comparar_sujeto(ep, "sub-x", malas_autores={0, 5}, electrodos_malos=["T8"])
    por = t.set_index("variante")
    assert por.loc["actual", "rechazadas"] == 2          # epoca 0 (VEO) y epoca 1 (T8); Fp1 se ignora
    assert por.loc["todos_canales", "rechazadas"] == 3   # ademas la epoca 2 (Fp1)
    assert por.loc["sin_veo", "rechazadas"] == 1         # solo la del canal T8
    assert por.loc["sin_malos", "rechazadas"] == 1       # solo la del VEO: ignora T8
    assert por.loc["actual", "coinciden"] == 1           # epoca 0 coincide con los autores


def test_sin_csv_se_omiten_variantes_con_malos(ep):
    t, _ = sensibilidad.comparar_sujeto(ep, "sub-x")
    assert "sin_malos" not in set(t["variante"])
    assert "coinciden" not in t.columns


def test_gradiente_positivo(ep):
    t, _ = sensibilidad.comparar_sujeto(ep, "sub-x")
    fila = t.set_index("variante").loc["actual"]
    assert fila["gradiente_E4_E1"] > 0
    assert fila["N2P2_E4"] > fila["N2P2_E1"]


def test_nivel_con_pocas_epocas_da_nan(ep):
    m = sensibilidad.medidas_por_epoca(ep)
    mantener = (ep.metadata["condicion"] != "E4").to_numpy() | (np.arange(len(ep)) < 2)
    res = sensibilidad.n2p2_por_nivel(m["v"], mantener)
    assert res["E4"][0] < sensibilidad.MIN_EPOCAS_NIVEL
    assert np.isnan(res["E4"][1])


def test_regresion_veo_quita_el_parpadeo_sin_descartar_la_epoca(ep):
    t, m = sensibilidad.comparar_sujeto(ep, "sub-x")
    por = t.set_index("variante")
    assert por.loc["regresion_veo", "rechazadas"] <= por.loc["actual", "rechazadas"]
    assert por.loc["regresion_veo", "gradiente_E4_E1"] > 0
    assert "corregido" in m


def test_coeficiente_recupera_la_propagacion_conocida():
    rng = np.random.default_rng(1)
    base = hacer_epocas(3)
    v = base.copy()
    d = v.get_data()
    veo = rng.normal(0, 50e-6, (len(v), len(v.times)))
    d[:, CANALES.index("VEO")] = veo
    d[:, CANALES.index("Cz")] += 0.3 * veo           # el VEO se propaga a Cz con coeficiente 0.3
    v._data = d
    beta, nombres = sensibilidad.coeficientes_veo(v)
    assert abs(dict(zip(nombres, beta))["Cz"] - 0.3) < 0.05
    corr, _ = sensibilidad.corregir_con_veo(v)
    dc = corr.get_data(picks="Cz")[:, 0, :]
    assert np.corrcoef(dc.ravel(), veo.ravel())[0, 1] < 0.1     # ya no queda parpadeo en Cz


def test_ventana_ignora_artefactos_fuera_de_la_ventana_analizada():
    """Un salto a 1.5 s en un canal no frontal descarta la epoca con 'actual' pero no con 'ventana'."""
    e = hacer_epocas()
    d = e.get_data()
    t = e.times
    d[5, CANALES.index("T8")] += 400e-6 * np.exp(-((t - 1.5) / 0.05) ** 2)
    e2 = mne.EpochsArray(d, e.info, tmin=-1.0, baseline=None, verbose="ERROR")
    e2.metadata = e.metadata
    por = sensibilidad.comparar_sujeto(e2, "sub-x")[0].set_index("variante")
    assert por.loc["ventana", "rechazadas"] == por.loc["actual", "rechazadas"] - 1
    assert por.loc["regresion_veo_ventana", "rechazadas"] <= por.loc["regresion_veo", "rechazadas"]


def test_veo_residual_no_marca_la_respuesta_ligada_al_estimulo():
    """Una onda de 140 uV pico a pico en el VEO, igual en todas las epocas, no es un parpadeo."""
    e = hacer_epocas()
    d = e.get_data()
    t = e.times
    d[:, CANALES.index("VEO")] += 70e-6 * np.sin(2 * np.pi * 3 * t) * ((t > 0.1) & (t < 0.6))
    e2 = mne.EpochsArray(d, e.info, tmin=-1.0, baseline=None, verbose="ERROR")
    e2.metadata = e.metadata
    por = sensibilidad.comparar_sujeto(e2, "sub-x")[0].set_index("variante")
    assert por.loc["ventana", "rechazadas"] > 60                          # casi todas por 'VEO'
    assert por.loc["ventana_veo_residual", "rechazadas"] <= 3             # quedan las de verdad


def test_lejanos_ignora_canales_cerca_de_los_ojos():
    e = hacer_epocas()
    d = e.get_data()
    t = e.times
    d[7, CANALES.index("Fz")] += 400e-6 * np.exp(-((t - 0.3) / 0.05) ** 2)
    e2 = mne.EpochsArray(d, e.info, tmin=-1.0, baseline=None, verbose="ERROR")
    e2.metadata = e.metadata
    por = sensibilidad.comparar_sujeto(e2, "sub-x")[0].set_index("variante")
    assert por.loc["ventana", "rechazadas"] == por.loc["ventana_lejanos", "rechazadas"] + 1
    assert "F7" in sensibilidad.canales_cerca_de_los_ojos(["F7", "Fz", "Cz", "Pz", "T8"])
    assert "Cz" not in sensibilidad.canales_cerca_de_los_ojos(["F7", "Fz", "Cz", "Pz", "T8"])


def test_parpadeo_fuera_del_tramo_n2p2_no_descarta_con_veo_n2p2():
    """Un parpadeo en el VEO a +0.7 s (dentro de la ventana de VEO pero despues de N2/P2) descarta la epoca
    con 'ventana' pero no con 'veo_n2p2'."""
    e = hacer_epocas()
    d = e.get_data()
    t = e.times
    d[9, CANALES.index("VEO")] += 400e-6 * np.exp(-((t - 0.7) / 0.05) ** 2)
    e2 = mne.EpochsArray(d, e.info, tmin=-1.0, baseline=None, verbose="ERROR")
    e2.metadata = e.metadata
    por = sensibilidad.comparar_sujeto(e2, "sub-x")[0].set_index("variante")
    assert por.loc["veo_n2p2", "rechazadas"] <= por.loc["ventana", "rechazadas"] - 1


def test_solo_cz_mira_unicamente_cz():
    """Un salto en T8 descarta la epoca con 'ventana' pero no con 'solo_cz'; uno en Cz descarta en las dos."""
    e = hacer_epocas()
    d = e.get_data()
    t = e.times
    d[11, CANALES.index("T8")] += 400e-6 * np.exp(-((t - 0.3) / 0.05) ** 2)
    d[12, CANALES.index("Cz")] += 400e-6 * np.exp(-((t - 0.3) / 0.05) ** 2)
    e2 = mne.EpochsArray(d, e.info, tmin=-1.0, baseline=None, verbose="ERROR")
    e2.metadata = e.metadata
    por = sensibilidad.comparar_sujeto(e2, "sub-x")[0].set_index("variante")
    assert por.loc["solo_cz", "rechazadas"] < por.loc["ventana", "rechazadas"]
    assert por.loc["solo_cz", "rechazadas"] >= 1                           # la de Cz sigue descartada