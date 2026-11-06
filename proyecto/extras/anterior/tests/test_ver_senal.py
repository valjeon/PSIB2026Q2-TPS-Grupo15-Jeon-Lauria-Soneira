"""Tests de ver_senal.py con epocas sinteticas (parpadeos que llegan mas a los canales frontales)."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
mne = pytest.importorskip("mne")

from extras.anterior import ver_senal  # noqa: E402

FS = 250
CANALES = ["Fp1", "Fp2", "F7", "F8", "Fz", "FC1", "Cz", "T7", "T8", "Pz", "P3", "O1", "VEO"]
PESO = {"Fp1": 0.8, "Fp2": 0.8, "F7": 0.4, "F8": 0.4, "Fz": 0.35, "FC1": 0.15, "Cz": 0.05}


def epocas():
    rng = np.random.default_rng(0)
    t = np.arange(int(-1 * FS), int(2 * FS)) / FS
    niveles = np.tile([1, 2, 3, 4], 10)
    x = rng.normal(0, 3e-6, (40, len(CANALES), len(t)))
    blink = 250e-6 * np.exp(-((t - 0.3) / 0.06) ** 2)
    for i in range(0, 40, 3):                       # 1 de cada 3 epocas tiene parpadeo (en todos los niveles)
        for j, c in enumerate(CANALES):
            x[i, j] += blink * (1.0 if c == "VEO" else PESO.get(c, 0.02))
    info = mne.create_info(CANALES, FS, ["eeg"] * (len(CANALES) - 1) + ["eog"])
    ep = mne.EpochsArray(x, info, tmin=-1.0, baseline=(-1, 0), verbose="ERROR")
    ep.metadata = pd.DataFrame({"condicion": [f"E{k}" for k in niveles]})
    return ep


def test_marca_los_parpadeos():
    v = ver_senal.preparar(epocas())
    con, ptp = ver_senal.marcar_parpadeos(v)
    assert set(np.flatnonzero(con)) == set(range(0, 40, 3))
    assert ptp[con].min() > 100 > ptp[~con].max()


def test_orden_adelante_atras():
    orden = ver_senal.ordenar_adelante_atras(["O1", "Cz", "Fp1", "Pz", "Fz"])
    assert orden[0] == "Fp1" and orden[-1] == "O1"


def _extra(v):
    con, _ = ver_senal.marcar_parpadeos(v)
    ptp, nombres = ver_senal.ptp_por_canal_en_ventana(v)
    return dict(zip(nombres, np.median(ptp[con], axis=0) - np.median(ptp[~con], axis=0)))


def test_con_la_referencia_original_la_influencia_decrece_hacia_atras():
    e = _extra(ver_senal.preparar(epocas(), "original"))
    assert e["Fp1"] > 5 * e["Pz"]


def test_con_referencia_promedio_los_canales_lejanos_reciben_el_parpadeo_invertido():
    """El promedio reparte el parpadeo frontal por todos los canales: Pz ya no queda libre."""
    original, promedio = _extra(ver_senal.preparar(epocas(), "original")), _extra(ver_senal.preparar(epocas()))
    assert promedio["Pz"] > 5 * original["Pz"]


def test_las_tres_figuras_se_generan():
    v = ver_senal.preparar(epocas())
    for fig in (ver_senal.fig_ejemplos(v, "E1", 2), ver_senal.fig_erp(v),
                ver_senal.fig_influencia(v, ver_senal.preparar(epocas(), "original"))):
        assert len(fig.axes) >= 2
        matplotlib.pyplot.close(fig)


def test_ejemplos_toma_marcadas_y_limpias():
    v = ver_senal.preparar(epocas())
    con, ptp = ver_senal.marcar_parpadeos(v)
    idx = np.flatnonzero(v.metadata["condicion"].to_numpy() == "E1")
    marcadas, limpias = ver_senal.elegir_ejemplos(idx, con, ptp, 2)
    assert all(con[i] for i in marcadas) and not any(con[i] for i in limpias)


def test_veo_con_respuesta_ligada_al_estimulo_no_es_parpadeo():
    """Una respuesta de 140 uV pico a pico en el VEO, igual en todas las epocas del nivel, supera el umbral
    pero no es un parpadeo: restandole el promedio del nivel no queda nada que marcar."""
    ep = epocas()
    d = ep.get_data()
    t = ep.times
    d[:, -1, :] += 70e-6 * np.sin(2 * np.pi * 3 * t) * ((t > 0.1) & (t < 0.6))
    e2 = mne.EpochsArray(d, ep.info, tmin=-1.0, baseline=None, verbose="ERROR")
    e2.metadata = ep.metadata
    v = ver_senal.preparar(e2)
    con, _ = ver_senal.marcar_parpadeos(v)
    sin_evocada = ver_senal.veo_ptp_sin_evocada(v) > 100
    assert con.sum() > sin_evocada.sum()
    tabla = ver_senal.tabla_veo(v)
    assert (tabla["ptp_VEO_promedio_uV"] > 100).all()


def test_comparacion_dentro_de_nivel_ignora_niveles_sin_ambos_tipos():
    v = ver_senal.preparar(epocas())
    con, _ = ver_senal.marcar_parpadeos(v)
    ptp, _ = ver_senal.ptp_por_canal_en_ventana(v)
    cond = v.metadata["condicion"].to_numpy()
    solo_e4 = con & (cond == "E4")
    _, _, usados, n_con, n_sin = ver_senal.medianas_dentro_de_nivel(ptp, solo_e4, cond)
    assert usados == ["E4"] and n_con == 4 and n_sin == 6
    with pytest.raises(ValueError):
        ver_senal.medianas_dentro_de_nivel(ptp, np.zeros(len(con), bool), cond)