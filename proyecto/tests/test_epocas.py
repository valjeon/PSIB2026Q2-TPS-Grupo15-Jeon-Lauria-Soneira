"""Pasos 1 y 2 de punta a punta con sujetos sinteticos en BrainVision (tests/sinteticos.py).

    python -m pytest tests/test_epocas.py -q        (necesita pybv: pip install -e ".[dev]")
"""
import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("pybv")

from procesamiento import config, epocas, pipeline  # noqa: E402
from sinteticos import escribir_sujeto  # noqa: E402


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    base = tmp_path_factory.mktemp("repo")
    datos = base / "data" / "ds005293"
    niveles = {s: escribir_sujeto(datos, s, semilla=k) for k, s in enumerate(["sub-001", "sub-002"])}
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "RAIZ_DATOS", datos)
    salida = base / "data" / "derived_mine"
    errores = pipeline.correr(["sub-001", "sub-002", "sub-099"], carpeta=salida)
    yield salida, errores, niveles
    mp.undo()


def test_ventana_y_linea_de_base_del_paper(corrida):
    salida, _, _ = corrida
    ep = mne.read_epochs(epocas.ruta_epocas("sub-001", salida), verbose="ERROR")
    assert ep.tmin == pytest.approx(-1.0) and ep.tmax == pytest.approx(2.0, abs=0.01)
    assert ep.info["sfreq"] == config.SFREQ_GUARDADO
    pre = ep.copy().crop(-1.0, 0.0).get_data(picks="eeg").mean(axis=2)
    assert np.abs(pre).max() < 1e-7                                          # linea de base -1 a 0 s
    assert not {"TP9", "TP10", "HEOR"} & set(ep.ch_names)
    assert ep.get_channel_types(picks=["VEO"]) == ["eog"]


def test_80_epocas_en_orden_y_con_su_nivel(corrida):
    salida, _, niveles = corrida
    ep = mne.read_epochs(epocas.ruta_epocas("sub-001", salida), verbose="ERROR")
    assert len(ep) == 80
    assert ep.metadata["nivel"].tolist() == niveles["sub-001"].tolist()       # mismo orden que los pulsos
    assert ep.metadata["ensayo"].tolist() == list(range(80))
    assert ep.metadata["sesion"].nunique() == 6
    assert (ep.metadata["retardo_ms"] == 4.0).all()                          # S3x -> S100: 2 muestras a 500 Hz


def test_el_lep_queda_alineado_y_crece_con_el_nivel(corrida):
    salida, _, _ = corrida
    ep = mne.read_epochs(epocas.ruta_epocas("sub-001", salida), verbose="ERROR")
    t = ep.times
    n2p2 = []
    for c in config.NIVELES:
        x = ep[c].average(picks="Cz").get_data()[0] * 1e6
        assert abs(t[np.argmin(np.where((t > 0.1) & (t < 0.3), x, np.inf))] - 0.22) < 0.02
        n2p2.append(x[(t > 0.3) & (t < 0.5)].max() - x[(t > 0.1) & (t < 0.3)].min())
    assert n2p2 == sorted(n2p2)


def test_resumen_figuras_y_sujeto_faltante(corrida):
    salida, errores, _ = corrida
    assert list(errores) == ["sub-099"] and "FileNotFoundError" in errores["sub-099"]
    r = pd.read_csv(salida / "epocas_resumen.csv").set_index("sujeto")
    assert list(r.index) == ["sub-001", "sub-002"]
    assert r["completo"].all() and (r[list(config.NIVELES)] == 20).all().all()
    assert (r["pico_50Hz_dB_antes"] > 10).all() and (r["pico_50Hz_dB_despues"] < 3).all()
    for s in ("sub-001", "sub-002"):
        for f in ("psd_filtrado.png", "promedios.png", "laser-metadata.csv"):
            assert (salida / s / f"{s}_{f}").exists()


def test_segunda_corrida_saltea_los_ya_extraidos(corrida, capsys):
    salida, _, _ = corrida
    assert pipeline.correr(["sub-001"], carpeta=salida) == {}
    assert "se saltea" in capsys.readouterr().out
