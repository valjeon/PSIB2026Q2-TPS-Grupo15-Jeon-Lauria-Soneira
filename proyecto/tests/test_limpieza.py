"""Pipeline (extraccion + limpieza) de punta a punta con sujetos sinteticos en BrainVision.

    python -m pytest tests/test_limpieza.py -q        (necesita pybv: pip install pybv)

Los sujetos sinteticos (tests/sinteticos.py) tienen: LEP en la ROI que crece con el nivel, 20 uV de red a 50 Hz,
O2 muy ruidoso, 9 parpadeos reflejos de 300 uV en el VEO a +0.3 s, parpadeos espontaneos de 200 uV antes de uno
de cada tres pulsos y un salto de 500 uV en Pz en el ensayo 40.
"""
import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("pybv")

from procesamiento import config, epocas, limpieza, pipeline  # noqa: E402
from sinteticos import GROSERO, PARPADEOS, escribir_csv_autores, escribir_sujeto  # noqa: E402


@pytest.fixture(scope="module")
def salida(tmp_path_factory):
    base = tmp_path_factory.mktemp("repo")
    datos = base / "data" / "ds005293"
    for k, s in enumerate(["sub-001", "sub-002"]):
        escribir_sujeto(datos, s, semilla=k)
    csv = base / "all experiments_preprocessing.csv"
    escribir_csv_autores(csv, {1: ("3 10", "T7 Tp7"), 2: ("\\", "\\")})
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "RAIZ_DATOS", datos)
    mp.setattr(config, "CARPETA_SALIDA", base / "data" / "derived_mine")
    mp.setattr(limpieza, "RUTA_CSV", csv)
    pipeline.main(["sub-001", "sub-002", "sub-099"])
    assert pipeline.sujetos_descargados() == ["sub-001", "sub-002"]
    yield base / "data" / "derived_mine"
    mp.undo()


def test_marcas_de_los_autores(salida):
    r = pd.read_csv(salida / "limpieza_resumen.csv").set_index("sujeto")
    assert r.loc["sub-001", "por_autores"] == 2 and r.loc["sub-001", "electrodos_autores"] == "T7"
    assert r.loc["sub-002", "por_autores"] == 0
    rech = pd.read_csv(salida / "sub-001" / "sub-001_rechazo.csv")
    assert not {2, 9} & set(rech["ensayo"])                         # ensayos 3 y 10 del CSV (desde 1)


def test_canal_ruidoso_fuera_de_la_referencia_y_roi_intacta(salida):
    r = pd.read_csv(salida / "limpieza_resumen.csv").set_index("sujeto")
    assert (r["malos_propios"] == "O2").all()
    assert r["roi_interpolados"].isna().all()                       # ningun canal de la ROI estaba malo
    ep = mne.read_epochs(limpieza.ruta_limpias("sub-001", salida), verbose="ERROR")
    assert set(ep.info["bads"]) == {"T7", "O2"}


def test_rechazo_por_motivo(salida):
    rech = pd.read_csv(salida / "sub-002" / "sub-002_rechazo.csv")
    motivo = dict(zip(rech["ensayo"], rech["motivo"].fillna("")))
    assert motivo[GROSERO] == "grosera"
    assert all(motivo[i] == "VEO" for i in PARPADEOS)                # los 9 parpadeos reflejos
    ep = mne.read_epochs(limpieza.ruta_limpias("sub-002", salida), verbose="ERROR")
    assert len(ep) == (rech["motivo"].fillna("") == "").sum()


def test_inclusion_y_resumenes(salida):
    r = pd.read_csv(salida / "limpieza_resumen.csv").set_index("sujeto")
    assert list(r.index) == ["sub-001", "sub-002"]                  # sub-099 no tiene datos: error, no frena
    # los parpadeos espontaneos (1 de cada 3 pulsos) caen en el tramo del VEO: se pierde ~45%
    assert not r["incluido"].any() and r["motivo_exclusion"].str.contains("perdidas").all()
    assert (pd.read_csv(salida / "epocas_resumen.csv")["epocas"] == 80).all()
    for s in ("sub-001", "sub-002"):
        assert (salida / s / f"{s}_limpieza.png").stat().st_size > 10_000


def test_regla_de_inclusion():
    ep = mne.EpochsArray(np.zeros((40, 2, 10)), mne.create_info(["Cz", "VEO"], 250, ["eeg", "eog"]),
                         verbose="ERROR")
    ep.metadata = pd.DataFrame({"condicion": ["E1"] * 10 + ["E2"] * 10 + ["E3"] * 10 + ["E4"] * 10})
    ok = {"roi_comprometida": False, "roi_interpolados": []}
    assert limpieza.incluir(60, 2, ok, ep, 80) == ["50% de epocas perdidas"]
    assert any("E4: 9" in m for m in limpieza.incluir(60, 2, ok, ep[:39], 80))
    assert any("ROI" in m for m in limpieza.incluir(60, 2, {"roi_comprometida": True, "roi_interpolados": ["Cz"]},
                                                     ep, 80))
    assert any("canales malos" in m for m in limpieza.incluir(60, 20, ok, ep, 80))
