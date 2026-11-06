"""Pipeline de punta a punta con sujetos sinteticos en BrainVision (tests/sinteticos.py).

    python -m pytest tests -q        (necesita pybv y scikit-learn)

Los sujetos sinteticos tienen: LEP en la ROI que crece con el nivel, 20 uV de red a 50 Hz, O2 muy ruidoso,
9 parpadeos reflejos de 300 uV en el VEO a +0.3 s y un salto de 500 uV en Pz en el ensayo 40. Sin parpadeos
espontaneos, para que queden incluidos y la etapa de grupo tenga con que trabajar.
"""
import pandas as pd
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("pybv")
pytest.importorskip("sklearn")

from procesamiento import config, limpieza, pipeline  # noqa: E402
from sinteticos import GROSERO, PARPADEOS, escribir_csv_autores, escribir_sujeto  # noqa: E402

SUJETOS = [f"sub-00{k}" for k in range(1, 7)]


@pytest.fixture(scope="module")
def salida(tmp_path_factory):
    base = tmp_path_factory.mktemp("repo")
    datos = base / "data" / "ds005293"
    for k, s in enumerate(SUJETOS):
        escribir_sujeto(datos, s, semilla=k, amplitud_uV=3 + k, espontaneos=False)
    csv = base / "all experiments_preprocessing.csv"
    escribir_csv_autores(csv, {1: ("3 10", "T7 Tp7")})
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "RAIZ_DATOS", datos)
    mp.setattr(config, "CARPETA_SALIDA", base / "data" / "derived_mine")
    mp.setattr(limpieza, "RUTA_CSV", csv)
    pipeline.main(SUJETOS + ["sub-099"])
    assert pipeline.sujetos_disponibles() == SUJETOS
    yield base / "data" / "derived_mine"
    mp.undo()


def test_limpieza_por_sujeto(salida):
    r = pd.read_csv(salida / "preprocesado_resumen.csv").set_index("sujeto")
    assert list(r.index) == SUJETOS                                  # sub-099 sin datos: error, no frena
    assert r.loc["sub-001", "por_autores"] == 2 and r.loc["sub-001", "interpolados_autores"] == "T7"
    assert (r["interpolados_propios"] == "O2").all()                 # el canal ruidoso
    assert (r["groseras"] == 1).all()                                # el salto en Pz
    ep = mne.read_epochs(limpieza.ruta_preprocesadas("sub-002", salida), verbose="ERROR")
    assert GROSERO not in set(ep.metadata["ensayo"])
    assert set(PARPADEOS) <= set(ep.metadata["ensayo"])              # los parpadeos los saca medicion, no limpieza


def test_rechazo_final_e_inclusion(salida):
    previa = pd.read_csv(salida / "inclusion_previa.csv")
    final = pd.read_csv(salida / "inclusion_final.csv")
    assert previa["incluido"].all() and final["incluido"].all()
    ens = pd.read_csv(salida / "medicion" / "rasgos_ensayos.csv")
    principal = ens[ens["sujeto"] == "sub-002"].set_index("ensayo")["rechazada_veo150"]
    assert principal.loc[sorted(PARPADEOS)].all()                    # los 9 parpadeos reflejos, por VEO
    sinveo = ens.set_index(["sujeto", "ensayo"])["rechazada_sinveo"]
    assert sinveo.sum() < ens["rechazada_veo150"].sum()              # sin VEO se rechaza menos


def test_lep_crece_con_la_intensidad(salida):
    v = pd.read_csv(salida / "localizador" / "ventanas.csv").set_index("componente")
    assert 0.19 < v.loc["N2", "latencia_s"] < 0.25 and 0.35 < v.loc["P2", "latencia_s"] < 0.41
    med = pd.read_csv(salida / "ensambles" / "medianas_niveles.csv").set_index("medida")
    assert med.loc["N2P2_uV", ["mediana_E1", "mediana_E2", "mediana_E3", "mediana_E4"]].is_monotonic_increasing
    res = pd.read_csv(salida / "estadistica" / "veo150" / "resumen.csv")
    assert len(res) > 0
    for variante in ("veo150", "veo200", "sinveo"):
        assert (salida / "estadistica" / variante / "resultados.png").exists()


def test_solo_grupo(salida):
    pipeline.main(["--grupo"])


def test_sin_crudo_se_reusan_las_epocas(salida):
    """Como lote.py: el crudo se borro y quedaron solo las epocas extraidas."""
    import shutil
    shutil.rmtree(config.RAIZ_DATOS / "sub-006")
    assert "sub-006" in pipeline.sujetos_disponibles()
    pipeline.main(["sub-006", "--sin-grupo"])                      # no necesita el crudo
    r = pd.read_csv(salida / "preprocesado_resumen.csv").set_index("sujeto")
    assert r.loc["sub-006", "segmentadas"] == 80
