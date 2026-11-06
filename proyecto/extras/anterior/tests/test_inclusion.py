"""Tests de la regla de inclusion de sujetos (pipeline.evaluar_inclusion)."""
import pandas as pd

from extras.anterior import pipeline2


def tabla(e1, e2, e3, e4):
    return pd.DataFrame({"condicion": ["E1", "E2", "E3", "E4"], "antes": 20,
                         "por_VEO": 0, "por_EEG": 0, "despues": [e1, e2, e3, e4],
                         "rechazadas": 20 - pd.Series([e1, e2, e3, e4])})


def test_incluye_si_cumple_en_E1_y_E4():
    assert pipeline2.evaluar_inclusion(tabla(13, 10, 6, 10)) == (True, "")   # E2 y E3 no cuentan


def test_excluye_si_E4_tiene_pocas():
    ok, motivo = pipeline2.evaluar_inclusion(tabla(13, 10, 6, 4))
    assert not ok and "E4: 4" in motivo and "E1" not in motivo


def test_excluye_si_falta_un_nivel():
    ok, motivo = pipeline2.evaluar_inclusion(pd.DataFrame({"condicion": ["E1"], "despues": [20]}))
    assert not ok and "E4: 0" in motivo


def test_fila_de_resumen_trae_la_decision():
    f = pipeline2.armar_fila("sub-006", "fir", tabla(13, 10, 6, 4), {}, 1, 1, 2)
    assert f["incluido"] is False and f["motivo_exclusion"]
