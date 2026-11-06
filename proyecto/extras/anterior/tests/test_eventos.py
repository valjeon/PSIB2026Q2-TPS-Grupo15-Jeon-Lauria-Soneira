import numpy as np
import pytest

from extras.anterior import eventos


def ev(*codigos):
    return np.array([[100 * (i + 1), 0, c] for i, c in enumerate(codigos)])


def test_pulso_laser_requiere_s100_inmediato():
    e = ev(32, 100, 33, 100, 34, 100, 35, 100)
    out = eventos.eventos_laser(e)
    assert out[:, 2].tolist() == [1, 2, 3, 4]
    assert out[:, 0].tolist() == [100, 300, 500, 700]  # muestra del S3x, no del S100


def test_ignora_otras_modalidades_y_s100_solo():
    e = ev(8, 100, 64, 100, 128, 100, 32, 12, 100)  # ningun S3x queda pegado a S100
    assert len(eventos.eventos_laser(e)) == 0


def test_sin_eventos_devuelve_forma_correcta():
    assert eventos.eventos_laser(np.empty((0, 3))).shape == (0, 3)


def test_contar_y_validar():
    e = ev(*([32, 100] * 20 + [33, 100] * 20 + [34, 100] * 20 + [35, 100] * 20))
    c = eventos.contar_por_nivel(eventos.eventos_laser(e))
    assert c == {"E1": 20, "E2": 20, "E3": 20, "E4": 20}
    assert eventos.validar_pulsos(c)
    c["E1"] = 14
    with pytest.warns(UserWarning):
        assert not eventos.validar_pulsos(c, sujeto="sub-999")
