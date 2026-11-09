import mne
import numpy as np
import pybv
import pytest

from extras.anterior import config, datos, eventos

CANALES = ["Fp1", "Cz", "Pz", "TP9", "TP10", "VEO", "HEOR"]


def escribir_sesion(raiz, sub, n, codigos):
    """Crea un registro BrainVision sintetico con los triggers dados (1 cada segundo)."""
    fs = 1000
    d = raiz / sub / f"ses-{n}" / "eeg"
    d.mkdir(parents=True)
    rng = np.random.default_rng(n)
    sig = rng.normal(0, 10e-6, (len(CANALES), fs * (len(codigos) + 2)))
    ann = np.array([[(i + 1) * fs, c] for i, c in enumerate(codigos)])  # [muestra, codigo]
    pybv.write_brainvision(data=sig * 1e6, sfreq=fs, ch_names=CANALES, ref_ch_names=None,
                           fname_base=f"{sub}_ses-{n}_task-{config.TAREA}_eeg", folder_out=d,
                           events=ann, unit="µV")


@pytest.fixture
def raiz(tmp_path):
    escribir_sesion(tmp_path, "sub-001", 2, [33, 100, 8, 100])
    escribir_sesion(tmp_path, "sub-001", 10, [32, 100, 35, 100])  # ses-10 debe ordenarse despues de ses-2
    return tmp_path


def test_listar(raiz):
    assert datos.listar_sujetos(raiz) == ["sub-001"]
    nombres = [p.parent.parent.name for p in datos.listar_sesiones("sub-001", raiz)]
    assert nombres == ["ses-2", "ses-10"]


def test_sujeto_faltante_da_mensaje_util(raiz):
    with pytest.raises(FileNotFoundError, match="descargar_sujeto"):
        datos.listar_sesiones("sub-077", raiz)


def test_cargar_sesion_saca_canales_y_marca_eog(raiz):
    raw = datos.cargar_sesion(datos.listar_sesiones("sub-001", raiz)[0])
    assert raw.ch_names == ["Fp1", "Cz", "Pz", "VEO"]
    assert raw.get_channel_types()[-1] == "eog"
    assert raw.info["sfreq"] == 1000  # no se remuestrea ni se filtra aca


def test_de_punta_a_punta_con_eventos(raiz):
    raws, nombres = datos.cargar_sujeto("sub-001", raiz)
    assert nombres == ["ses-2", "ses-10"]
    assert eventos.pulsos_laser(raws[0])[:, 2].tolist() == [2]
    assert eventos.pulsos_laser(raws[1])[:, 2].tolist() == [1, 4]


def test_verificar_cli(raiz, monkeypatch, capsys):
    from extras.anterior import verificar
    monkeypatch.setattr(config, "RAIZ_DATOS", raiz)
    assert verificar.main(["sub-001"]) == 1  # solo 3 pulsos: no coincide con 20 por nivel
    assert "NO COINCIDE" in capsys.readouterr().out


def test_config_respeta_variable_de_entorno(tmp_path, monkeypatch):
    monkeypatch.setenv("PSIB_DATOS", str(tmp_path))
    assert config._buscar_datos() == tmp_path
