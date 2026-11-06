"""Tests de pipeline.py con un sujeto sintetico escrito en disco (formato BrainVision).

Correr desde la carpeta que contiene `procesamiento/` y `tests/`:
    python -m pytest tests/test_pipeline.py -q
Necesita pybv (pip install -e ".[dev]").
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mne")
pybv = pytest.importorskip("pybv")

from extras.anterior import config, pipeline  # noqa: E402

FS = 1000
CANALES = ["Fp1", "Fp2", "Fz", "FCz", "Cz", "CPz", "Pz", "C3", "C4", "O1", "TP9", "TP10", "VEO", "HEOR"]
CODIGOS = {1: 32, 2: 33, 3: 34, 4: 35}


def escribir_sujeto(raiz, sub="sub-001", semilla=0):
    """6 sesiones, 80 pulsos (20 por nivel). En Cz hay un LEP cuya amplitud crece con el nivel."""
    rng = np.random.default_rng(semilla)
    niveles = rng.permutation(np.repeat([1, 2, 3, 4], 20))
    for n, grupo in enumerate(np.array_split(niveles, 6), start=1):
        dur = 6 * len(grupo) + 10
        t = np.arange(dur * FS) / FS
        x = rng.normal(0, 3e-6, (len(CANALES), len(t)))
        ann = []
        for i, nivel in enumerate(grupo):
            m = int((5 + 6 * i) * FS)
            ann += [[m, CODIGOS[nivel]], [m + 3, config.CODIGO_CONFIRMACION]]
            tt = t - m / FS
            x[CANALES.index("Cz")] += nivel * 4e-6 * (-np.exp(-((tt - 0.22) / 0.03) ** 2)
                                                      + 1.5 * np.exp(-((tt - 0.38) / 0.05) ** 2))
        d = raiz / sub / f"ses-{n}" / "eeg"
        d.mkdir(parents=True)
        pybv.write_brainvision(data=x, sfreq=FS, ch_names=CANALES, ref_ch_names=None,
                               fname_base=f"{sub}_ses-{n}_task-{config.TAREA}_eeg",
                               folder_out=d, events=np.array(ann), unit="µV")


@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    base = tmp_path_factory.mktemp("pipe")
    datos_dir = base / "data" / "ds005293"
    escribir_sujeto(datos_dir)
    return datos_dir, base / "salida"


@pytest.fixture
def datos_sinteticos(entorno, monkeypatch):
    datos_dir, salida = entorno
    monkeypatch.setattr(config, "RAIZ_DATOS", datos_dir)
    return salida


def test_un_sujeto_de_punta_a_punta(datos_sinteticos):
    salida = datos_sinteticos
    fila = pipeline.procesar_sujeto("sub-001", carpeta=salida)
    assert fila["segmentadas"] == 80 and fila["completo"]
    assert fila["limpias"] == 80 and fila["rechazadas"] == 0
    assert [fila[f"{c}_limpias"] for c in config.NIVELES] == [20, 20, 20, 20]
    n2p2 = [fila[f"N2P2_{c}_uV"] for c in config.NIVELES]
    assert n2p2 == sorted(n2p2) and n2p2[0] < n2p2[-1]            # crece con el nivel, como lo inyectamos
    base = salida / "sub-001"
    for nombre in ("sub-001_laser-epo.fif", "sub-001_laser-metadata.csv", "sub-001_limpias-epo.fif",
                   "sub-001_rechazo.csv", "sub-001_rasgos.csv", "qc", "qc_limpias", "limpieza"):
        assert (base / nombre).exists(), nombre
    # rasgos: una fila por ensayo limpio; el N2-P2 medio crece con el nivel
    r = pd.read_csv(base / "sub-001_rasgos.csv")
    assert len(r) == 80 and fila["ensayos_con_rasgos"] == 80
    medias = r.groupby("nivel")["N2P2_uV"].mean()
    assert medias.is_monotonic_increasing and medias.iloc[-1] > medias.iloc[0]


def test_segunda_corrida_reusa_las_epocas(datos_sinteticos, capsys):
    pipeline.procesar_sujeto("sub-001", carpeta=datos_sinteticos, con_qc=False)
    assert "ya existen" in capsys.readouterr().out


def test_lote_sigue_si_un_sujeto_falla_y_guarda_el_resumen(datos_sinteticos):
    filas = pipeline.correr(["sub-001", "sub-099"], carpeta=datos_sinteticos, con_qc=False)
    assert [f["estado"] == "ok" for f in filas] == [True, False]
    assert "FileNotFoundError" in filas[1]["estado"]
    csv = pd.read_csv(datos_sinteticos / "resumen_pipeline.csv")
    assert csv["sujeto"].tolist() == ["sub-001", "sub-099"]
    pipeline.correr(["sub-001"], carpeta=datos_sinteticos, con_qc=False)   # reemplaza, no duplica
    assert pd.read_csv(datos_sinteticos / "resumen_pipeline.csv")["sujeto"].tolist() == ["sub-001", "sub-099"]


def test_filtro_iir_sale_en_su_subcarpeta(datos_sinteticos):
    fila = pipeline.procesar_sujeto("sub-001", filtro="iir", carpeta=datos_sinteticos, con_qc=False)
    assert fila["filtro"] == "iir" and fila["segmentadas"] == 80
    assert (datos_sinteticos / "iir" / "sub-001" / "sub-001_limpias-epo.fif").exists()


def test_main_junta_los_rasgos_de_todos_los_sujetos(datos_sinteticos, monkeypatch):
    monkeypatch.setattr(config, "CARPETA_SALIDA", datos_sinteticos)
    assert pipeline.main(["sub-001", "--sin-qc"]) == 0
    todos = pd.read_csv(datos_sinteticos / "rasgos_todos.csv")
    assert len(todos) == 80 and todos["sujeto"].unique().tolist() == ["sub-001"]


def test_main_devuelve_error_si_falla(datos_sinteticos, monkeypatch):
    monkeypatch.setattr(config, "CARPETA_SALIDA", datos_sinteticos)
    assert pipeline.main(["sub-099", "--sin-qc"]) == 1
