"""Tests de segmentacion.py y qc.py con datos sinteticos (no hace falta bajar nada).

Correr desde la carpeta que contiene `procesamiento/` y `tests/`:
    python -m pytest tests -q
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from extras.anterior import config, qc, segmentacion  # noqa: E402

SFREQ = 1000.0
CANALES = ["Fz", "FCz", "Cz", "CPz", "Pz", "C3", "C4", "VEO"]


def raw_sintetico(ev_laser, dur_s=60.0, semilla=0):
    """Ruido + un 'LEP' en Cz despues de cada pulso: N2 (-) a 220 ms y P2 (+) a 380 ms,
    con amplitud proporcional al nivel."""
    rng = np.random.default_rng(semilla)
    n = int(dur_s * SFREQ)
    x = rng.normal(0, 2e-6, (len(CANALES), n))
    t = np.arange(int(0.8 * SFREQ)) / SFREQ
    cz = CANALES.index("Cz")
    for muestra, _, nivel in ev_laser:
        lep = nivel * 4e-6 * (-np.exp(-((t - 0.22) / 0.03) ** 2) + 1.5 * np.exp(-((t - 0.38) / 0.05) ** 2))
        fin = min(muestra + len(t), n)
        x[cz, muestra:fin] += lep[: fin - muestra]
    info = mne.create_info(CANALES, SFREQ, ["eeg"] * 7 + ["eog"])
    raw = mne.io.RawArray(x, info, verbose="ERROR")
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    return raw


def eventos_ejemplo():
    """8 pulsos (2 por nivel) cada 6 s, desde el segundo 3."""
    niveles = [1, 2, 3, 4, 4, 3, 2, 1]
    return np.array([[3000 + 6000 * i, 0, k] for i, k in enumerate(niveles)])


# --- funciones puras ---------------------------------------------------------

def test_retardos_un_valor_por_pulso():
    ev = np.array([[100, 0, 32], [103, 0, 100], [500, 0, 7], [900, 0, 35], [902, 0, 100], [950, 0, 100]])
    assert segmentacion.retardos_confirmacion(ev).tolist() == [3, 2]


def test_metadata_alineada():
    ev = eventos_ejemplo()
    meta = segmentacion.armar_metadata(ev, SFREQ, "sub-x", "ses-1", retardos=np.full(len(ev), 2))
    assert len(meta) == len(ev)
    assert meta["condicion"].tolist() == ["E1", "E2", "E3", "E4", "E4", "E3", "E2", "E1"]
    assert meta["onset_s"].iloc[1] == pytest.approx(9.0)
    assert (meta["retardo_ms"] == 2.0).all()


def test_metadata_rechaza_retardos_desalineados():
    with pytest.raises(ValueError):
        segmentacion.armar_metadata(eventos_ejemplo(), SFREQ, "s", "ses-1", retardos=[1, 2])


def test_nombre_sesion():
    assert segmentacion.nombre_sesion("data/sub-001/ses-12/eeg/x_eeg.vhdr") == "ses-12"


# --- epocado -----------------------------------------------------------------

def test_epocas_ventana_y_metadata():
    ev = eventos_ejemplo()
    raw = raw_sintetico(ev)
    meta = segmentacion.armar_metadata(ev, SFREQ, "sub-x", "ses-1")
    ep = segmentacion.epocas_sesion(raw, ev, meta)
    assert len(ep) == 8
    assert ep.tmin == pytest.approx(-1.0) and ep.tmax == pytest.approx(2.0)
    assert ep.metadata["nivel"].tolist() == ev[:, 2].tolist()
    assert set(ep.event_id) == set(config.NIVELES)


def test_pulso_en_el_borde_se_va_con_su_fila():
    ev = eventos_ejemplo()
    ev[-1, 0] = int(59.5 * SFREQ)            # sin 2 s de post-estimulo
    raw = raw_sintetico(ev)
    meta = segmentacion.armar_metadata(ev, SFREQ, "sub-x", "ses-1")
    ep = segmentacion.epocas_sesion(raw, ev, meta)
    assert len(ep) == 7
    assert ep.metadata["ensayo_sesion"].tolist() == list(range(7))


def test_concatenar_sesiones_incompletas():
    """Una sesion sin E4: igual se concatena porque el event_id es el mismo."""
    ev1 = eventos_ejemplo()
    ev2 = ev1[ev1[:, 2] != 4]
    eps = [segmentacion.epocas_sesion(raw_sintetico(e, semilla=i), e,
                                      segmentacion.armar_metadata(e, SFREQ, "sub-x", f"ses-{i}"))
           for i, e in enumerate([ev1, ev2], start=1)]
    ep = mne.concatenate_epochs(eps, verbose="ERROR")
    assert len(ep) == len(ev1) + len(ev2)
    assert ep.metadata.groupby("sesion").size().to_dict() == {"ses-1": 8, "ses-2": 6}


# --- qc ----------------------------------------------------------------------

def test_qc_encuentra_el_lep_inyectado(tmp_path):
    ev = eventos_ejemplo()
    raw = raw_sintetico(ev)
    meta = segmentacion.armar_metadata(ev, SFREQ, "sub-x", "ses-1", retardos=np.full(len(ev), 1))
    ep = segmentacion.epocas_sesion(raw, ev, meta)
    ep.metadata = ep.metadata.assign(ensayo=np.arange(len(ep)))

    resumen = qc.correr_qc(ep, "sub-x", carpeta=tmp_path)

    assert (tmp_path / "sub-x" / "qc" / "sub-x_3_erp_Cz_por_nivel.png").exists()
    assert resumen["N2_lat_ms"].between(190, 250).all()
    assert resumen["P2_lat_ms"].between(340, 420).all()
    # amplitud N2-P2 crece con el nivel (asi lo construimos)
    assert resumen.sort_values("condicion")["N2P2_uV"].is_monotonic_increasing


# --- filtros -----------------------------------------------------------------

@pytest.mark.parametrize("metodo", ["fir", "iir"])
def test_filtro_saca_50hz_y_continua_y_deja_10hz(metodo):
    from extras.anterior import filtros
    t = np.arange(int(30 * SFREQ)) / SFREQ
    x = 30e-6 + 10e-6 * np.sin(2 * np.pi * 10 * t) + 10e-6 * np.sin(2 * np.pi * 50 * t)
    info = mne.create_info(["Cz", "VEO"], SFREQ, ["eeg", "eog"])
    raw = mne.io.RawArray(np.vstack([x, x]), info, verbose="ERROR")
    y = filtros.filtrar(raw, metodo=metodo, copiar=True).get_data()[:, 5000:-5000]  # sin bordes
    espectro = np.abs(np.fft.rfft(y, axis=1))
    f = np.fft.rfftfreq(y.shape[1], 1 / SFREQ)
    a10 = espectro[:, np.argmin(abs(f - 10))]
    a50 = espectro[:, np.argmin(abs(f - 50))]
    assert np.all(a50 < 0.05 * a10)              # 50 Hz atenuado > 26 dB respecto de 10 Hz
    assert np.all(abs(y.mean(axis=1)) < 1e-6)    # sin offset de continua
    assert raw.get_data()[0].mean() > 20e-6      # copiar=True no toca el original