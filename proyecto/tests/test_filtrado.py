"""Tests del paso 1 (filtrado.py)."""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from procesamiento import config, filtrado  # noqa: E402


def test_regla_de_pulsos_S3x_seguido_de_S100():
    ev = np.array([[100, 0, 32], [102, 0, 100],      # pulso E1
                   [500, 0, 35], [503, 0, 100],      # pulso E4
                   [900, 0, 33], [905, 0, 7],        # S33 sin confirmacion: no es laser
                   [950, 0, 100],                    # S100 suelto
                   [1200, 0, 34], [1201, 0, 100]])   # pulso E3
    pulsos, retardos = filtrado.pulsos_laser(ev)
    assert pulsos[:, 0].tolist() == [100, 500, 1200]
    assert pulsos[:, 2].tolist() == [1, 4, 3]
    assert retardos.tolist() == [2, 3, 1]


def _raw(red_uV=20.0, deriva_uV=0.0):
    fs = 500
    t = np.arange(60 * fs) / fs
    rng = np.random.default_rng(0)
    x = rng.normal(0, 4e-6, (3, len(t))) + red_uV * 1e-6 * np.sin(2 * np.pi * 50 * t)
    x += deriva_uV * 1e-6 * np.sin(2 * np.pi * 0.1 * t)
    raw = mne.io.RawArray(x, mne.create_info(["Fz", "Cz", "VEO"], fs, ["eeg", "eeg", "eog"]), verbose="ERROR")
    return raw


def test_el_filtro_saca_la_red_y_la_deriva():
    raw = _raw(red_uV=20, deriva_uV=200)
    f, antes = filtrado.psd_mediana(raw)
    _, despues = filtrado.psd_mediana(filtrado.filtrar(raw, copiar=True))
    r = filtrado.resumen_psd(f, antes, despues)
    assert r["pico_50Hz_dB_antes"] > 20 and r["pico_50Hz_dB_despues"] < 3
    # el pasa-altos de 1 Hz tiene su banda de transicion entre 0 y 1 Hz: atenua la deriva, no la borra
    assert r["bajo_1Hz_pct_antes"] > 50 and r["bajo_1Hz_pct_despues"] < r["bajo_1Hz_pct_antes"] / 4


def test_el_filtro_no_toca_la_banda_del_LEP():
    raw = _raw(red_uV=0)
    t = raw.times
    raw._data[1] += 10e-6 * np.sin(2 * np.pi * 5 * t)               # 5 Hz, dentro de la banda
    filt = filtrado.filtrar(raw, copiar=True)
    medio = slice(5 * 500, -5 * 500)
    ref = np.sin(2 * np.pi * 5 * t[medio])
    amp = lambda r: 2 * np.mean(r.get_data(picks="Cz")[0, medio] * ref)     # amplitud del componente de 5 Hz
    assert amp(filt) == pytest.approx(amp(raw), rel=0.02)
    assert config.PASA_ALTOS_HZ == 1.0 and config.PASA_BAJOS_HZ == 100.0
