"""Rasgos por ensayo que se agregan a los de medicion.py (N2, P2, N2-P2): latencia de N2 y rasgos espectrales.

Lee y actualiza derived_mine/medicion/rasgos_ensayos.csv (una fila por epoca de los sujetos medidos), agregando:
    lat_N2_ms     instante del minimo de la senal de la ROI (30 Hz) dentro de la ventana de N2
    gamma_ERS_dB  cambio de potencia 60-95 Hz en la ROI, 200-500 ms, respecto del pre-estimulo (exploratorio)
    beta_ERS_dB   cambio de potencia 14-30 Hz en la ROI, 100-500 ms (exploratorio)
Bandas, canales y ventanas como Zhuang et al. 2026 (NeuroImage), que usa estos mismos datasets.

Espectrales: sobre la senal de 1-100 Hz (sin el pasa-bajos de 30 Hz, que borraria gamma), referencia sin
frontales y linea de base. STFT por ensayo (ventana Hanning de 250 ms, paso de 20 ms); potencia de cada
celda tiempo-frecuencia; cambio = 10*log10(potencia en la ventana / potencia en la linea de base), en dB
(> 0: ERS, aumento; < 0: ERD, caida). Linea de base espectral: -400 a -100 ms.

La latencia de P2 no se mide: P2 es ancha y poco marcada, y en un ensayo individual su maximo cae casi en
cualquier punto de la ventana (refleja el ruido, no la respuesta).

Lo llama pipeline.py en la etapa de grupo (python -m procesamiento.pipeline --grupo).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from procesamiento import config, limpieza, localizador

CANALES_ROI = getattr(config, "CANALES_ROI", ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"])
VENTANA_STFT_S = 0.25
PASO_STFT_S = 0.02
BASE_ESPECTRAL = (-0.4, -0.1)
BANDAS = {   # nombre: (canales, (f_min, f_max) Hz, (t_min, t_max) s)
    "gamma_ERS_dB": (CANALES_ROI, (60, 95), (0.2, 0.5)),
}
COLUMNAS_NUEVAS = ["lat_N2_ms"] + list(BANDAS)


def latencia_n2(v, ventana_n2):
    """Latencia (ms) del minimo de la senal de la ROI en la ventana de N2, por epoca. `v`: epocas a 30 Hz."""
    t = v.times
    m = (t >= ventana_n2[0]) & (t <= ventana_n2[1])
    senal = v.get_data(picks=[c for c in CANALES_ROI if c in v.ch_names]).mean(axis=1)
    return t[m][np.argmin(senal[:, m], axis=1)] * 1000


def cambio_potencia(x, sfreq, tmin, banda, ventana, base=BASE_ESPECTRAL):
    """Cambio de potencia (dB) por epoca. x: epocas x tiempo (senal de un canal o de la ROI)."""
    nper = int(round(VENTANA_STFT_S * sfreq))
    paso = int(round(PASO_STFT_S * sfreq))
    f, tt, p = spectrogram(x, fs=sfreq, window="hann", nperseg=nper, noverlap=nper - paso,
                           scaling="density", mode="psd", axis=-1)           # epocas x f x t
    tt = tt + tmin                                                          # centros en tiempo de la epoca
    mf = (f >= banda[0]) & (f <= banda[1])
    pot = lambda a, b: p[:, mf][:, :, (tt >= a) & (tt <= b)].mean(axis=(1, 2))
    return 10 * np.log10(pot(*ventana) / pot(*base))


def rasgos_sujeto(sub, ventanas):
    """DataFrame con 'ensayo' y las columnas nuevas, para las epocas de para_analisis del sujeto."""
    v30, _ = limpieza.para_analisis(sub)                       # 30 Hz: para la latencia
    vfull, _ = limpieza.para_analisis(sub, pasa_bajos=False)   # 1-100 Hz: para los espectrales
    d = pd.DataFrame({"sujeto": sub, "ensayo": v30.metadata["ensayo"].to_numpy(),
                      "lat_N2_ms": latencia_n2(v30, ventanas["N2"])})
    sf, tmin = vfull.info["sfreq"], vfull.tmin
    for nombre, (canales, banda, ventana) in BANDAS.items():
        x = vfull.get_data(picks=[c for c in canales if c in vfull.ch_names]).mean(axis=1) * 1e6
        d[nombre] = cambio_potencia(x, sf, tmin, banda, ventana)
    return d.round(3)


def correr(carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    ruta = carpeta / "medicion" / "rasgos_ensayos.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No esta {ruta}. Corre antes: python -m procesamiento.pipeline --grupo")
    base = pd.read_csv(ruta).drop(columns=COLUMNAS_NUEVAS, errors="ignore")
    ventanas = localizador.leer_ventanas(carpeta)
    nuevos = pd.concat([rasgos_sujeto(s, ventanas) for s in base["sujeto"].unique()], ignore_index=True)
    todo = base.merge(nuevos, on=["sujeto", "ensayo"], how="left", validate="one_to_one")
    todo.to_csv(ruta, index=False)
    print(f"Rasgos agregados a {ruta}: {', '.join(COLUMNAS_NUEVAS)}")
    print(todo.groupby("condicion")[["N2P2_uV"] + COLUMNAS_NUEVAS].median().round(2).to_string())
    return todo
