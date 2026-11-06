"""Paso 1: filtrado de la senal continua de cada sesion, con lo elegido en el analisis previo.

Pasabanda FIR de 1 a 100 Hz y notch FIR de 48 a 52 Hz, sobre la senal continua (antes de cortar en epocas).
La lectura de las sesiones y de los pulsos es la misma del analisis previo (analisis.py).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from procesamiento import analisis, config

CANALES = ["eeg", "eog"]     # el VEO se filtra igual que el EEG
# Los cortes estan en config.py: PASA_ALTOS_HZ, PASA_BAJOS_HZ, NOTCH_HZ, NOTCH_ANCHO_HZ


def filtrar(raw):
    """Pasabanda 1-100 Hz y notch 48-52 Hz, FIR de fase cero, sobre una copia de la sesion continua."""
    filt = raw.copy().filter(config.PASA_ALTOS_HZ, config.PASA_BAJOS_HZ, picks=CANALES,
                             method="fir", fir_design="firwin", verbose="ERROR")
    filt.notch_filter(config.NOTCH_HZ, notch_widths=config.NOTCH_ANCHO_HZ, picks=CANALES,
                      method="fir", fir_design="firwin", verbose="ERROR")
    return filt


def fig_antes_despues(raw, filt, titulo):
    """PSD media entre canales EEG antes y despues del filtro, para comprobar que hizo lo que se queria."""
    freqs, antes = analisis.psd_media(raw)
    _, despues = analisis.psd_media(filt)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    m = freqs >= 0.5
    ax.axvspan(config.PASA_ALTOS_HZ, config.PASA_BAJOS_HZ, color="#e5f4e4", alpha=0.6, zorder=0)
    for f in (config.PASA_ALTOS_HZ, config.NOTCH_HZ, config.PASA_BAJOS_HZ):
        ax.axvline(f, color="k", ls="--", lw=0.8)
    ax.semilogx(freqs[m], 10 * np.log10(antes[m]), color="#9a9993", lw=1.2, label="crudo")
    ax.semilogx(freqs[m], 10 * np.log10(despues[m]), color="#2a78d6", lw=1.4, label="filtrado")
    ax.set_xlim(0.3, freqs.max())
    ax.grid(True, which="both", color="0.9", lw=0.6)
    ax.set_xlabel("frecuencia (Hz)")
    ax.set_ylabel("densidad espectral (dB re 1 uV²/Hz)")
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def guardar_verificacion(sub, sesion=0, carpeta=None):
    """Filtra una sesion del sujeto y guarda la PSD antes y despues en derived_mine/filtrado/.

    sesion: posicion en la lista de sesiones (0 = la primera). Devuelve la ruta de la figura.
    """
    vhdr = analisis.listar_sesiones(sub)[sesion]
    ses = vhdr.parent.parent.name                       # .../sub-004/ses-1/eeg/x.vhdr -> "ses-1"
    raw = analisis.cargar_sesion(vhdr)
    fig = fig_antes_despues(raw, filtrar(raw), f"{sub} {ses}: PSD antes y despues del filtro "
                                               f"({config.PASA_ALTOS_HZ:g}-{config.PASA_BAJOS_HZ:g} Hz, notch "
                                               f"{config.NOTCH_HZ - config.NOTCH_ANCHO_HZ / 2:g}-"
                                               f"{config.NOTCH_HZ + config.NOTCH_ANCHO_HZ / 2:g} Hz)")
    destino = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "filtrado"
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{sub}_{ses}_psd_filtro.png"
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return ruta


def leer_y_filtrar(vhdr):
    """Paso 1 completo para una sesion: lee el crudo, encuentra los pulsos y filtra la senal continua.

    Devuelve (sesion filtrada, pulsos [[muestra, 0, nivel]], nombre de la sesion, sesion cruda).
    La cruda se devuelve para poder armar la metadata o comparar antes y despues; quien llama la libera.
    """
    raw = analisis.cargar_sesion(vhdr)
    ev = analisis.pulsos_laser(raw)
    return filtrar(raw), ev, vhdr.parent.parent.name, raw
