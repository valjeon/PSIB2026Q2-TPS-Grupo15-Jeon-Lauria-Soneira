"""Filtrado de la senal continua de cada sesion (antes de cortar en épocas).
Pasabanda FIR de fase cero de 1-100 Hz y notch FIR de 48-52 Hz.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from procesamiento import analisis, config

CANALES = ["eeg", "eog"]


def filtrar(raw):
    """Pasabanda 1-100 Hz y notch 48-52 Hz (FIR de fase cero) sobre una copia de la señal continua."""
    filt = raw.copy().filter(config.PASA_ALTOS_HZ, config.PASA_BAJOS_HZ, picks=CANALES,
                             method="fir", fir_design="firwin", verbose="ERROR")    # MNE aplica el FIR de fase lineal sin desfase
    filt.notch_filter(config.NOTCH_HZ, notch_widths=config.NOTCH_ANCHO_HZ, picks=CANALES,
                      method="fir", fir_design="firwin", verbose="ERROR")
    return filt # se filtra solo la copia, no el crudo


def fig_antes_despues(raw, filt, titulo):
    """PSD media de los canales EEG antes y despues del filtro."""
    freqs, antes = analisis.psd_media(raw)      # PSD del crudo (promedio entre canales)
    _, despues = analisis.psd_media(filt)       # PSD del filtrado
    fig, ax = plt.subplots(figsize=(8, 4.6))
    desde_05 = freqs >= 0.5         # se omite < 0,5 Hz porque en escala logarítmica no se lee

    ax.axvspan(config.PASA_ALTOS_HZ, config.PASA_BAJOS_HZ, color="#e5f4e4", alpha=0.6, zorder=0)    # banda de paso
    for f in (config.PASA_ALTOS_HZ, config.NOTCH_HZ, config.PASA_BAJOS_HZ):
        ax.axvline(f, color="k", ls="--", lw=0.8)   # líneas en 1, 50 y 100 Hz

    ax.semilogx(freqs[desde_05], 10 * np.log10(antes[desde_05]), color="#9a9993", lw=1.2, label="crudo")    # dB
    ax.semilogx(freqs[desde_05], 10 * np.log10(despues[desde_05]), color="#2a78d6", lw=1.4, label="filtrado")
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
    """Filtra una sesion del sujeto y guarda su PSD de antes y despues en derived_mine/filtrado/. Devuelve la ruta de la figura."""
    vhdr = analisis.listar_sesiones(sub)[sesion]    # sesion
    ses = vhdr.parent.parent.name
    raw = analisis.cargar_sesion(vhdr)

    ini_notch = config.NOTCH_HZ - config.NOTCH_ANCHO_HZ / 2          # 48 Hz
    fin_notch = config.NOTCH_HZ + config.NOTCH_ANCHO_HZ / 2          # 52 Hz
    titulo = (f"{sub} {ses}: PSD antes y despues del filtro "
              f"({config.PASA_ALTOS_HZ:g}-{config.PASA_BAJOS_HZ:g} Hz, notch {ini_notch:g}-{fin_notch:g} Hz)")
    fig = fig_antes_despues(raw, filtrar(raw), titulo)

    destino = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "filtrado"
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{sub}_{ses}_psd_filtro.png"
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return ruta


def leer_y_filtrar(vhdr):
    """Para cada sesion: lee el crudo, encuentra los pulsos y filtra la señal continua.
    Devuelve sesion filtrada, pulsos, nombre de la sesion, sesion cruda.¿"""
    raw = analisis.cargar_sesion(vhdr)      # crudo
    ev = analisis.pulsos_laser(raw)         # pulsos
    # devuelve también el crudo en caso de que se quiera comparar o armar la metadata, se puede elegir si devolverlo o no
    return filtrar(raw), ev, vhdr.parent.parent.name, raw
