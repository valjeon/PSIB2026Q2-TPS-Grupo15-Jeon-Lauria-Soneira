"""Filtros: la logica de exploracion/filtrado.py convertida en funciones.

- filtrar(raw): lo que usa el pipeline. Pasabanda 1-100 Hz + notch 50 Hz (48-52),
  FIR firwin, exactamente los parametros de filtrado.py.
- metodo="iir": la alternativa para la comparacion FIR vs IIR del cronograma
  (Butterworth de fase cero).
- fig_espectros / fig_tramo: las figuras de filtrado.py, como funciones.

No lee archivos: recibe un Raw ya cargado (datos.cargar_sesion).
"""
import matplotlib.pyplot as plt
import numpy as np

from . import config

CANALES_FILTRO = ["eeg", "eog"]     # el EOG tambien se filtra: la ICA lo usa como referencia


def filtrar(raw, metodo="fir", copiar=False):
    """Pasabanda + notch sobre EEG y EOG. Modifica raw (o una copia si copiar=True).

    metodo="fir": firwin, fase lineal y cero (MNE compensa el retardo). Es el de filtrado.py.
    metodo="iir": Butterworth de orden config.ORDEN_IIR aplicado ida y vuelta (fase cero).
    """
    raw = raw.copy() if copiar else raw
    if metodo == "fir":
        raw.filter(config.PASA_ALTOS_HZ, config.PASA_BAJOS_HZ, picks=CANALES_FILTRO,
                   method="fir", fir_design="firwin", verbose="ERROR")
        raw.notch_filter(config.NOTCH_HZ, notch_widths=config.NOTCH_ANCHO_HZ, picks=CANALES_FILTRO,
                         method="fir", fir_design="firwin", verbose="ERROR")
    elif metodo == "iir":
        iir = dict(order=config.ORDEN_IIR, ftype="butter", output="sos")
        raw.filter(config.PASA_ALTOS_HZ, config.PASA_BAJOS_HZ, picks=CANALES_FILTRO,
                   method="iir", iir_params=iir, verbose="ERROR")
        raw.notch_filter(config.NOTCH_HZ, notch_widths=config.NOTCH_ANCHO_HZ, picks=CANALES_FILTRO,
                         method="iir", iir_params=iir, verbose="ERROR")
    else:
        raise ValueError(f"metodo debe ser 'fir' o 'iir', no {metodo!r}")
    return raw


def fig_espectros(raw, raw_filt, fmax=120):
    """PSD antes y despues (las dos primeras figuras de filtrado.py), en una sola figura."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, r, titulo in ((axes[0], raw, "Antes del filtrado"), (axes[1], raw_filt, "Despues del filtrado")):
        r.compute_psd(fmax=fmax, picks="eeg", verbose="ERROR").plot(axes=ax, show=False)
        ax.set_title(titulo)
    fig.tight_layout()
    return fig


def fig_tramo(raw, raw_filt, canales=("Fz", "Cz", "Pz"), inicio=100, fin=110):
    """Un tramo de senal sin filtrar vs filtrada (la tercera figura de filtrado.py)."""
    fig, axes = plt.subplots(len(canales), 1, figsize=(12, 7), sharex=True)
    for ax, canal in zip(np.atleast_1d(axes), canales):
        antes = raw.copy().crop(inicio, fin).get_data(picks=canal)[0] * 1e6
        despues = raw_filt.copy().crop(inicio, fin).get_data(picks=canal)[0] * 1e6
        t = np.linspace(inicio, fin, len(antes))
        ax.plot(t, antes, label="Sin filtrar", alpha=0.6)
        ax.plot(t, despues, label="Filtrada", linewidth=0.8)
        ax.set_ylabel(f"{canal} (uV)")
        ax.legend(loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("Tiempo (s)")
    fig.tight_layout()
    return fig