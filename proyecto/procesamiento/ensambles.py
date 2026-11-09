"""Promedios por ensamble: por sujeto y nivel, y gran promedio por nivel.

Entrada: epocas finales (sub-XXX_finales-epo.fif) de los sujetos incluidos (inclusion_final.csv), ya limpias,
filtradas a 30 Hz y con línea de base.

1. Por sujeto y nivel: promedio inhomogeneo: cada epoca pesa 1/sigma_i^2, con sigma_i^2 estimada en el pre-estimulo.
   Sobre la ROI se mide: amplitud media de N2 y P2 en sus ventanas, N2-P2, y latencias de N2 y P2, y la SNR del 
   ensamble (ruido estimado con subensambles par/impar).
2. Gran promedio por nivel: promedio ordinario de los ensambles de los sujetos (cada sujeto pesa igual).

Salida en derived_mine/ensambles/: ensambles_sujetos.csv, medianas_niveles.csv, gran_promedio_niveles-ave.fif,
figuras de grupo y una figura por sujeto en sujetos/.
Lo llama pipeline.py en la etapa de grupo (python -m procesamiento.pipeline --grupo).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from procesamiento import config, inclusion, localizador

CANALES_ROI = config.CANALES_ROI
MEDIDAS = ["N2_uV", "P2_uV", "N2P2_uV", "lat_N2_ms", "lat_P2_ms"]   # lo que se mide por nivel


COLORES = {"E1": "#9ecae1", "E2": "#4292c6", "E3": "#ef6548", "E4": "#b30000"}


def _con_posicion(inst):
    """Deja solo canales EEG con posicion valida (si no, plot_topomap falla)."""
    sin_pos = []
    for ch in inst.info["chs"]:
        es_eeg = ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH      # se fija si el canal es de EEG
        xyz = ch["loc"][:3]                                   # coordenadas del electrodo en el montaje
        sin_posicion = np.isnan(xyz).any() or not np.any(xyz)  # si no se conoce la posición son NaN o (0,0,0)
        if es_eeg and sin_posicion:
            sin_pos.append(ch["ch_name"])       # arma la lista de los que no tienen posición válida
    if sin_pos:
        print(f"  aviso: sin posicion en el montaje, fuera del topomapa: {sin_pos}")
    return inst.copy().pick("eeg", exclude=sin_pos)     # arma una copia, deja los canales de EEG excluyendo los de
                                                        # posición inválida


def _roi(evoked):
    """Promedio entre todos los canales de la ROI (uV).
    Puede hacer el promedio de un sujeto y nivel o el gran promedio de un nivel."""
    canales = [c for c in CANALES_ROI if c in evoked.ch_names]
    return evoked.get_data(picks=canales).mean(axis=0) * 1e6


def _evoked(datos, info, tmin, nave, comment):
    """Guarda un array ya promediado como objeto Evoked de MNE."""
    return mne.EvokedArray(datos, info, tmin=tmin, nave=nave, comment=comment, verbose="ERROR")


def medidas_roi(evoked, ventanas):
    """Medidas del N2 y P2 sobre la senal de la ROI de un ensamble."""
    t, x = evoked.times, _roi(evoked)                 # tiempos (s) y senal de la ROI (uV)
    (a_n2, b_n2), (a_p2, b_p2) = ventanas["N2"], ventanas["P2"]     # separa los límites de las ventanas en 4 variables
    en_n2 = (t >= a_n2) & (t <= b_n2)                  # mascara: muestras dentro de la ventana de N2
    en_p2 = (t >= a_p2) & (t <= b_p2)                  # idem para P2
    n2 = x[en_n2].mean()                               # amplitud media de N2
    p2 = x[en_p2].mean()                               # amplitud media de P
    lat_n2 = t[en_n2][np.argmin(x[en_n2])]             # tiempo del minimo dentro de la ventana de N2
    lat_p2 = t[en_p2][np.argmax(x[en_p2])]             # tiempo del maximo dentro de la ventana de P2
    return {"N2_uV": n2, "P2_uV": p2,
            "N2P2_uV": p2 - n2,                        # amplitud pico a pico del complejo
            "lat_N2_ms": lat_n2 * 1000, "lat_P2_ms": lat_p2 * 1000}


def snr_par_impar(x, t, ventanas, picks_roi):
    """SNR (dB) del ensamble de la ROI en las ventanas de N2 y P2, con el ruido estimado por par/impar."""
    roi = lambda prom: prom[picks_roi].mean(axis=0)                  # lambda hace un promedio entre los canales de la ROI pero con arrays
    total = roi(localizador.promedio_ponderado(x, t))                 # ensamble con todas las epocas (de un sujeto y nivel)
    pares = roi(localizador.promedio_ponderado(x[0::2], t))           # epocas pares
    impares = roi(localizador.promedio_ponderado(x[1::2], t))         # epocas impares
    # pares e impares tienen la misma señal y ruidos independientes: se cancela la señal y
    # deja un ruido con la misma varianza que el del ensamble total
    error = (pares - impares) / 2               # se divide x2 para que quede el mismo tamaño que el ruido del ensamble
    en_ventanas = np.zeros(len(t), bool)                              # mascara de 0s
    for ini, fin in ventanas.values():
        en_ventanas |= (t >= ini) & (t <= fin)                        # marca en la máscara las muestras que caen dentro de las ventanas de N2 o P2
    p_senal = np.mean(total[en_ventanas] ** 2)                        # potencia media del ensamble
    p_ruido = np.mean(error[en_ventanas] ** 2)                        # potencia media del error
    return 10 * np.log10(p_senal / p_ruido)                           # cociente de potencias --> medida del SNR


def ensambles_sujeto(sub, ventanas, carpeta):
    """Ensamble de un nivel, por sujeto. Devuelve el nivel y una fila de medidas por nivel."""
    ep = mne.read_epochs(inclusion.ruta_finales(sub, carpeta), verbose="ERROR").pick("eeg")     # épocas finales
    t, info = ep.times, ep.info                     # tiempos y metadatos de los canales
    picks_roi = [ep.ch_names.index(c) for c in CANALES_ROI if c in ep.ch_names]     # índices de la ROI
    cond = ep.metadata["condicion"].to_numpy()              # niveles de cada época
    datos = ep.get_data()
    evs, filas = {}, []
    for c in config.NIVELES:
        x = datos[cond == c]        # saca las épocas correspondientes a cada nivel
        if len(x) < 2:
            continue
        prom = localizador.promedio_ponderado(x, t)                  # promedio inhomogeneo
        evs[c] = _evoked(prom, info, ep.tmin, len(x), f"{sub} {c}")   # se guarda como Evoked
        medidas = medidas_roi(evs[c], ventanas)                      
        snr = snr_par_impar(x, t, ventanas, picks_roi)               
        filas.append({"sujeto": sub, "condicion": c, "n_epocas": len(x),
                      **{k: round(float(v), 3) for k, v in medidas.items()},   # ** agrega las medidas como columnas
                      "snr_dB": round(float(snr), 2)})
    return evs, filas


def fig_erp_niveles(evs, titulo, ventanas):
    """Figura de la señal de la ROI de cada nivel superpuesta, con las ventanas de N2 y P2 sombreadas."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for c, e in evs.items():                                        # una curva por nivel
        ax.plot(e.times * 1000, _roi(e), color=COLORES[c], lw=1.8, label=f"{c} (n={e.nave})")
    for ini, fin in ventanas.values():                               # franjas en las ventanas de N2 y P2
        ax.axvspan(ini * 1000, fin * 1000, color="0.92", zorder=0)
    ax.axvline(0, color="k", lw=0.6)                    # instante del pulso
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-200, 1000)
    ax.set_xlabel("tiempo desde el pulso (ms)")
    ax.set_ylabel("ROI (uV)")
    ax.set_title(titulo)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_topomapas(gas, ventanas):
    """Topomapas de la amplitud media de N2 y P2 por nivel."""
    niveles = list(gas)
    fig, axes = plt.subplots(2, len(niveles), figsize=(2.8 * len(niveles), 5.6))
    con_pos = {c: _con_posicion(e) for c, e in gas.items()}          # gran promedio de cada nivel
    info = con_pos[niveles[0]].info                                  # posiciones de los canales
    for fila, comp in enumerate(("N2", "P2")):
        ini, fin = ventanas[comp]
        vals = {}
        for c, e in con_pos.items():
            ventana = e.copy().crop(ini, fin)                        # el gran promedio solo dentro de la ventana
            vals[c] = ventana.data.mean(axis=1) * 1e6                # media en el tiempo -> un valor por canal (uV)
        lim = max(np.abs(v).max() for v in vals.values())            # escala simetrica comun
        for col, c in enumerate(niveles):                            # un topomapa por nivel
            im, _ = mne.viz.plot_topomap(vals[c], info, axes=axes[fila, col], show=False,
                                         cmap="RdBu_r",
                                         vlim=(-lim, lim),
                                         extrapolate="local")         # no extrapola mas alla de los electrodos
            axes[fila, col].set_title(f"{comp} {c}", fontsize=10)
        fig.colorbar(im, ax=axes[fila].tolist(), shrink=0.8, label="uV")
    (ini_n2, fin_n2), (ini_p2, fin_p2) = ventanas["N2"], ventanas["P2"]
    fig.suptitle(f"Gran promedio por nivel: N2 {ini_n2 * 1000:.0f}-{fin_n2 * 1000:.0f} ms, "
                 f"P2 {ini_p2 * 1000:.0f}-{fin_p2 * 1000:.0f} ms")
    return fig


def correr(sujetos=None, carpeta=None):
    """"Ensambles de todos los sujetos incluídos, gran promedio por nivel, figuras y tablas."""
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    sujetos = sujetos or inclusion.incluidos("final", carpeta)      # los sujetos que pasaron la inclusión final
    ventanas = localizador.leer_ventanas(carpeta)
    salida = carpeta / "ensambles"
    (salida / "sujetos").mkdir(parents=True, exist_ok=True)         # crea carpetas de los sujetos, sino existen
    print(f"Ensambles de {len(sujetos)} sujetos (inhomogeneo por nivel; ordinario entre sujetos)")
    # ensamble por sujeto y nivel:
    todos, filas = {}, []
    for sub in sujetos:
        evs, f = ensambles_sujeto(sub, ventanas, carpeta)
        todos[sub] = evs
        filas += f                          # suma las filas de este sujeto a la lista
        mne.write_evokeds(salida / "sujetos" / f"{sub}_ensambles-ave.fif", list(evs.values()),
                          overwrite=True, verbose="ERROR")          # guarda los 4 ensambles en un .fif
        fig = fig_erp_niveles(evs, f"{sub}: ensamble por nivel (ROI)", ventanas)
        fig.savefig(salida / "sujetos" / f"{sub}_ensambles.png", dpi=110)
        plt.close("all")
    tabla = pd.DataFrame(filas)         # una fila por sujeto y nivel
    # gran promedio por nivel --> promedio ordinario de los ensambles (cada sujeto pesa igual)
    gas = {}
    for c in config.NIVELES:
        evs = [todos[s][c] for s in todos if c in todos[s]]          # el ensamble de nivel c de cada sujeto
        prom = np.mean([e.get_data() for e in evs], axis=0)
        gas[c] = _evoked(prom, evs[0].info, evs[0].tmin, len(evs), f"gran promedio {c}")   # nave = n de sujetos
    mne.write_evokeds(salida / "gran_promedio_niveles-ave.fif", list(gas.values()), overwrite=True,
                      verbose="ERROR")
    fig_erp_niveles(gas, f"Gran promedio por nivel ({len(todos)} sujetos), ROI", ventanas) \
        .savefig(salida / "gran_promedio_niveles.png", dpi=130)
    fig_topomapas(gas, ventanas).savefig(salida / "topomapas_niveles.png", dpi=130)
    plt.close("all")
    # mediana entre sujetos de cada medida por nivel
    filas_med = []
    for m in MEDIDAS:
        ancho = tabla.pivot(index="sujeto", columns="condicion", values=m)   # sujetos x niveles
        ancho = ancho[list(config.NIVELES)].dropna()
        filas_med.append({"medida": m, "n_sujetos": len(ancho),
                          **{f"mediana_{c}": round(ancho[c].median(), 2) for c in config.NIVELES}})
    medianas = pd.DataFrame(filas_med)                               # una fila por medida
    tabla.to_csv(salida / "ensambles_sujetos.csv", index=False)
    medianas.to_csv(salida / "medianas_niveles.csv", index=False)
    # resumen en consola
    print("\nMediana entre sujetos por nivel (ensambles):")
    with pd.option_context("display.float_format", lambda x: f"{x:.3g}"):
        print(medianas.to_string(index=False))
    snr = tabla.groupby("condicion")["snr_dB"].median()              # mediana de la SNR por nivel
    print("SNR del ensamble (mediana por nivel, dB): " + "  ".join(f"{c} {snr[c]:.1f}" for c in config.NIVELES))
    print(f"\nGuardado en {salida}")
    return tabla, medianas
