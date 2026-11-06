"""Promedios por ensamble: por sujeto y nivel, y gran promedio por nivel.

Sujetos: los incluidos en la etapa final (inclusion_final.csv). Epocas: sub-XXX_finales-epo.fif (ya con ref.
sin frontales, 30 Hz y linea de base; sin las rechazadas).

1. Ensamble por sujeto y nivel: promedio INHOMOGENEO, caso 2 (varianza del ruido variable): cada epoca pesa
   1/sigma_i^2, con sigma_i^2 estimada en el pre-estimulo (localizador.promedio_ponderado).
   Ruido del promedio (subensambles par/impar): (promedio pares - promedio impares) / 2 -> el PE se cancela y
   queda el error. SNR (dB) = potencia del ensamble / potencia del error, en las ventanas de N2 y P2, en la ROI.
   Sobre la senal de la ROI del ensamble: amplitud media de N2 y P2 en sus ventanas, N2-P2, y latencias de
   N2 (minimo) y P2 (maximo). En el ensamble la latencia de P2 si es confiable (el promedio reduce el ruido).
2. Gran promedio por nivel: promedio ORDINARIO de los ensambles de los sujetos (cada sujeto pesa igual).
   Figuras: ERP de la ROI por nivel y topomapas de N2 y P2 por nivel (amplitud media en cada ventana).

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

CANALES_ROI = getattr(config, "CANALES_ROI", ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"])
MEDIDAS = ["N2_uV", "P2_uV", "N2P2_uV", "lat_N2_ms", "lat_P2_ms"]


COLORES = {"E1": "#9ecae1", "E2": "#4292c6", "E3": "#ef6548", "E4": "#b30000"}   # (era qc.COLORES)


def _con_posicion(inst):
    """Deja solo canales EEG con posicion valida (si no, plot_topomap falla). (Era qc._con_posicion.)"""
    sin_pos = [ch["ch_name"] for ch in inst.info["chs"]
               if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH
               and (np.isnan(ch["loc"][:3]).any() or not np.any(ch["loc"][:3]))]
    if sin_pos:
        print(f"  aviso: sin posicion en el montaje, fuera del topomapa: {sin_pos}")
    return inst.copy().pick("eeg", exclude=sin_pos)


def _roi(evoked):
    return evoked.get_data(picks=[c for c in CANALES_ROI if c in evoked.ch_names]).mean(axis=0) * 1e6


def _evoked(datos, info, tmin, nave, comment):
    return mne.EvokedArray(datos, info, tmin=tmin, nave=nave, comment=comment, verbose="ERROR")


def medidas_roi(evoked, ventanas):
    """Medidas sobre la senal de la ROI de un ensamble."""
    t, x = evoked.times, _roi(evoked)
    m = {c: (t >= a) & (t <= b) for c, (a, b) in ventanas.items()}
    n2, p2 = x[m["N2"]].mean(), x[m["P2"]].mean()
    return {"N2_uV": n2, "P2_uV": p2, "N2P2_uV": p2 - n2,
            "lat_N2_ms": t[m["N2"]][np.argmin(x[m["N2"]])] * 1000,
            "lat_P2_ms": t[m["P2"]][np.argmax(x[m["P2"]])] * 1000}


def snr_par_impar(x, t, ventanas, picks_roi):
    """SNR (dB) del ensamble en la ROI, con el ruido estimado por subensambles par/impar."""
    total = localizador.promedio_ponderado(x, t)[picks_roi].mean(axis=0)
    a = localizador.promedio_ponderado(x[0::2], t)[picks_roi].mean(axis=0)
    b = localizador.promedio_ponderado(x[1::2], t)[picks_roi].mean(axis=0)
    error = (a - b) / 2
    m = np.zeros(len(t), bool)
    for (i, f) in ventanas.values():
        m |= (t >= i) & (t <= f)
    return 10 * np.log10((total[m] ** 2).mean() / (error[m] ** 2).mean())


def ensambles_sujeto(sub, ventanas, carpeta):
    ep = mne.read_epochs(inclusion.ruta_finales(sub, carpeta), verbose="ERROR").pick("eeg")
    t, info = ep.times, ep.info
    picks_roi = [ep.ch_names.index(c) for c in CANALES_ROI if c in ep.ch_names]
    cond = ep.metadata["condicion"].to_numpy()
    evs, filas = {}, []
    for c in config.NIVELES:
        x = ep.get_data()[cond == c]
        if len(x) < 2:
            continue
        evs[c] = _evoked(localizador.promedio_ponderado(x, t), info, ep.tmin, len(x), f"{sub} {c}")
        filas.append({"sujeto": sub, "condicion": c, "n_epocas": len(x),
                      **{k: round(float(v), 3) for k, v in medidas_roi(evs[c], ventanas).items()},
                      "snr_dB": round(float(snr_par_impar(x, t, ventanas, picks_roi)), 2)})
    return evs, filas


def fig_erp_niveles(evs, titulo, ventanas):
    fig, ax = plt.subplots(figsize=(8, 4))
    for c, e in evs.items():
        ax.plot(e.times * 1000, _roi(e), color=COLORES[c], lw=1.8, label=f"{c} (n={e.nave})")
    for (a, b) in ventanas.values():
        ax.axvspan(a * 1000, b * 1000, color="0.92", zorder=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-200, 1000)
    ax.set_xlabel("tiempo desde el pulso (ms)")
    ax.set_ylabel("ROI (uV)")
    ax.set_title(titulo)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_topomapas(gas, ventanas):
    """Amplitud media de N2 y P2 por nivel; misma escala en cada fila."""
    niveles = list(gas)
    fig, axes = plt.subplots(2, len(niveles), figsize=(2.8 * len(niveles), 5.6))
    e0 = _con_posicion(next(iter(gas.values())))
    for fila, comp in enumerate(("N2", "P2")):
        a, b = ventanas[comp]
        vals = {c: _con_posicion(e).copy().crop(a, b).data.mean(axis=1) * 1e6 for c, e in gas.items()}
        lim = max(np.abs(v).max() for v in vals.values())
        for col, c in enumerate(niveles):
            im, _ = mne.viz.plot_topomap(vals[c], e0.info, axes=axes[fila, col], show=False, cmap="RdBu_r",
                                         vlim=(-lim, lim), extrapolate="local")
            axes[fila, col].set_title(f"{comp} {c}", fontsize=10)
        fig.colorbar(im, ax=axes[fila].tolist(), shrink=0.8, label="uV")
    (n2a, n2b), (p2a, p2b) = ventanas["N2"], ventanas["P2"]
    fig.suptitle(f"Gran promedio por nivel: N2 {n2a * 1000:.0f}-{n2b * 1000:.0f} ms, "
                 f"P2 {p2a * 1000:.0f}-{p2b * 1000:.0f} ms")
    return fig


def correr(sujetos=None, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    sujetos = sujetos or inclusion.incluidos("final", carpeta)
    ventanas = localizador.leer_ventanas(carpeta)
    salida = carpeta / "ensambles"
    (salida / "sujetos").mkdir(parents=True, exist_ok=True)
    print(f"Ensambles de {len(sujetos)} sujetos (inhomogeneo caso 2 por nivel; ordinario entre sujetos)")

    todos, filas = {}, []
    for sub in sujetos:
        evs, f = ensambles_sujeto(sub, ventanas, carpeta)
        todos[sub] = evs
        filas += f
        mne.write_evokeds(salida / "sujetos" / f"{sub}_ensambles-ave.fif", list(evs.values()),
                          overwrite=True, verbose="ERROR")
        fig_erp_niveles(evs, f"{sub}: ensamble por nivel (ROI)", ventanas) \
            .savefig(salida / "sujetos" / f"{sub}_ensambles.png", dpi=110)
        plt.close("all")
    tabla = pd.DataFrame(filas)

    gas = {}
    for c in config.NIVELES:
        evs = [todos[s][c] for s in todos if c in todos[s]]
        gas[c] = _evoked(np.mean([e.get_data() for e in evs], axis=0), evs[0].info, evs[0].tmin, len(evs),
                         f"gran promedio {c}")
    mne.write_evokeds(salida / "gran_promedio_niveles-ave.fif", list(gas.values()), overwrite=True,
                      verbose="ERROR")
    fig_erp_niveles(gas, f"Gran promedio por nivel ({len(todos)} sujetos), ROI", ventanas) \
        .savefig(salida / "gran_promedio_niveles.png", dpi=130)
    fig_topomapas(gas, ventanas).savefig(salida / "topomapas_niveles.png", dpi=130)
    plt.close("all")

    ancho = {m: tabla.pivot(index="sujeto", columns="condicion", values=m)[list(config.NIVELES)].dropna()
             for m in MEDIDAS}
    fr = pd.DataFrame([{"medida": m, "n_sujetos": len(a),
                        **{f"mediana_{c}": round(a[c].median(), 2) for c in config.NIVELES}}
                       for m, a in ancho.items()])
    tabla.to_csv(salida / "ensambles_sujetos.csv", index=False)
    fr.to_csv(salida / "medianas_niveles.csv", index=False)

    print("\nMediana entre sujetos por nivel (ensambles):")
    with pd.option_context("display.float_format", lambda x: f"{x:.3g}"):
        print(fr.to_string(index=False))
    print(f"SNR del ensamble (mediana por nivel, dB): "
          + "  ".join(f"{c} {tabla[tabla.condicion == c]['snr_dB'].median():.1f}" for c in config.NIVELES))
    print(f"\nGuardado en {salida}")
    return tabla, fr
