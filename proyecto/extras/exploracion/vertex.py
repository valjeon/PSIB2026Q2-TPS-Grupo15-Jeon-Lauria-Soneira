"""¿El maximo del LEP esta en el vertex? Sin ningun criterio de rechazo propio.

Parte de las epocas segmentadas con las marcas de los autores aplicadas (electrodos interpolados, epocas
de movimiento fuera), referencia promedio y pasa-bajos 30 Hz (como qc.py). Por nivel se resume con la
MEDIANA de las epocas (un parpadeo ocasional casi no la mueve; el reflejo, que esta en casi todas, si).

Figura 1  perfil por la linea media (Fpz -> Oz): ERP por nivel en cada canal y, a la derecha, la amplitud
          N2-P2 segun la posicion. LEP: sube hasta Cz y despues baja. Ojo: maxima en Fpz y cae hacia atras.
Figura 2  mapa de amplitud N2-P2 por canal, un mapa por nivel con la misma escala; marca el canal maximo.
Consola   canal con mayor N2-P2 por nivel y puesto de Cz; canales sospechosos (pico a pico atipico).

N2-P2 de un canal = maximo en VENTANA_P2 menos minimo en VENTANA_N2 (como qc.picos_n2_p2).

Uso desde proyecto/:
    python -m exploracion.vertex sub-010 sub-025 sub-050
    python -m exploracion.vertex sub-010 --promedio      (media en vez de mediana, para comparar)
    python -m exploracion.vertex sub-010 --guardar       (guarda las figuras en derived_mine/vertex/)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from extras.anterior import config, preprocesado, qc

# Linea media de adelante hacia atras. No hay FCz en el montaje: se usa el promedio de FC1 y FC2.
LINEA_MEDIA = ["Fpz", "Fz", "FCz*", "Cz", "CPz", "Pz", "POz", "Oz"]
VIRTUALES = {"FCz*": ["FC1", "FC2"]}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def epocas_para_ver(sub, regresion=None, carpeta=None, ref_sin_frontales=None):
    """Epocas preprocesadas (autores + canales propios + groseras), [regresion VEO], ref. promedio, 30 Hz."""
    return preprocesado.para_analisis(sub, regresion, carpeta, ref_sin_frontales)[0]


def resumen_por_nivel(v, estadistico="mediana"):
    """{'E1': Evoked, ...} con la mediana (o la media) de las epocas de cada nivel."""
    f = np.median if estadistico == "mediana" else np.mean
    out = {}
    for c in config.NIVELES:
        idx = np.flatnonzero((v.metadata["condicion"] == c).to_numpy())
        if len(idx) == 0:
            continue
        e = v[idx]
        out[c] = mne.EvokedArray(f(e.get_data(), axis=0), e.info, tmin=e.tmin, nave=len(e),
                                 comment=c, verbose="ERROR")
    return out


def senal_canal(evoked, canal):
    """Senal (uV) de un canal real o virtual (promedio de varios)."""
    nombres = VIRTUALES.get(canal, [canal])
    return evoked.get_data(picks=nombres).mean(axis=0) * 1e6


def n2p2(x, t):
    """Amplitud N2-P2 (uV) de una senal o de una matriz canales x tiempo."""
    mn = (t >= config.VENTANA_N2[0]) & (t <= config.VENTANA_N2[1])
    mp = (t >= config.VENTANA_P2[0]) & (t <= config.VENTANA_P2[1])
    return x[..., mp].max(axis=-1) - x[..., mn].min(axis=-1)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def fig_linea_media(evs, sub, estadistico):
    canales = [c for c in LINEA_MEDIA if all(n in next(iter(evs.values())).ch_names
                                               for n in VIRTUALES.get(c, [c]))]
    fig = plt.figure(figsize=(12, 1.3 * len(canales) + 1))
    gs = fig.add_gridspec(len(canales), 2, width_ratios=[2.2, 1], wspace=0.25)
    t = next(iter(evs.values())).times
    lim = max(np.abs(senal_canal(e, c)[(t > -0.2) & (t < 1)]).max() for e in evs.values() for c in canales)
    for i, canal in enumerate(canales):
        ax = fig.add_subplot(gs[i, 0])
        for c, e in evs.items():
            ax.plot(t * 1000, senal_canal(e, canal), color=qc.COLORES[c], lw=1.4, label=f"{c} (n={e.nave})")
        for (a, b) in (config.VENTANA_N2, config.VENTANA_P2):
            ax.axvspan(a * 1000, b * 1000, color="0.92", zorder=0)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlim(-200, 1000)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(canal, rotation=0, ha="right", va="center", fontsize=10)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper right")
        if i < len(canales) - 1:
            ax.set_xticklabels([])
    ax.set_xlabel("tiempo desde el pulso (ms)")

    ax = fig.add_subplot(gs[:, 1])
    for c, e in evs.items():
        amp = [n2p2(senal_canal(e, canal), t) for canal in canales]
        ax.plot(amp, range(len(canales)), "o-", color=qc.COLORES[c], label=c)
    ax.set_yticks(range(len(canales)), canales)
    ax.invert_yaxis()
    ax.set_xlabel("amplitud N2-P2 (uV)")
    ax.set_title("adelante (arriba) -> atras (abajo)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"{sub}: linea media ({estadistico} de las epocas, ref. promedio, 30 Hz)")
    return fig


def fig_mapas_n2p2(evs, sub, estadistico):
    e0 = qc._con_posicion(next(iter(evs.values())))
    t = e0.times
    amps = {c: n2p2(qc._con_posicion(e).get_data() * 1e6, t) for c, e in evs.items()}
    fuera = [i for i, c in enumerate(e0.ch_names) if c in config.CANALES_SIN_CRITERIO_EEG]
    vmax = max(np.delete(a, fuera).max() for a in amps.values())    # escala sin el ojo
    fig, axes = plt.subplots(1, len(amps), figsize=(3.2 * len(amps), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (c, a) in zip(axes, amps.items()):
        mascara = np.zeros(len(a), bool)
        a_sin = a.copy()
        a_sin[fuera] = -np.inf
        mascara[a_sin.argmax()] = True
        im, _ = mne.viz.plot_topomap(a, e0.info, axes=ax, show=False, cmap="Reds", extrapolate="local", vlim=(0, vmax),
                                     mask=mascara, mask_params=dict(marker="*", markersize=14,
                                                                    markerfacecolor="k"))
        ax.set_title(f"{c}: max en {e0.ch_names[a_sin.argmax()]}", fontsize=10)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="N2-P2 (uV)")
    fig.suptitle(f"{sub}: amplitud N2-P2 por canal ({estadistico}); * = canal maximo")
    return fig, amps, e0.ch_names


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

def analizar(sub, estadistico="mediana", guardar=False, regresion=False, ref_sin_frontales=True):
    config_txt = f"{'con' if regresion else 'sin'} regresion, ref. {'sin frontales' if ref_sin_frontales else 'promedio'}"
    print(f"\n=== {sub} ({config_txt}) ===")
    v = epocas_para_ver(sub, regresion, ref_sin_frontales=ref_sin_frontales)
    evs = resumen_por_nivel(v, estadistico)
    titulo = f"{estadistico}, {config_txt}"
    f1 = fig_linea_media(evs, sub, titulo)
    f2, amps, nombres = fig_mapas_n2p2(evs, sub, titulo)

    print("Canales con mayor N2-P2 por nivel (y puesto de Cz):")
    nombres_arr = np.array(nombres)
    validos = ~np.isin(nombres_arr, config.CANALES_SIN_CRITERIO_EEG)
    for c, a in amps.items():
        orden = np.argsort(-np.where(validos, a, -np.inf))[:validos.sum()]
        top = ", ".join(f"{nombres[i]} {a[i]:.1f}" for i in orden[:5])
        puesto = int(np.flatnonzero(nombres_arr[orden] == config.CANAL_QC)[0]) + 1
        print(f"  {c}: {top}   | Cz puesto {puesto} de {validos.sum()} (sin Fp/AF)")

    if guardar:
        carpeta = Path(config.CARPETA_SALIDA) / "vertex"
        carpeta.mkdir(parents=True, exist_ok=True)
        for nombre, fig in (("linea_media", f1), ("mapas_n2p2", f2)):
            fig.savefig(carpeta / f"{sub}_{nombre}_{estadistico}.png", dpi=130)
        print(f"Figuras en {carpeta}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Paso 1: ¿el maximo del LEP esta en el vertex?")
    p.add_argument("sujetos", nargs="+")
    p.add_argument("--regresion", action="store_true", help="con la regresion del VEO (descartada)")
    p.add_argument("--ref-promedio", action="store_true", help="referencia promedio con todos los canales")
    p.add_argument("--promedio", action="store_true", help="media de las epocas en vez de mediana")
    p.add_argument("--guardar", action="store_true", help="guardar las figuras en derived_mine/vertex/")
    a = p.parse_args(argv)
    for sub in a.sujetos:
        analizar(sub, "media" if a.promedio else "mediana", a.guardar, not a.ref_promedio)
    plt.show()


if __name__ == "__main__":
    main()