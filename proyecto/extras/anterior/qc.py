"""Control de calidad de las epocas: figuras y una tabla resumen por sujeto.

Las figuras se hacen sobre una COPIA con referencia promedio y pasa-bajos de 30 Hz,
que es como se miran los LEPs. Las epocas guardadas no se tocan.

Uso desde la carpeta que contiene `procesamiento/`:
    python -m extras.anterior.qc sub-001            # segmenta (si hace falta) y hace las figuras
    python -m extras.anterior.qc sub-001 --rehacer  # vuelve a segmentar aunque exista el .fif
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from . import config, segmentacion

COLORES = {"E1": "#9ecae1", "E2": "#4292c6", "E3": "#ef6548", "E4": "#b30000"}


# ---------------------------------------------------------------------------
# Preparacion para mirar
# ---------------------------------------------------------------------------

def copia_para_ver(ep):
    """Copia con referencia promedio y pasa-bajos 30 Hz (solo para figuras)."""
    v = ep.copy()
    v.set_eeg_reference("average", projection=False, verbose="ERROR")
    v.filter(None, config.PASA_BAJOS_LEP_HZ, picks="eeg", verbose="ERROR")
    return v


def evocados_por_nivel(v):
    """{'E1': Evoked, ...} solo con los niveles que tienen epocas."""
    presentes = set(v.metadata["condicion"])
    return {c: v[c].average() for c in config.NIVELES if c in presentes}


def picos_n2_p2(evoked, canal=None):
    """Latencia (ms) y amplitud (uV) de N2 (minimo) y P2 (maximo) en sus ventanas."""
    canal = config.CANAL_QC if canal is None else canal
    x = evoked.get_data(picks=canal)[0] * 1e6
    t = evoked.times
    out = {}
    for nombre, (a, b), f in (("N2", config.VENTANA_N2, np.argmin), ("P2", config.VENTANA_P2, np.argmax)):
        m = (t >= a) & (t <= b)
        i = f(x[m])
        out[f"{nombre}_lat_ms"] = t[m][i] * 1000
        out[f"{nombre}_amp_uV"] = x[m][i]
    out["N2P2_uV"] = out["P2_amp_uV"] - out["N2_amp_uV"]
    return out


def pico_a_pico(ep):
    """Maximo (sobre canales EEG) del pico a pico de cada epoca, en uV."""
    x = ep.get_data(picks="eeg")
    return (x.max(axis=2) - x.min(axis=2)).max(axis=1) * 1e6


# ---------------------------------------------------------------------------
# Figuras (cada una devuelve la figura; guardar() las escribe a disco)
# ---------------------------------------------------------------------------

def fig_conteos(meta):
    """Epocas por nivel y sesion: tienen que sumar 20 por nivel."""
    tabla = meta.pivot_table(index="sesion", columns="condicion", values="ensayo",
                             aggfunc="count", fill_value=0)
    tabla = tabla.reindex(sorted(tabla.index, key=lambda s: int(s.split("-")[1])))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    tabla.plot.bar(ax=ax, color=[COLORES[c] for c in tabla.columns], rot=0)
    total = meta.groupby("condicion").size()
    ax.set_title("Epocas por sesion  |  total: " + ", ".join(f"{k}={v}" for k, v in total.items()))
    ax.set_ylabel("n epocas")
    fig.tight_layout()
    return fig


def fig_retardo(meta):
    """Histograma del retardo S3x -> S100. Decide cual trigger es el tiempo cero."""
    r = meta["retardo_ms"].dropna()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(r, bins=30, color="#555")
    ax.set_xlabel("retardo S3x -> S100 (ms)")
    ax.set_ylabel("n pulsos")
    ax.set_title(f"mediana {r.median():.1f} ms, rango {r.min():.1f}-{r.max():.1f} ms")
    fig.tight_layout()
    return fig


def fig_erp_canal(evs, canal=None):
    """ERP en Cz por nivel, con las ventanas de N2 y P2 sombreadas."""
    canal = config.CANAL_QC if canal is None else canal
    fig, ax = plt.subplots(figsize=(8, 4))
    for c, e in evs.items():
        ax.plot(e.times * 1000, e.get_data(picks=canal)[0] * 1e6,
                color=COLORES[c], lw=1.8, label=f"{c} (n={e.nave})")
    for (a, b), nombre in ((config.VENTANA_N2, "N2"), (config.VENTANA_P2, "P2")):
        ax.axvspan(a * 1000, b * 1000, color="0.9", zorder=0)
        ax.text((a + b) / 2 * 1000, 0.98, nombre, transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="0.4")
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-200, 1000)
    ax.set_xlabel("tiempo desde el pulso (ms)")
    ax.set_ylabel(f"{canal} (uV, ref. promedio)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_imagen_ensayos(v, canal=None):
    """Ensayo por ensayo en Cz, ordenado por nivel: muestra ensayos raros y si el LEP es consistente."""
    canal = config.CANAL_QC if canal is None else canal
    orden = np.argsort(v.metadata["nivel"].to_numpy(), kind="stable")
    figs = v.plot_image(picks=canal, order=orden, show=False, title=f"{canal}: ensayos ordenados E1 -> E4")
    return figs[0]


def fig_topomapas(evoked_todos, picos):
    """Mapas en las latencias de N2 y P2 (promedio de todos los niveles)."""
    e = _con_posicion(evoked_todos)
    tiempos = [picos["N2_lat_ms"] / 1000, picos["P2_lat_ms"] / 1000]
    return e.plot_topomap(times=tiempos, ch_type="eeg", show=False)


def fig_pico_a_pico(ptp, meta):
    """Distribucion del pico a pico por epoca: para elegir el umbral de rechazo despues de la ICA."""
    fig, ax = plt.subplots(figsize=(8, 3))
    for c in config.NIVELES:
        m = (meta["condicion"] == c).to_numpy()
        ax.scatter(np.flatnonzero(m), ptp[m], s=14, color=COLORES[c], label=c)
    for u in config.UMBRALES_PTP_UV:
        ax.axhline(u, ls="--", color="0.5", lw=0.8)
        ax.text(len(ptp), u, f" {u} uV", va="center", fontsize=8, color="0.4")
    ax.set_xlabel("ensayo")
    ax.set_ylabel("max pico a pico (uV)")
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    return fig


def _con_posicion(inst):
    """Deja solo canales EEG con posicion valida (si no, plot_topomap falla)."""
    sin_pos = [ch["ch_name"] for ch in inst.info["chs"]
               if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH
               and (np.isnan(ch["loc"][:3]).any() or not np.any(ch["loc"][:3]))]
    if sin_pos:
        print(f"  aviso: sin posicion en el montaje, fuera del topomapa: {sin_pos}")
    return inst.copy().pick("eeg", exclude=sin_pos)


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------

def tabla_resumen(evs, ptp, meta):
    filas = []
    for c, e in evs.items():
        m = (meta["condicion"] == c).to_numpy()
        filas.append({"condicion": c, "n": e.nave, **picos_n2_p2(e),
                      "ptp_mediana_uV": float(np.median(ptp[m]))})
    return pd.DataFrame(filas).round(1)

def correr_qc(ep, sub, carpeta=None, subcarpeta="qc"):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / sub / subcarpeta
    carpeta.mkdir(parents=True, exist_ok=True)
    meta = ep.metadata.reset_index(drop=True)

    v = copia_para_ver(ep)
    evs = evocados_por_nivel(v)
    todos = v.average()
    picos = picos_n2_p2(todos)
    ptp = pico_a_pico(v)

    figuras = {
        "1_conteos": fig_conteos(meta),
        "2_retardo_S3x_S100": fig_retardo(meta),
        "3_erp_Cz_por_nivel": fig_erp_canal(evs),
        "4_ensayos_Cz": fig_imagen_ensayos(v),
        "5_topomapas_N2_P2": fig_topomapas(todos, picos),
        "6_pico_a_pico": fig_pico_a_pico(ptp, meta),
    }
    for nombre, fig in figuras.items():
        fig.savefig(carpeta / f"{sub}_{nombre}.png", dpi=130)
        plt.close(fig)

    resumen = tabla_resumen(evs, ptp, meta)
    resumen.to_csv(carpeta / f"{sub}_resumen.csv", index=False)
    print(f"\nPromedio de todos los niveles en {config.CANAL_QC}: "
          f"N2 {picos['N2_amp_uV']:.1f} uV a {picos['N2_lat_ms']:.0f} ms, "
          f"P2 {picos['P2_amp_uV']:.1f} uV a {picos['P2_lat_ms']:.0f} ms")
    print(resumen.to_string(index=False))
    print(f"\nFiguras en {carpeta}")
    return resumen


def main(argv=None):
    p = argparse.ArgumentParser(description="Figuras de control de un sujeto")
    p.add_argument("sujeto")
    p.add_argument("--sesiones", nargs="*", help="solo estas sesiones, ej. ses-1")
    p.add_argument("--rehacer", action="store_true", help="volver a segmentar aunque exista el .fif")
    p.add_argument("--etapa", choices=["crudas", "limpias"], default="crudas")
    a = p.parse_args(argv)
    matplotlib.use("Agg") 
    if a.etapa == "limpias":
        ruta = segmentacion.ruta_epocas(a.sujeto).with_name(f"{a.sujeto}_limpias-epo.fif")
        print(f"Leyendo {ruta}")
        correr_qc(mne.read_epochs(ruta, verbose="ERROR"), a.sujeto, subcarpeta="qc_limpias")
        return                  # guardar PNG sin abrir ventanas

    ruta = segmentacion.ruta_epocas(a.sujeto)
    if ruta.exists() and not a.rehacer and not a.sesiones:
        print(f"Leyendo {ruta}")
        ep = mne.read_epochs(ruta, verbose="ERROR")
    else:
        print(f"Segmentando {a.sujeto}...")
        ep = segmentacion.epocas_sujeto(a.sujeto, sesiones=a.sesiones)
        if not a.sesiones:
            segmentacion.guardar_epocas(ep, a.sujeto)
    correr_qc(ep, a.sujeto)


if __name__ == "__main__":
    main()