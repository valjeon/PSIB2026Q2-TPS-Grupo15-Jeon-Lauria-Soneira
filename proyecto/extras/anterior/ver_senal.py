"""Como se ve la senal de EEG con los parpadeos TAL CUAL (sin descartar ni corregir nada).

Parte de las epocas sin rechazar (sub-XXX_laser-epo.fif), les pone referencia promedio y un pasa-bajos de
30 Hz (solo para mirar) y marca como "con parpadeo" las epocas donde el VEO supera UMBRAL_VEO_UV dentro de
VENTANA_VEO. Hace tres figuras en derived_mine/sub-XXX/senal/:
    1_ejemplos      epocas con y sin parpadeo del mismo nivel: VEO arriba y canales apilados de adelante hacia atras
    2_erp           promedios por nivel en Fp1, Fz, Cz y Pz: todas las epocas (lleno) y sin las marcadas (punteado)
    3_influencia    cuanto sube el pico a pico de cada canal cuando hay parpadeo, ordenado de adelante hacia atras
Sirve para ver cuanto llega el parpadeo a cada canal y si los canales lejos de los ojos casi no lo reciben.

Uso desde proyecto/:
    python -m extras.anterior.ver_senal sub-009 [--nivel E4] [--n 2]
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from extras.anterior import artefactos, config, qc, segmentacion

CLARO, OSCURO = "#2a78d6", "#eb6834"          # sin parpadeo / con parpadeo (azul y naranja, siempre en este orden)
TINTA, TINTA_SEC = "#0b0b0b", "#52514e"
CANALES_EJEMPLO = ["Fp1", "Fp2", "F7", "F8", "Fz", "FC1", "Cz", "T7", "T8", "Pz", "P3", "O1"]
CANALES_ERP = ["Fp1", "Fz", "Cz", "Pz"]


# ---------------------------------------------------------------------------
# Preparacion
# ---------------------------------------------------------------------------

def preparar(ep, referencia="promedio"):
    """Copia para mirar con pasa-bajos de 30 Hz (EEG y VEO).

    referencia: 'promedio' (la del pipeline) u 'original' (la del registro, sin re-referenciar).
    Con referencia promedio el parpadeo de los canales frontales se reparte, con signo opuesto, entre
    todos los demas; con la original no.
    """
    base = artefactos.con_referencia_promedio(ep) if referencia == "promedio" else ep
    return artefactos._pasa_bajos(base)


def posicion_y(nombres):
    """Posicion adelante-atras (metros, mas grande = mas adelante) de cada canal; NaN si no esta en el montaje."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    pos = {k.lower(): v for k, v in pos.items()}
    return np.array([pos[n.lower()][1] if n.lower() in pos else np.nan for n in nombres])


def ordenar_adelante_atras(nombres):
    y = posicion_y(nombres)
    orden = np.argsort(-np.where(np.isnan(y), -np.inf, y), kind="stable")
    return [nombres[i] for i in orden]


def veo_ptp(v, ventana=None):
    """Pico a pico del VEO (uV) dentro de `ventana` (por defecto VENTANA_VEO), por epoca."""
    w = v.copy().crop(*(config.VENTANA_VEO if ventana is None else ventana)).get_data(
        picks=config.CANAL_EOG)[:, 0, :]
    return (w.max(axis=1) - w.min(axis=1)) * 1e6


def veo_ptp_sin_evocada(v, ventana=None):
    """Pico a pico del VEO (uV) en VENTANA_VEO despues de restarle a cada epoca el VEO promedio de su nivel.

    El VEO tambien registra una respuesta ligada al estimulo (el LEP llega atenuado al ojo, y el parpadeo
    reflejo tambien esta atado al pulso). Esa parte es igual en todas las epocas del nivel y no es un
    parpadeo "de verdad": con esto solo queda lo que varia de epoca en epoca.
    """
    w = v.copy().crop(*(config.VENTANA_VEO if ventana is None else ventana)).get_data(
        picks=config.CANAL_EOG)[:, 0, :]
    cond = v.metadata["condicion"].to_numpy()
    r = w.copy()
    for c in set(cond):
        i = cond == c
        r[i] -= w[i].mean(axis=0)
    return (r.max(axis=1) - r.min(axis=1)) * 1e6


def tabla_veo(v, umbral_uV=None):
    """Por nivel: epocas, marcadas por VEO, marcadas por VEO sin la parte evocada, y pico a pico del VEO promedio."""
    umbral = config.UMBRAL_VEO_UV if umbral_uV is None else umbral_uV
    cond = v.metadata["condicion"].to_numpy()
    p, r = veo_ptp(v), veo_ptp_sin_evocada(v)
    w = v.copy().crop(*config.VENTANA_VEO).get_data(picks=config.CANAL_EOG)[:, 0, :] * 1e6
    filas = []
    for c in config.NIVELES:
        i = cond == c
        if not i.any():
            continue
        medio = w[i].mean(axis=0)
        filas.append({"nivel": c, "epocas": int(i.sum()), "marcadas_VEO": int((p[i] > umbral).sum()),
                      "marcadas_sin_evocada": int((r[i] > umbral).sum()),
                      "ptp_VEO_promedio_uV": round(float(medio.max() - medio.min()), 1)})
    return pd.DataFrame(filas)


def medianas_dentro_de_nivel(ptp, con, cond, minimo=2):
    """Mediana por canal de las epocas con y sin parpadeo, comparando DENTRO de cada nivel y promediando.

    Las epocas marcadas son casi todas de E4, donde el LEP es mas grande: compararlas con todas las demas
    mezclaria el efecto del parpadeo con el de la intensidad. Solo cuentan los niveles con al menos
    `minimo` epocas de cada tipo. Devuelve (con, sin, niveles usados, n con, n sin).
    """
    con_l, sin_l, usados, n_con, n_sin = [], [], [], 0, 0
    for c in config.NIVELES:
        i = cond == c
        a, b = i & con, i & ~con
        if a.sum() >= minimo and b.sum() >= minimo:
            con_l.append(np.median(ptp[a], axis=0))
            sin_l.append(np.median(ptp[b], axis=0))
            usados.append(c)
            n_con, n_sin = n_con + int(a.sum()), n_sin + int(b.sum())
    if not usados:
        raise ValueError("Ningun nivel tiene epocas con y sin parpadeo para comparar dentro del nivel")
    return np.mean(con_l, axis=0), np.mean(sin_l, axis=0), usados, n_con, n_sin


def marcar_parpadeos(v, umbral_uV=None):
    """(bool por epoca: con parpadeo, pico a pico del VEO por epoca)."""
    umbral = config.UMBRAL_VEO_UV if umbral_uV is None else umbral_uV
    p = veo_ptp(v)
    return p > umbral, p


def ptp_por_canal_en_ventana(v):
    """(epocas x canales EEG) pico a pico en uV dentro de VENTANA_VEO, y los nombres de canal."""
    e = v.copy().pick("eeg").crop(*config.VENTANA_VEO)
    x = e.get_data()
    return (x.max(axis=2) - x.min(axis=2)) * 1e6, e.ch_names


def elegir_ejemplos(nivel_idx, con, ptp, n):
    """Indices de n epocas con parpadeo (las de VEO cerca de la mediana de las marcadas) y n sin parpadeo."""
    marcadas = [i for i in nivel_idx if con[i]]
    limpias = [i for i in nivel_idx if not con[i]]
    if marcadas:
        mediana = np.median(ptp[marcadas])
        marcadas = sorted(marcadas, key=lambda i: abs(ptp[i] - mediana))[:n]
    limpias = sorted(limpias, key=lambda i: ptp[i])[:n]
    return sorted(marcadas), sorted(limpias)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=TINTA_SEC, labelsize=8)
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_axisbelow(True)


def fig_ejemplos(v, nivel="E4", n=2, canales=None, paso_uV=100):
    """Columnas: epocas con parpadeo y sin parpadeo del mismo nivel. Arriba el VEO; abajo el EEG apilado."""
    con, ptp = marcar_parpadeos(v)
    cond = v.metadata["condicion"].to_numpy()
    idx = np.flatnonzero(cond == nivel)
    marcadas, limpias = elegir_ejemplos(idx, con, ptp, n)
    if not marcadas:
        warnings.warn(f"No hay epocas con parpadeo en {nivel}: se muestran solo epocas sin parpadeo. Proba otro --nivel.")
    columnas = [(i, True) for i in marcadas] + [(i, False) for i in limpias]
    if not columnas:
        raise ValueError(f"No hay epocas del nivel {nivel}")
    pedidos = CANALES_EJEMPLO if canales is None else canales
    chs = ordenar_adelante_atras([c for c in pedidos if c in v.ch_names])
    fig, axes = plt.subplots(2, len(columnas), figsize=(3.4 * len(columnas) + 1.6, 7.5), sharex=True,
                             squeeze=False, gridspec_kw={"height_ratios": [1, 4]}, layout="constrained")
    t = v.times
    for k, (i, es_blink) in enumerate(columnas):
        color = OSCURO if es_blink else CLARO
        ax_v, ax = axes[0, k], axes[1, k]
        ax_v.plot(t, v.get_data(picks=config.CANAL_EOG)[i, 0] * 1e6, color=color, lw=1.1)
        ax_v.axhline(0, color="0.6", lw=0.5)
        ax_v.set_title(f"{'CON' if es_blink else 'SIN'} parpadeo · {nivel} · época {i}\n"
                       f"VEO {ptp[i]:.0f} µV en la ventana", fontsize=9, color=TINTA)
        for ax_ in (ax_v, ax):
            ax_.axvspan(*config.VENTANA_VEO, color="0.93", zorder=0)
            ax_.axvline(0, color=TINTA_SEC, lw=0.8)
            _estilo(ax_)
        for j, c in enumerate(chs):
            y = -j * paso_uV
            ax.plot(t, v.get_data(picks=c)[i, 0] * 1e6 + y, color=color, lw=0.9)
        ax.set_yticks([-j * paso_uV for j in range(len(chs))])
        ax.set_yticklabels(chs if k == 0 else [])
        ax.set_xlabel("tiempo desde el pulso (s)", fontsize=8, color=TINTA_SEC)
        ax.set_xlim(t[0], t[-1])
        if k == 0:
            ax_v.set_ylabel("VEO (µV)", fontsize=8, color=TINTA_SEC)
            ax.plot([t[-1] - 0.15] * 2, [0, paso_uV], color=TINTA, lw=2, clip_on=False)
            ax.text(t[-1] - 0.18, paso_uV / 2, f"{paso_uV} µV", ha="right", va="center", fontsize=8, color=TINTA)
    fig.suptitle("EEG sin descartar ni corregir nada (ref. promedio, 30 Hz). Franja gris: ventana de parpadeo "
                 f"{config.VENTANA_VEO[0]:g} a {config.VENTANA_VEO[1]:g} s. Canales de adelante hacia atrás.",
                 fontsize=9, color=TINTA_SEC)
    return fig


def fig_erp(v, canales=None, ventana=(-0.2, 1.0)):
    """Promedio por nivel en cada canal: todas las epocas (lleno) y sin las marcadas por VEO (punteado)."""
    con, _ = marcar_parpadeos(v)
    cond = v.metadata["condicion"].to_numpy()
    chs = [c for c in (CANALES_ERP if canales is None else canales) if c in v.ch_names] + [config.CANAL_EOG]
    fig, axes = plt.subplots(len(chs), 1, figsize=(8.5, 2.3 * len(chs)), sharex=True, layout="constrained")
    t = v.times
    m = (t >= ventana[0]) & (t <= ventana[1])
    for ax, c in zip(np.atleast_1d(axes), chs):
        x = v.get_data(picks=c)[:, 0, :] * 1e6
        for nivel in config.NIVELES:
            sel = cond == nivel
            if sel.sum() == 0:
                continue
            col = qc.COLORES[nivel]
            etiqueta = f"{nivel}: {sel.sum()} épocas, {(sel & ~con).sum()} sin parpadeo"
            ax.plot(t[m], x[sel].mean(axis=0)[m], color=col, lw=1.6, label=etiqueta)
            if (sel & ~con).sum() > 0:
                ax.plot(t[m], x[sel & ~con].mean(axis=0)[m], color=col, lw=1.2, ls=":")
        ax.axvline(0, color=TINTA_SEC, lw=0.8)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_ylabel(f"{c} (µV)", fontsize=8, color=TINTA_SEC)
        _estilo(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    np.atleast_1d(axes)[0].set_title("Promedio por nivel: todas las épocas (línea llena) y sin las marcadas "
                                     "por VEO (punteado)", fontsize=9, color=TINTA)
    np.atleast_1d(axes)[-1].set_xlabel("tiempo desde el pulso (s)", fontsize=8, color=TINTA_SEC)
    return fig


def fig_influencia(v, v_original=None):
    """Cuanto sube el pico a pico de cada canal (en la ventana de parpadeo) en las epocas marcadas por VEO.

    Arriba: mediana por canal en epocas con y sin parpadeo (referencia promedio), de adelante hacia atras.
    Abajo: el aumento (con - sin) segun la posicion adelante-atras del canal; si se pasa `v_original`
    (mismas epocas sin re-referenciar) se agrega esa serie para ver el efecto de la referencia.
    """
    con, _ = marcar_parpadeos(v)
    cond = v.metadata["condicion"].to_numpy()
    ptp, nombres = ptp_por_canal_en_ventana(v)
    con_med, sin_med, usados, n_con, n_sin = medianas_dentro_de_nivel(ptp, con, cond)
    orden = ordenar_adelante_atras(nombres)
    ix = [nombres.index(c) for c in orden]
    ignorados = set(getattr(config, "CANALES_SIN_CRITERIO_EEG", []))
    etiquetas = [c + ("*" if c in ignorados else "") for c in orden]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1.2, 1]},
                                 layout="constrained")
    x = np.arange(len(ix))
    a1.bar(x - 0.2, sin_med[ix], width=0.4, color=CLARO, label=f"sin parpadeo (n={n_sin})")
    a1.bar(x + 0.2, con_med[ix], width=0.4, color=OSCURO, label=f"con parpadeo (n={n_con})")
    a1.set_xticks(x)
    a1.set_xticklabels(etiquetas, rotation=90, fontsize=6)
    a1.set_ylabel("pico a pico mediano en la ventana (µV)", fontsize=8, color=TINTA_SEC)
    a1.set_title("Canales de adelante hacia atrás (* = no cuentan en el criterio de EEG). Comparación dentro de "
                 f"cada nivel ({', '.join(usados)}), para no mezclar el parpadeo con la intensidad", fontsize=9,
                 color=TINTA)
    a1.legend(frameon=False, fontsize=8)
    _estilo(a1)
    y_mm = posicion_y(nombres) * 1000
    extra = con_med - sin_med
    ok = np.isfinite(y_mm)
    a2.scatter(y_mm[ok], extra[ok], s=22, color=OSCURO, edgecolors="white", linewidths=0.6, zorder=3,
               label="referencia promedio (la del pipeline)")
    if v_original is not None:
        po, _ = ptp_por_canal_en_ventana(v_original)
        co, so, _, _, _ = medianas_dentro_de_nivel(po, con, cond)
        extra_o = co - so
        a2.scatter(y_mm[ok], extra_o[ok], s=22, color="#1baf7a", edgecolors="white", linewidths=0.6, zorder=3,
                   label="referencia original del registro")
        a2.legend(frameon=False, fontsize=8)
    for c in CANALES_EJEMPLO:
        if c in nombres and np.isfinite(y_mm[nombres.index(c)]):
            k = nombres.index(c)
            a2.annotate(c, (y_mm[k], extra[k]), xytext=(4, 3), textcoords="offset points", fontsize=7,
                        color=TINTA_SEC)
    a2.axhline(0, color="0.6", lw=0.6)
    a2.set_xlabel("posición adelante-atrás del electrodo (mm; más a la derecha = más cerca de los ojos)",
                  fontsize=8, color=TINTA_SEC)
    a2.set_ylabel("aumento con parpadeo (µV)", fontsize=8, color=TINTA_SEC)
    _estilo(a2)
    return fig


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------

def correr(sub, nivel="E4", n=2, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    ep = mne.read_epochs(segmentacion.ruta_epocas(sub, carpeta), verbose="ERROR")
    v = preparar(ep)
    con, ptp = marcar_parpadeos(v)
    cond = v.metadata["condicion"].to_numpy()
    print(f"{sub}: épocas con VEO > {config.UMBRAL_VEO_UV} µV en {config.VENTANA_VEO} s, con y sin la parte "
          "del VEO ligada al estímulo (promedio del nivel):")
    print(tabla_veo(v).to_string(index=False))
    salida = carpeta / sub / "senal"
    salida.mkdir(parents=True, exist_ok=True)
    figuras = {"1_ejemplos": fig_ejemplos(v, nivel, n), "2_erp": fig_erp(v),
               "3_influencia": fig_influencia(v, preparar(ep, "original"))}
    for nombre, fig in figuras.items():
        fig.savefig(salida / f"{sub}_{nombre}.png", dpi=130)
        plt.close(fig)
    print(f"Figuras en {salida}")
    return salida


def main(argv=None):
    p = argparse.ArgumentParser(description="Ver la senal de EEG con los parpadeos sin tocar")
    p.add_argument("sujeto")
    p.add_argument("--nivel", default="E4", choices=list(config.NIVELES))
    p.add_argument("--n", type=int, default=2, help="epocas de ejemplo con y sin parpadeo")
    a = p.parse_args(argv)
    matplotlib.use("Agg")
    correr(a.sujeto, a.nivel, a.n)


if __name__ == "__main__":
    main()