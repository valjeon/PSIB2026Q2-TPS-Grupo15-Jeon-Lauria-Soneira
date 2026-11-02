"""Localizador colapsado (Luck & Gaspelin 2017): ROI y latencias de N2/P2 sin mirar la diferencia entre niveles.

Sujetos: los incluidos en la etapa previa (inclusion_previa.csv).
Por sujeto: epocas preprocesadas leidas con preprocesado.para_analisis (ref. sin frontales, 30 Hz, linea de base),
    promedio ponderado por 1/varianza del ruido pre-estímulo (inhomogéneo, caso 2) en cada nivel -> ERP colapsado.
Grupo: promedio de los ERP colapsados (cada sujeto pesa igual).

1. Latencias: minimo (N2) y maximo (P2) del gran promedio en VENTANA_N2 / VENTANA_P2, sobre el promedio de
   los canales de la ROI a priori (config.CANALES_ROI).
2. Medida con polaridad por canal: media en +-MEDIA_PICO_S alrededor de P2 menos la de N2. El ojo (positivo
   en la latencia de N2) resta en vez de sumar.
3. ROI de los datos: canales (sin Fp/AF) con medida >= FRACCION_ROI x maximo del gran promedio.
4. Verificacion de grupo: canal maximo por sujeto (sin Fp/AF) y si cae en la ROI a priori; ROI a priori vs
   resto de los canales, Wilcoxon pareado entre sujetos.

Salida en derived_mine/localizador/: resumen por sujeto, medida por canal, gran promedio (-ave.fif) y figuras.

Uso desde proyecto/:
    python -m procesamiento.localizador
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from procesamiento import config, inclusion, preprocesado

CANALES_ROI = getattr(config, "CANALES_ROI", ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"])
MEDIA_PICO_S = getattr(config, "MEDIA_PICO_S", 0.02)        # +-20 ms alrededor de cada pico
FRACCION_ROI = getattr(config, "FRACCION_ROI_DATOS", 0.8)   # canales con >= 80% del maximo
FRACCION_VENTANA = getattr(config, "FRACCION_VENTANA", 0.5)  # ventanas: ancho a media altura de N2 y P2


# ---------------------------------------------------------------------------
# ERP colapsado por sujeto y gran promedio
# ---------------------------------------------------------------------------

def promedio_ponderado(x, t):
    """Promedio inhomogeneo, caso 2 (varianza del ruido variable): cada epoca pesa 1/sigma_i^2, con sigma_i^2
    estimada en el pre-estimulo (donde no hay PE). Pesos normalizados (suman 1), uno por epoca y canal.
    x: epocas x canales x tiempo."""
    var = x[:, :, t < 0].var(axis=2)                       # epocas x canales
    w = (1 / var) / (1 / var).sum(axis=0, keepdims=True)
    return (w[:, :, None] * x).sum(axis=0)


def erp_colapsado(sub):
    """ERP del sujeto: promedio ponderado por 1/varianza del ruido pre-estímulo (inhomogéneo, caso 2) en cada nivel."""
    v, _ = preprocesado.para_analisis(sub)
    v = v.copy().pick("eeg")
    cond = v.metadata["condicion"].to_numpy()
    promedios = [promedio_ponderado(v.get_data()[cond == c], v.times) for c in config.NIVELES if (cond == c).any()]
    return mne.EvokedArray(np.mean(promedios, axis=0), v.info, tmin=v.tmin, nave=len(v),
                           comment=f"{sub} colapsado", verbose="ERROR")


def gran_promedio(erps):
    """Promedio de los ERP colapsados (todos tienen los mismos canales: los malos estan interpolados)."""
    datos = np.mean([e.get_data() for e in erps], axis=0)
    return mne.EvokedArray(datos, erps[0].info, tmin=erps[0].tmin, nave=len(erps),
                           comment="gran promedio colapsado", verbose="ERROR")


# ---------------------------------------------------------------------------
# Latencias y medida por canal
# ---------------------------------------------------------------------------

def senal_roi(evoked, canales=None):
    canales = CANALES_ROI if canales is None else canales
    return evoked.get_data(picks=[c for c in canales if c in evoked.ch_names]).mean(axis=0)


def latencias(evoked):
    """(latencia N2, latencia P2) en s: minimo y maximo de la senal de la ROI a priori en sus ventanas."""
    t, x = evoked.times, senal_roi(evoked)
    out = []
    for (a, b), f in ((config.VENTANA_N2, np.argmin), (config.VENTANA_P2, np.argmax)):
        m = (t >= a) & (t <= b)
        out.append(float(t[m][f(x[m])]))
    return tuple(out)


def ventana_media_altura(evoked, t_pico, fraccion=None):
    """(inicio, fin) en s del tramo continuo alrededor de t_pico donde la senal de la ROI supera
    fraccion x su valor en el pico, con el mismo signo (ancho a media altura con fraccion = 0.5)."""
    fraccion = FRACCION_VENTANA if fraccion is None else fraccion
    t, x = evoked.times, senal_roi(evoked)
    i = int(np.argmin(np.abs(t - t_pico)))
    umbral = fraccion * x[i]
    dentro = x >= umbral if x[i] > 0 else x <= umbral
    a = b = i
    while a > 0 and dentro[a - 1]:
        a -= 1
    while b < len(t) - 1 and dentro[b + 1]:
        b += 1
    return float(t[a]), float(t[b])


def leer_ventanas(carpeta=None):
    """{'N2': (inicio, fin), 'P2': (inicio, fin)} en s, segun localizador/ventanas.csv."""
    ruta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "localizador" / "ventanas.csv"
    t = pd.read_csv(ruta).set_index("componente")
    return {c: (float(t.loc[c, "inicio_s"]), float(t.loc[c, "fin_s"])) for c in t.index}


def medida_canales(evoked, t_n2, t_p2, ancho=None):
    """P2 - N2 (uV) por canal: medias en +-ancho alrededor de cada latencia."""
    ancho = MEDIA_PICO_S if ancho is None else ancho
    t, x = evoked.times, evoked.get_data() * 1e6
    media = lambda tc: x[:, (t >= tc - ancho) & (t <= tc + ancho)].mean(axis=1)
    return media(t_p2) - media(t_n2)


def validos(nombres):
    """Mascara de canales que compiten en el localizador (todos menos Fp/AF)."""
    return ~np.isin(np.array(nombres), config.CANALES_SIN_CRITERIO_EEG)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def fig_erp(ga, t_n2, t_p2, ventanas=()):
    fig, ax = plt.subplots(figsize=(8, 4))
    t = ga.times * 1000
    for c in ga.ch_names:
        ax.plot(t, ga.get_data(picks=c)[0] * 1e6, color="0.85", lw=0.6)
    ax.plot(t, senal_roi(ga) * 1e6, color="tab:red", lw=2, label="ROI a priori (promedio)")
    for (a, b) in (config.VENTANA_N2, config.VENTANA_P2):
        ax.axvspan(a * 1000, b * 1000, color="0.93", zorder=0)
    for (a, b) in ventanas:
        ax.axvspan(a * 1000, b * 1000, color="tab:red", alpha=0.15, zorder=0)
    for tc, nombre in ((t_n2, "N2"), (t_p2, "P2")):
        ax.axvline(tc * 1000, color="k", ls="--", lw=0.8)
        ax.text(tc * 1000, 0.97, f" {nombre} {tc * 1000:.0f} ms", transform=ax.get_xaxis_transform(), va="top")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-200, 1000)
    ax.set_xlabel("tiempo desde el pulso (ms)")
    ax.set_ylabel("uV (ref. sin frontales)")
    ax.set_title(f"Gran promedio colapsado ({ga.nave} sujetos); gris: todos los canales")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_mapa(ga, medida, roi_datos):
    nombres = np.array(ga.ch_names)
    ok = validos(nombres)
    vmax = medida[ok].max()
    fig, ax = plt.subplots(figsize=(5, 5))
    mascara = np.isin(nombres, CANALES_ROI)
    mne.viz.plot_topomap(medida, ga.info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vmax, vmax),
                         extrapolate="local", mask=mascara,
                         mask_params=dict(marker="o", markersize=7, markerfacecolor="none",
                                          markeredgecolor="k", markeredgewidth=1.5))
    ax.set_title("P2 - N2 por canal (gran promedio)\n"
                 f"circulos: ROI a priori | ROI datos: {' '.join(roi_datos)}", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

def correr(sujetos=None, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    salida = carpeta / "localizador"
    salida.mkdir(parents=True, exist_ok=True)
    sujetos = sujetos or inclusion.incluidos("previa", carpeta)
    print(f"Localizador con {len(sujetos)} sujetos (referencia sin frontales: {config.REF_SIN_FRONTALES})")

    erps = {}
    for sub in sujetos:
        erps[sub] = erp_colapsado(sub)
    ga = gran_promedio(list(erps.values()))
    t_n2, t_p2 = latencias(ga)
    v_n2, v_p2 = ventana_media_altura(ga, t_n2), ventana_media_altura(ga, t_p2)

    nombres = np.array(ga.ch_names)
    ok = validos(nombres)
    en_roi = np.isin(nombres, CANALES_ROI)
    medida = medida_canales(ga, t_n2, t_p2)
    maximo = medida[ok].max()
    roi_datos = [n for n, m, v in zip(nombres, medida, ok) if v and m >= FRACCION_ROI * maximo]

    filas = []
    for sub, e in erps.items():
        m = medida_canales(e, t_n2, t_p2)
        i_max = int(np.argmax(np.where(ok, m, -np.inf)))
        filas.append({"sujeto": sub, "canal_max": nombres[i_max], "max_en_roi": bool(en_roi[i_max]),
                      "medida_roi_uV": round(float(m[en_roi].mean()), 2),
                      "medida_resto_uV": round(float(m[ok & ~en_roi].mean()), 2)})
    tabla = pd.DataFrame(filas)
    p = wilcoxon(tabla["medida_roi_uV"], tabla["medida_resto_uV"], alternative="greater").pvalue \
        if len(tabla) >= 6 else np.nan

    # guardar
    tabla.to_csv(salida / "localizador_sujetos.csv", index=False)
    pd.DataFrame({"canal": nombres, "medida_uV": medida.round(2), "en_roi_a_priori": en_roi,
                  "en_roi_datos": np.isin(nombres, roi_datos), "compite": ok}) \
        .sort_values("medida_uV", ascending=False).to_csv(salida / "localizador_canales.csv", index=False)
    pd.DataFrame({"componente": ["N2", "P2"], "latencia_s": [t_n2, t_p2], "inicio_s": [v_n2[0], v_p2[0]],
                  "fin_s": [v_n2[1], v_p2[1]]}).to_csv(salida / "ventanas.csv", index=False)
    ga.save(salida / "gran_promedio_colapsado-ave.fif", overwrite=True, verbose="ERROR")
    fig_erp(ga, t_n2, t_p2, (v_n2, v_p2)).savefig(salida / "gran_promedio_erp.png", dpi=130)
    fig_mapa(ga, medida, roi_datos).savefig(salida / "mapa_P2_N2.png", dpi=130)
    plt.close("all")

    # informar
    orden = np.argsort(-np.where(ok, medida, -np.inf))[:8]
    print(f"\nLatencias (gran promedio, ROI a priori): N2 {t_n2 * 1000:.0f} ms, P2 {t_p2 * 1000:.0f} ms")
    print(f"Ventanas ({FRACCION_VENTANA:.0%} del pico): N2 {v_n2[0] * 1000:.0f}-{v_n2[1] * 1000:.0f} ms, "
          f"P2 {v_p2[0] * 1000:.0f}-{v_p2[1] * 1000:.0f} ms")
    print("Canales con mayor P2-N2 (sin Fp/AF): " + ", ".join(f"{nombres[i]} {medida[i]:.1f}" for i in orden))
    print(f"ROI de los datos (>= {FRACCION_ROI:.0%} del maximo): {' '.join(roi_datos)}")
    print(f"  en comun con la ROI a priori: {' '.join(c for c in roi_datos if c in CANALES_ROI) or '-'}")
    print(f"  solo en datos: {' '.join(c for c in roi_datos if c not in CANALES_ROI) or '-'}"
          f"   |   solo a priori: {' '.join(c for c in CANALES_ROI if c not in roi_datos) or '-'}")
    n_en = int(tabla["max_en_roi"].sum())
    print(f"Sujetos con el maximo dentro de la ROI a priori: {n_en} de {len(tabla)} ({100 * n_en / len(tabla):.0f}%)")
    print(f"ROI a priori vs resto (Wilcoxon pareado, una cola): mediana {tabla['medida_roi_uV'].median():.1f} "
          f"vs {tabla['medida_resto_uV'].median():.1f} uV, p = {p:.2g}")
    fuera = tabla[~tabla["max_en_roi"]]
    if len(fuera):
        print("Maximo fuera de la ROI: " + ", ".join(f"{s} ({c})" for s, c in zip(fuera["sujeto"], fuera["canal_max"])))
    print(f"\nGuardado en {salida}")
    return ga, tabla, roi_datos, (t_n2, t_p2)


def main(argv=None):
    p = argparse.ArgumentParser(description="Localizador colapsado: ROI y latencias de N2/P2")
    p.add_argument("sujetos", nargs="*", help="por defecto, los incluidos en inclusion_previa.csv")
    a = p.parse_args(argv)
    matplotlib.use("Agg")
    correr(a.sujetos or None)


if __name__ == "__main__":
    main()