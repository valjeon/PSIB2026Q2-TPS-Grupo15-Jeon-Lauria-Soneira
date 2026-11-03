"""Forma de onda del VEO por nivel: ¿registra parpadeo o el propio LEP?

Si el VEO separa E1 de E4 casi tan bien como la ROI, puede ser porque (a) hay parpadeo reflejo, o (b) el
electrodo VEO capta por conduccion de volumen el campo del LEP. Se distinguen por la forma de onda:
    LEP en el VEO   bifasica, con las latencias de N2 y P2 de la ROI (~200 y ~370 ms), duracion como el LEP;
                    se parece a la ROI (o a la ROI invertida) en forma y tiempo.
    parpadeo        una sola deflexion lenta y grande (cientos de uV), que empieza ~150-250 ms y dura
                    300-500 ms; no tiene la forma bifasica del LEP.

Por sujeto y nivel: promedio inhomogeneo caso 2 (como ensambles.py) del VEO (30 Hz) y de la ROI, con los
ensayos validos (sub-XXX_finales-epo.fif). Gran promedio por nivel: ordinario entre sujetos.
Se informa, para cada nivel: pico del VEO (latencia, amplitud), su ancho a media altura, el valor del VEO en
las latencias de N2 y P2 de la ROI, y la correlacion de forma VEO-ROI entre 0 y 600 ms.

Salida en derived_mine/forma_veo/: forma_veo.png, forma_veo.csv, gran_promedio_veo-ave.fif.

Uso desde proyecto/:
    python -m procesamiento.forma_veo
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from procesamiento import config, inclusion, localizador, qc

CANALES_ROI = getattr(config, "CANALES_ROI", ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"])
TRAMO_FORMA = (0.0, 0.6)


def ensambles_sujeto(sub, carpeta):
    """{nivel: (veo, roi)} en uV: promedio inhomogeneo caso 2 de cada senal."""
    ep = mne.read_epochs(inclusion.ruta_finales(sub, carpeta), verbose="ERROR")
    t = ep.times
    veo = ep.copy().pick(config.CANAL_EOG).filter(None, config.PASA_BAJOS_LEP_HZ, picks="all",
                                                  verbose="ERROR").get_data() * 1e6        # epocas x 1 x t
    roi = ep.get_data(picks=[c for c in CANALES_ROI if c in ep.ch_names]).mean(axis=1, keepdims=True) * 1e6
    cond = ep.metadata["condicion"].to_numpy()
    out = {}
    for c in config.NIVELES:
        m = cond == c
        if m.sum() >= 2:
            out[c] = (localizador.promedio_ponderado(veo[m], t)[0], localizador.promedio_ponderado(roi[m], t)[0])
    return t, out


def ancho_media_altura(t, x, i):
    """Duracion (ms) del tramo continuo alrededor de i donde |x| >= 50% de |x[i]| con el mismo signo."""
    dentro = (x >= 0.5 * x[i]) if x[i] > 0 else (x <= 0.5 * x[i])
    a = b = i
    while a > 0 and dentro[a - 1]:
        a -= 1
    while b < len(x) - 1 and dentro[b + 1]:
        b += 1
    return (t[b] - t[a]) * 1000


def correr(sujetos=None, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    sujetos = sujetos or inclusion.incluidos("final", carpeta)
    lat = pd.read_csv(carpeta / "localizador" / "ventanas.csv").set_index("componente")["latencia_s"]
    ventanas = localizador.leer_ventanas(carpeta)

    por_nivel = {c: {"veo": [], "roi": []} for c in config.NIVELES}
    t = None
    for sub in sujetos:
        t, evs = ensambles_sujeto(sub, carpeta)
        for c, (veo, roi) in evs.items():
            por_nivel[c]["veo"].append(veo)
            por_nivel[c]["roi"].append(roi)
    ga = {c: {k: np.mean(v[k], axis=0) for k in ("veo", "roi")} for c, v in por_nivel.items() if v["veo"]}

    filas = []
    mt = (t >= TRAMO_FORMA[0]) & (t <= TRAMO_FORMA[1])
    i_n2, i_p2 = int(np.argmin(np.abs(t - lat["N2"]))), int(np.argmin(np.abs(t - lat["P2"])))
    for c, s in ga.items():
        veo, roi = s["veo"], s["roi"]
        i = np.flatnonzero(mt)[np.argmax(np.abs(veo[mt]))]
        filas.append({"condicion": c, "n_sujetos": len(por_nivel[c]["veo"]),
                      "veo_pico_ms": round(t[i] * 1000), "veo_pico_uV": round(float(veo[i]), 1),
                      "veo_ancho_media_altura_ms": round(ancho_media_altura(t, veo, i)),
                      "veo_en_lat_N2_uV": round(float(veo[i_n2]), 1), "veo_en_lat_P2_uV": round(float(veo[i_p2]), 1),
                      "roi_N2_uV": round(float(roi[i_n2]), 1), "roi_P2_uV": round(float(roi[i_p2]), 1),
                      "corr_forma_veo_roi": round(float(np.corrcoef(veo[mt], roi[mt])[0, 1]), 2)})
    tabla = pd.DataFrame(filas)

    salida = carpeta / "forma_veo"
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "forma_veo.csv", index=False)
    info = mne.create_info(["VEO", "ROI"], 1 / (t[1] - t[0]), ["eog", "eeg"])
    mne.write_evokeds(salida / "gran_promedio_veo-ave.fif",
                      [mne.EvokedArray(np.vstack([s["veo"], s["roi"]]) * 1e-6, info, tmin=t[0], comment=c,
                                       verbose="ERROR") for c, s in ga.items()], overwrite=True, verbose="ERROR")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k, nombre in ((axes[0], "veo", "VEO"), (axes[1], "roi", "ROI")):
        for c, s in ga.items():
            ax.plot(t * 1000, s[k], color=qc.COLORES[c], lw=1.8, label=c)
        ax.set_title(f"{nombre}: gran promedio por nivel")
    for c in ("E3", "E4"):
        if c in ga:
            norm = lambda x: x / np.abs(x[mt]).max()
            axes[2].plot(t * 1000, norm(ga[c]["veo"]), color=qc.COLORES[c], lw=1.8, label=f"VEO {c}")
            axes[2].plot(t * 1000, norm(ga[c]["roi"]), color=qc.COLORES[c], lw=1.2, ls="--", label=f"ROI {c}")
    axes[2].set_title("Forma normalizada (cada una a su maximo)")
    for ax in axes:
        for (a, b) in ventanas.values():
            ax.axvspan(a * 1000, b * 1000, color="0.93", zorder=0)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlim(-200, 1000)
        ax.set_xlabel("tiempo desde el pulso (ms)")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("uV")
    fig.suptitle(f"VEO vs ROI ({len(sujetos)} sujetos); gris: ventanas de N2 y P2")
    fig.tight_layout()
    fig.savefig(salida / "forma_veo.png", dpi=130)
    plt.close("all")

    print(f"{len(sujetos)} sujetos | latencias de la ROI: N2 {lat['N2'] * 1000:.0f} ms, P2 {lat['P2'] * 1000:.0f} ms")
    print(tabla.to_string(index=False))
    print(f"\nGuardado en {salida}")
    return tabla


def main(argv=None):
    p = argparse.ArgumentParser(description="Forma de onda del VEO por nivel: parpadeo o LEP")
    p.add_argument("sujetos", nargs="*", help="por defecto, los incluidos en inclusion_final.csv")
    a = p.parse_args(argv)
    matplotlib.use("Agg")
    correr(a.sujetos or None)


if __name__ == "__main__":
    main()