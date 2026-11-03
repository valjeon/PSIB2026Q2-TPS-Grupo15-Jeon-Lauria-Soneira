"""Control de la gamma-ERS: ¿es actividad cortical central o musculo?

El musculo (temporal, sobresalto o retirada de la mano) tambien crece con la intensidad y cae en 60-95 Hz,
pero se ve sobre todo en canales temporales/laterales, lejos del vertex. Se compara:
    1. Mapa de la gamma-ERS por nivel (gran promedio): cortical -> centrada en la ROI; musculo -> bordes/temporales.
    2. ROI vs canales temporales (TEMPORALES): rho de Spearman nivel-gamma por sujeto en cada region, y
       Wilcoxon pareado ROI vs temporales. Si la ROI sigue a la intensidad mas que los temporales, apoya el
       origen cortical; si los temporales siguen igual o mas, probablemente es musculo.
Mismos parametros que rasgos_ensayo.py (60-95 Hz, 200-500 ms, linea de base -400 a -100 ms, STFT 250 ms),
sobre la senal de 1-100 Hz, con los ensayos validos de la variante principal (medicion.datos_variante).

Ojo: en varios sujetos T7/T8/TP7/TP8 estan interpolados (marcas de los autores); se informa cuantos.

Salida en derived_mine/control_gamma/: gamma_canales.csv (gamma por sujeto, nivel y canal), comparacion.csv,
mapa_gamma_niveles.png.

Uso desde proyecto/:
    python -m procesamiento.control_gamma
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from procesamiento import config, medicion, preprocesado, qc, rasgos_ensayo

CANALES_ROI = rasgos_ensayo.CANALES_ROI
TEMPORALES = ["FT7", "FT8", "T7", "T8", "TP7", "TP8"]
BANDA, VENTANA = (60, 95), (0.2, 0.5)


def gamma_sujeto(sub, ensayos_validos):
    """(info, matriz epocas x canales de gamma-ERS en dB, niveles) para los ensayos validos del sujeto."""
    v, _ = preprocesado.para_analisis(sub, pasa_bajos=False)
    v = v[np.flatnonzero(v.metadata["ensayo"].isin(ensayos_validos).to_numpy())].pick("eeg")
    x = v.get_data() * 1e6                                                 # epocas x canales x t
    g = np.stack([rasgos_ensayo.cambio_potencia(x[:, i], v.info["sfreq"], v.tmin, BANDA, VENTANA)
                  for i in range(x.shape[1])], axis=1)
    return v.info, g, v.metadata["condicion"].to_numpy()


def correr(carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    d = medicion.datos_variante(carpeta=carpeta)
    salida = carpeta / "control_gamma"
    salida.mkdir(parents=True, exist_ok=True)
    resumen = pd.read_csv(carpeta / "preprocesado_resumen.csv", dtype=str).fillna("").set_index("sujeto")

    filas, comp, mapas, info = [], [], {c: [] for c in config.NIVELES}, None
    for sub, g in d.groupby("sujeto"):
        info, gam, cond = gamma_sujeto(sub, g["ensayo"])
        nombres = info.ch_names
        roi = [nombres.index(c) for c in CANALES_ROI if c in nombres]
        tem = [nombres.index(c) for c in TEMPORALES if c in nombres]
        nivel = pd.Series(cond).map(config.NIVELES).to_numpy()
        for c in config.NIVELES:
            m = cond == c
            if m.any():
                mapas[c].append(gam[m].mean(axis=0))
                filas += [{"sujeto": sub, "condicion": c, "canal": n, "gamma_dB": round(float(val), 3)}
                          for n, val in zip(nombres, gam[m].mean(axis=0))]
        interp = (resumen.loc[sub, "interpolados_autores"] + " " + resumen.loc[sub, "interpolados_propios"]
                  if sub in resumen.index else "").split()
        comp.append({"sujeto": sub,
                     "rho_roi": spearmanr(nivel, gam[:, roi].mean(axis=1))[0],
                     "rho_temporales": spearmanr(nivel, gam[:, tem].mean(axis=1))[0],
                     "temporales_interpolados": " ".join(c for c in TEMPORALES if c in interp)})
    comp = pd.DataFrame(comp)
    pd.DataFrame(filas).to_csv(salida / "gamma_canales.csv", index=False)
    comp.round(3).to_csv(salida / "comparacion.csv", index=False)

    # mapa por nivel (gran promedio de los sujetos)
    ga = {c: np.mean(v, axis=0) for c, v in mapas.items() if v}
    ev = mne.EvokedArray(np.zeros((len(info.ch_names), 1)), info, verbose="ERROR")
    ok = qc._con_posicion(ev).ch_names
    idx = [info.ch_names.index(c) for c in ok]
    lim = max(np.abs(m[idx]).max() for m in ga.values())
    fig, axes = plt.subplots(1, len(ga), figsize=(3 * len(ga), 3.4))
    for ax, (c, m) in zip(axes, ga.items()):
        im, _ = mne.viz.plot_topomap(m[idx], qc._con_posicion(ev).info, axes=ax, show=False, cmap="RdBu_r",
                                     vlim=(-lim, lim), extrapolate="local",
                                     mask=np.isin(ok, CANALES_ROI + TEMPORALES),
                                     mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="k",
                                                      markersize=5))
        ax.set_title(f"gamma-ERS {c}", fontsize=10)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="dB (60-95 Hz, 200-500 ms)")
    fig.suptitle("Gran promedio de la gamma-ERS por nivel; circulos: ROI y temporales")
    fig.savefig(salida / "mapa_gamma_niveles.png", dpi=130)
    plt.close("all")

    # informe
    med = lambda c, chans: np.median([m[[info.ch_names.index(x) for x in chans if x in info.ch_names]].mean()
                                      for m in mapas[c]])
    print(f"{len(comp)} sujetos (variante {medicion.PRINCIPAL})")
    print("Gamma-ERS (dB), mediana entre sujetos:")
    for nombre, chans in (("ROI", CANALES_ROI), ("temporales", TEMPORALES)):
        print(f"  {nombre:10s} " + "  ".join(f"{c} {med(c, chans):5.2f}" for c in config.NIVELES))
    p = wilcoxon(comp["rho_roi"], comp["rho_temporales"]).pvalue
    print(f"rho de Spearman nivel-gamma (mediana): ROI {comp['rho_roi'].median():.3f}, "
          f"temporales {comp['rho_temporales'].median():.3f}  (Wilcoxon pareado ROI vs temporales: p = {p:.2g})")
    n_int = (comp["temporales_interpolados"] != "").sum()
    print(f"Sujetos con algun canal temporal interpolado: {n_int} de {len(comp)}")
    print(f"\nGuardado en {salida}")
    return comp


def main(argv=None):
    argparse.ArgumentParser(description="Control de la gamma-ERS: cortical vs muscular").parse_args(argv)
    matplotlib.use("Agg")
    correr()


if __name__ == "__main__":
    main()