"""Control del ojo: el resultado de la ROI (N2-P2 sigue a la intensidad), ¿se explica por el parpadeo reflejo?

El parpadeo reflejo crece con la intensidad. Si el VEO solo separara E1 de E4 tan bien como el N2-P2 de la ROI,
o si el N2-P2 dejara de separarlos al sacarle la parte que explica el VEO, el resultado podria ser ojo.

Rasgos por ensayo (ensayos validos de la variante; medicion.datos_variante):
    N2P2_uV        N2-P2 en la ROI (de medicion.py)
    VEO            log del pico a pico del VEO entre el inicio de la ventana de N2 y el fin de la de P2
    N2P2_sin_VEO   residuo del N2-P2 tras restarle (por regresion lineal, dentro de cada sujeto y con todos
                   sus ensayos, sin usar el nivel) la parte explicada por el VEO
Pruebas:
    1. AUROC E1 vs E4 dejando un sujeto afuera (estadistica.auroc_sujetos) para N2P2, VEO, N2P2_sin_VEO y
       N2P2 + VEO. Wilcoxon pareado entre sujetos: AUROC de N2P2 > AUROC de VEO.
    2. Fuga ensayo a ensayo: dentro de cada sujeto y nivel, Spearman entre el VEO y el N2-P2 de cada ensayo
       (con el nivel fijo, la intensidad no entra). Cerca de 0: el parpadeo no arrastra al N2-P2.

Salida en derived_mine/control_veo/<variante>/: auroc_sujetos.csv, fuga.csv, resumen.csv.

Uso desde proyecto/:
    python -m procesamiento.control_veo                    (variante principal)
    python -m procesamiento.control_veo --variante sinveo  (la mas exigente: no se rechazo ningun parpadeo)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from procesamiento import config, estadistica, localizador, medicion, preprocesado


def veo_por_ensayo(sub, ventanas):
    """DataFrame (ensayo, VEO): log del pico a pico del VEO (30 Hz) de N2 a P2."""
    v, _ = preprocesado.para_analisis(sub)
    a, b = ventanas["N2"][0], ventanas["P2"][1]
    e = v.copy().pick(config.CANAL_EOG).filter(None, config.PASA_BAJOS_LEP_HZ, picks="all", verbose="ERROR")
    x = e.copy().crop(a, b).get_data()[:, 0, :] * 1e6
    return pd.DataFrame({"ensayo": v.metadata["ensayo"].to_numpy(), "VEO": np.log(x.max(axis=1) - x.min(axis=1))})


def residuo(y, x):
    """y menos su ajuste lineal sobre x (minimos cuadrados)."""
    a, b = np.polyfit(x, y, 1)
    return y - (a * x + b)


def correr(variante=None, carpeta=None):
    variante = medicion.PRINCIPAL if variante is None else variante
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    d = medicion.datos_variante(variante, carpeta)
    ventanas = localizador.leer_ventanas(carpeta)

    partes = []
    for sub, g in d.groupby("sujeto"):
        g = g.merge(veo_por_ensayo(sub, ventanas), on="ensayo", how="left")
        g["N2P2_sin_VEO"] = residuo(g["N2P2_uV"].to_numpy(), g["VEO"].to_numpy())
        partes.append(g)
    d = pd.concat(partes, ignore_index=True)

    conjuntos = {"N2P2_uV": ["N2P2_uV"], "VEO": ["VEO"], "N2P2_sin_VEO": ["N2P2_sin_VEO"],
                 "N2P2_mas_VEO": ["N2P2_uV", "VEO"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc = estadistica.auroc_sujetos(d, conjuntos)

    fuga = []
    for (sub, c), g in d.groupby(["sujeto", "condicion"]):
        if len(g) >= 8:
            fuga.append({"sujeto": sub, "condicion": c, "rho_VEO_N2P2": spearmanr(g["VEO"], g["N2P2_uV"])[0]})
    fuga = pd.DataFrame(fuga)

    filas = []
    for nombre in conjuntos:
        a = auc[nombre].dropna()
        ic = estadistica.bootstrap_media(a)
        filas.append({"rasgos": nombre, "auroc_media": round(a.mean(), 3), "auroc_ic95": f"[{ic[0]:.3f}; {ic[1]:.3f}]",
                      "p_vs_azar": wilcoxon(a - 0.5, alternative="greater").pvalue})
    resumen = pd.DataFrame(filas)
    p_roi_vs_veo = wilcoxon(auc["N2P2_uV"], auc["VEO"], alternative="greater").pvalue

    salida = carpeta / "control_veo" / variante
    salida.mkdir(parents=True, exist_ok=True)
    auc.to_csv(salida / "auroc_sujetos.csv", index=False)
    fuga.round(3).to_csv(salida / "fuga.csv", index=False)
    resumen.to_csv(salida / "resumen.csv", index=False)

    print(f"Variante {variante}: {d['sujeto'].nunique()} sujetos, {len(d)} ensayos")
    with pd.option_context("display.float_format", lambda x: f"{x:.3g}"):
        print(resumen.to_string(index=False))
    print(f"AUROC N2P2 > AUROC VEO (Wilcoxon pareado entre sujetos): p = {p_roi_vs_veo:.2g}")
    print("Fuga VEO -> N2-P2 dentro de cada nivel (rho de Spearman, mediana entre sujetos): "
          + "  ".join(f"{c} {fuga[fuga.condicion == c]['rho_VEO_N2P2'].median():.3f}" for c in config.NIVELES))
    print(f"\nGuardado en {salida}")
    return resumen


def main(argv=None):
    p = argparse.ArgumentParser(description="Control del ojo: ROI vs VEO para separar E1 de E4")
    p.add_argument("--variante", default=None, help=f"por defecto {medicion.PRINCIPAL}; tambien veo200, sinveo")
    a = p.parse_args(argv)
    correr(a.variante)


if __name__ == "__main__":
    main()