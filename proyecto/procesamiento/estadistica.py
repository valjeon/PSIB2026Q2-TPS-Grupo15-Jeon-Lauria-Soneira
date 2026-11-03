"""Estadistica por ensayo: relacion graduada con la intensidad (Spearman) y discriminacion E1 vs E4 (AUROC).

Datos: medicion.datos_variante(variante): ensayos no rechazados de los sujetos incluidos en la variante.
Rasgos: PRINCIPAL (N2-P2) + SECUNDARIOS + CONTROL (ver abajo).

1. Spearman (todos los niveles): por sujeto, rho entre el nivel (1-4) y el rasgo de cada ensayo.
   Grupo: Wilcoxon de una muestra sobre los rho de los sujetos (bilateral: rho distinto de 0).
   Signo esperado: N2-P2, P2, gamma > 0; N2 (mas negativa) y latencia de N2 (mas corta) < 0.
2. AUROC E1 vs E4, dejando un sujeto afuera: regresion logistica entrenada con los ensayos E1/E4 de los demas
   sujetos y probada en el que quedo afuera -> un AUROC por sujeto y por rasgo, mas un modelo combinado
   (rasgos de COMBINADO; sin N2-P2, que es P2 - N2, ni el control).
   Grupo: Wilcoxon de una muestra (AUROC > 0.5) y bootstrap de los sujetos (IC 95% del AUROC medio).

Salida en derived_mine/estadistica/<variante>/: spearman_sujetos.csv, auroc_sujetos.csv, resumen.csv, figuras.

Uso desde proyecto/:
    python -m procesamiento.estadistica                     (variante principal)
    python -m procesamiento.estadistica --variante veo200   (sensibilidad; tambien: sinveo)
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from procesamiento import config, medicion

PRINCIPAL = "N2P2_uV"
SECUNDARIOS = ["N2_uV", "P2_uV", "lat_N2_ms", "gamma_ERS_dB"]
CONTROL = []
COMBINADO = ["N2_uV", "P2_uV", "lat_N2_ms", "gamma_ERS_dB"]
NIVELES_AUROC = ("E1", "E4")
N_BOOTSTRAP = 2000
SEMILLA = 0


def rol(rasgo):
    return "principal" if rasgo == PRINCIPAL else "control" if rasgo in CONTROL else "secundario"


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------

def spearman_sujetos(d, rasgos):
    filas = []
    for sub, g in d.groupby("sujeto"):
        nivel = g["condicion"].map(config.NIVELES)
        fila = {"sujeto": sub}
        for r in rasgos:
            ok = g[r].notna()
            fila[r] = spearmanr(nivel[ok], g.loc[ok, r])[0] if ok.sum() > 3 else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# AUROC dejando un sujeto afuera
# ---------------------------------------------------------------------------

def _modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def auroc_sujetos(d, conjuntos):
    """{nombre: [columnas]} -> DataFrame con un AUROC por sujeto (fila) y conjunto de rasgos (columna)."""
    e = d[d["condicion"].isin(NIVELES_AUROC)]
    y = (e["condicion"] == NIVELES_AUROC[1]).to_numpy().astype(int)
    sujetos = e["sujeto"].to_numpy()
    out = {"sujeto": sorted(set(sujetos))}
    for nombre, cols in conjuntos.items():
        x = e[cols].to_numpy()
        valido = ~np.isnan(x).any(axis=1)
        aucs = []
        for s in out["sujeto"]:
            te, tr = (sujetos == s) & valido, (sujetos != s) & valido
            if len(set(y[te])) < 2:
                aucs.append(np.nan)
                continue
            m = _modelo().fit(x[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(x[te])[:, 1]))
        out[nombre] = aucs
    return pd.DataFrame(out)


def bootstrap_media(valores, n=N_BOOTSTRAP, semilla=SEMILLA):
    """IC 95% de la media, remuestreando los sujetos con reposicion."""
    v = np.asarray(valores, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(semilla)
    medias = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


# ---------------------------------------------------------------------------
# Resumen y figuras
# ---------------------------------------------------------------------------

def resumir(rho, auc, rasgos):
    filas = []
    for r in list(rasgos) + ["combinado"]:
        f = {"rasgo": r, "rol": "combinado" if r == "combinado" else rol(r)}
        if r in rho:
            v = rho[r].dropna()
            f.update(rho_mediana=round(v.median(), 3), rho_n_positivos=f"{int((v > 0).sum())}/{len(v)}",
                     p_spearman=wilcoxon(v).pvalue if len(v) >= 6 else np.nan)
        if r in auc:
            a = auc[r].dropna()
            ic = bootstrap_media(a)
            f.update(auroc_media=round(a.mean(), 3), auroc_ic95=f"[{ic[0]:.3f}; {ic[1]:.3f}]",
                     p_auroc=wilcoxon(a - 0.5, alternative="greater").pvalue if len(a) >= 6 else np.nan)
        filas.append(f)
    return pd.DataFrame(filas)


def figura(rho, auc, rasgos, variante):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, r in enumerate(rasgos):
        v = rho[r].dropna()
        a1.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=12, alpha=0.6)
        a1.hlines(v.median(), i - 0.3, i + 0.3, color="k", lw=2)
    a1.axhline(0, color="0.5", lw=0.8)
    a1.set_xticks(range(len(rasgos)), rasgos, rotation=35, ha="right")
    a1.set_ylabel("rho de Spearman (por sujeto)")
    a1.set_title("Relacion con el nivel (E1-E4); linea: mediana")
    cols = list(rasgos) + ["combinado"]
    for i, r in enumerate(cols):
        a = auc[r].dropna()
        ic = bootstrap_media(a)
        a2.errorbar(i, a.mean(), yerr=[[a.mean() - ic[0]], [ic[1] - a.mean()]], fmt="o", color="k", capsize=4)
        a2.scatter(np.full(len(a), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(a)), a, s=10,
                   alpha=0.4, color="tab:blue")
    a2.axhline(0.5, color="0.5", lw=0.8, ls="--")
    a2.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    a2.set_ylabel("AUROC E1 vs E4 (sujeto afuera)")
    a2.set_title("Discriminacion; negro: media e IC 95% bootstrap")
    fig.suptitle(f"Variante {variante}")
    fig.tight_layout()
    return fig


def correr(variante=None, carpeta=None):
    variante = medicion.PRINCIPAL if variante is None else variante
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    d = medicion.datos_variante(variante, carpeta)
    rasgos = [r for r in [PRINCIPAL] + SECUNDARIOS + CONTROL if r in d]
    faltan = [r for r in [PRINCIPAL] + SECUNDARIOS + CONTROL if r not in d]
    if faltan:
        print(f"Aviso: faltan rasgos {faltan} (corre python -m procesamiento.rasgos_ensayo)")
    conjuntos = {r: [r] for r in rasgos}
    comb = [c for c in COMBINADO if c in d]
    if comb:
        conjuntos["combinado"] = comb
    print(f"Variante {variante}: {d['sujeto'].nunique()} sujetos, {len(d)} ensayos")

    rho = spearman_sujetos(d, rasgos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc = auroc_sujetos(d, conjuntos)
    resumen = resumir(rho, auc, rasgos)

    salida = carpeta / "estadistica" / variante
    salida.mkdir(parents=True, exist_ok=True)
    rho.to_csv(salida / "spearman_sujetos.csv", index=False)
    auc.to_csv(salida / "auroc_sujetos.csv", index=False)
    resumen.to_csv(salida / "resumen.csv", index=False)
    figura(rho, auc, rasgos, variante).savefig(salida / "resultados.png", dpi=130)
    plt.close("all")

    with pd.option_context("display.float_format", lambda x: f"{x:.3g}"):
        print(resumen.to_string(index=False))
    print(f"\nGuardado en {salida}")
    return resumen


def main(argv=None):
    p = argparse.ArgumentParser(description="Spearman y AUROC E1 vs E4 por ensayo")
    p.add_argument("--variante", default=None, help=f"por defecto {medicion.PRINCIPAL}; tambien veo200, sinveo")
    a = p.parse_args(argv)
    matplotlib.use("Agg")
    correr(a.variante)


if __name__ == "__main__":
    main()