"""Estadistica por ensayo: relacion graduada con la intensidad (Spearman) y discriminacion E1 vs E4 (AUROC).

Datos: medicion.datos_variante(variante): ensayos no rechazados de los sujetos incluidos en la variante.
Rasgos: PRINCIPAL (N2-P2) + SECUNDARIOS + CONTROL (ver abajo).

1. Spearman (todos los niveles): por sujeto, rho entre el nivel (1-4) y el rasgo de cada ensayo.
   Grupo: Wilcoxon de una muestra sobre los rho de los sujetos (bilateral: rho distinto de 0).
   Signo esperado: N2-P2, P2, gamma > 0; N2 (mas negativa) y latencia de N2 (mas corta) < 0.
2. AUROC E1 vs E4 directo, por sujeto: probabilidad de que un ensayo de E4 tenga un valor mayor que uno de E1
   en el sentido esperado (SIGNO), es decir el U de Mann-Whitney normalizado. No se ajusta nada con los datos:
   el sentido se fija a priori, asi que el resultado de cada sujeto ya es el de un sujeto nuevo.
   Grupo: Wilcoxon de una muestra (AUROC > 0.5) y bootstrap de los sujetos (IC 95% del AUROC medio).

Salida en derived_mine/estadistica/<variante>/: spearman_sujetos.csv, auroc_sujetos.csv, resumen.csv, figuras.

Lo llama pipeline.py en la etapa de grupo (python -m procesamiento.pipeline --grupo).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

from procesamiento import config, medicion

PRINCIPAL = "N2P2_uV"
SECUNDARIOS = ["N2_uV", "P2_uV", "lat_N2_ms", "gamma_ERS_dB"]
CONTROL = []
SIGNO = {"N2P2_uV": 1, "N2_uV": -1, "P2_uV": 1, "lat_N2_ms": -1, "gamma_ERS_dB": 1}   # sentido esperado con la intensidad
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
# AUROC directo por sujeto
# ---------------------------------------------------------------------------

def auroc_sujetos(d, rasgos):
    """DataFrame con un AUROC E1 vs E4 por sujeto (fila) y rasgo (columna), con el rasgo multiplicado por su
    SIGNO como puntaje (U de Mann-Whitney normalizado; 0.5 = no discrimina)."""
    e = d[d["condicion"].isin(NIVELES_AUROC)]
    filas = []
    for sub, g in e.groupby("sujeto"):
        y = (g["condicion"] == NIVELES_AUROC[1]).to_numpy().astype(int)
        fila = {"sujeto": sub}
        for r in rasgos:
            ok = g[r].notna().to_numpy()
            fila[r] = (roc_auc_score(y[ok], SIGNO.get(r, 1) * g[r].to_numpy()[ok])
                       if len(set(y[ok])) == 2 else np.nan)
        filas.append(fila)
    return pd.DataFrame(filas)


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
    for r in rasgos:
        f = {"rasgo": r, "rol": rol(r)}
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
    cols = list(rasgos)
    for i, r in enumerate(cols):
        a = auc[r].dropna()
        ic = bootstrap_media(a)
        a2.errorbar(i, a.mean(), yerr=[[a.mean() - ic[0]], [ic[1] - a.mean()]], fmt="o", color="k", capsize=4)
        a2.scatter(np.full(len(a), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(a)), a, s=10,
                   alpha=0.4, color="tab:blue")
    a2.axhline(0.5, color="0.5", lw=0.8, ls="--")
    a2.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    a2.set_ylabel("AUROC E1 vs E4 (por sujeto)")
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
        print(f"Aviso: faltan rasgos {faltan} (los agrega rasgos_ensayo.py en la etapa de grupo)")
    print(f"Variante {variante}: {d['sujeto'].nunique()} sujetos, {len(d)} ensayos")

    rho = spearman_sujetos(d, rasgos)
    auc = auroc_sujetos(d, rasgos)
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