"""Compara sujetos ya procesados: rechazo por nivel, N2-P2 por nivel y separacion E4 vs E1.

Lee, para cada sujeto, lo que deja el pipeline en derived_mine/sub-XXX/:
    sub-XXX_rechazo.csv   (artefactos.py)
    sub-XXX_rasgos.csv    (rasgos por ensayo: columnas condicion, N2P2_uV, ...)

Uso desde proyecto/:
    python -m exploracion.comparar_sujetos sub-001 sub-002 sub-003
    (o:  python exploracion/comparar_sujetos.py sub-001 sub-002 sub-003)

Guarda las tablas en derived_mine/comparacion_sujetos/.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

NIVELES = ["E1", "E2", "E3", "E4"]


def carpeta_salida():
    """derived_mine (misma regla que config.py, sin depender de importar el paquete)."""
    try:
        from extras.anterior import config
        return Path(config.CARPETA_SALIDA)
    except ImportError:
        base = Path(os.environ.get("PSIB_DATOS", "../data/ds005293"))
        return base.parent / "derived_mine"


def auroc(a, b):
    """P(un ensayo de `a` > uno de `b`). 0.5 = no separa; 1 = separa perfecto."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return mannwhitneyu(a, b).statistic / (len(a) * len(b))


def resumen_sujeto(sub, base):
    fila = {"sujeto": sub}
    r = base / sub / f"{sub}_rechazo.csv"
    if r.exists():
        t = pd.read_csv(r).set_index("condicion")
        for c in NIVELES:
            if c in t.index:
                fila[f"quedan_{c}"] = int(t.loc[c, "despues"])
                fila[f"pct_rech_{c}"] = round(100 * t.loc[c, "rechazadas"] / t.loc[c, "antes"])
        if "por_VEO" in t:
            fila["por_VEO_E4"] = int(t.loc["E4", "por_VEO"]) if "E4" in t.index else None
            fila["por_VEO_E1a3"] = int(t.loc[[c for c in NIVELES[:3] if c in t.index], "por_VEO"].sum())
    g = base / sub / f"{sub}_rasgos.csv"
    if g.exists():
        t = pd.read_csv(g)
        if "N2P2_uV" in t:
            medias = t.groupby("condicion")["N2P2_uV"].mean()
            for c in NIVELES:
                if c in medias:
                    fila[f"N2P2_{c}"] = round(medias[c], 1)
            fila["AUROC_E4vsE1"] = round(auroc(t.loc[t.condicion == "E4", "N2P2_uV"],
                                               t.loc[t.condicion == "E1", "N2P2_uV"]), 2)
            fila["AUROC_E4vsE3"] = round(auroc(t.loc[t.condicion == "E4", "N2P2_uV"],
                                               t.loc[t.condicion == "E3", "N2P2_uV"]), 2)
    return fila


def main(argv=None):
    p = argparse.ArgumentParser(description="Comparar sujetos procesados")
    p.add_argument("sujetos", nargs="+")
    a = p.parse_args(argv)
    base = carpeta_salida()
    tabla = pd.DataFrame([resumen_sujeto(s, base) for s in a.sujetos]).set_index("sujeto")

    pd.set_option("display.width", 200)
    bloques = {
        "Epocas que quedan / % rechazado por nivel": [c for c in tabla if c.startswith(("quedan_", "pct_rech_"))],
        "Rechazos por VEO (parpadeo en la ventana del LEP)": [c for c in tabla if c.startswith("por_VEO")],
        "N2-P2 medio por ensayo en Cz (uV)": [c for c in tabla if c.startswith("N2P2_")],
        "Separacion por N2-P2 (AUROC)": [c for c in tabla if c.startswith("AUROC")],
    }
    for titulo, cols in bloques.items():
        if cols:
            print(f"\n{titulo}\n{tabla[cols].to_string()}")

    if "N2P2_E4" in tabla and tabla["N2P2_E4"].notna().sum() > 1:
        v = tabla["N2P2_E4"].dropna()
        print(f"\nVariabilidad entre sujetos del N2-P2 en E4: de {v.min()} a {v.max()} uV "
              f"(cociente max/min = {v.max() / v.min():.1f}).")
        print("  Si el cociente es grande (> ~2), un umbral fijo en uV no sirve para todos: "
              "hay que normalizar por sujeto o usar rasgos relativos.")

    salida = base / "comparacion_sujetos"
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "resumen_por_sujeto.csv")
    print(f"\nTabla guardada en {salida / 'resumen_por_sujeto.csv'}")


if __name__ == "__main__":
    main()