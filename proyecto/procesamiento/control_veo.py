"""Control: el clasificador E1 vs E4, ¿se apoya en la respuesta cortical (Cz) o en el parpadeo (VEO)?

El parpadeo reflejo crece con la intensidad del pulso. Si un clasificador que SOLO ve el pico a pico del
VEO (ninguna senal de Cz) separa E1 de E4 tan bien como uno que ve el N2-P2 de Cz, el resultado de Cz no
se puede atribuir a la respuesta cortical.

Parte de las epocas sin rechazar (sub-XXX_laser-epo.fif) y se queda con las que pasan el criterio
`centrales`: pico a pico menor a UMBRAL en VENTANA_EEG en cada canal de CANALES_CRITERIO (Cz, C3, C4).
Por epoca calcula el N2-P2 (30 Hz, referencia promedio) de cada canal central y el pico a pico del VEO:
    Cz, C3, C4        N2-P2 del canal
    *_corr            lo mismo despues de restarle a la senal la parte explicada por el VEO (regresion,
                      como `regresion_veo_ventana` de sensibilidad). Es un piso: la regresion tambien
                      puede restar respuesta cerebral que llega al VEO. Sin corregir es el techo.
    Cz+C3+C4          los tres canales juntos
    VEO               pico a pico del VEO en VENTANA_VEO (en logaritmo)
    VEO_n2p2          pico a pico del VEO en la ventana del N2-P2 (0.12-0.50 s)
No se usa el VEO "sin la parte evocada": para restarla hay que conocer el nivel, o sea la etiqueta.

Dos formas de medir la separacion E1 vs E4 (AUROC, 0.5 = azar):
    por sujeto   regresion logistica con validacion cruzada estratificada repetida dentro del sujeto
    agrupado     regresion logistica entrenada en los demas sujetos y probada en uno (leave-one-subject-out)

Uso desde proyecto/:
    python -m procesamiento.control_veo sub-004 sub-005 ...
"""
import argparse
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from procesamiento import config, marcas_autores, segmentacion, sensibilidad

NIVELES_COMPARADOS = ("E1", "E4")
UMBRAL_EEG_UV = 100
CANALES_CRITERIO = ("Cz", "C3", "C4")
CONJUNTOS = {"Cz": ["n2p2_Cz"], "Cz_corr": ["n2p2_Cz_corr"], "C3": ["n2p2_C3"], "C3_corr": ["n2p2_C3_corr"],
             "C4": ["n2p2_C4"], "C4_corr": ["n2p2_C4_corr"], "Cz+C3+C4": ["n2p2_Cz", "n2p2_C3", "n2p2_C4"],
             "VEO": ["veo"], "VEO_n2p2": ["veo_n2p2"]}
REPETICIONES, PLIEGUES, SEMILLA = 20, 5, 0


def _modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def n2p2_canal(v, canal):
    """N2-P2 (uV) por epoca en un canal: maximo en VENTANA_P2 menos minimo en VENTANA_N2."""
    x = v.get_data(picks=canal)[:, 0, :] * 1e6
    t = v.times
    mn = (t >= config.VENTANA_N2[0]) & (t <= config.VENTANA_N2[1])
    mp = (t >= config.VENTANA_P2[0]) & (t <= config.VENTANA_P2[1])
    return x[:, mp].max(axis=1) - x[:, mn].min(axis=1)


def rasgos_por_epoca(ep):
    """DataFrame con una fila por epoca limpia segun `centrales`: nivel (E1..E4) y los rasgos."""
    m = sensibilidad.medidas_por_epoca(ep)
    mc = sensibilidad.medidas_por_epoca(ep, corregir_veo=True)
    cols = [i for i, c in enumerate(m["canales"]) if c in CANALES_CRITERIO]
    malas = m["ptp_ventana"][:, cols].max(axis=1) > UMBRAL_EEG_UV
    d = {"nivel": ep.metadata["condicion"].to_numpy()}
    for c in CANALES_CRITERIO:
        d[f"n2p2_{c}"] = n2p2_canal(m["v"], c)
        d[f"n2p2_{c}_corr"] = n2p2_canal(mc["v"], c)
    d["veo"] = np.log(m["ptp_veo"])                  # log: el pico a pico es muy asimetrico
    d["veo_n2p2"] = np.log(m["ptp_veo_n2p2"])
    return pd.DataFrame(d)[~malas].reset_index(drop=True)


def auroc_dentro_del_sujeto(df, columnas):
    """AUROC de E4 vs E1 con validacion cruzada repetida dentro de un sujeto (NaN si hay muy pocas epocas)."""
    d = df[df["nivel"].isin(NIVELES_COMPARADOS)]
    y = (d["nivel"] == NIVELES_COMPARADOS[1]).to_numpy().astype(int)
    pliegues = min(PLIEGUES, int(y.sum()), int((1 - y).sum()))
    if pliegues < 2 or d[columnas].isna().any().any():
        return np.nan
    x = d[columnas].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=pliegues, n_repeats=REPETICIONES, random_state=SEMILLA)
    aucs = []
    for tr, te in cv.split(x, y):
        mod = _modelo().fit(x[tr], y[tr])
        aucs.append(roc_auc_score(y[te], mod.predict_proba(x[te])[:, 1]))
    return float(np.mean(aucs))


def auroc_leave_one_subject_out(tabla, columnas):
    """AUROC en cada sujeto con un modelo entrenado en los demas. Devuelve {sujeto: auroc}."""
    d = tabla[tabla["nivel"].isin(NIVELES_COMPARADOS)].dropna(subset=columnas)
    y = (d["nivel"] == NIVELES_COMPARADOS[1]).to_numpy().astype(int)
    out = {}
    for s in d["sujeto"].unique():
        te = (d["sujeto"] == s).to_numpy()
        if len(set(y[~te])) < 2 or len(set(y[te])) < 2:
            out[s] = np.nan
            continue
        mod = _modelo().fit(d.loc[~te, columnas].to_numpy(), y[~te])
        out[s] = float(roc_auc_score(y[te], mod.predict_proba(d.loc[te, columnas].to_numpy())[:, 1]))
    return out


def comparar(tabla):
    """Tabla (una fila por sujeto y conjunto de rasgos) con los dos AUROC."""
    filas = []
    for nombre, cols in CONJUNTOS.items():
        loso = auroc_leave_one_subject_out(tabla, cols)
        for s, g in tabla.groupby("sujeto"):
            filas.append({"sujeto": s, "rasgos": nombre,
                          "AUROC_dentro_del_sujeto": auroc_dentro_del_sujeto(g, cols),
                          "AUROC_leave_one_subject_out": loso.get(s, np.nan)})
    return pd.DataFrame(filas)


def correr(sujetos, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    salida = carpeta / "control_veo"
    salida.mkdir(parents=True, exist_ok=True)
    partes = []
    for sub in sujetos:
        ruta = segmentacion.ruta_epocas(sub, carpeta)
        if not ruta.exists():
            print(f"{sub}: no estan las epocas ({ruta}). Corre antes el pipeline.")
            continue
        ep = mne.read_epochs(ruta, verbose="ERROR")
        ep = marcas_autores.aplicar_marcas(ep, marcas_autores.marcas_de(sub))
        if config.CANAL_EOG not in ep.ch_names:
            print(f"{sub}: no tiene canal {config.CANAL_EOG}, se omite")
            continue
        df = rasgos_por_epoca(ep)
        df.insert(0, "sujeto", sub)
        partes.append(df)
    if not partes:
        return None
    tabla = pd.concat(partes, ignore_index=True)
    res = comparar(tabla)
    res.to_csv(salida / "control_veo.csv", index=False)
    tabla.to_csv(salida / "rasgos_por_epoca.csv", index=False)

    for col, titulo in (("AUROC_dentro_del_sujeto", "Dentro del sujeto (validacion cruzada repetida)"),
                        ("AUROC_leave_one_subject_out", "Agrupado (leave-one-subject-out)")):
        ancho = res.pivot(index="sujeto", columns="rasgos", values=col)[list(CONJUNTOS)]
        print(f"\nAUROC E4 vs E1 - {titulo}   (0.5 = azar)")
        print(ancho.round(2).to_string())
        print("mediana:  " + "  ".join(f"{c}={ancho[c].median():.2f}" for c in ancho))
    print(f"\nGuardado en {salida}")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description="Control: Cz vs VEO como rasgo para separar E1 de E4")
    p.add_argument("sujetos", nargs="+")
    a = p.parse_args(argv)
    warnings.simplefilter("ignore", FutureWarning)
    correr(a.sujetos)


if __name__ == "__main__":
    main()