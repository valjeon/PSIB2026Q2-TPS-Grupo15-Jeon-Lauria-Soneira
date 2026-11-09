"""Tabla para elegir los umbrales del rechazo final SIN mirar el efecto: solo cuenta epocas y sujetos.

Para cada sujeto incluido en la etapa previa (inclusion_previa.csv), con las epocas de para_analisis()
(ref. sin frontales, 30 Hz, linea de base) y en el tramo de -1 s al final de la ventana de P2 (ventanas.csv):
    EEG: pico a pico maximo entre los canales de la ROI
    VEO: pico a pico del VEO (pasa-bajos 30 Hz)
y para cada combinacion de umbrales candidatos cuenta:
    - epocas rechazadas por nivel (sumadas entre sujetos)
    - sujetos que caerian por el criterio 1 (> MAX_PCT_EPOCAS_PERDIDAS % de 80 perdidas, contando autores y
      groseras) y por el criterio 3 (< MIN_EPOCAS_LIMPIAS en E1 o E4)
No usa amplitudes del LEP ni la diferencia entre niveles.

Salida en derived_mine/umbrales/: umbrales_resumen.csv (una fila por combinacion) y
umbrales_por_sujeto.csv (una fila por sujeto y combinacion).

Uso desde proyecto/:
    python -m extras.anterior.umbrales_rechazo
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from extras.anterior import config, inclusion, localizador, preprocesado

CANDIDATOS_EEG_UV = [100, 150, 200]
CANDIDATOS_VEO_UV = [100, 150, 200, None]          # None = sin rechazo por VEO
CANALES_ROI = getattr(config, "CANALES_ROI", ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"])


def tramo_rechazo(carpeta=None):
    """(-1 s, fin de la ventana de P2)."""
    return (config.TMIN, localizador.leer_ventanas(carpeta)["P2"][1])


def picos_por_epoca(sub, tramo):
    """DataFrame: una fila por epoca con nivel, pico a pico max en la ROI y pico a pico del VEO (uV)."""
    v, _ = preprocesado.para_analisis(sub)
    w = v.copy().crop(*tramo)
    roi = w.get_data(picks=[c for c in CANALES_ROI if c in w.ch_names]) * 1e6
    d = pd.DataFrame({"condicion": v.metadata["condicion"].to_numpy(),
                      "ptp_roi_uV": (roi.max(axis=2) - roi.min(axis=2)).max(axis=1)})
    if config.CANAL_EOG in w.ch_names:
        e = w.copy().pick(config.CANAL_EOG).filter(None, config.PASA_BAJOS_LEP_HZ, picks="all",
                                                   verbose="ERROR").get_data()[:, 0, :] * 1e6
        d["ptp_veo_uV"] = e.max(axis=1) - e.min(axis=1)
    else:
        d["ptp_veo_uV"] = np.nan
    return d


def evaluar(sujetos=None, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    sujetos = sujetos or inclusion.incluidos("previa", carpeta)
    tramo = tramo_rechazo(carpeta)
    esperadas = getattr(config, "EPOCAS_ESPERADAS", 80)
    max_pct = getattr(config, "MAX_PCT_EPOCAS_PERDIDAS", 25)
    print(f"{len(sujetos)} sujetos; tramo de rechazo {tramo[0]:.2f} a {tramo[1]:.3f} s")

    medidas = {sub: picos_por_epoca(sub, tramo) for sub in sujetos}
    filas = []
    for u_eeg, u_veo in product(CANDIDATOS_EEG_UV, CANDIDATOS_VEO_UV):
        for sub, d in medidas.items():
            malas = d["ptp_roi_uV"] > u_eeg
            if u_veo is not None:
                malas |= d["ptp_veo_uV"] > u_veo
            quedan = d[~malas]["condicion"].value_counts()
            fila = {"umbral_eeg_uV": u_eeg, "umbral_veo_uV": u_veo if u_veo is not None else "sin VEO",
                    "sujeto": sub, "rechazadas": int(malas.sum())}
            for c in config.NIVELES:
                fila[f"rech_{c}"] = int((malas & (d["condicion"] == c)).sum())
                fila[f"quedan_{c}"] = int(quedan.get(c, 0))
            fila["pct_perdidas"] = round(100 * (esperadas - (~malas).sum()) / esperadas, 1)
            fila["cae_c1"] = fila["pct_perdidas"] > max_pct
            fila["cae_c3"] = any(fila[f"quedan_{c}"] < config.MIN_EPOCAS_LIMPIAS for c in config.NIVELES_MINIMOS)
            filas.append(fila)
    por_sujeto = pd.DataFrame(filas)

    g = por_sujeto.groupby(["umbral_eeg_uV", "umbral_veo_uV"], sort=False)
    resumen = g.agg(**{f"rech_{c}": (f"rech_{c}", "sum") for c in config.NIVELES},
                    caen_c1=("cae_c1", "sum"), caen_c3=("cae_c3", "sum"))
    resumen["caen_total"] = g.apply(lambda t: int((t["cae_c1"] | t["cae_c3"]).sum()), include_groups=False)
    resumen["quedan_sujetos"] = len(sujetos) - resumen["caen_total"]
    resumen = resumen.reset_index()

    salida = carpeta / "umbrales"
    salida.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(salida / "umbrales_resumen.csv", index=False)
    por_sujeto.to_csv(salida / "umbrales_por_sujeto.csv", index=False)
    print("\nEpocas rechazadas por nivel (suma entre sujetos) y sujetos que caerian por los criterios 1 y 3:")
    print(resumen.to_string(index=False))
    print(f"\nGuardado en {salida}")
    return resumen, por_sujeto


def main(argv=None):
    p = argparse.ArgumentParser(description="Tabla de conteos para elegir los umbrales del rechazo final")
    p.add_argument("sujetos", nargs="*")
    a = p.parse_args(argv)
    evaluar(a.sujetos or None)


if __name__ == "__main__":
    main()