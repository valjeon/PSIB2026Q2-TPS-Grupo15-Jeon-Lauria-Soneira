"""Diagnostico del rechazo por EEG: de donde sale cada epoca descartada.

Para cada epoca que supera el umbral de EEG (el mismo criterio que artefactos.py: 30 Hz, ref. promedio,
sin los canales frontales de config.CANALES_SIN_CRITERIO_EEG) dice en que canal fue el mayor pico a pico,
cuanto valio y en que instantes estuvieron el maximo y el minimo. Las clasifica asi:
    LEP     canal central y maximo y minimo (los dos) entre 0.1 y 0.6 s: parece la propia respuesta
            (el LEP es bifasico: N2 y P2 caen dentro de esa ventana; un artefacto suele ser de un solo lado)
    pre     el pico dominante (el que mas se aleja de la mediana) esta antes del pulso
    tarde   el pico dominante esta despues de 0.8 s: movimiento o deriva
    otro    el resto

Si en un nivel casi todo es 'LEP', el umbral fijo esta rechazando respuestas grandes y sesga contra los
mejores respondedores. Si es 'tarde' o 'pre', el rechazo es razonable.

Uso desde proyecto/:
    python -m extras.anterior.diagnostico sub-009 [sub-006 ...]
"""
import argparse

import mne
import numpy as np
import pandas as pd

from extras.anterior import artefactos, config, marcas_autores, segmentacion

CANALES_CENTRALES = ["Fz", "F1", "F2", "FC1", "FC2", "Cz", "C1", "C2", "CP1", "CP2", "CPz", "Pz", "P1", "P2"]
VENTANA_LEP = (0.1, 0.6)
TARDE_S = 0.8


def clasificar(canal, t_max, t_min, t_pico):
    """'LEP', 'pre', 'tarde' u 'otro'. t_pico: instante del pico dominante (el mas lejos de la mediana)."""
    a, b = VENTANA_LEP
    if canal in CANALES_CENTRALES and a <= t_max <= b and a <= t_min <= b:
        return "LEP"
    if t_pico < 0:
        return "pre"
    if t_pico > TARDE_S:
        return "tarde"
    return "otro"


def epocas_rechazadas_por_eeg(ep, umbral_uV=None):
    """DataFrame con una fila por epoca rechazada por el criterio EEG (sin VEO)."""
    umbral = config.UMBRAL_RECHAZO_UV if umbral_uV is None else umbral_uV
    v = artefactos._pasa_bajos(artefactos.con_referencia_promedio(ep))
    excluir = set(getattr(config, "CANALES_SIN_CRITERIO_EEG", []))
    canales = [c for c in v.copy().pick("eeg").ch_names if c not in excluir]
    x = v.get_data(picks=canales) * 1e6                                  # epocas x canales x tiempo
    t = v.times
    ptp = x.max(axis=2) - x.min(axis=2)
    cz = v.get_data(picks=config.CANAL_QC)[:, 0, :] * 1e6 if config.CANAL_QC in v.ch_names else None
    filas = []
    for i in np.flatnonzero(ptp.max(axis=1) > umbral):
        j = int(ptp[i].argmax())
        t_max, t_min = float(t[x[i, j].argmax()]), float(t[x[i, j].argmin()])
        t_pico = float(t[np.abs(x[i, j] - np.median(x[i, j])).argmax()])
        filas.append({"epoca": int(i), "nivel": ep.metadata["condicion"].iloc[i], "canal": canales[j],
                      "ptp_uV": round(float(ptp[i, j])), "t_max_ms": round(t_max * 1000),
                      "t_min_ms": round(t_min * 1000), "t_pico_ms": round(t_pico * 1000),
                      "ptp_Cz_uV": round(float(cz[i].max() - cz[i].min())) if cz is not None else np.nan,
                      "tipo": clasificar(canales[j], t_max, t_min, t_pico)})
    return pd.DataFrame(filas, columns=["epoca", "nivel", "canal", "ptp_uV", "t_max_ms", "t_min_ms",
                                        "t_pico_ms", "ptp_Cz_uV", "tipo"])


def resumen_por_nivel(tabla):
    """Cuantas epocas rechazadas hay de cada tipo en cada nivel."""
    if tabla.empty:
        return pd.DataFrame()
    r = pd.crosstab(tabla["nivel"], tabla["tipo"]).reindex(list(config.NIVELES)).fillna(0).astype(int)
    for c in ("LEP", "pre", "tarde", "otro"):
        if c not in r:
            r[c] = 0
    r["total"] = r.sum(axis=1)
    return r[["total", "LEP", "pre", "tarde", "otro"]]


def main(argv=None):
    p = argparse.ArgumentParser(description="De donde sale cada epoca rechazada por EEG")
    p.add_argument("sujetos", nargs="+")
    a = p.parse_args(argv)
    for sub in a.sujetos:
        ep = mne.read_epochs(segmentacion.ruta_epocas(sub), verbose="ERROR")
        ep = marcas_autores.aplicar_marcas(ep, marcas_autores.marcas_de(sub))
        t = epocas_rechazadas_por_eeg(ep)
        print(f"\\n=== {sub}: {len(t)} epocas rechazadas por EEG (umbral {config.UMBRAL_RECHAZO_UV} uV) ===")
        if t.empty:
            continue
        print(t.to_string(index=False))
        print("\\nPor nivel y tipo:")
        print(resumen_por_nivel(t).to_string())
        print("\\nCanales que mas rechazan:", t["canal"].value_counts().head(5).to_dict())


if __name__ == "__main__":
    main()
