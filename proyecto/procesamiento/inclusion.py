"""Criterios de inclusion de sujetos, definidos ANTES de ver los datos (config.py). Nunca miran el efecto.

Etapa 'previa' (despues de limpieza.py; decide quien entra al localizador):
    2. canales interpolados (autores + propios) > MAX_PCT_CANALES_INTERPOLADOS % de los EEG    (PREP)
    4. Cz interpolado, o mas de MAX_ROI_INTERPOLADOS canales de la ROI interpolados
    5. sin canal VEO
Etapa 'final' (despues del rechazo final; decide quien entra al analisis): ademas de 2, 4 y 5,
    1. epocas perdidas > MAX_PCT_EPOCAS_PERDIDAS % de EPOCAS_ESPERADAS (80): las que faltan desde la
       segmentacion, las de los autores, las groseras y las del rechazo final cuentan como perdidas
    3. menos de MIN_EPOCAS_LIMPIAS epocas en algun nivel de NIVELES_MINIMOS (E1, E4)

Lee:
    derived_mine/preprocesado_resumen.csv   (limpieza.py)
    sub-XXX_preprocesadas-epo.fif            (canales EEG y VEO)
    etapa final: las epocas despues del rechazo final, en la ruta que devuelve ruta_finales(sub)
Escribe derived_mine/inclusion_<etapa>.csv: una fila por sujeto con el valor de cada criterio, si pasa,
y el motivo de exclusion.

Lo llama pipeline.py en la etapa de grupo (python -m procesamiento.pipeline --grupo).
"""
from pathlib import Path

import mne
import pandas as pd

from procesamiento import config, limpieza

EPOCAS_ESPERADAS = getattr(config, "EPOCAS_ESPERADAS", 4 * config.PULSOS_POR_NIVEL)
MAX_PCT_EPOCAS_PERDIDAS = getattr(config, "MAX_PCT_EPOCAS_PERDIDAS", 25)
MAX_PCT_CANALES_INTERPOLADOS = getattr(config, "MAX_PCT_CANALES_INTERPOLADOS", 25)
MAX_ROI_INTERPOLADOS = getattr(config, "MAX_ROI_INTERPOLADOS", 2)
CANALES_ROI = getattr(config, "CANALES_ROI", ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"])


def ruta_finales(sub, carpeta=None):
    """Epocas despues del rechazo final (las va a generar el paso de rechazo, todavia no definido)."""
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    return carpeta / sub / f"{sub}_finales-epo.fif"


def _resumen(carpeta):
    ruta = carpeta / "preprocesado_resumen.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No esta {ruta}. Corre antes: python -m procesamiento.pipeline")
    return pd.read_csv(ruta, dtype={"interpolados_autores": str, "interpolados_propios": str}).fillna("")


def criterios_previos(fila, ep):
    """Criterios 2, 4 y 5. `fila`: la del sujeto en preprocesado_resumen.csv; `ep`: sus epocas preprocesadas."""
    interp = list(dict.fromkeys(fila["interpolados_autores"].split() + fila["interpolados_propios"].split()))
    n_eeg = len(mne.pick_types(ep.info, eeg=True))
    roi_interp = [c for c in interp if c in CANALES_ROI]
    pct_interp = round(100 * len(interp) / n_eeg, 1)
    out = {
        "c0_completo_ok": int(fila["segmentadas"]) == EPOCAS_ESPERADAS,
        "canales_interpolados": len(interp),
        "pct_canales_interpolados": pct_interp,
        "c2_canales_ok": pct_interp <= MAX_PCT_CANALES_INTERPOLADOS,
        "roi_interpolados": " ".join(roi_interp),
        "c4_roi_ok": (config.CANAL_QC not in roi_interp) and len(roi_interp) <= MAX_ROI_INTERPOLADOS,
        "c5_veo_ok": config.CANAL_EOG in ep.ch_names,
    }
    motivos = []
    if not out["c2_canales_ok"]:
        motivos.append(f"{pct_interp}% de canales interpolados (> {MAX_PCT_CANALES_INTERPOLADOS}%)")
    if not out["c4_roi_ok"]:
        motivos.append(f"ROI interpolada: {' '.join(roi_interp)}")
    if not out["c5_veo_ok"]:
        motivos.append("sin canal VEO")
    if not out["c0_completo_ok"]:
        motivos.append(f"{int(fila['segmentadas'])} epocas segmentadas (se esperaban {EPOCAS_ESPERADAS})")
    return out, motivos


def criterios_finales(finales):
    """Criterios 1 y 3 a partir de las epocas que quedan despues del rechazo final."""
    perdidas = EPOCAS_ESPERADAS - len(finales)
    pct = round(100 * perdidas / EPOCAS_ESPERADAS, 1)
    por_nivel = finales.metadata["condicion"].value_counts()
    minimos = {c: int(por_nivel.get(c, 0)) for c in config.NIVELES_MINIMOS}
    out = {"epocas_finales": len(finales), "epocas_perdidas": perdidas, "pct_epocas_perdidas": pct,
           "c1_epocas_ok": pct <= MAX_PCT_EPOCAS_PERDIDAS,
           **{f"finales_{c}": int(por_nivel.get(c, 0)) for c in config.NIVELES},
           "c3_niveles_ok": all(n >= config.MIN_EPOCAS_LIMPIAS for n in minimos.values())}
    motivos = []
    if not out["c1_epocas_ok"]:
        motivos.append(f"{pct}% de epocas perdidas (> {MAX_PCT_EPOCAS_PERDIDAS}% de {EPOCAS_ESPERADAS})")
    if not out["c3_niveles_ok"]:
        faltan = ", ".join(f"{c}: {n}" for c, n in minimos.items() if n < config.MIN_EPOCAS_LIMPIAS)
        motivos.append(f"menos de {config.MIN_EPOCAS_LIMPIAS} epocas en {faltan}")
    return out, motivos


def evaluar(etapa="previa", sujetos=None, carpeta=None):
    """DataFrame con una fila por sujeto. etapa: 'previa' (2, 4, 5) o 'final' (1 a 5)."""
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    res = _resumen(carpeta).set_index("sujeto")
    sujetos = sujetos or list(res.index)
    filas = []
    for sub in sujetos:
        fila = {"sujeto": sub}
        if sub not in res.index or not limpieza.ruta_preprocesadas(sub, carpeta).exists():
            filas.append({**fila, "incluido": False, "motivo": "sin preprocesar"})
            continue
        ep = mne.read_epochs(limpieza.ruta_preprocesadas(sub, carpeta), preload=False, verbose="ERROR")
        r = res.loc[sub]
        fila["segmentadas"] = int(r["segmentadas"])
        valores, motivos = criterios_previos(r, ep)
        fila.update(valores)
        if etapa == "final":
            ruta = ruta_finales(sub, carpeta)
            if not ruta.exists():
                motivos.append("sin epocas del rechazo final")
            else:
                v, m = criterios_finales(mne.read_epochs(ruta, preload=False, verbose="ERROR"))
                fila.update(v)
                motivos += m
        fila["incluido"] = not motivos
        fila["motivo"] = "; ".join(motivos)
        filas.append(fila)
    return pd.DataFrame(filas)


def incluidos(etapa="previa", carpeta=None):
    """Lista de sujetos incluidos segun el ultimo inclusion_<etapa>.csv guardado."""
    ruta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / f"inclusion_{etapa}.csv"
    t = pd.read_csv(ruta)
    return t[t["incluido"]]["sujeto"].tolist()
