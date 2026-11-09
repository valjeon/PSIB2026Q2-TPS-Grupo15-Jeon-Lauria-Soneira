"""Criterios de inclusion de sujetos, definidos en config.py.

Etapa 'previa' (despues de limpieza.py; decide quien entra al localizador):
    - No tener las 80 épocas esperadas después de la segmentación.
    - Más de MAX_PCT_CANALES_INTERPOLADOS % de canales de EEG interpolados.
    - Cz interpolado o más de MAX_ROI_INTERPOLADOS canales de la ROI interpolados.
    - Sin canal VEO
Etapa 'final' (despues del rechazo final de medicion.py, decide quien entra al analisis), ademas de las anteriores:
    - Más de MAX_PCT_EPOCAS_PERDIDAS % de épocas perdidas
    - Menos de MIN_EPOCAS_LIMPIAS épocas en E1 o en E4.

Un sujeto queda excluído si falla cualquier criterio. El resultado va a inclusion_<etapa>.csv.
"""
from pathlib import Path

import mne
import pandas as pd

from procesamiento import config, limpieza

EPOCAS_ESPERADAS = config.EPOCAS_ESPERADAS
MAX_PCT_EPOCAS_PERDIDAS = config.MAX_PCT_EPOCAS_PERDIDAS
MAX_PCT_CANALES_INTERPOLADOS = config.MAX_PCT_CANALES_INTERPOLADOS
MAX_ROI_INTERPOLADOS = config.MAX_ROI_INTERPOLADOS
CANALES_ROI = config.CANALES_ROI


def ruta_finales(sub, carpeta=None):
    """Ruta de las épocas que quedan después del rechazo final, guardadas por medicion.py."""
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    return carpeta / sub / f"{sub}_finales-epo.fif"


def _resumen(carpeta):
    """Lee preprocesado_resumen.csv (limpieza.py).
    Saca información de los canales interpolados y las épocas segmentadas de cada objeto."""
    ruta = carpeta / "preprocesado_resumen.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No esta {ruta}. Corre antes: python -m procesamiento.pipeline")
    return pd.read_csv(ruta, dtype={"interpolados_autores": str, "interpolados_propios": str}).fillna("") # dtype transforma en str los canales


def criterios_previos(fila, ep):
    """Toma las épocas que quedaron después del rechazo final y devuelve los valores de cada criterio (los finales)
    y los motivos de exclusión (en caso de que se excluya)."""
    # canales interpolados totales (autores+propios)
    interp = list(dict.fromkeys(fila["interpolados_autores"].split() + fila["interpolados_propios"].split()))
    n_eeg = len(mne.pick_types(ep.info, eeg=True))      # cantidad de canales EEG por sujeto
    pct_interp = round(100 * len(interp) / n_eeg, 1)    # % de canales interpolados
    roi_interp = [c for c in interp if c in CANALES_ROI]    # interpolados en la ROI
    segmentadas = int(fila["segmentadas"])      # segmentadas antes de descartar

    ok_c1 = segmentadas == EPOCAS_ESPERADAS
    ok_c2 = pct_interp <= MAX_PCT_CANALES_INTERPOLADOS      # no más del 25% de los canales interpolados
    ok_c3 = config.CANAL_QC not in roi_interp and len(roi_interp) <= MAX_ROI_INTERPOLADOS   # debe estar Cz y no debe tener más de 2 canales de la ROI interpolados
    ok_c4 = config.CANAL_EOG in ep.ch_names

    out = { "c1_completo_ok": ok_c1,
        "canales_interpolados": len(interp),
        "pct_canales_interpolados": pct_interp,
        "c2_canales_ok": ok_c2,
        "roi_interpolados": " ".join(roi_interp),
        "c3_roi_ok": ok_c3,
        "c4_veo_ok": ok_c4}
    
    motivos = []
    if not ok_c1:
        motivos.append(f"{segmentadas} epocas segmentadas (se esperaban {EPOCAS_ESPERADAS})")
    if not ok_c2:
        motivos.append(f"{pct_interp}% de canales interpolados (> {MAX_PCT_CANALES_INTERPOLADOS}%)")
    if not ok_c3:
        motivos.append(f"ROI interpolada: {' '.join(roi_interp)}")
    if not ok_c4:
        motivos.append("sin canal VEO")
    return out, motivos


def criterios_finales(finales):
    """Criterios 1 y 3 a partir de las epocas que quedan despues del rechazo final."""
    n_finales = len(finales)        # épocas que quedaron después del rechazo
    perdidas = EPOCAS_ESPERADAS - n_finales 
    pct = round(100 * perdidas / EPOCAS_ESPERADAS, 1)

    conteo = finales.metadata["condicion"].value_counts()   # épocas que quedan en cada nivel
    por_nivel = {c: int(conteo.get(c, 0)) for c in config.NIVELES}      
    minimos = {c: por_nivel[c] for c in config.NIVELES_MINIMOS}     # toma del conteo solo las épocas de E1 y E4

    ok_c5 = pct <= MAX_PCT_EPOCAS_PERDIDAS          # no perdió más del 25% del total de las épocas
    ok_c6 = all(n >= config.MIN_EPOCAS_LIMPIAS for n in minimos.values())       # al menos 10 épocas en E1 y en E4
    
    out = {"epocas_finales": n_finales, "epocas_perdidas": perdidas, "pct_epocas_perdidas": pct,
           "c5_epocas_ok": ok_c5,
           **{f"finales_{c}": n for c, n in por_nivel.items()},
           "c6_niveles_ok": ok_c6}
    
    motivos = []
    if not ok_c5:
        motivos.append(f"{pct}% de epocas perdidas (> {MAX_PCT_EPOCAS_PERDIDAS}% de {EPOCAS_ESPERADAS})")
    if not ok_c6:
        faltan = ", ".join(f"{c}: {n}" for c, n in minimos.items() if n < config.MIN_EPOCAS_LIMPIAS)
        motivos.append(f"menos de {config.MIN_EPOCAS_LIMPIAS} epocas en {faltan}")
    return out, motivos


def evaluar(etapa="previa", sujetos=None, carpeta=None):
    """Aplica los criterios a cada sujeto.
    Devuelve un DataFrame con una fila por sujeto, con lo medido en cada criterio, 'incluido' y 'motivo'."""
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    resumen = _resumen(carpeta).set_index("sujeto")     # toma preprocesado_resumen.csv
    sujetos = sujetos or list(resumen.index)

    filas = []
    for sub in sujetos:
        fila = {"sujeto": sub}
        ruta_pre = limpieza.ruta_preprocesadas(sub, carpeta)
        if sub not in resumen.index or not ruta_pre.exists():       # si no pasó por limpieza.py no se puede evaluar
            filas.append({**fila, "incluido": False, "motivo": "sin preprocesar"})
            continue

        # criterios previos --> canales interpolados, épocas segmentadas y VEO
        ep = mne.read_epochs(ruta_pre, preload=False, verbose="ERROR")
        datos_sub = resumen.loc[sub]        # fila de preprocesado_resumen.csv del sujeto
        fila["segmentadas"] = int(datos_sub["segmentadas"])
        valores, motivos = criterios_previos(datos_sub, ep)
        fila.update(valores)        # agrega las columnas de los criterios

        # criterios finales --> épocas que quedaron después del rechazo final de medicion.py
        if etapa == "final":
            ruta_fin = ruta_finales(sub, carpeta)
            if not ruta_fin.exists():
                motivos.append("sin epocas del rechazo final")
            else:
                finales = mne.read_epochs(ruta_fin, preload=False, verbose="ERROR")
                valores_fin, motivos_fin = criterios_finales(finales)
                fila.update(valores_fin)
                motivos += motivos_fin      # se suman a los motivos de la etapa previa

        fila["incluido"] = not motivos      # ningun motivo --> sujeto inlcuído
        fila["motivo"] = "; ".join(motivos)
        filas.append(fila)
    return pd.DataFrame(filas)


def incluidos(etapa="previa", carpeta=None):
    """Lista de sujetos incluidos segun el ultimo inclusion_<etapa>.csv guardado."""
    ruta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / f"inclusion_{etapa}.csv"
    tabla = pd.read_csv(ruta)       # una fila por sujeto (armada por evaluar)
    return tabla.loc[tabla["incluido"], "sujeto"].tolist()      # devuelve las filas con incluido == True
