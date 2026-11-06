"""Paso 2: extraccion de epocas, como en el paper.

Por sesion: leer (analisis.py) -> pulsos -> filtrar la senal continua (filtrado.py) -> epocas de -1000 a 2000 ms
con linea de base -1000 a 0 ms. Despues se juntan las seis sesiones, se remuestrea a 250 Hz y se guarda.
No se rechaza ni se re-referencia nada: eso es la limpieza, un paso aparte.
"""
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from procesamiento import analisis, config, filtrado


def metadata_sesion(raw, ev, sub, ses):
    """Una fila por pulso: sujeto, sesion, nivel (1-4), condicion (E1-E4) y momento del pulso en la sesion (s)."""
    nombre = {v: k for k, v in config.NIVELES.items()}
    return pd.DataFrame({"sujeto": sub, "sesion": ses, "nivel": ev[:, 2],
                         "condicion": [nombre[n] for n in ev[:, 2]],
                         "onset_s": (ev[:, 0] - raw.first_samp) / raw.info["sfreq"]})


def epocas_sesion(filt, ev, metadata):
    """Epocas de una sesion ya filtrada: de TMIN a TMAX, con linea de base BASELINE."""
    return mne.Epochs(filt, ev, event_id=config.NIVELES, tmin=config.TMIN, tmax=config.TMAX,
                      baseline=config.BASELINE, metadata=metadata, reject=None, preload=True,
                      on_missing="ignore", verbose="ERROR")


def extraer_sujeto(sub, raiz=None):
    """Todas las sesiones del sujeto -> un solo Epochs (1-100 Hz, notch, referencia original, 250 Hz)."""
    lista = []
    for vhdr in analisis.listar_sesiones(sub, raiz):
        filt, ev, ses, raw = filtrado.leer_y_filtrar(vhdr)
        if len(ev) == 0:
            print(f"  {sub} {ses}: sin pulsos laser, se saltea")
            continue
        ep = epocas_sesion(filt, ev, metadata_sesion(raw, ev, sub, ses))
        if len(ep) < len(ev):
            print(f"  {sub} {ses}: {len(ev) - len(ep)} pulsos sin ventana completa (borde del archivo)")
        lista.append(ep)
        del raw, filt
    if not lista:
        raise RuntimeError(f"{sub}: ninguna sesion tuvo pulsos laser")
    ep = lista[0] if len(lista) == 1 else mne.concatenate_epochs(lista, verbose="ERROR")
    with mne.use_log_level("ERROR"):
        ep.metadata = ep.metadata.reset_index(drop=True).assign(ensayo=np.arange(len(ep)))
    if config.SFREQ_GUARDADO and ep.info["sfreq"] > config.SFREQ_GUARDADO:
        # remuestrear corre un poco la media del pre-estimulo: se vuelve a aplicar la linea de base
        ep = ep.resample(config.SFREQ_GUARDADO, verbose="ERROR").apply_baseline(config.BASELINE, verbose="ERROR")
    return ep


def ruta_epocas(sub, carpeta=None):
    """derived_mine/sub-XXX/sub-XXX_laser-epo.fif (el mismo nombre que usaba el pipeline anterior)."""
    return Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / sub / f"{sub}_laser-epo.fif"


def guardar(ep, sub, carpeta=None):
    """Guarda las epocas y su metadata (CSV, para mirarla sin abrir MNE). Devuelve la ruta del .fif."""
    ruta = ruta_epocas(sub, carpeta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ep.save(ruta, overwrite=True, verbose="ERROR")
    ep.metadata.to_csv(ruta.with_name(f"{sub}_laser-metadata.csv"), index=False)
    return ruta


def resumen(ep, sub):
    """Una fila con lo que hay que mirar: sesiones, epocas totales y por nivel."""
    por_nivel = ep.metadata["condicion"].value_counts()
    return {"sujeto": sub, "sesiones": ep.metadata["sesion"].nunique(), "epocas": len(ep),
            **{c: int(por_nivel.get(c, 0)) for c in config.NIVELES},
            "completo": len(ep) == config.EPOCAS_ESPERADAS}


def guardar_resumen(filas, nombre="epocas_resumen.csv", carpeta=None):
    """Escribe derived_mine/<nombre> con una fila por sujeto, conservando los de corridas anteriores."""
    destino = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / nombre
    nuevo = pd.DataFrame(filas)
    if destino.exists():
        viejo = pd.read_csv(destino)
        nuevo = pd.concat([viejo[~viejo["sujeto"].isin(nuevo["sujeto"])], nuevo]).sort_values("sujeto")
    nuevo.to_csv(destino, index=False)
    return destino
