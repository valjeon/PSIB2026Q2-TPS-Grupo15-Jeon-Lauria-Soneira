"""Segmentacion: de registros crudos a epocas de pulsos laser (-1 a +2 s).

Flujo por sujeto (una sesion a la vez para no llenar la memoria):

    por cada sesion:  cargar -> pulsos -> filtrar (continuo) -> epocar -> metadata -> liberar
    por sujeto:       concatenar epocas -> validar 80 pulsos -> guardar -epo.fif

No hace ICA ni rechazo de ensayos (eso es artefactos.py). Tampoco re-referencia:
las epocas guardadas quedan con la referencia original del registro.

Uso desde la carpeta que contiene `procesamiento/`:
    python -m extras.anterior.segmentacion sub-001
"""
import argparse
import re
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from . import config, datos, eventos, filtros


# ---------------------------------------------------------------------------
# Funciones puras (sin archivos): se prueban con arreglos sinteticos
# ---------------------------------------------------------------------------

def retardos_confirmacion(events, codigos=None, confirmacion=None):
    """Muestras entre cada trigger S3x y el S100 que lo confirma.

    Misma regla que eventos.eventos_laser, asi que devuelve un valor por pulso
    y en el mismo orden. Sirve para decidir si el tiempo cero es S3x o S100:
    si el retardo es grande o variable, conviene epocar en el S100.
    """
    codigos = config.CODIGOS_LASER if codigos is None else codigos
    confirmacion = config.CODIGO_CONFIRMACION if confirmacion is None else confirmacion
    events = np.asarray(events, dtype=int).reshape(-1, 3)
    return np.array([events[i, 0] - events[i - 1, 0]
                     for i in range(1, len(events))
                     if events[i, 2] == confirmacion and events[i - 1, 2] in codigos],
                    dtype=int)


def armar_metadata(ev_laser, sfreq, sujeto, sesion, retardos=None, first_samp=0):
    """Tabla con una fila por pulso, alineada con ev_laser.

    Columnas: sujeto, sesion, ensayo_sesion, nivel (1..4), condicion ('E1'..'E4'),
    muestra, onset_s (segundos desde el inicio del archivo), retardo_ms (S3x -> S100).

    Mano, rating y potencia del laser NO estan en los triggers: se agregan despues
    cruzando con el archivo de conducta del dataset (ver respuesta en el chat).
    """
    ev_laser = np.asarray(ev_laser, dtype=int).reshape(-1, 3)
    nombre_de = {v: k for k, v in config.NIVELES.items()}
    n = len(ev_laser)
    if retardos is None:
        retardos = np.full(n, np.nan)
    if len(retardos) != n:
        raise ValueError(f"{len(retardos)} retardos para {n} pulsos: la regla de pulsos no coincide")
    return pd.DataFrame({
        "sujeto": sujeto,
        "sesion": sesion,
        "ensayo_sesion": np.arange(n),
        "nivel": ev_laser[:, 2],
        "condicion": [nombre_de[k] for k in ev_laser[:, 2]],
        "muestra": ev_laser[:, 0],
        "onset_s": (ev_laser[:, 0] - first_samp) / sfreq,
        "retardo_ms": np.asarray(retardos, dtype=float) / sfreq * 1000.0,
    })


def nombre_sesion(vhdr):
    """'ses-3' a partir de la ruta del .vhdr."""
    return re.search(r"(ses-\d+)", Path(vhdr).as_posix()).group(1)


# ---------------------------------------------------------------------------
# Epocado
# ---------------------------------------------------------------------------

def epocas_sesion(raw, ev_laser, metadata, tmin=None, tmax=None, baseline=None):
    """Epocas de una sesion ya filtrada.

    - event_id completo (E1..E4) con on_missing='ignore': todas las sesiones quedan
      con el mismo event_id, que es lo que pide concatenate_epochs.
    - La metadata se pasa al constructor: si un pulso cae muy cerca del borde del
      archivo y MNE descarta esa epoca, la fila correspondiente se descarta con ella.
    - reject=None: aca no se rechaza nada (eso es de artefactos.py).
    """
    tmin = config.TMIN if tmin is None else tmin
    tmax = config.TMAX if tmax is None else tmax
    baseline = config.BASELINE if baseline is None else baseline
    return mne.Epochs(raw, ev_laser, event_id=config.NIVELES, tmin=tmin, tmax=tmax,
                      baseline=baseline, metadata=metadata.reset_index(drop=True),
                      reject=None, preload=True, on_missing="ignore", verbose="ERROR")


def epocas_sujeto(sub, raiz=None, filtrar=None, sesiones=None):
    """Recorre las sesiones del sujeto y devuelve un solo Epochs con todo.

    filtrar: funcion Raw -> Raw; por defecto filtros.filtrar (FIR 1-100 Hz + notch 48-52).
    sesiones: lista opcional de nombres ('ses-1', ...) para procesar solo algunas.
    """
    filtrar = filtros.filtrar if filtrar is None else filtrar
    lista = []
    cuentas = {k: 0 for k in config.NIVELES}

    for vhdr in datos.listar_sesiones(sub, raiz):
        ses = nombre_sesion(vhdr)
        if sesiones is not None and ses not in sesiones:
            continue
        raw = datos.cargar_sesion(vhdr)
        todos = eventos.eventos_de_raw(raw)
        ev = eventos.eventos_laser(todos)
        if len(ev) == 0:
            warnings.warn(f"{sub} {ses}: sin pulsos laser, se saltea")
            continue
        for k, v in eventos.contar_por_nivel(ev).items():
            cuentas[k] += v

        meta = armar_metadata(ev, raw.info["sfreq"], sub, ses,
                              retardos=retardos_confirmacion(todos), first_samp=raw.first_samp)
        raw = filtrar(raw)                      # filtrar el continuo ANTES de cortar
        ep = epocas_sesion(raw, ev, meta)
        if len(ep) < len(ev):
            warnings.warn(f"{sub} {ses}: {len(ev) - len(ep)} pulsos sin ventana completa (borde del archivo)")
        lista.append(ep)
        print(f"  {ses}: {len(ep)} epocas  {eventos.contar_por_nivel(ev)}")
        del raw                                 # liberar memoria antes de la siguiente sesion

    if not lista:
        raise RuntimeError(f"{sub}: ninguna sesion tuvo pulsos laser")

    if sesiones is None:                        # la validacion de 80 solo tiene sentido con todo
        eventos.validar_pulsos(cuentas, sujeto=sub)

    ep = lista[0] if len(lista) == 1 else mne.concatenate_epochs(lista, verbose="ERROR")
    ep.metadata = ep.metadata.assign(ensayo=np.arange(len(ep)))
    return ep


def ruta_epocas(sub, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    return carpeta / sub / f"{sub}_laser-epo.fif"


def guardar_epocas(ep, sub, carpeta=None):
    if config.SFREQ_GUARDADO and ep.info["sfreq"] > config.SFREQ_GUARDADO:
        ep = ep.copy().resample(config.SFREQ_GUARDADO, verbose="ERROR")
    ruta = ruta_epocas(sub, carpeta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ep.save(ruta, overwrite=True, verbose="ERROR")
    ep.metadata.to_csv(ruta.with_name(f"{sub}_laser-metadata.csv"), index=False)
    return ruta


def main(argv=None):
    p = argparse.ArgumentParser(description="Epocas de pulsos laser de un sujeto")
    p.add_argument("sujeto", help="ej. sub-001")
    p.add_argument("--sesiones", nargs="*", help="solo estas sesiones, ej. ses-1 ses-2")
    a = p.parse_args(argv)
    print(f"Segmentando {a.sujeto}...")
    ep = epocas_sujeto(a.sujeto, sesiones=a.sesiones)
    ruta = guardar_epocas(ep, a.sujeto)
    print(f"{len(ep)} epocas guardadas en {ruta}")
    print(ep.metadata.groupby("condicion").size().to_string())


if __name__ == "__main__":
    main()