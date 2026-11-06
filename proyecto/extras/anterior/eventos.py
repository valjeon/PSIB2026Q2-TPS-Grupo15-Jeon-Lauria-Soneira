"""Eventos: identificar los pulsos laser en los triggers del registro.

Funciones puras sobre arreglos de MNE (muestra, 0, codigo): no leen archivos,
asi se prueban sin datos reales.
"""
import warnings
import mne
import numpy as np
from extras.anterior import config

def eventos_laser(events, codigos=None, confirmacion=None):
    """Devuelve [[muestra, 0, nivel]] para cada trigger S3x seguido inmediatamente de S100.

    events: arreglo (n, 3) de MNE, ordenado por muestra.
    La muestra devuelta es la del trigger S3x (no la del S100).
    """
    codigos = config.CODIGOS_LASER if codigos is None else codigos
    confirmacion = config.CODIGO_CONFIRMACION if confirmacion is None else confirmacion
    events = np.asarray(events, dtype=int).reshape(-1, 3)
    out = [[events[i - 1, 0], 0, codigos[events[i - 1, 2]]]
           for i in range(1, len(events))
           if events[i, 2] == confirmacion and events[i - 1, 2] in codigos]
    return np.array(out, dtype=int).reshape(-1, 3)


def eventos_de_raw(raw):
    """Todos los triggers del Raw como arreglo (n, 3), usando los codigos numericos de BrainVision."""
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    return events


def pulsos_laser(raw):
    """Pulsos laser de un Raw: [[muestra, 0, nivel]]."""
    return eventos_laser(eventos_de_raw(raw))


def contar_por_nivel(eventos):
    """{'E1': n, ..., 'E4': n} a partir de un arreglo de eventos laser."""
    niveles = np.asarray(eventos, dtype=int).reshape(-1, 3)[:, 2]
    return {nombre: int((niveles == k).sum()) for nombre, k in config.NIVELES.items()}


def validar_pulsos(cuentas, esperado=None, sujeto=""):
    """Avisa (sin cortar) si los conteos totales por nivel no son los del dataset (20 por nivel).

    cuentas: dict {'E1': n, ...} sumado sobre todas las sesiones del sujeto.
    Devuelve True si coincide con lo esperado.
    """
    esperado = config.PULSOS_POR_NIVEL if esperado is None else esperado
    mal = {k: v for k, v in cuentas.items() if v != esperado}
    if mal:
        warnings.warn(f"{sujeto or 'sujeto'}: se esperaban {esperado} pulsos por nivel y hay {cuentas}")
    return not mal
