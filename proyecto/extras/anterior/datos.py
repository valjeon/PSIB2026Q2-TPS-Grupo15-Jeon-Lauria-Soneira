"""Datos: ubicar y leer los registros crudos (BrainVision) de ds005293.

No filtra ni epoca: eso es de filtros.py y segmentacion.py. Aca solo se lee,
se sacan los canales descartados y se deja listo el canal de ojo y el montaje.
"""
import re
from pathlib import Path

import mne
from extras.anterior import config

def listar_sujetos(raiz=None):
    """IDs de sujetos que tienen al menos un registro crudo descargado, ej. ['sub-001', 'sub-004']."""
    raiz = Path(config.RAIZ_DATOS if raiz is None else raiz)
    subs = sorted(p.name for p in raiz.glob("sub-*") if p.is_dir() and list(p.glob("ses-*/eeg/*_eeg.vhdr")))
    return subs


def listar_sesiones(sub, raiz=None):
    """Rutas .vhdr de las sesiones de un sujeto, ordenadas por numero de sesion (ses-1..ses-6)."""
    raiz = Path(config.RAIZ_DATOS if raiz is None else raiz)

    def numero(p):
        return int(re.search(r"ses-(\d+)", p.as_posix()).group(1))

    vhdrs = sorted(raiz.glob(f"{sub}/ses-*/eeg/*_eeg.vhdr"), key=numero)
    if not vhdrs:
        raise FileNotFoundError(
            f"No hay registros de {sub} en {raiz}. Bajalos con descargar_sujeto "
            f"(ej. bash descargar_sujeto.sh {sub}).")
    return vhdrs


def cargar_sesion(vhdr, descartar=None):
    """Lee una sesion cruda: sin filtrar, sin remuestrear.

    - saca los canales descartados (TP9, TP10, HEOR)
    - marca VEO como canal de ojo (EOG)
    - asigna posiciones estandar 10-05 (para mapas topograficos)
    """
    descartar = config.CANALES_DESCARTADOS if descartar is None else descartar
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    raw.drop_channels([c for c in descartar if c in raw.ch_names])
    if config.CANAL_EOG in raw.ch_names:
        raw.set_channel_types({config.CANAL_EOG: "eog"})
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    return raw


def cargar_sujeto(sub, raiz=None):
    """Todas las sesiones de un sujeto: devuelve (lista de Raw, lista de nombres 'ses-1'...)."""
    vhdrs = listar_sesiones(sub, raiz)
    raws = [cargar_sesion(v) for v in vhdrs]
    nombres = [re.search(r"(ses-\d+)", v.as_posix()).group(1) for v in vhdrs]
    return raws, nombres
