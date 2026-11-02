"""Marcas manuales de los autores (CSV 'all experiments_preprocessing.csv'): epocas y electrodos malos.

Los autores borraron a mano ensayos con artefactos grandes e interpolaron electrodos malos (paso
"Manually deleting bad trials and interpolating bad channels" de code/preprocessing_95.m); el CSV es el
registro de eso. Este modulo:
    leer_marcas        lee el CSV para TODOS los sujetos de un experimento -> {sub: Marcas}
    aplicar_marcas     interpola los electrodos malos y descarta las epocas malas de un sujeto
    verificar_orden    controla que los indices de epoca de los autores coincidan con los nuestros
La columna 'bad ica' no se usa (requiere la ICA de los autores; el trabajo es sin ICA).

Uso desde proyecto/ (solo imprime, no guarda nada):
    python -m procesamiento.marcas_autores sub-010 sub-050
    python -m procesamiento.marcas_autores            (todos los sujetos con epocas ya segmentadas)
"""
import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

from procesamiento import config, segmentacion

# El CSV esta en la raiz del repo (al lado de data/)
RUTA_CSV = Path(config.RAIZ_DATOS).parent.parent / "all experiments_preprocessing.csv"
EXPERIMENTO = "Experiment 6"        # ds005293 = Experiment 6_95 By BP
DESPLAZAMIENTOS = range(-3, 4)      # corrimientos de indice que se prueban en verificar_orden


@dataclass
class Marcas:
    epocas: set = field(default_factory=set)        # 0-based (los autores numeran 1..80)
    electrodos: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lectura del CSV
# ---------------------------------------------------------------------------

def _lista(texto):
    """'33 70 72' -> ['33', '70', '72'];  '\\' o vacio -> []."""
    return [x for x in texto.replace(",", " ").split() if x != "\\"]

_CANONICOS = None

def _normalizar(nombre):
    """'Tp7' -> 'TP7': el CSV no respeta las mayusculas del montaje 10-05."""
    global _CANONICOS
    if _CANONICOS is None:
        _CANONICOS = {c.lower(): c for c in mne.channels.make_standard_montage("standard_1005").ch_names}
    return _CANONICOS.get(nombre.lower(), nombre)

def leer_marcas(ruta_csv=None, experimento=EXPERIMENTO):
    """{'sub-010': Marcas(...), ...} para todos los sujetos del experimento.

    Las columnas se buscan por nombre en la fila 2 (no por posicion): en el CSV no todos los
    experimentos tienen el mismo orden (el 8 tiene 'bad electrode' antes que 'bad epoch').
    """
    ruta = Path(RUTA_CSV if ruta_csv is None else ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No esta el CSV de los autores: {ruta}")
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.reader(f))
    titulos, nombres = filas[0], [c.strip().lower() for c in filas[1]]

    inicio = next((i for i, t in enumerate(titulos) if t.strip().startswith(experimento)), None)
    if inicio is None:
        raise ValueError(f"No encuentro '{experimento}' en la primera fila del CSV")
    fin = next((i for i in range(inicio + 1, len(titulos)) if titulos[i].strip()), len(titulos))
    col = {nombres[i]: i for i in range(inicio, fin)}
    c_sub, c_ep, c_el = col["sub_id"], col["bad epoch"], col["bad electrode"]

    marcas = {}
    for fila in filas[2:]:
        if len(fila) <= max(c_sub, c_ep, c_el) or not fila[c_sub].strip().isdigit():
            continue
        sub = f"sub-{int(fila[c_sub]):03d}"
        marcas[sub] = Marcas(epocas={int(x) - 1 for x in _lista(fila[c_ep]) if x.isdigit()},
                             electrodos=[_normalizar(c) for c in _lista(fila[c_el])])
    return marcas


_CACHE = {}

def marcas_de(sub, ruta_csv=None):
    """Marcas de un sujeto (vacias si no esta en el CSV). El CSV se lee una sola vez."""
    if ruta_csv not in _CACHE:
        _CACHE[ruta_csv] = leer_marcas(ruta_csv)
    return _CACHE[ruta_csv].get(sub, Marcas())

# ---------------------------------------------------------------------------
# Verificacion del orden de las epocas
# ---------------------------------------------------------------------------

def pico_a_pico_por_epoca(ep, excluir=()):
    """Maximo pico a pico (uV) entre canales EEG por epoca, pasa-bajos 30 Hz, epoca completa.

    Sin referencia promedio: los electrodos malos todavia no estan interpolados.
    """
    from procesamiento import artefactos
    v = artefactos._pasa_bajos(ep)
    picks = [c for c in v.copy().pick("eeg").ch_names if c not in set(excluir)]
    x = v.get_data(picks=picks) * 1e6
    return (x.max(axis=2) - x.min(axis=2)).max(axis=1)


def verificar_orden(ep, marcas, desplazamientos=DESPLAZAMIENTOS):
    """Percentil medio (0-100) del pico a pico de las epocas marcadas, para cada corrimiento de indice.

    Si el orden coincide, con corrimiento 0 las epocas marcadas deberian ser las mas grandes
    (percentil alto) y con otros corrimientos el percentil deberia caer. Devuelve {corrimiento: percentil}.
    """
    ptp = pico_a_pico_por_epoca(ep, list(marcas.electrodos) + list(config.CANALES_SIN_CRITERIO_EEG))
    rango = ptp.argsort().argsort() / (len(ptp) - 1) * 100       # percentil de cada epoca
    out = {}
    for d in desplazamientos:
        idx = [i + d for i in marcas.epocas if 0 <= i + d < len(ptp)]
        out[d] = float(np.mean(rango[idx])) if idx else np.nan
    return out, ptp


# ---------------------------------------------------------------------------
# Aplicacion
# ---------------------------------------------------------------------------

def aplicar_marcas(ep, marcas, n_esperado=None):
    """Copia de `ep` con los electrodos malos interpolados y las epocas malas descartadas.

    Las epocas se descartan por la columna 'ensayo' de la metadata (indice global 0..79). Si el sujeto
    no tiene n_esperado epocas, no se descarta ninguna (el orden no es confiable) y se avisa.
    Interpolar va ANTES de la referencia promedio: si no, el canal malo contamina a todos.
    """
    n_esperado = 4 * config.PULSOS_POR_NIVEL if n_esperado is None else n_esperado
    out = ep.copy()

    malos = [c for c in marcas.electrodos if c in out.ch_names]
    faltan = [c for c in marcas.electrodos if c not in out.ch_names]
    if faltan:
        print(f"  aviso: electrodos marcados que no estan en el registro: {faltan}")
    if malos:
        out.info["bads"] = sorted(set(out.info["bads"]) | set(malos))
        out.interpolate_bads(reset_bads=True, verbose="ERROR")

    if marcas.epocas:
        if len(out) != n_esperado:
            print(f"  aviso: {len(out)} epocas (se esperaban {n_esperado}): NO se descartan las marcadas")
        else:
            quitar = np.flatnonzero(out.metadata["ensayo"].isin(marcas.epocas).to_numpy())
            out.drop(quitar, reason="autores", verbose="ERROR")
    return out


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

def sujetos_segmentados(carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    return sorted(p.name for p in carpeta.glob("sub-*") if segmentacion.ruta_epocas(p.name, carpeta).exists())


def main(argv=None):
    p = argparse.ArgumentParser(description="Marcas de los autores: verificar el orden de las epocas")
    p.add_argument("sujetos", nargs="*", help="por defecto, todos los que tienen epocas segmentadas")
    p.add_argument("--csv", default=None, help=f"CSV de los autores (por defecto {RUTA_CSV.name} en la raiz)")
    a = p.parse_args(argv)

    marcas = leer_marcas(a.csv)
    for sub in a.sujetos or sujetos_segmentados():
        m = marcas.get(sub)
        print(f"\n=== {sub} ===")
        if m is None:
            print("  no esta en el CSV de los autores")
            continue
        print(f"  epocas malas (1-based): {sorted(i + 1 for i in m.epocas) or 'ninguna'}")
        print(f"  electrodos malos: {m.electrodos or 'ninguno'}")
        ruta = segmentacion.ruta_epocas(sub)
        if not ruta.exists():
            print(f"  no estan las epocas ({ruta}). Corre antes la segmentacion.")
            continue
        ep = mne.read_epochs(ruta, verbose="ERROR")
        print(f"  epocas propias: {len(ep)}")
        if not m.epocas:
            continue
        perc, ptp = verificar_orden(ep, m)
        print("  percentil medio del pico a pico de las marcadas, segun corrimiento de indice:")
        print("   " + "  ".join(f"{d:+d}: {v:5.1f}" for d, v in perc.items()))
        mejor = max(perc, key=lambda d: -np.inf if np.isnan(perc[d]) else perc[d])
        veredicto = "el orden coincide" if mejor == 0 else f"OJO: el maximo esta en {mejor:+d}"
        print(f"  -> {veredicto}  (mediana de todas: {np.median(ptp):.0f} uV; "
              f"marcadas: {np.round(ptp[sorted(m.epocas)]).astype(int).tolist()} uV)")


if __name__ == "__main__":
    main()