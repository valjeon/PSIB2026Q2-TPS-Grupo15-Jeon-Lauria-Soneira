"""Paso 3: limpieza de las epocas, sin ICA (lo pide la consigna) y sin mirar el nivel de intensidad.

Ocupa el lugar de preprocesado.py (csoneira): deja las epocas listas para la etapa de grupo, que hace el
rechazo final por ROI y VEO (medicion.py, con sus variantes de sensibilidad). En orden (limpiar_sujeto):
    1. aplicar_marcas    marcas manuales de los autores: saca sus epocas malas y marca sus electrodos malos
    2. canales_malos     canales planos o ruidosos en el pre-estimulo que los autores no marcaron
    3. referenciar       interpola solo los malos de la ROI; referencia promedio sin frontopolares ni malos
    4. epocas_groseras   pico a pico > 300 uV en la cabeza (sin frontopolares ni malos), de -1 a 1 s
Salida: derived_mine/sub-XXX/sub-XXX_preprocesadas-epo.fif y una fila por sujeto en
derived_mine/preprocesado_resumen.csv (los mismos nombres que usaba preprocesado.py).
"""
import csv
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from procesamiento import config, epocas

# CSV de los autores, en la raiz del repo (junto a la carpeta data/)
RUTA_CSV = Path(config.RAIZ_DATOS).parent.parent / "all experiments_preprocessing.csv"
EXPERIMENTO = "Experiment 6"          # ds005293 = "Experiment 6_95 By BP" en la primera fila del CSV


def _lista(texto):
    """'33 70 72' -> ['33', '70', '72'];  '\\' o vacio -> []."""
    return [x for x in texto.replace(",", " ").split() if x != "\\"]


def _nombres_10_05():
    """Nombres de electrodos del sistema 10-05 como los escribe MNE (colin27_1005 desde MNE 1.13; antes standard_1005)."""
    try:
        return mne.channels.make_standard_montage("colin27_1005").ch_names
    except ValueError:
        return mne.channels.make_standard_montage("standard_1005").ch_names


def leer_marcas(ruta_csv=None, experimento=EXPERIMENTO):
    """Marcas de los autores: {'sub-004': (epocas malas, [electrodos malos]), ...}.

    Las epocas vienen numeradas desde 1 en el CSV y se devuelven desde 0, como el 'ensayo' de la metadata.
    Los nombres de electrodos se pasan a la forma de MNE ('Tp7' -> 'TP7').
    """
    ruta = Path(RUTA_CSV if ruta_csv is None else ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No esta el CSV de los autores: {ruta}")
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.reader(f))
    # fila 0: titulos de experimento (uno cada varias columnas); fila 1: nombres de columna
    titulos, nombres = filas[0], [c.strip().lower() for c in filas[1]]
    inicio = next((i for i, t in enumerate(titulos) if t.strip().startswith(experimento)), None)
    if inicio is None:
        raise ValueError(f"No encuentro '{experimento}' en la primera fila del CSV")
    fin = next((i for i in range(inicio + 1, len(titulos)) if titulos[i].strip()), len(titulos))
    col = {nombres[i]: i for i in range(inicio, fin)}
    c_sub, c_ep, c_el = col["sub_id"], col["bad epoch"], col["bad electrode"]
    canonico = {c.lower(): c for c in _nombres_10_05()}
    marcas = {}
    for fila in filas[2:]:
        if len(fila) <= max(c_sub, c_ep, c_el) or not fila[c_sub].strip().isdigit():
            continue
        epocas_malas = {int(x) - 1 for x in _lista(fila[c_ep]) if x.isdigit()}
        electrodos = [canonico.get(c.lower(), c) for c in _lista(fila[c_el])]
        marcas[f"sub-{int(fila[c_sub]):03d}"] = (epocas_malas, electrodos)
    return marcas


def aplicar_marcas(ep, sub, marcas):
    """Saca las epocas malas de los autores y marca sus electrodos malos (todavia no los interpola).

    Las epocas se sacan solo si el sujeto tiene las EPOCAS_ESPERADAS (80): si falta alguna, la numeracion de
    los autores ya no coincide con la nuestra y se sacarian las equivocadas. Los electrodos quedan en
    info['bads'] para interpolarlos despues junto con los canales malos propios.
    Devuelve (epocas, registro) con lo que se hizo.
    """
    ep = ep.copy()                                     # no tocar las epocas que recibe
    ep_malas, electrodos = marcas.get(sub, (set(), []))
    electrodos = [c for c in electrodos if c in ep.ch_names]
    registro = {"epocas_autores": [], "electrodos_autores": electrodos, "aviso": ""}
    if ep_malas and len(ep) == config.EPOCAS_ESPERADAS:
        quitar = ep.metadata["ensayo"].isin(ep_malas).to_numpy()
        registro["epocas_autores"] = sorted(ep.metadata.loc[quitar, "ensayo"].tolist())
        ep = ep[~quitar]
    elif ep_malas:
        registro["aviso"] = (f"{len(ep)} epocas (se esperaban {config.EPOCAS_ESPERADAS}): "
                             "no se sacan las epocas de los autores")
    ep.info["bads"] = list(dict.fromkeys(ep.info["bads"] + electrodos))
    return ep, registro


def _z_robusto(x, eje):
    """(x - mediana) / (1.4826 * MAD) a lo largo de `eje`: como un z, pero sin que lo arrastren los atipicos."""
    med = np.median(x, axis=eje, keepdims=True)
    mad = np.median(np.abs(x - med), axis=eje, keepdims=True) * 1.4826
    return (x - med) / np.where(mad > 0, mad, np.inf)


def canales_malos(ep):
    """Canales malos propios: los que no marcaron los autores. Devuelve (lista de malos, tabla por canal).

    Se mira el pico a pico de cada canal en el pre-estimulo (-1 a 0 s), epoca por epoca, con la referencia original
    y sin los frontopolares (su amplitud es ocular) ni los ya marcados en info['bads']. Un canal es malo si:
      - plano: su pico a pico tipico es menor que FACTOR_CANAL_PLANO x la mediana de los canales, o
      - ruidoso: su pico a pico tipico tiene z robusto > Z_CANAL_MALO entre canales, y es atipico (z > Z_CANAL_MALO
        dentro de su epoca) en mas de FRAC_EPOCAS_CANAL_MALO de las epocas.
    Si es atipico solo en algunas epocas no se marca: esas epocas las saca el rechazo de epocas.
    """
    fuera = set(ep.info["bads"]) | set(config.CANALES_SIN_CRITERIO_EEG)
    canales = [c for c in ep.copy().pick("eeg").ch_names if c not in fuera]
    x = ep.copy().crop(config.TMIN, 0.0).get_data(picks=canales) * 1e6
    ptp = x.max(axis=2) - x.min(axis=2)                          # epocas x canales
    tipico = np.median(ptp, axis=0)                              # un valor por canal
    z = _z_robusto(tipico, eje=0)
    pct = (_z_robusto(ptp, eje=1) > config.Z_CANAL_MALO).mean(axis=0) * 100
    plano = tipico < config.FACTOR_CANAL_PLANO * np.median(tipico)
    ruidoso = (z > config.Z_CANAL_MALO) & (pct > config.FRAC_EPOCAS_CANAL_MALO * 100)
    tabla = pd.DataFrame({"canal": canales, "ptp_tipico_uV": tipico.round(1), "z": z.round(1),
                          "pct_epocas_atipicas": pct.round(0),
                          "decision": np.select([plano, ruidoso], ["malo (plano)", "malo (ruidoso)"], default="ok")})
    return [c for c, m in zip(canales, plano | ruidoso) if m], tabla


def referenciar(ep, malos_propios):
    """Interpola solo los canales malos de la ROI y pone referencia promedio sin frontopolares ni canales malos.

    Los malos fuera de la ROI no se interpolan (no se miden): se dejan afuera del promedio de la referencia,
    para que su ruido no se le reste a la ROI. Devuelve (epocas, registro).
    """
    ep = ep.copy()
    malos = list(dict.fromkeys(ep.info["bads"] + list(malos_propios)))
    roi_malos = [c for c in malos if c in config.CANALES_ROI]
    if roi_malos:
        ep.info["bads"] = roi_malos
        ep.interpolate_bads(reset_bads=True, verbose="ERROR")
    otros = [c for c in malos if c not in roi_malos]
    ep.info["bads"] = otros                                   # quedan marcados, fuera de la referencia
    excluir = set(otros) | set(config.CANALES_SIN_CRITERIO_EEG)
    ref = [c for c in ep.copy().pick("eeg", exclude=[]).ch_names if c not in excluir]
    ep.set_eeg_reference(ref, verbose="ERROR")
    registro = {"roi_interpolados": roi_malos, "fuera_de_referencia": otros, "canales_referencia": len(ref),
                "roi_comprometida": config.CANAL_QC in roi_malos or len(roi_malos) > config.MAX_ROI_INTERPOLADOS}
    return ep, registro


def epocas_groseras(ep):
    """Indices de las epocas con pico a pico > UMBRAL_GROSERO_UV en algun canal EEG sin frontopolares ni malos,
    en VENTANA_GROSERO, sobre una copia con pasa-bajos de 30 Hz. Solo lo catastrofico: el rechazo fino por ROI
    y VEO lo hace medicion.py."""
    fuera = set(config.CANALES_SIN_CRITERIO_EEG) | set(ep.info["bads"])
    cabeza = [c for c in ep.copy().pick("eeg").ch_names if c not in fuera]
    x = (ep.copy().filter(None, config.PASA_BAJOS_LEP_HZ, picks="eeg", verbose="ERROR")
         .crop(*config.VENTANA_GROSERO).get_data(picks=cabeza) * 1e6)
    return np.flatnonzero((x.max(axis=2) - x.min(axis=2)).max(axis=1) > config.UMBRAL_GROSERO_UV)


def limpiar_sujeto(ep, sub, marcas):
    """Pasos 1 a 4. Devuelve (epocas preprocesadas, fila de preprocesado_resumen.csv)."""
    n0 = len(ep)
    ep, r_marcas = aplicar_marcas(ep, sub, marcas)
    malos, _ = canales_malos(ep)
    ep, r_ref = referenciar(ep, malos)
    groseras = epocas_groseras(ep)
    ep = ep.copy().drop(groseras, reason="grosera", verbose="ERROR")
    quedan = ep.metadata["condicion"].value_counts()
    fila = {"sujeto": sub, "segmentadas": n0, "por_autores": len(r_marcas["epocas_autores"]),
            "groseras": len(groseras), "quedan": len(ep),
            **{f"quedan_{c}": int(quedan.get(c, 0)) for c in config.NIVELES},
            # nombres de preprocesado.py: inclusion.py cuenta estos canales (criterios 2 y 4)
            "interpolados_autores": " ".join(r_marcas["electrodos_autores"]), "interpolados_propios": " ".join(malos),
            "roi_interpolados": " ".join(r_ref["roi_interpolados"]),
            "fuera_de_referencia": " ".join(r_ref["fuera_de_referencia"]), "aviso": r_marcas["aviso"]}
    return ep, fila


def ruta_preprocesadas(sub, carpeta=None):
    """derived_mine/sub-XXX/sub-XXX_preprocesadas-epo.fif (el nombre que usaba preprocesado.py)."""
    return epocas.ruta_epocas(sub, carpeta).with_name(f"{sub}_preprocesadas-epo.fif")


def guardar(ep, sub, carpeta=None):
    ruta = ruta_preprocesadas(sub, carpeta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ep.save(ruta, overwrite=True, verbose="ERROR")
    return ruta


def para_analisis(sub, regresion=None, carpeta=None, ref_sin_frontales=None, pasa_bajos=True):
    """Epocas preprocesadas listas para medir, con la misma firma que preprocesado.para_analisis (devuelve
    (epocas, None); regresion y ref_sin_frontales se ignoran: no hay regresion y la referencia ya esta puesta).

    Se interpolan los canales malos de fuera de la ROI (asi todos los sujetos tienen los mismos canales para el
    localizador, los mapas y el gran promedio; en la ROI no cambia nada), y despues pasa-bajos de 30 Hz
    (si pasa_bajos) y linea de base.
    """
    ep = mne.read_epochs(ruta_preprocesadas(sub, carpeta), verbose="ERROR")
    if ep.info["bads"]:
        ep.interpolate_bads(reset_bads=True, verbose="ERROR")
    if pasa_bajos:
        ep.filter(None, config.PASA_BAJOS_LEP_HZ, picks="eeg", verbose="ERROR")
    return ep.apply_baseline(config.BASELINE, verbose="ERROR"), None
