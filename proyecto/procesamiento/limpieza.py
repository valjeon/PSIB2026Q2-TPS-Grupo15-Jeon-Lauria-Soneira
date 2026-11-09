"""Paso 3: limpieza de las epocas, sin ICA y sin mirar el nivel de intensidad.

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
EXPERIMENTO = "Experiment 6" # ds005293 = "Experiment 6_95 By BP" en la primera fila del CSV


# convierte un texto del CSV en una lista
def _lista(texto):
    return [x for x in texto.replace(",", " ").split() if x != "\\"]


# devuelve la lista de nombres de electrodos del sistema 10-05
def _nombres_10_05():
    try:
        return mne.channels.make_standard_montage("colin27_1005").ch_names
    except ValueError:
        return mne.channels.make_standard_montage("standard_1005").ch_names


# funcion que lee el CSV de los autores y devuelve un diccionario con las épocas malas y los electrodos malos para cada sujeto
def leer_marcas(ruta_csv=None, experimento=EXPERIMENTO):
    ruta = Path(RUTA_CSV if ruta_csv is None else ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No esta el CSV de los autores: {ruta}") # devuelve error si no encuentra el csv
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.reader(f)) # lee las listas
    # fila 0: titulos de experimento (uno cada varias columnas); fila 1: nombres de columna
    titulos, nombres = filas[0], [c.strip().lower() for c in filas[1]] # separa titulos y nombres
    inicio = next((i for i, t in enumerate(titulos) if t.strip().startswith(experimento)), None)
    if inicio is None:
        raise ValueError(f"No encuentro '{experimento}' en la primera fila del CSV")
    fin = next((i for i in range(inicio + 1, len(titulos)) if titulos[i].strip()), len(titulos))
    col = {nombres[i]: i for i in range(inicio, fin)}
    c_sub, c_ep, c_el = col["sub_id"], col["bad epoch"], col["bad electrode"]
    canonico = {c.lower(): c for c in _nombres_10_05()} # pone todo en minuscula para que MNE lo reconozca
    marcas = {}
    for fila in filas[2:]:
        # recorre las filas de datos
        if len(fila) <= max(c_sub, c_ep, c_el) or not fila[c_sub].strip().isdigit(): # saltea las filas cortas o que no tienen un número de sujeto.
            continue
        epocas_malas = {int(x) - 1 for x in _lista(fila[c_ep]) if x.isdigit()} # convierte las épocas malas a números y les resta 1: los autores numeran desde 1 y la metadata (ensayo) desde 0.
        electrodos = [canonico.get(c.lower(), c) for c in _lista(fila[c_el])] # corrige los nombres de los electrodos
        marcas[f"sub-{int(fila[c_sub]):03d}"] = (epocas_malas, electrodos)
    return marcas


def aplicar_marcas(ep, sub, marcas):
    ep = ep.copy() # hace una copia para no tocar las epocas que recibe
    ep_malas, electrodos = marcas.get(sub, (set(), []))
    electrodos = [c for c in electrodos if c in ep.ch_names] # se queda con la informacion de los electrodos del registro
    registro = {"epocas_autores": [], "electrodos_autores": electrodos, "aviso": ""}
    if ep_malas and len(ep) == config.EPOCAS_ESPERADAS: # saca las épocas malas, pero solo si el sujeto tiene las 80 épocas esperadas
        quitar = ep.metadata["ensayo"].isin(ep_malas).to_numpy()
        registro["epocas_autores"] = sorted(ep.metadata.loc[quitar, "ensayo"].tolist())
        ep = ep[~quitar]
    elif ep_malas: # si falta alguna la numeración de los autores ya no coincide con la nuestra y se sacarían épocas equivocadas
        # en ese caso no saca ninguna y deja un aviso
        registro["aviso"] = (f"{len(ep)} epocas (se esperaban {config.EPOCAS_ESPERADAS}): "
                             "no se sacan las epocas de los autores")
    ep.info["bads"] = list(dict.fromkeys(ep.info["bads"] + electrodos)) # marca los electrodos malos en info["bads"]
    return ep, registro


# medida estadistica: z-score robusto
# indica cuán atípico es cada valor pero sin que los propios valores atípicos lo distorsionen.
def _z_robusto(x, eje):
    med = np.median(x, axis=eje, keepdims=True) # usa la mediana
    mad = np.median(np.abs(x - med), axis=eje, keepdims=True) * 1.4826
    # MAD: mediana de las distancias a la mediana 
    # en lugar del desvio
    return (x - med) / np.where(mad > 0, mad, np.inf)


# funcion que busca canales planos o ruidosos que los autores no hayan marcado
def canales_malos(ep):
    fuera = set(ep.info["bads"]) | set(config.CANALES_SIN_CRITERIO_EEG)
    canales = [c for c in ep.copy().pick("eeg").ch_names if c not in fuera] # elige que canales evaluar
    # todos los EEG menos los frontopolares (su amplitud alta se debe a los ojos, no a que estén malos) y los ya marcados qu eno esten en fuera
    x = ep.copy().crop(config.TMIN, 0.0).get_data(picks=canales) * 1e6 # toma solo el pre-estimulo para evaluar la calidad del canal
    # calcula el pico a pico de cada canal en cada época
    ptp = x.max(axis=2) - x.min(axis=2) # epocas x canales
    tipico = np.median(ptp, axis=0) # calcula el valor típico de cada canal
    z = _z_robusto(tipico, eje=0) # z: cuán atípico es el valor típico del canal respecto de los otros canales
    pct = (_z_robusto(ptp, eje=1) > config.Z_CANAL_MALO).mean(axis=0) * 100 
    # pct: en qué porcentaje de las épocas el canal es atípico dentro de esa época, comparado con los otros canales de la misma época
    plano = tipico < config.FACTOR_CANAL_PLANO * np.median(tipico) # plano si su valor típico es menor que una fracción de la mediana de los canale
    ruidoso = (z > config.Z_CANAL_MALO) & (pct > config.FRAC_EPOCAS_CANAL_MALO * 100) # ruidoso si tiene z alto y además es atípico en muchas épocas
    tabla = pd.DataFrame({"canal": canales, "ptp_tipico_uV": tipico.round(1), "z": z.round(1),
                          "pct_epocas_atipicas": pct.round(0),
                          "decision": np.select([plano, ruidoso], ["malo (plano)", "malo (ruidoso)"], default="ok")})
    return [c for c, m in zip(canales, plano | ruidoso) if m], tabla
# Arma una tabla por canal con los valores y la decisión


# Junta los malos de los autores (que ya están en bads) con los propios, sin repetir
def referenciar(ep, malos_propios):
    ep = ep.copy()
    malos = list(dict.fromkeys(ep.info["bads"] + list(malos_propios))) 
    roi_malos = [c for c in malos if c in config.CANALES_ROI] # Interpola solo los malos que están en la ROI
    if roi_malos:
        ep.info["bads"] = roi_malos # los malos fuera de la ROI no se interpolan
        ep.interpolate_bads(reset_bads=True, verbose="ERROR")
    otros = [c for c in malos if c not in roi_malos]
    ep.info["bads"] = otros # quedan marcados, fuera de la referencia
    excluir = set(otros) | set(config.CANALES_SIN_CRITERIO_EEG)
    ref = [c for c in ep.copy().pick("eeg", exclude=[]).ch_names if c not in excluir]
    # Re-referencia
    # la nueva referencia es el promedio de los canales EEG sin los frontopolares (por los ojos) y sin los malos que quedaron
    ep.set_eeg_reference(ref, verbose="ERROR")
    registro = {"roi_interpolados": roi_malos, "fuera_de_referencia": otros, "canales_referencia": len(ref),
                "roi_comprometida": config.CANAL_QC in roi_malos or len(roi_malos) > config.MAX_ROI_INTERPOLADOS}
    return ep, registro
# devuelve las épocas y un registro con:
# qué canales de la ROI se interpolaron;
# cuáles quedaron fuera de la referencia;
# cuántos canales formaron la referencia


# funcion que busca épocas catastróficas como un electrodo que se soltó, un movimiento brusco
def epocas_groseras(ep):
    fuera = set(config.CANALES_SIN_CRITERIO_EEG) | set(ep.info["bads"])
    cabeza = [c for c in ep.copy().pick("eeg").ch_names if c not in fuera] # elige canales de la cabeza (no los malos)
    # pasa-bajos de 30 Hz (para medir sin el ruido rápido) y recorta a la ventana VENTANA_GROSERO (−1 a 1 s).
    x = (ep.copy().filter(None, config.PASA_BAJOS_LEP_HZ, picks="eeg", verbose="ERROR")
         .crop(*config.VENTANA_GROSERO).get_data(picks=cabeza) * 1e6) 
    # calcula el pico a pico de cada canal en cada época y se queda con el máximo por época
    return np.flatnonzero((x.max(axis=2) - x.min(axis=2)).max(axis=1) > config.UMBRAL_GROSERO_UV)
# El umbral es alto a propósito porque el rechazo fino lo hace medicion.py


# se juntan todos los pasos previos
def limpiar_sujeto(ep, sub, marcas):
    n0 = len(ep)
    ep, r_marcas = aplicar_marcas(ep, sub, marcas)
    malos, _ = canales_malos(ep)
    ep, r_ref = referenciar(ep, malos)
    groseras = epocas_groseras(ep)
    ep = ep.copy().drop(groseras, reason="grosera", verbose="ERROR")
    quedan = ep.metadata["condicion"].value_counts() # Cuenta cuántas épocas quedan por nivel despues de aplicar las funciones
    # Arma la fila del resumen
    # épocas iniciales
    # sacadas por los autores
    # groseras
    # las que quedan (en total y por nivel)
    # canales interpolados de cada tipo
    # canales fuera de la referencia y avisos
    fila = {"sujeto": sub, "segmentadas": n0, "por_autores": len(r_marcas["epocas_autores"]),
            "groseras": len(groseras), "quedan": len(ep),
            **{f"quedan_{c}": int(quedan.get(c, 0)) for c in config.NIVELES},
            # nombres de preprocesado.py: inclusion.py cuenta estos canales (criterios 2 y 4)
            "interpolados_autores": " ".join(r_marcas["electrodos_autores"]), "interpolados_propios": " ".join(malos),
            "roi_interpolados": " ".join(r_ref["roi_interpolados"]),
            "fuera_de_referencia": " ".join(r_ref["fuera_de_referencia"]), "aviso": r_marcas["aviso"]}
    return ep, fila
# devuelve las épocas limpias y la fila


# cambia el nombre de la carpeta
def ruta_preprocesadas(sub, carpeta=None):
    return epocas.ruta_epocas(sub, carpeta).with_name(f"{sub}_preprocesadas-epo.fif")


# crea la carpeta si no existe, guarda las épocas limpias y devuelve la ruta.
def guardar(ep, sub, carpeta=None):
    ruta = ruta_preprocesadas(sub, carpeta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ep.save(ruta, overwrite=True, verbose="ERROR")
    return ruta


# funcion que usan los otros archivos (localizador.py, rasgos_ensayo.py, etc.) para leer las épocas listas para medir
def para_analisis(sub, regresion=None, carpeta=None, ref_sin_frontales=None, pasa_bajos=True):
    ep = mne.read_epochs(ruta_preprocesadas(sub, carpeta), verbose="ERROR")
    if ep.info["bads"]:
        ep.interpolate_bads(reset_bads=True, verbose="ERROR")
        # interpola los canales malos que quedaban, los de fuera de la ROI
        # así todos los sujetos tienen los mismos canales (hace falta para el gran promedio y los mapas)
        # en la ROI no cambia nada, porque ya estaba interpolada.
    if pasa_bajos:
        ep.filter(None, config.PASA_BAJOS_LEP_HZ, picks="eeg", verbose="ERROR")
        # Si pasa_bajos=True, aplica el pasa-bajos de 30 Hz (para el LEP)
    return ep.apply_baseline(config.BASELINE, verbose="ERROR"), None
