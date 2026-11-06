"""Paso 3: limpieza de las epocas, sin ICA (lo pide la consigna) y sin mirar el nivel de intensidad.

Parte de las epocas extraidas (sub-XXX_laser-epo.fif). Por ahora:
    1. marcas manuales de los autores: sus epocas malas y sus electrodos malos.
"""
import csv
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from procesamiento import config

# CSV de los autores, en la raiz del repo (junto a la carpeta data/)
RUTA_CSV = Path(config.RAIZ_DATOS).parent.parent / "all experiments_preprocessing.csv"
EXPERIMENTO = "Experiment 6"          # ds005293 = "Experiment 6_95 By BP" en la primera fila del CSV


def _lista(texto):
    """'33 70 72' -> ['33', '70', '72'];  '\\' o vacio -> []."""
    return [x for x in texto.replace(",", " ").split() if x != "\\"]


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
    canonico = {c.lower(): c for c in mne.channels.make_standard_montage("standard_1005").ch_names}
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


def _ptp(ep, canales, tramo):
    """Pico a pico (uV) por epoca en el tramo, el mayor entre los `canales`, sobre una copia con pasa-bajos de 30 Hz."""
    x = (ep.copy().filter(None, config.PASA_BAJOS_LEP_HZ, picks=["eeg", "eog"], verbose="ERROR")
         .crop(*tramo).get_data(picks=canales) * 1e6)
    return (x.max(axis=2) - x.min(axis=2)).max(axis=1)


def medir_epocas(ep):
    """Una fila por epoca con los tres pico a pico que deciden el rechazo (copia con pasa-bajos de 30 Hz):
        ptp_cabeza_uV  el mayor entre los canales EEG sin frontopolares ni malos, de -1 a 1 s (epocas groseras)
        ptp_roi_uV     el mayor entre los canales de la ROI, de -1 s al fin de la ventana de P2
        ptp_veo_uV     el del VEO, en el mismo tramo
    """
    fuera = set(config.CANALES_SIN_CRITERIO_EEG) | set(ep.info["bads"])
    cabeza = [c for c in ep.copy().pick("eeg").ch_names if c not in fuera]
    roi = [c for c in config.CANALES_ROI if c in ep.ch_names]
    tramo = (config.TMIN, config.VENTANA_P2[1])
    medidas = ep.metadata[["ensayo", "condicion"]].reset_index(drop=True).copy()
    medidas["ptp_cabeza_uV"] = _ptp(ep, cabeza, config.VENTANA_GROSERO).round(1)
    medidas["ptp_roi_uV"] = _ptp(ep, roi, tramo).round(1)
    medidas["ptp_veo_uV"] = (_ptp(ep, [config.CANAL_EOG], tramo).round(1)
                             if config.CANAL_EOG in ep.ch_names else np.nan)
    return medidas


def rechazar(ep, medidas):
    """Descarta las epocas que superan algun umbral y anota el motivo (el primero que se cumple):
        grosera  ptp_cabeza_uV > UMBRAL_GROSERO_UV (300)
        ROI      ptp_roi_uV    > UMBRAL_RECHAZO_UV (100)
        VEO      ptp_veo_uV    > UMBRAL_VEO_UV (150)
    Devuelve (epocas que quedan, medidas con la columna 'motivo': vacia si la epoca queda).
    """
    medidas = medidas.copy()
    medidas["motivo"] = np.select(
        [medidas["ptp_cabeza_uV"] > config.UMBRAL_GROSERO_UV,
         medidas["ptp_roi_uV"] > config.UMBRAL_RECHAZO_UV,
         medidas["ptp_veo_uV"] > config.UMBRAL_VEO_UV],
        ["grosera", "ROI", "VEO"], default="")
    return ep[(medidas["motivo"] == "").to_numpy()], medidas
 
 
def incluir(n_eeg, n_malos, registro_ref, limpias, n_segmentadas):
    """Regla de inclusion del sujeto. Devuelve la lista de motivos de exclusion (vacia = se incluye)."""
    motivos = []
    if n_segmentadas != config.EPOCAS_ESPERADAS:
        motivos.append(f"{n_segmentadas} epocas extraidas (se esperaban {config.EPOCAS_ESPERADAS})")
    if config.CANAL_EOG not in limpias.ch_names:
        motivos.append("sin canal VEO")
    pct_malos = 100 * n_malos / n_eeg
    if pct_malos > config.MAX_PCT_CANALES_MALOS:
        motivos.append(f"{pct_malos:.0f}% de canales malos")
    if registro_ref["roi_comprometida"]:
        motivos.append(f"ROI interpolada: {' '.join(registro_ref['roi_interpolados'])}")
    pct_perdidas = 100 * (config.EPOCAS_ESPERADAS - len(limpias)) / config.EPOCAS_ESPERADAS
    if pct_perdidas > config.MAX_PCT_EPOCAS_PERDIDAS:
        motivos.append(f"{pct_perdidas:.0f}% de epocas perdidas")
    quedan = limpias.metadata["condicion"].value_counts()
    pocas = [f"{c}: {int(quedan.get(c, 0))}" for c in config.NIVELES_MINIMOS
             if quedan.get(c, 0) < config.MIN_EPOCAS_LIMPIAS]
    if pocas:
        motivos.append(f"menos de {config.MIN_EPOCAS_LIMPIAS} epocas en " + ", ".join(pocas))
    return motivos
 
 
def limpiar_sujeto(ep, sub, marcas):
    """Los pasos 1 a 6 en orden. Devuelve (epocas limpias, epocas antes del rechazo, medidas por epoca, fila resumen)."""
    n0 = len(ep)
    ep, r_marcas = aplicar_marcas(ep, sub, marcas)
    malos, _ = canales_malos(ep)
    ep, r_ref = referenciar(ep, malos)
    limpias, medidas = rechazar(ep, medir_epocas(ep))
    n_eeg = len(mne.pick_types(ep.info, eeg=True, exclude=[]))
    n_malos = len(r_ref["roi_interpolados"]) + len(r_ref["fuera_de_referencia"])
    motivos = incluir(n_eeg, n_malos, r_ref, limpias, n0)
    quedan = limpias.metadata["condicion"].value_counts()
    fila = {"sujeto": sub, "extraidas": n0, "por_autores": len(r_marcas["epocas_autores"]),
            **{f"por_{m}": int((medidas["motivo"] == m).sum()) for m in ("grosera", "ROI", "VEO")},
            "limpias": len(limpias), **{c: int(quedan.get(c, 0)) for c in config.NIVELES},
            "electrodos_autores": " ".join(r_marcas["electrodos_autores"]), "malos_propios": " ".join(malos),
            "roi_interpolados": " ".join(r_ref["roi_interpolados"]), "canales_referencia": r_ref["canales_referencia"],
            "incluido": not motivos, "motivo_exclusion": "; ".join(motivos), "aviso": r_marcas["aviso"]}
    return limpias, ep, medidas, fila
 
 
def ruta_limpias(sub, carpeta=None):
    return epocas.ruta_epocas(sub, carpeta).with_name(f"{sub}_limpias-epo.fif")
 
 
def guardar(limpias, ep_antes, medidas, sub, carpeta=None):
    """Guarda las epocas limpias, la tabla de rechazo y la figura de control. Devuelve la ruta de las limpias."""
    ruta = ruta_limpias(sub, carpeta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    limpias.save(ruta, overwrite=True, verbose="ERROR")
    medidas.to_csv(ruta.with_name(f"{sub}_rechazo.csv"), index=False)
    fig = fig_control(ep_antes, limpias, medidas, sub)
    fig.savefig(ruta.with_name(f"{sub}_limpieza.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return ruta
 
 
COLORES = {"E1": "#9ecae1", "E2": "#4292c6", "E3": "#ef6548", "E4": "#b30000"}
 
 
def fig_control(ep_antes, limpias, medidas, sub):
    """Control de la limpieza (no decide nada): el pico a pico de cada epoca en la ROI y en el VEO con sus umbrales
    (circulo lleno = queda, x = rechazada), y el promedio de la ROI por nivel antes y despues de rechazar."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1, 1, 1.3]})
    queda = medidas["motivo"] == ""
    for ax, col, umbral, nombre in ((axes[0], "ptp_roi_uV", config.UMBRAL_RECHAZO_UV, "ROI"),
                                    (axes[1], "ptp_veo_uV", config.UMBRAL_VEO_UV, "VEO")):
        for c, color in COLORES.items():
            m = (medidas["condicion"] == c).to_numpy()
            ax.scatter(medidas.loc[m & queda, "ensayo"], medidas.loc[m & queda, col], s=16, color=color, label=c)
            ax.scatter(medidas.loc[m & ~queda, "ensayo"], medidas.loc[m & ~queda, col], s=30, color=color, marker="x")
        ax.axhline(umbral, color="0.3", ls="--", lw=0.9)
        ax.set_yscale("log")
        ax.set_ylabel(f"pico a pico {nombre} (uV)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    axes[1].set_xlabel("ensayo (x = rechazado; linea: umbral)")
    roi = [c for c in config.CANALES_ROI if c in limpias.ch_names]
    ax = axes[2]
    for c, color in COLORES.items():
        if (ep_antes.metadata["condicion"] == c).any():
            antes = ep_antes[c].copy().filter(None, config.PASA_BAJOS_LEP_HZ, verbose="ERROR")
            ax.plot(antes.times * 1000, antes.average(picks=roi).get_data().mean(axis=0) * 1e6,
                    color=color, lw=1, ls=":")
        if (limpias.metadata["condicion"] == c).any():
            despues = limpias[c].copy().filter(None, config.PASA_BAJOS_LEP_HZ, verbose="ERROR")
            ax.plot(despues.times * 1000, despues.average(picks=roi).get_data().mean(axis=0) * 1e6,
                    color=color, lw=1.8, label=f"{c} (n={len(despues)})")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-200, 1000)
    ax.set_xlabel("tiempo desde el pulso (ms)   ·   punteado: antes del rechazo, lleno: despues")
    ax.set_ylabel("ROI (uV)")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{sub}: control de la limpieza", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    return fig