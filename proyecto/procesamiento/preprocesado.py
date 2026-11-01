"""Limpieza comun a todos los sujetos, sin usar el nivel (E1..E4): se puede aplicar antes del localizador.

Guardado (preparar_sujeto -> sub-XXX_preprocesadas-epo.fif), con la referencia original:
    1. epocas segmentadas (sub-XXX_laser-epo.fif)
    2. se sacan las epocas malas de los autores (solo si el sujeto tiene las 80; si no, no es confiable)
    3. se interpolan JUNTOS los electrodos malos de los autores y los propios (malos_propios.csv)
    4. se sacan las epocas con artefactos groseros: pico a pico > UMBRAL_GROSERO_UV en algun canal EEG
       (sin frontopolares, pasa-bajos 30 Hz, en VENTANA_GROSERO). Umbral laxo: solo lo catastrofico.

Al leer para analizar (para_analisis), opcionales y en este orden:
    5. [opcional, apagado: config.REGRESION_VEO] regresion del VEO (Gratton), beta estimado solo en tramos
       sin LEP. Se descarto: en los 3 sujetos piloto el VEO explica 6-34% de la varianza de los frontales
       (pocos parpadeos espontaneos), el beta queda mal estimado y en sub-025 le restaba LEP a Cz.
    6. referencia promedio SIN los frontales (config.REF_SIN_FRONTALES): su actividad ocular no se
       reparte a Cz a traves de la referencia
    7. pasa-bajos 30 Hz y linea de base (config.BASELINE)
Se dejan fuera del archivo guardado para poder comparar con y sin regresion.

Registro: derived_mine/preprocesado_resumen.csv (una fila por sujeto: que se saco y cuanto queda por nivel).

Uso desde proyecto/:
    python -m procesamiento.preprocesado sub-010 sub-025 sub-050
    python -m procesamiento.preprocesado                     (todos los sujetos segmentados)
"""
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from procesamiento import artefactos, canales_malos, config, marcas_autores, segmentacion

UMBRAL_GROSERO_UV = getattr(config, "UMBRAL_GROSERO_UV", 300)
VENTANA_GROSERO = getattr(config, "VENTANA_GROSERO", (config.TMIN, 1.0))   # tramo que se analiza
# Tramos sin respuesta evocada donde se estima el beta del VEO (parpadeos espontaneos): antes del pulso
# y despues de 1 s. Asi la regresion no confunde el LEP con ojo aunque los dos crezcan con la intensidad.
VENTANAS_BETA = getattr(config, "VENTANAS_BETA_VEO", [(config.TMIN, 0.0), (1.0, config.TMAX)])


def ruta_preprocesadas(sub, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    return carpeta / sub / f"{sub}_preprocesadas-epo.fif"


# ---------------------------------------------------------------------------
# Pasos 2-4 (se guardan)
# ---------------------------------------------------------------------------

def epocas_groseras(ep, umbral_uV=None, ventana=None):
    """Indices de epocas con pico a pico > umbral en algun canal EEG (sin frontopolares). Y el maximo por epoca."""
    umbral_uV = UMBRAL_GROSERO_UV if umbral_uV is None else umbral_uV
    ventana = VENTANA_GROSERO if ventana is None else ventana
    canales = [c for c in ep.copy().pick("eeg").ch_names if c not in set(config.CANALES_SIN_CRITERIO_EEG)]
    x = artefactos._pasa_bajos(ep).crop(*ventana).get_data(picks=canales) * 1e6
    ptp = (x.max(axis=2) - x.min(axis=2)).max(axis=1)
    return np.flatnonzero(ptp > umbral_uV), ptp


def preparar(ep, sub, carpeta=None):
    """Pasos 2-4 sobre las epocas segmentadas. Devuelve (epocas, dict de resumen)."""
    m = marcas_autores.marcas_de(sub)
    propios = canales_malos.malos_propios(sub, carpeta)
    n0 = len(ep)
    conteo = lambda e: e.metadata["condicion"].value_counts()

    # 2-3: marcas de los autores con los electrodos propios sumados (se interpolan todos juntos)
    juntos = marcas_autores.Marcas(epocas=m.epocas, electrodos=list(dict.fromkeys(m.electrodos + propios)))
    ep = marcas_autores.aplicar_marcas(ep, juntos)
    n_autores = n0 - len(ep)

    # 4: artefactos groseros
    malas, ptp = epocas_groseras(ep)
    groseras = ep[malas].metadata["condicion"].value_counts() if len(malas) else pd.Series(dtype=int)
    ep = ep.copy().drop(malas, reason="grosero", verbose="ERROR")

    fila = {"sujeto": sub, "segmentadas": n0, "por_autores": n_autores, "groseras": len(malas),
            "quedan": len(ep), "interpolados_autores": " ".join(m.electrodos),
            "interpolados_propios": " ".join(propios),
            "ptp_mediana_uV": round(float(np.median(ptp)), 1), "umbral_grosero_uV": UMBRAL_GROSERO_UV}
    quedan = conteo(ep)
    for c in config.NIVELES:
        fila[f"groseras_{c}"] = int(groseras.get(c, 0))
        fila[f"quedan_{c}"] = int(quedan.get(c, 0))
    return ep, fila


def guardar_resumen(fila, carpeta):
    ruta = carpeta / "preprocesado_resumen.csv"
    nueva = pd.DataFrame([fila])
    if ruta.exists():
        viejo = pd.read_csv(ruta)
        nueva = pd.concat([viejo[viejo["sujeto"] != fila["sujeto"]], nueva], ignore_index=True)
    nueva.sort_values("sujeto").to_csv(ruta, index=False)
    return ruta


def preparar_sujeto(sub, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    ep = mne.read_epochs(segmentacion.ruta_epocas(sub, carpeta), verbose="ERROR")
    ep, fila = preparar(ep, sub, carpeta)
    ep.save(ruta_preprocesadas(sub, carpeta), overwrite=True, verbose="ERROR")
    guardar_resumen(fila, carpeta)
    return ep, fila


# ---------------------------------------------------------------------------
# Pasos 5-7 (al leer)
# ---------------------------------------------------------------------------

def regresion_veo(ep, ventanas=None):
    """Resta de cada canal EEG la parte explicada por el VEO. Devuelve (copia, {canal: beta}).

    Beta (uno por canal, minimos cuadrados) se estima solo en `ventanas` (tramos sin LEP: parpadeos
    espontaneos) y se aplica a la epoca completa. No usa el nivel.
    """
    ventanas = VENTANAS_BETA if ventanas is None else ventanas
    t = ep.times
    m = np.zeros(len(t), bool)
    for a, b in ventanas:
        m |= (t >= a) & (t <= b)
    eeg = ep.copy().pick("eeg")
    x, e = eeg.get_data(), ep.get_data(picks=config.CANAL_EOG)[:, 0, :]
    xr = x[:, :, m].transpose(1, 0, 2).reshape(x.shape[1], -1)               # canales x (epocas*tiempo)
    er = e[:, m].reshape(-1)
    xr = xr - xr.mean(axis=1, keepdims=True)
    er = er - er.mean()
    beta = (xr @ er) / (er @ er)
    out = ep.copy()
    datos = out.get_data()
    for b, nombre in zip(beta, eeg.ch_names):
        datos[:, out.ch_names.index(nombre), :] -= b * e
    out._data = datos
    return out, dict(zip(eeg.ch_names, np.round(beta, 3)))


def r2_veo(ep, canales=("Fp1", "Fp2", "AF7", "AF8"), ventanas=None):
    """% de la varianza de los frontopolares que explica el VEO en los tramos sin LEP (mediana entre canales)."""
    ventanas = VENTANAS_BETA if ventanas is None else ventanas
    t = ep.times
    m = np.zeros(len(t), bool)
    for a, b in ventanas:
        m |= (t >= a) & (t <= b)
    e = ep.get_data(picks=config.CANAL_EOG)[:, 0, m].reshape(-1)
    r2 = []
    for c in canales:
        if c in ep.ch_names:
            x = ep.get_data(picks=c)[:, 0, m].reshape(-1)
            r2.append(np.corrcoef(x, e)[0, 1] ** 2 * 100)
    return round(float(np.median(r2)), 1)


def para_analisis(sub, regresion=None, carpeta=None, ref_sin_frontales=None):
    """Epocas preprocesadas listas para mirar/medir: [regresion VEO] -> ref. promedio [sin Fp/AF] -> 30 Hz -> linea de base."""
    regresion = config.REGRESION_VEO if regresion is None else regresion
    ref_sin_frontales = config.REF_SIN_FRONTALES if ref_sin_frontales is None else ref_sin_frontales
    ep = mne.read_epochs(ruta_preprocesadas(sub, carpeta), verbose="ERROR")
    beta = None
    if regresion and config.CANAL_EOG in ep.ch_names:
        ep, beta = regresion_veo(ep)
    if ref_sin_frontales:      # promedio de referencia sin Fp/AF: el ojo no se reparte a todos los canales
        fuera = set(config.CANALES_SIN_CRITERIO_EEG)
        ref = [c for c in ep.copy().pick("eeg").ch_names if c not in fuera]
        v = ep.copy().set_eeg_reference(ref, projection=False, verbose="ERROR")
    else:
        v = artefactos.con_referencia_promedio(ep)
    v = v.filter(None, config.PASA_BAJOS_LEP_HZ, picks="eeg", verbose="ERROR")
    return v.apply_baseline(config.BASELINE, verbose="ERROR"), beta


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Limpieza comun (autores + canales propios + groseras)")
    p.add_argument("sujetos", nargs="*", help="por defecto, todos los que tienen epocas segmentadas")
    a = p.parse_args(argv)
    print(f"Umbral grosero: {UMBRAL_GROSERO_UV} uV en {VENTANA_GROSERO} s (sin frontopolares, 30 Hz)")
    for sub in a.sujetos or marcas_autores.sujetos_segmentados():
        ep, f = preparar_sujeto(sub)
        print(f"\n=== {sub} ===  {f['segmentadas']} -> {f['quedan']} epocas "
              f"(autores {f['por_autores']}, groseras {f['groseras']}; ptp mediano {f['ptp_mediana_uV']} uV)")
        print("  interpolados: autores [" + (f["interpolados_autores"] or "-") + "]  propios ["
              + (f["interpolados_propios"] or "-") + "]")
        print("  groseras por nivel: " + "  ".join(f"{c} {f[f'groseras_{c}']}" for c in config.NIVELES)
              + "   |   quedan: " + "  ".join(f"{c} {f[f'quedan_{c}']}" for c in config.NIVELES))
    print(f"\nResumen: {Path(config.CARPETA_SALIDA) / 'preprocesado_resumen.csv'}")


if __name__ == "__main__":
    main()