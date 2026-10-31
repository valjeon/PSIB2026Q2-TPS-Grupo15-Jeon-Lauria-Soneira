"""Sensibilidad del criterio de rechazo de artefactos (sin ICA).

Para cada sujeto toma las epocas SIN rechazar (sub-XXX_laser-epo.fif) y compara varios criterios:
cuantas epocas quedan por nivel, cuanto vale el N2-P2 en Cz por nivel, si el gradiente con la
intensidad se mantiene y cuanto coinciden las epocas rechazadas con las que marcaron los autores.
No modifica ningun archivo de epocas; solo escribe CSV en derived_mine/sensibilidad/.

IMPORTANTE: esto sirve para ENTENDER el efecto de cada criterio. El criterio final se elige por
razones de calidad de senal (parpadeos, canales malos), no por el valor que de el clasificador.

Variantes (VEO = ojo vertical, ventana critica config.VENTANA_VEO; EEG = pico a pico de epoca completa;
"frontales" = config.CANALES_SIN_CRITERIO_EEG, que el criterio de EEG ignora porque los decide el VEO):
    actual          lo que hace artefactos.py: VEO > 100 uV o EEG > 100 uV sin los canales frontales
    todos_canales   como 'actual' pero el EEG usa todos los canales (criterio anterior)
    veo_150         VEO > 150 uV
    sin_veo         solo EEG (sin frontales)
    sin_malos       como 'actual' y ademas ignora los electrodos malos de los autores
    sin_malos_150   VEO > 150 uV y EEG sin frontales ni malos de los autores
    ventana         como 'actual' pero el EEG se mide solo en la ventana que se analiza (VENTANA_EEG, -1 a 1 s:
                    N2-P2 y rasgos pre/post), no en toda la epoca (-1 a 2 s)
    ventana_veo_residual   como 'ventana' pero el VEO se mide sin la parte ligada al estimulo (se le resta el VEO
                    promedio de su nivel): el VEO tambien registra el LEP, y eso no es un parpadeo
    ventana_lejanos        como 'ventana' pero el criterio de EEG solo mira canales a menos de Y_MAX_LEJANOS_MM
                    hacia adelante (centro y parte posterior de la cabeza), lejos de los ojos
    ventana_veo_residual_lejanos   las dos cosas a la vez
    solo_cz         se mira solo Cz: descarta si el pico a pico de Cz (en VENTANA_EEG) supera el umbral; no usa VEO
    solo_cz_veo_residual   lo mismo y ademas descarta por VEO sin la parte ligada al estimulo
    veo_n2p2        como 'ventana' pero el VEO se mira solo en VENTANA_N2P2 (0.12 a 0.5 s): un parpadeo fuera del
                    tramo donde se miden N2 y P2 no descarta la epoca (aproxima "recortar el intervalo del parpadeo")
    veo_residual_n2p2   lo mismo con el VEO sin la parte ligada al estimulo
    regresion_veo   NO descarta por VEO: resta de cada canal EEG la parte explicada por el VEO (Gratton);
                    sigue descartando por EEG > 100 uV (sin frontales), ya con la senal corregida
    regresion_veo_ventana   regresion_veo con el EEG medido solo en VENTANA_EEG
Las dos ultimas necesitan el CSV de los autores (--csv-autores).

Ademas, 'fuga_veo': correlacion de Spearman (dentro de cada nivel, sin rechazar) entre el pico a pico
de VEO y el N2-P2 de cada ensayo en Cz. Si es alta, parte del N2-P2 puede ser parpadeo.

Uso desde proyecto/:
    python -m procesamiento.sensibilidad sub-004 sub-005 sub-006 \\
        --csv-autores ../ruta/supplement-data/CSV/"all experiments_preprocessing.csv"
"""
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from procesamiento import artefactos, config, marcas_autores, qc, segmentacion
from procesamiento.ver_senal import posicion_y, veo_ptp, veo_ptp_sin_evocada

VENTANA_EEG = getattr(config, "VENTANA_EEG", None) or (-1.0, 1.0)   # s; tramo que se usa de verdad (pre -1..0, post 0..1)
# tramo donde N2 y P2 se miden (union de VENTANA_N2 y VENTANA_P2): un parpadeo fuera de ahi no afecta el N2-P2
VENTANA_N2P2 = (min(config.VENTANA_N2[0], config.VENTANA_P2[0]), max(config.VENTANA_N2[1], config.VENTANA_P2[1]))
Y_MAX_LEJANOS_MM = 30         # canales mas adelantados que esto (Fz, F*, AF*, Fp*) se consideran cerca de los ojos
MIN_EPOCAS_NIVEL = 3          # con menos epocas en un nivel no se calcula el N2-P2 de ese nivel

CANALES_CENTRALES_CRITERIO = ("Cz", "C3", "C4")   # variante `centrales`: lejos de los ojos, N2-P2 visible


def _v(uv=100, ue=100, frontales=True, malos=False, ventana=False, residual=False, lejanos=False, n2p2=False,
       solo_cz=False, centrales=False):
    """Definicion de una variante: umbral VEO (None = sin VEO), umbral EEG, ignorar frontales de config,
    ignorar electrodos malos de los autores, medir el EEG solo en VENTANA_EEG, usar el VEO sin la parte
    evocada, mirar solo los canales lejanos a los ojos, mirar el VEO solo en VENTANA_N2P2, y mirar solo el canal CANAL_QC (Cz) en el criterio de EEG,
    o solo CANALES_CENTRALES_CRITERIO (Cz, C3, C4)."""
    return dict(uv=uv, ue=ue, frontales=frontales, malos=malos, ventana=ventana, residual=residual,
                lejanos=lejanos, n2p2=n2p2, solo_cz=solo_cz, centrales=centrales)


VARIANTES = {
    "actual": _v(),
    "todos_canales": _v(frontales=False),
    "veo_150": _v(uv=150),
    "sin_veo": _v(uv=None),
    "sin_malos": _v(malos=True),
    "sin_malos_150": _v(uv=150, malos=True),
    "ventana": _v(ventana=True),
    "ventana_veo_residual": _v(ventana=True, residual=True),
    "ventana_lejanos": _v(ventana=True, lejanos=True),
    "ventana_veo_residual_lejanos": _v(ventana=True, residual=True, lejanos=True),
    "veo_n2p2": _v(ventana=True, n2p2=True),
    "veo_residual_n2p2": _v(ventana=True, residual=True, n2p2=True),
    "solo_cz": _v(uv=None, ventana=True, solo_cz=True),
    "solo_cz_veo_residual": _v(ventana=True, residual=True, solo_cz=True),
    "centrales": _v(uv=None, ventana=True, centrales=True),
}
# variantes con la senal corregida por regresion de VEO: nombre -> medir el EEG solo en VENTANA_EEG
VARIANTES_REGRESION = {"regresion_veo": False, "regresion_veo_ventana": True}


# ---------------------------------------------------------------------------
# Medidas por epoca (una sola vez por sujeto)
# ---------------------------------------------------------------------------

def coeficientes_veo(v):
    """Coeficiente de propagacion del VEO a cada canal EEG (uno por canal), como en Gratton et al. 1983.

    Se estima sobre la senal SIN la respuesta evocada: a cada ensayo se le resta el promedio de su nivel,
    asi la regresion no intenta explicar el LEP (que tambien aparece, atenuado, en el VEO).
    Devuelve (coeficientes, nombres de canales EEG).
    """
    eeg = v.copy().pick("eeg")
    x, e = eeg.get_data(), v.get_data(picks=config.CANAL_EOG)[:, 0, :]
    cond = v.metadata["condicion"].to_numpy()
    xr, er = x.copy(), e.copy()
    for c in set(cond):
        i = cond == c
        xr[i] -= x[i].mean(axis=0)
        er[i] -= e[i].mean(axis=0)
    xr = xr.transpose(1, 0, 2).reshape(x.shape[1], -1)       # canales x (ensayos * tiempo)
    er = er.reshape(-1)
    xr -= xr.mean(axis=1, keepdims=True)
    er = er - er.mean()
    return (xr @ er) / (er @ er), eeg.ch_names


def corregir_con_veo(v):
    """Copia de `v` con la parte del VEO restada de cada canal EEG. Devuelve (copia, {canal: coeficiente})."""
    beta, nombres = coeficientes_veo(v)
    out = v.copy()
    e = v.get_data(picks=config.CANAL_EOG)[:, 0, :]
    datos = out.get_data()
    for b, nombre in zip(beta, nombres):
        datos[:, out.ch_names.index(nombre), :] -= b * e
    out._data = datos
    return out, dict(zip(nombres, beta))


def medidas_por_epoca(ep, corregir_veo=False):
    """Prepara lo que usan todas las variantes. `ep`: epocas crudas del sujeto.

    Devuelve dict con: v (copia con ref. promedio y pasa-bajos 30 Hz), ptp (n_epocas x n_canales EEG,
    epoca completa, uV), canales (nombres), ptp_veo (uV en VENTANA_VEO), n2p2 (por ensayo, Cz, uV).
    corregir_veo: antes de medir, resta del EEG la parte explicada por el VEO (agrega la clave 'beta').
    """
    v = artefactos._pasa_bajos(artefactos.con_referencia_promedio(ep))
    beta = None
    if corregir_veo:
        v, beta = corregir_con_veo(v)
    eeg = v.copy().pick("eeg")
    x = eeg.get_data()
    ptp = (x.max(axis=2) - x.min(axis=2)) * 1e6
    xw = eeg.copy().crop(*VENTANA_EEG).get_data()
    ptp_ventana = (xw.max(axis=2) - xw.min(axis=2)) * 1e6
    if config.CANAL_EOG in v.ch_names:
        w = v.copy().crop(*config.VENTANA_VEO).get_data(picks=config.CANAL_EOG)[:, 0, :]
        ptp_veo = (w.max(axis=1) - w.min(axis=1)) * 1e6
        ptp_veo_residual = veo_ptp_sin_evocada(v)
        ptp_veo_n2p2 = veo_ptp(v, VENTANA_N2P2)
        ptp_veo_residual_n2p2 = veo_ptp_sin_evocada(v, VENTANA_N2P2)
    else:
        ptp_veo = ptp_veo_residual = ptp_veo_n2p2 = ptp_veo_residual_n2p2 = np.full(len(v), np.nan)
    cz = v.get_data(picks=config.CANAL_QC)[:, 0, :] * 1e6
    t = v.times
    mn = (t >= config.VENTANA_N2[0]) & (t <= config.VENTANA_N2[1])
    mp = (t >= config.VENTANA_P2[0]) & (t <= config.VENTANA_P2[1])
    n2p2 = cz[:, mp].max(axis=1) - cz[:, mn].min(axis=1)
    return {"v": v, "ptp": ptp, "ptp_ventana": ptp_ventana, "canales": eeg.ch_names, "ptp_veo": ptp_veo,
            "ptp_veo_residual": ptp_veo_residual, "ptp_veo_n2p2": ptp_veo_n2p2,
            "ptp_veo_residual_n2p2": ptp_veo_residual_n2p2, "n2p2": n2p2, "beta": beta}


def canales_cerca_de_los_ojos(nombres, y_max_mm=None):
    """Canales mas adelantados que y_max_mm (por defecto Y_MAX_LEJANOS_MM). Los que no tienen posicion no se tocan."""
    y_max = Y_MAX_LEJANOS_MM if y_max_mm is None else y_max_mm
    y = posicion_y(nombres) * 1000
    return [n for n, yy in zip(nombres, y) if np.isfinite(yy) and yy > y_max]


def malas_segun_variante(m, umbral_veo, umbral_eeg, ignorar, ventana=False, residual=False, n2p2=False):
    """Vector booleano: True = la epoca se rechaza.

    ventana: medir el EEG solo en VENTANA_EEG. residual: medir el VEO sin la parte ligada al estimulo.
    """
    cols = [i for i, c in enumerate(m["canales"]) if c not in set(ignorar)]
    ptp = m["ptp_ventana"] if ventana else m["ptp"]
    por_eeg = ptp[:, cols].max(axis=1) > umbral_eeg
    ptp_veo = m["ptp_veo" + ("_residual" if residual else "") + ("_n2p2" if n2p2 else "")]
    por_veo = (ptp_veo > umbral_veo) if umbral_veo is not None else np.zeros(len(por_eeg), bool)
    return por_veo | por_eeg


def n2p2_por_nivel(v, mantener):
    """N2-P2 (uV) y n por nivel, del promedio de las epocas que se mantienen."""
    out = {}
    cond = v.metadata["condicion"].to_numpy()
    for c in config.NIVELES:
        idx = np.flatnonzero((cond == c) & mantener)
        if len(idx) < MIN_EPOCAS_NIVEL:
            out[c] = (len(idx), np.nan)
            continue
        ev = v[idx].average()
        out[c] = (len(idx), qc.picos_n2_p2(ev)["N2P2_uV"])
    return out


# ---------------------------------------------------------------------------
# Comparacion
# ---------------------------------------------------------------------------

def _fila(sub, nombre, malas, m, v, nivel, malas_autores):
    """Fila de resultados de una variante a partir del vector de epocas rechazadas."""
    res = n2p2_por_nivel(v, ~malas)
    fila = {"sujeto": sub, "variante": nombre, "rechazadas": int(malas.sum()), "limpias": int((~malas).sum())}
    for c in config.NIVELES:
        fila[f"n_{c}"], fila[f"N2P2_{c}"] = res[c]
    fila["gradiente_E4_E1"] = fila["N2P2_E4"] - fila["N2P2_E1"]
    ok = ~malas
    fila["spearman_nivel_N2P2"] = (spearmanr(nivel[ok], m["n2p2"][ok])[0]
                                   if ok.sum() >= 8 and len(set(nivel[ok])) > 1 else np.nan)
    if malas_autores is not None:
        propias = set(np.flatnonzero(malas))
        fila["autores_malas"] = len(malas_autores)
        fila["coinciden"] = len(propias & malas_autores)
        fila["solo_nuestras"] = len(propias) - fila["coinciden"]
    return fila


def comparar_sujeto(ep, sub, malas_autores=None, electrodos_malos=()):
    """DataFrame con una fila por variante para este sujeto (y las medidas, para fuga_veo)."""
    m = medidas_por_epoca(ep)
    nivel = ep.metadata["condicion"].map(config.NIVELES).to_numpy()
    frontales = list(getattr(config, "CANALES_SIN_CRITERIO_EEG", []))
    filas = []
    cerca_ojos = canales_cerca_de_los_ojos(m["canales"])
    for nombre, d in VARIANTES.items():
        if d["malos"] and not electrodos_malos:
            continue                      # sin CSV de los autores no se puede calcular
        ignorar = ((frontales if d["frontales"] else []) + (list(electrodos_malos) if d["malos"] else [])
                   + (cerca_ojos if d["lejanos"] else [])
                   + ([c for c in m["canales"] if c != config.CANAL_QC] if d["solo_cz"] else [])
                   + ([c for c in m["canales"] if c not in CANALES_CENTRALES_CRITERIO] if d["centrales"] else []))
        malas = malas_segun_variante(m, d["uv"], d["ue"], ignorar, d["ventana"], d["residual"], d["n2p2"])
        filas.append(_fila(sub, nombre, malas, m, m["v"], nivel, malas_autores))
    if config.CANAL_EOG in ep.ch_names:
        mc = medidas_por_epoca(ep, corregir_veo=True)
        m["corregido"] = mc
        for nombre, ventana in VARIANTES_REGRESION.items():
            malas = malas_segun_variante(mc, None, 100, frontales, ventana)
            filas.append(_fila(sub, nombre, malas, mc, mc["v"], nivel, malas_autores))
    return pd.DataFrame(filas), m


def fuga_veo(m, ep, corregido=False):
    """Spearman (por nivel, epocas sin rechazar) entre el pico a pico de VEO y el N2-P2 del ensayo."""
    cond = ep.metadata["condicion"].to_numpy()
    if corregido:
        m = {**m, "n2p2": m["corregido"]["n2p2"]}
    out = {}
    for c in config.NIVELES:
        i = cond == c
        ok = i & np.isfinite(m["ptp_veo"])
        out[c] = spearmanr(m["ptp_veo"][ok], m["n2p2"][ok])[0] if ok.sum() >= 8 else np.nan
    return out


def correr(sujetos, csv_autores=None, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    salida = carpeta / "sensibilidad"
    salida.mkdir(parents=True, exist_ok=True)
    tablas, fugas = [], []
    for sub in sujetos:
        ruta = segmentacion.ruta_epocas(sub, carpeta)
        if not ruta.exists():
            print(f"{sub}: no estan las epocas ({ruta}). Corre antes el pipeline.")
            continue
        m_aut = marcas_autores.marcas_de(sub, csv_autores)
        ep = marcas_autores.aplicar_marcas(mne.read_epochs(ruta, verbose="ERROR"), m_aut)
        t, m = comparar_sujeto(ep, sub)
        electrodos = m_aut.electrodos
        t["electrodos_malos_autores"] = " ".join(electrodos)
        tablas.append(t)
        fila_f = {"sujeto": sub, **{f"rho_VEO_N2P2_{c}": r for c, r in fuga_veo(m, ep).items()}}
        if "corregido" in m:
            fila_f.update({f"rho_corregido_{c}": r for c, r in fuga_veo(m, ep, corregido=True).items()})
            b = m["corregido"]["beta"]
            fila_f["beta_VEO_Cz"] = b.get(config.CANAL_QC, np.nan)
        fugas.append(fila_f)
        print(f"\n=== {sub}  (electrodos malos de los autores: {' '.join(electrodos) or 'sin dato'}) ===")
        cols = ["variante", "rechazadas", "n_E1", "n_E2", "n_E3", "n_E4",
                "N2P2_E1", "N2P2_E2", "N2P2_E3", "N2P2_E4", "gradiente_E4_E1"]
        cols += [c for c in ("coinciden", "autores_malas", "solo_nuestras") if c in t]
        print(t[cols].round(1).to_string(index=False))
    if not tablas:
        return None, None
    todo = pd.concat(tablas, ignore_index=True)
    fuga = pd.DataFrame(fugas)
    todo.to_csv(salida / "sensibilidad.csv", index=False)
    fuga.to_csv(salida / "fuga_veo.csv", index=False)
    print("\nFuga VEO -> N2-P2 (Spearman por nivel, todas las epocas; 'corregido' = tras la regresion):")
    print(fuga.round(2).to_string(index=False))
    print(f"\nGuardado en {salida}")
    return todo, fuga


def main(argv=None):
    p = argparse.ArgumentParser(description="Sensibilidad del criterio de rechazo de artefactos")
    p.add_argument("sujetos", nargs="+")
    p.add_argument("--csv-autores", default=None,
                   help="'all experiments_preprocessing.csv' del repo de los autores (supplement-data/CSV)")
    a = p.parse_args(argv)
    correr(a.sujetos, a.csv_autores)


if __name__ == "__main__":
    main()