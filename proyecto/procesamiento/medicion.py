"""Rechazo final y medicion por ensayo en la ROI, para la variante principal y las de sensibilidad.

Sujetos: los incluidos en la etapa previa. Epocas: limpieza.para_analisis (ref. sin frontales, 30 Hz,
linea de base). Ventanas de N2/P2: localizador/ventanas.csv.

Por epoca (independiente del rechazo):
    N2_uV, P2_uV   amplitud media de la senal de la ROI (promedio de config.CANALES_ROI) en cada ventana
    N2P2_uV        P2 - N2
    ptp_roi_uV     pico a pico maximo entre los canales de la ROI, de -1 s al fin de la ventana de P2
    ptp_veo_uV     pico a pico del VEO (30 Hz) en el mismo tramo
Rechazo por variante: ptp_roi > UMBRAL_RECHAZO_UV  o  ptp_veo > umbral VEO de la variante
    principal:     config.UMBRAL_VEO_UV                      (guarda sub-XXX_finales-epo.fif)
    sensibilidad:  config.UMBRALES_VEO_SENSIBILIDAD          (None = sin rechazo por VEO)
Inclusion por variante: criterios 1 y 3 (inclusion.criterios_finales) sobre las epocas que quedan.

Salida en derived_mine/medicion/:
    rasgos_ensayos.csv       una fila por epoca: metadata, rasgos y 'rechazada_<variante>' por variante
    inclusion_variantes.csv  una fila por sujeto y variante: epocas por nivel, criterios 1 y 3, incluido
La estadistica de cada variante usa las filas no rechazadas de los sujetos incluidos en esa variante.

Lo llama pipeline.py en la etapa de grupo (python -m procesamiento.pipeline --grupo).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from procesamiento import config, inclusion, limpieza, localizador

CANALES_ROI = getattr(config, "CANALES_ROI", ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"])
UMBRAL_EEG = getattr(config, "UMBRAL_RECHAZO_UV", 100)
UMBRAL_VEO = getattr(config, "UMBRAL_VEO_UV", 150)
UMBRALES_SENSIBILIDAD = getattr(config, "UMBRALES_VEO_SENSIBILIDAD", [200, None])


def nombre_variante(umbral_veo):
    return "sinveo" if umbral_veo is None else f"veo{umbral_veo:g}"


VARIANTES = {nombre_variante(u): u for u in [UMBRAL_VEO] + list(UMBRALES_SENSIBILIDAD)}
PRINCIPAL = nombre_variante(UMBRAL_VEO)


def _media_en(x, t, ventana):
    a, b = ventana
    return x[:, (t >= a) & (t <= b)].mean(axis=1)


def medir_sujeto(sub, ventanas):
    """(epocas para analisis, DataFrame con una fila por epoca: metadata, rasgos y picos para el rechazo)."""
    v, _ = limpieza.para_analisis(sub)
    t = v.times
    tramo = (config.TMIN, ventanas["P2"][1])
    m_tramo = (t >= tramo[0]) & (t <= tramo[1])

    roi = v.get_data(picks=[c for c in CANALES_ROI if c in v.ch_names]) * 1e6       # epocas x canales x t
    senal = roi.mean(axis=1)                                                        # senal de la ROI
    d = v.metadata.reset_index(drop=True)[[c for c in ("sujeto", "sesion", "ensayo", "nivel", "condicion")
                                          if c in v.metadata]].copy()
    d["sujeto"] = sub
    d["N2_uV"] = _media_en(senal, t, ventanas["N2"])
    d["P2_uV"] = _media_en(senal, t, ventanas["P2"])
    d["N2P2_uV"] = d["P2_uV"] - d["N2_uV"]
    r = roi[:, :, m_tramo]
    d["ptp_roi_uV"] = (r.max(axis=2) - r.min(axis=2)).max(axis=1)
    if config.CANAL_EOG in v.ch_names:
        e = v.copy().pick(config.CANAL_EOG).filter(None, config.PASA_BAJOS_LEP_HZ, picks="all",
                                                   verbose="ERROR").get_data()[:, 0, m_tramo] * 1e6
        d["ptp_veo_uV"] = e.max(axis=1) - e.min(axis=1)
    else:
        d["ptp_veo_uV"] = np.nan
    for nombre, u in VARIANTES.items():
        mala = d["ptp_roi_uV"] > UMBRAL_EEG
        if u is not None:
            mala |= d["ptp_veo_uV"] > u
        d[f"rechazada_{nombre}"] = mala.to_numpy()
    return v, d.round(3)


def correr(sujetos=None, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    sujetos = sujetos or inclusion.incluidos("previa", carpeta)
    ventanas = localizador.leer_ventanas(carpeta)
    print(f"{len(sujetos)} sujetos | ventanas N2 {ventanas['N2']} s, P2 {ventanas['P2']} s | "
          f"rechazo: ROI > {UMBRAL_EEG} uV o VEO > umbral | variantes: {list(VARIANTES)} (principal {PRINCIPAL})")

    tablas, filas_inc = [], []
    for sub in sujetos:
        v, d = medir_sujeto(sub, ventanas)
        tablas.append(d)
        for nombre in VARIANTES:
            quedan = v[np.flatnonzero(~d[f"rechazada_{nombre}"].to_numpy())]
            valores, motivos = inclusion.criterios_finales(quedan)
            filas_inc.append({"sujeto": sub, "variante": nombre, **valores,
                              "incluido": not motivos, "motivo": "; ".join(motivos)})
            if nombre == PRINCIPAL:
                quedan.save(inclusion.ruta_finales(sub, carpeta), overwrite=True, verbose="ERROR")

    rasgos = agregar_estimulo(pd.concat(tablas, ignore_index=True))   # suma laser_J y rating a cada epoca
    inc = pd.DataFrame(filas_inc)
    salida = carpeta / "medicion"
    salida.mkdir(parents=True, exist_ok=True)
    rasgos.to_csv(salida / "rasgos_ensayos.csv", index=False)
    inc.to_csv(salida / "inclusion_variantes.csv", index=False)

    print("\nPor variante: epocas rechazadas por nivel (suma) y sujetos incluidos (criterios 1 y 3)")
    for nombre in VARIANTES:
        rech = rasgos[rasgos[f"rechazada_{nombre}"]]["condicion"].value_counts()
        i = inc[inc["variante"] == nombre]
        print(f"  {nombre:7s} rechazadas " + "  ".join(f"{c} {int(rech.get(c, 0))}" for c in config.NIVELES)
              + f"  |  incluidos {int(i['incluido'].sum())} de {len(i)}")
    excl = inc[(inc["variante"] == PRINCIPAL) & ~inc["incluido"]]
    for _, f in excl.iterrows():
        print(f"    excluido ({PRINCIPAL}) {f['sujeto']}: {f['motivo']}")
    print(f"\nGuardado en {salida}. Epocas finales ({PRINCIPAL}): sub-XXX_finales-epo.fif "
          f"(la inclusion final la hace pipeline.py)")
    return rasgos, inc


def agregar_estimulo(rasgos, ruta=None):
    """Agrega a cada epoca la energia del pulso (laser_J) y el puntaje de dolor informado (rating, NRS 0-10).

    Fuente: CSV por ensayo de los autores (config.RUTA_RATINGS). Ese CSV lista, en orden, los ensayos que
    quedan despues de sacar las epocas malas de los autores y rellena el final con ceros. Entonces su k-esima
    fila con energia > 0 es el k-esimo ensayo (columna 'ensayo', 0..79) que no esta en las marcas.
    Control: dentro de cada sujeto el orden de las energias tiene que coincidir con el nivel E1-E4; si no
    coincide (por ejemplo, un sujeto con menos de 80 epocas extraidas) ese sujeto queda sin estos datos.
    Si no esta el CSV, devuelve la tabla sin cambios.
    """
    ruta = Path(getattr(config, "RUTA_RATINGS", "") if ruta is None else ruta)
    if not ruta.is_file():                                   # sin CSV: el pipeline sigue como antes
        print(f"Aviso: no esta {ruta}: se sigue sin energia ni puntaje por ensayo")
        return rasgos
    csv_aut = pd.read_csv(ruta)
    marcas = limpieza.leer_marcas()                          # epocas malas de los autores, numeradas desde 0
    filas = []
    for sub in rasgos["sujeto"].unique():
        # filas del sujeto con energia > 0 (las de energia 0 son relleno), en el orden de los ensayos
        g = csv_aut[(csv_aut["ID"] == int(sub[4:])) & (csv_aut["laser_power"] > 0)].sort_values("Trial_Num")
        malas = marcas.get(sub, (set(), []))[0]
        quedan = [i for i in range(config.EPOCAS_ESPERADAS) if i not in malas]   # ensayos que no marcaron
        if len(quedan) == len(g):                            # misma cantidad: se pueden emparejar en orden
            filas.append(pd.DataFrame({"sujeto": sub, "ensayo": quedan, "laser_J": g["laser_power"].to_numpy(),
                                       "rating": g["Painrating"].to_numpy()}))
    t = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["sujeto", "ensayo", "laser_J", "rating"])
    out = rasgos.drop(columns=["laser_J", "rating"], errors="ignore").merge(t, on=["sujeto", "ensayo"], how="left")
    # control: el orden de las energias de cada sujeto tiene que ser el de E1-E4; si no, se descarta ese sujeto
    rango = out.groupby("sujeto")["laser_J"].rank(method="dense")
    mal = out.loc[out["laser_J"].notna() & (rango != out["condicion"].map(config.NIVELES)), "sujeto"].unique()
    out.loc[out["sujeto"].isin(mal), ["laser_J", "rating"]] = np.nan
    sin = sorted(set(out.loc[out["laser_J"].isna(), "sujeto"]))
    print(f"Energia y puntaje por ensayo: {out['rating'].notna().sum()} epocas con puntaje"
          + (f"; sin alinear: {sin}" if sin else ""))
    return out


def datos_variante(variante=None, carpeta=None):
    """Rasgos de los ensayos no rechazados de los sujetos incluidos en la variante (para la estadistica)."""
    variante = PRINCIPAL if variante is None else variante
    salida = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "medicion"
    rasgos = pd.read_csv(salida / "rasgos_ensayos.csv")
    inc = pd.read_csv(salida / "inclusion_variantes.csv")
    ok = inc[(inc["variante"] == variante) & inc["incluido"]]["sujeto"]
    return rasgos[rasgos["sujeto"].isin(ok) & ~rasgos[f"rechazada_{variante}"]].reset_index(drop=True)
