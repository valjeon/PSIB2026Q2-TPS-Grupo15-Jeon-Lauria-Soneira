"""Rechazo final de épocas y medicion por ensayo en la ROI.

Para cada época de los sujetos incluídos en la etapa previa se mide:
    N2_uV, P2_uV, N2P2_uV   amplitud media de la señal de la ROI en las ventanas del localizador, y P2-N2
    ptp_roi_uV, ptp_veo_uV  pico a pico máximo de la señal en la ROI y del VEO
se agrega también la energía del pulso (laser_J) y el puntaje de dolor informado (rating, NRS 0-10)

Una época se rechaza si ptp_roi > 100 uV o ptp_VEO > umbral de VEO de la variante. Se calculan tres variantes
sobre las mismas épocas: veo150 (principal), veo200 y sinveo (sin rechazo por VEO), las dos últimas variantes
son para ver si el resultado depende del rechazo por parpadeo.

Salida en derived_mine/medicion/: rasgos_ensayos.csv (una fila por epoca) e inclusion_variantes.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from procesamiento import config, inclusion, limpieza, localizador

CANALES_ROI = config.CANALES_ROI
UMBRAL_EEG = config.UMBRAL_RECHAZO_UV
UMBRAL_VEO = config.UMBRAL_VEO_UV
UMBRALES_SENSIBILIDAD = config.UMBRALES_VEO_SENSIBILIDAD


def nombre_variante(umbral_veo):
    """Nombre corto de una variante segun su umbral de VEO: 150 -> veo150, 200 -> veo200, None -> sinveo."""
    return "sinveo" if umbral_veo is None else f"veo{umbral_veo:g}"

# creación de las variantes según el umbral de VEO
VARIANTES = {nombre_variante(u): u for u in [UMBRAL_VEO] + list(UMBRALES_SENSIBILIDAD)}
PRINCIPAL = nombre_variante(UMBRAL_VEO)


def _media_en(x, t, ventana):
    """Amplitud media de cada epoca dentro de una ventana."""
    ini, fin = ventana
    en_ventana = (t >= ini) & (t <= fin)    # toma las muestras dentro de una ventana
    return x[:, en_ventana].mean(axis=1)    # promedio en el tiempo, para cada epoca


def medir_sujeto(sub, ventanas):
    """Mide cada epoca de un sujeto y se fija si se rechaza en cada variante."""
    v, _ = limpieza.para_analisis(sub)      # epocas preprocesadas en limpieza
    t = v.times
    ini_tramo, fin_tramo = config.TMIN, ventanas["P2"][1]   # tramo del rechazo: desde -1s hasta el fin de la ventana de P2
    en_tramo = (t >= ini_tramo) & (t <= fin_tramo)

    canales = [c for c in CANALES_ROI if c in v.ch_names]
    roi = v.get_data(picks=canales) * 1e6       # epocas x canales x t
    senal = roi.mean(axis=1)        # promedio de los canales de la ROI (señal)

    # tabla: una fila por epoca con su sujeto, sesion, ensayo, nivel y condicion
    columnas = [c for c in ("sujeto", "sesion", "ensayo", "nivel", "condicion") if c in v.metadata]
    d = v.metadata.reset_index(drop=True)[columnas].copy()      # copia de la metadata
    d["sujeto"] = sub

    # rasgos del LEP en cada epoca
    d["N2_uV"] = _media_en(senal, t, ventanas["N2"])
    d["P2_uV"] = _media_en(senal, t, ventanas["P2"])
    d["N2P2_uV"] = d["P2_uV"] - d["N2_uV"]

    # pico a pico para aplicar el rechazo: el mayor entre los canales de la ROI (cada canal por separado)
    roi_tramo = roi[:, :, en_tramo]
    ptp_por_canal = roi_tramo.max(axis=2) - roi_tramo.min(axis=2)
    d["ptp_roi_uV"] = ptp_por_canal.max(axis=1)     # busca el peor canal de cada época
    
    if config.CANAL_EOG in v.ch_names:  # pico a pico del VEO
        veo = v.copy().pick(config.CANAL_EOG).filter(None, config.PASA_BAJOS_LEP_HZ, picks="all", verbose="ERROR")
        e = veo.get_data()[:, 0, en_tramo] * 1e6        # epocas de un solo canal
        d["ptp_veo_uV"] = e.max(axis=1) - e.min(axis=1)
    else:
        d["ptp_veo_uV"] = np.nan        # si no tiene VEO no se va a poder rechazar por parpadeo

    # rechazo en cada variante, siempre se aplica ROI > 100 uV antes, después el VEO
    for nombre, umbral_veo in VARIANTES.items():
        mala = d["ptp_roi_uV"] > UMBRAL_EEG     # rechaza por umbral EEG
        if umbral_veo is not None:
            mala |= d["ptp_veo_uV"] > umbral_veo    # rechaza por umbral VEO
        d[f"rechazada_{nombre}"] = mala.to_numpy()  # distintas columnas segun la variante de rechazo
    return v, d.round(3)


def correr(sujetos=None, carpeta=None):
    """Mide todas las epocas, aplica el rechazo de cada variante y la inclusion.
    Guarda las tablas y las epocas finales."""
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    sujetos = sujetos or inclusion.incluidos("previa", carpeta)     # sujetos que pasaron la etapa previa
    ventanas = localizador.leer_ventanas(carpeta)               # ventanas de N2 y P2 del localizador
    print(f"{len(sujetos)} sujetos | ventanas N2 {ventanas['N2']} s, P2 {ventanas['P2']} s | "
          f"rechazo: ROI > {UMBRAL_EEG} uV o VEO > umbral | variantes: {list(VARIANTES)} (principal {PRINCIPAL})")

    # se hace la medición, rechazo e inclusion de cada sujeto
    tablas, filas_inc = [], []
    for sub in sujetos:
        v, d = medir_sujeto(sub, ventanas)      # epocas y tabla de rasgos y rechazo
        tablas.append(d)        # se agrega a la tabla vacía
        for nombre in VARIANTES:        # variantes de rechazo de VEO
            no_rechazadas = ~d[f"rechazada_{nombre}"].to_numpy()    # True en las epocas que se quedan (definido luego de aplicar medir_sujeto)
            quedan = v[np.flatnonzero(no_rechazadas)]   # devuelve los indices de las epocas que se quedan
            valores, motivos = inclusion.criterios_finales(quedan)  # criterios finales con las epocas que quedaron (estan en inclusion.py)
            filas_inc.append({"sujeto": sub, "variante": nombre, **valores,
                              "incluido": not motivos, "motivo": "; ".join(motivos)})
            if nombre == PRINCIPAL:     # solo las que pasan la principal se guardan como epocas finales, el resto son para analizar sensibilidad de los umbrales
                quedan.save(inclusion.ruta_finales(sub, carpeta), overwrite=True, verbose="ERROR")

    # tablas con todos los sujetos juntos
    rasgos = pd.concat(tablas, ignore_index=True)       # tabla: una fila por epoca, todos los sujetos
    rasgos = agregar_estimulo(rasgos)   # agrega laser_J y rating a cada epoca
    inc = pd.DataFrame(filas_inc)       # otra tabla: una fila por sujeto y variante
    salida = carpeta / "medicion"
    salida.mkdir(parents=True, exist_ok=True)
    rasgos.to_csv(salida / "rasgos_ensayos.csv", index=False)   # crea los csv
    inc.to_csv(salida / "inclusion_variantes.csv", index=False)

    # resumen en consola
    print("\nPor variante: epocas rechazadas por nivel (suma) y sujetos incluidos")
    for nombre in VARIANTES:
        rech = rasgos.loc[rasgos[f"rechazada_{nombre}"], "condicion"].value_counts()    # rechazadas por nivel
        inc_var = inc[inc["variante"] == nombre]  # extrae las filas de inc, solo de esa variante
        print(f"  {nombre:7s} rechazadas " + "  ".join(f"{c} {int(rech.get(c, 0))}" for c in config.NIVELES)
              + f"  |  incluidos {int(inc_var['incluido'].sum())} de {len(inc_var)}")
    excluidos = inc[(inc["variante"] == PRINCIPAL) & ~inc["incluido"]]   # sujetos excluidos de la variante principal
    for _, fila in excluidos.iterrows():
        print(f"    excluido ({PRINCIPAL}) {fila['sujeto']}: {fila['motivo']}")
    print(f"\nGuardado en {salida}. Epocas finales ({PRINCIPAL}): sub-XXX_finales-epo.fif "
          f"(la inclusion final la hace pipeline.py)")
    return rasgos, inc
# rasgos: una fila por epoca y por sujeto
# inc: una fila por sujeto y por variante


def agregar_estimulo(rasgos, ruta=None):
    """Agrega a cada epoca la energia del pulso (laser_J, J) y el puntaje de dolor informado (rating, NRS 0-10).

    Los toma del CSV por ensayo de los autores (config.RUTA_RATINGS); si no esta, devuelve la tabla sin cambios.
    """
    ruta = Path(config.RUTA_RATINGS if ruta is None else ruta)
    if not ruta.is_file():                                   # sin csv el pipeline sigue igual
        print(f"Aviso: no esta {ruta}: se sigue sin energia ni puntaje por ensayo")
        return rasgos
    csv_aut = pd.read_csv(ruta)
    marcas = limpieza.leer_marcas()      # epocas malas de los autores, se usa para saber a que ensayo corresponde cada fila de los autores

    # se empareja cada fila del csv con su ensayo correspondiente
    # se hace porque el csv solo lista los ensayos despues de sacar las epocas malas que detectaron, se alteran los indices,
    # el final se ese csv tiene las filas restantes con energia 0
    filas = []
    for sub in rasgos["sujeto"].unique():
        # filas del sujeto con energia > 0 (las de energia 0 son relleno), en el orden de los ensayos
        id_num = int(sub[4:])       # extrae el numero del sujeto
        es_del_sujeto = csv_aut["ID"] == id_num     # se fija si la fila es del sujeto correspondiente del for
        no_relleno = csv_aut["laser_power"] > 0
        g = csv_aut[es_del_sujeto & no_relleno].sort_values("Trial_Num")    # una fila por ensayo que los autores conservaron, en orden
        malas = marcas.get(sub, (set(), []))[0]     # ensayos que descartaron los autores
        quedan = [i for i in range(config.EPOCAS_ESPERADAS) if i not in malas]   # ensayos que quedaron, en orden
        if len(quedan) == len(g):                            # misma cantidad: se pueden emparejar en orden
            filas.append(pd.DataFrame({"sujeto": sub, "ensayo": quedan, "laser_J": g["laser_power"].to_numpy(),
                                       "rating": g["Painrating"].to_numpy()}))
        # si no coinciden, el sujeto queda sin estos datos, no se puede hacer el emparejamiento
    if filas:   # si hay al menos un sujeto emparejado
        t = pd.concat(filas, ignore_index=True)              # sujeto, ensayo, laser_J, rating
    else:
        t = pd.DataFrame(columns=["sujeto", "ensayo", "laser_J", "rating"])     # tabla vacía con las mismas columnas

    # agregar laser_J y rating a la tabla de epocas (rasgos), por sujeto y epoca
    out = rasgos.drop(columns=["laser_J", "rating"], errors="ignore")   # saca columnas viejas de laser_J y rating si las hubiera
    out = out.merge(t, on=["sujeto", "ensayo"], how="left")     # agrega los datos

    # control: el orden de las energias de cada sujeto tiene que ser el de E1-E4; si no, se descarta ese sujeto
    rango = out.groupby("sujeto")["laser_J"].rank(method="dense")   # la energia mas baja es 1, y la mas alta es 4
    no_coincide = out["laser_J"].notna() & (rango != out["condicion"].map(config.NIVELES))  # se fija si el orden de energia coincide con el nivel
                                                                                            # sino, es porque el emparejamiento con el csv quedo corrido
    mal = out.loc[no_coincide, "sujeto"].unique()   # sujetos con algun ensayo desalineado
    out.loc[out["sujeto"].isin(mal), ["laser_J", "rating"]] = np.nan    # se borran los datos de los sujetos desalineados

    sin = sorted(set(out.loc[out["laser_J"].isna(), "sujeto"])) # sujetos que quedaron sin datos
    print(f"Energia y puntaje por ensayo: {out['rating'].notna().sum()} epocas con puntaje"
          + (f"; sin alinear: {sin}" if sin else ""))
    return out
# out: tabla rasgos + laser_J + rating (NaN si no hay alguno de esos datos)

def datos_variante(variante=None, carpeta=None):
    """Rasgos de los ensayos no rechazados de los sujetos incluidos en la variante.
    Se usan para los parametros estadisticos"""
    variante = PRINCIPAL if variante is None else variante  # veo150 por defecto
    salida = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "medicion"
    rasgos = pd.read_csv(salida / "rasgos_ensayos.csv")     # una fila por epoca
    inc = pd.read_csv(salida / "inclusion_variantes.csv")   # una fila por sujeto y variante

    
    es_variante = inc["variante"] == variante   # sujetos de esta variante
    incluidos = inc.loc[es_variante & inc["incluido"], "sujeto"]    # sujetos de esta variante e incluidos

    # epocas de esos sujetos que no fueron rechazados por esta variante
    de_incluido = rasgos["sujeto"].isin(incluidos)      # True si la epoca es de un sujeto incluido
    no_rechazada = ~rasgos[f"rechazada_{variante}"]     # True si la epoca no es rechazada
    return rasgos[de_incluido & no_rechazada].reset_index(drop=True)
# devuelve una tabla con una fila por epoca valida de esta variante, mismas columnas de rasgos_ensayos.csv
