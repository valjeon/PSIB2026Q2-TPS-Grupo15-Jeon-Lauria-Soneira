"""Paso 2: extraccion de epocas, como en el paper.

Por sesion: leer (analisis.py) -> pulsos -> filtrar la senal continua (filtrado.py) -> epocas de -1000 a 2000 ms
con linea de base -1000 a 0 ms. Despues se juntan las seis sesiones, se remuestrea a 250 Hz y se guarda.
No se rechaza ni se re-referencia nada: eso es la limpieza, un paso aparte.
"""
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from procesamiento import analisis, config, filtrado


# función que arma una tabla con una fila por pulso laser de una sesion
def metadata_sesion(raw, ev, sub, ses):
    nombre = {v: k for k, v in config.NIVELES.items()} # hace el proceso inverso de obtención de intensidades (pasa de numero 1 2 3 4 a E1 E2 E3 E4)
    return pd.DataFrame({"sujeto": sub, "sesion": ses, "nivel": ev[:, 2], # nivel es el codgio de cada intensidad
                         "condicion": [nombre[n] for n in ev[:, 2]], # condicion es el nombre de cada intensidad
                         "onset_s": (ev[:, 0] - raw.first_samp) / raw.info["sfreq"]}) # momento en el que ocurre el pulso dentor de la sesión (en segs)
                        # MNE cuenta las muestras sumando first_samp (un desfasaje del inicio del archivo)
 

# corta la señal continua (ya filtrada) en épocas alrededor de cada pulso
def epocas_sesion(filt, ev, metadata):
    return mne.Epochs(filt, ev, event_id=config.NIVELES, tmin=config.TMIN, tmax=config.TMAX, # ventana Tmin Tmax es -1 y 2 segs respectivamente
                      # event_id=config.NIVELES para saber qué códigos corresponden a E1–E4
                      baseline=config.BASELINE, metadata=metadata, reject=None, preload=True, # no descarta nada y copia los datos a memoria
                      # BASELINE le resta a cada canal su promedio antes del estímulo
                      on_missing="ignore", verbose="ERROR") # para que no dé error si en esa sesión falta algún nivel


# función principal
# procesa todas las sesiones de un sujeto y devuelve un único objeto con todas sus épocas
def extraer_sujeto(sub, raiz=None):
    lista = []
    for vhdr in analisis.listar_sesiones(sub, raiz): # para cada sesion:
        filt, ev, ses, raw = filtrado.leer_y_filtrar(vhdr) # lee el archivo, filtra la señal continua y detecta los pulsos
        if len(ev) == 0:
            print(f"  {sub} {ses}: sin pulsos laser, se saltea") # si no hay pulsos, avisa y pasa a la siguiente sesión
            continue
        ep = epocas_sesion(filt, ev, metadata_sesion(raw, ev, sub, ses)) # arma la metadata y corta las épocas con las dos funciones anteriores
        if len(ep) < len(ev): # si salieron menos épocas que pulsos avisa
            # esto pasa cuando un pulso está tan cerca del principio o del final del archivo que no entra la ventana completa
            # MNE lo descarta
            print(f"  {sub} {ses}: {len(ev) - len(ep)} pulsos sin ventana completa (borde del archivo)")
        lista.append(ep) # agrega las épocas a la lista
        del raw, filt # borra el registro continuo para liberar memoria antes de la siguiente sesión
    if not lista:
        raise RuntimeError(f"{sub}: ninguna sesion tuvo pulsos laser") # si ninguna sesión tuvo pulsos, corta con un error
    ep = lista[0] if len(lista) == 1 else mne.concatenate_epochs(lista, verbose="ERROR") # junta las sesiones en un solo objeto
    with mne.use_log_level("ERROR"):
        ep.metadata = ep.metadata.reset_index(drop=True).assign(ensayo=np.arange(len(ep))) # se le asigna un número único por época que se mantiene aunque después se descarten épocas.
    if config.SFREQ_GUARDADO and ep.info["sfreq"] > config.SFREQ_GUARDADO:
        # remuestrear corre un poco la media del pre-estimulo: se vuelve a aplicar la linea de base
        ep = ep.resample(config.SFREQ_GUARDADO, verbose="ERROR").apply_baseline(config.BASELINE, verbose="ERROR")
        # se remuestrea a 250 Hz si la frec original es mayor para que no pesen tanto
        # como el remuestreo mueve un poco el promedio del pre‑estímulo, vuelve a aplicar la línea de base
    return ep


# arma la ruta del archivo donde se guardan las épocas
def ruta_epocas(sub, carpeta=None):
    return Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / sub / f"{sub}_laser-epo.fif"


def guardar(ep, sub, carpeta=None):
    ruta = ruta_epocas(sub, carpeta) # obtiene la ruta del archivo
    ruta.parent.mkdir(parents=True, exist_ok=True) # crea la carpeta del sujeto
    ep.save(ruta, overwrite=True, verbose="ERROR") # guarda las epocas en .fif
    ep.metadata.to_csv(ruta.with_name(f"{sub}_laser-metadata.csv"), index=False) # guarda la metadata en .csv
    return ruta # devuelve la ruta de las epocas (.fif)


# devuelve un diccionario con la info del sujeto
def resumen(ep, sub):
    por_nivel = ep.metadata["condicion"].value_counts() # cuenta cuantas epocas hay de cada condicion
    # arma diccionario con todas esas caracteristicas
    return {"sujeto": sub, "sesiones": ep.metadata["sesion"].nunique(), "epocas": len(ep),
            **{c: int(por_nivel.get(c, 0)) for c in config.NIVELES},
            "completo": len(ep) == config.EPOCAS_ESPERADAS} 


# función que escribe la tabla resumen de todos los sujetos en un CSV
def guardar_resumen(filas, nombre="epocas_resumen.csv", carpeta=None):
    destino = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / nombre
    nuevo = pd.DataFrame(filas)
    if destino.exists():
        viejo = pd.read_csv(destino)
        nuevo = pd.concat([viejo[~viejo["sujeto"].isin(nuevo["sujeto"])], nuevo]).sort_values("sujeto")
    nuevo.to_csv(destino, index=False)
    return destino
