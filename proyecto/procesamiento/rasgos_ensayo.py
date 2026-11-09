"""
Lo llama pipeline.py en la etapa de grupo (python -m procesamiento.pipeline --grupo).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from procesamiento import config, limpieza, localizador

#se definen los canales a analizar
CANALES_ROI = getattr(config, "CANALES_ROI", ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"])
VENTANA_STFT_S = 0.25
PASO_STFT_S = 0.02
BASE_ESPECTRAL = (-0.4, -0.1)
BANDAS = {   # nombre: (canales, (f_min, f_max) Hz, (t_min, t_max) s)
    "gamma_ERS_dB": (CANALES_ROI, (60, 95), (0.2, 0.5)),
}
COLUMNAS_NUEVAS = ["lat_N2_ms"] + list(BANDAS)


def latencia_n2(v, ventana_n2):
    #Latencia (ms) del minimo de la senal de la ROI en la ventana de N2, por epoca
    t = v.times
    m = (t >= ventana_n2[0]) & (t <= ventana_n2[1]) #se obtienen los instantes dentro de la ventana N2
    senal = v.get_data(picks=[c for c in CANALES_ROI if c in v.ch_names]).mean(axis=1)
    #extrae las señales de los canales de la ROI y calcula su promedio espacial
    return t[m][np.argmin(senal[:, m], axis=1)] * 1000 #devuelve la posición del mínimo y se multiplica por 1000 para expresar el resultado en milisegundos


def cambio_potencia(x, sfreq, tmin, banda, ventana, base=BASE_ESPECTRAL):
    #función que realiza el análisis espectral
    nper = int(round(VENTANA_STFT_S * sfreq))
    paso = int(round(PASO_STFT_S * sfreq))
    #convierte la duración de las ventanas y el desplazamiento a cantidad de muestras, según la frecuencia de muestreo
    f, tt, p = spectrogram(x, fs=sfreq, window="hann", nperseg=nper, noverlap=nper - paso,
                           scaling="density", mode="psd", axis=-1)           #se calcula el espectrograma
    tt = tt + tmin #centros en tiempo de la epoca
    mf = (f >= banda[0]) & (f <= banda[1]) #selecciona las frecuencias de la banda a analizar
    pot = lambda a, b: p[:, mf][:, :, (tt >= a) & (tt <= b)].mean(axis=(1, 2)) #calcula la potencia media de cada ensayo dentro de la banda de frecuencias y la ventana temporal
    return 10 * np.log10(pot(*ventana) / pot(*base)) #calcula el cambio relativo de potencia


def rasgos_sujeto(sub, ventanas):
    #se reunen los calculos anteriores para un sujeto
    v30, _ = limpieza.para_analisis(sub)  # 30 Hz para la latencia
    vfull, _ = limpieza.para_analisis(sub, pasa_bajos=False)   # 1-100 Hz para los espectrales, la gamma está en 60-95 Hz
    d = pd.DataFrame({"sujeto": sub, "ensayo": v30.metadata["ensayo"].to_numpy(),
                      "lat_N2_ms": latencia_n2(v30, ventanas["N2"])})
    #se construye un data frame donde cada fila representa un ensayo y contiene su identificación y latencia N2
    sf, tmin = vfull.info["sfreq"], vfull.tmin
    for nombre, (canales, banda, ventana) in BANDAS.items():
        x = vfull.get_data(picks=[c for c in canales if c in vfull.ch_names]).mean(axis=1) * 1e6
        #extrae los canales de la ROI, calcula el promedio espacial y convierte la señal de voltios a microvoltios
        d[nombre] = cambio_potencia(x, sf, tmin, banda, ventana) #calcula el cambio de potencia para cada ensayo
    return d.round(3) #devuelve el dataframe redondeado


def correr(carpeta=None):
    #función que integra el módulo con el resto del pipeline
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    ruta = carpeta / "medicion" / "rasgos_ensayos.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No esta {ruta}. Corre antes: python -m procesamiento.pipeline --grupo")
    base = pd.read_csv(ruta).drop(columns=COLUMNAS_NUEVAS, errors="ignore") #carga el archivo y elimina las columnas que va a recalcular
    ventanas = localizador.leer_ventanas(carpeta) #obtiene las ventanas
    nuevos = pd.concat([rasgos_sujeto(s, ventanas) for s in base["sujeto"].unique()], ignore_index=True) #calcula los nuevos rasgos para todos los sujetos
    todo = base.merge(nuevos, on=["sujeto", "ensayo"], how="left", validate="one_to_one") #combina las mediciones originales con las nuevas, utilizando como identificadores el sujeto y el número de ensayo
    todo.to_csv(ruta, index=False)
    print(f"Rasgos agregados a {ruta}: {', '.join(COLUMNAS_NUEVAS)}")
    print(todo.groupby("condicion")[["N2P2_uV"] + COLUMNAS_NUEVAS].median().round(2).to_string()) #muestra estadísticas descriptivas agrupadas por condición, usando la mediana
    return todo
