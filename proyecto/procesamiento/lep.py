"""Paso 4: LEP de cada sujeto para cada intensidad (E1..E4), medido en la ROI.

Etapa de grupo: se corre despues de la limpieza, con las epocas limpias de los sujetos incluidos.
"""
import numpy as np
from procesamiento import config


# prepara una copia de la sepocas para medir el LEP
def para_lep(ep):
    # copia las epocas, les aplica un pasabajos de 30Hz y vuelve a aplicar la linea de base
    return (ep.copy().filter(None, config.PASA_BAJOS_LEP_HZ, picks=["eeg", "eog"], verbose="ERROR")
            .apply_baseline(config.BASELINE, verbose="ERROR"))


# calcula el promedio de las épocas dándole menos peso a las más ruidosas
# metodologia de potenicales evocados 
def promedio_ponderado(x, t):
    var = x[:, :, t < 0].var(axis=2) # estima el ruido de cada epoca al tomar la varianza de un tiempo antes del estimulo (solo EEG ruidoso)
    # epocas x canales
    w = (1 / var) / (1 / var).sum(axis=0, keepdims=True) # cada epoca recibe un peso de 1/varianza 
    # mas ruidos = menos peso
    # despues multiplica cada epoca x su peso y suma sobre las epocas
    return (w[:, :, None] * x).sum(axis=0) # canales x tiempo


# calcula la relación señal/ruido (en dB) del LEP promedio en la región de interés (ROI)
# el ruido se estima sin necesidad de saber cuál es la respuesta verdadera
def snr_par_impar(x, t, tramo, roi):
    roi_de = lambda y: promedio_ponderado(y, t)[roi].mean(axis=0) # promedio de la ROI
    senal = roi_de(x) # aplica el promedio a todas las epocas (LEP promedio de la ROI)
    ruido = (roi_de(x[0::2]) - roi_de(x[1::2])) / 2 # arma 2 subpromedio (pares e impares)
    # la respuesta está igual en los dos subpromedios, así que se cancela al restar
    # solo queda el ruido
    return 10 * np.log10((senal[tramo] ** 2).mean() / (ruido[tramo] ** 2).mean()) 
    # hace la potencia media (el valor al cuadrado promediado) de la señal y del ruido solo dentro de tramo
    # divide una por otra y pasa a decibeles