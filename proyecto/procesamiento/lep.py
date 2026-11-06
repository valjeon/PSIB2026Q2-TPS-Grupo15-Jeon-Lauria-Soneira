"""Paso 4: LEP de cada sujeto para cada intensidad (E1..E4), medido en la ROI.

Etapa de grupo: se corre despues de la limpieza, con las epocas limpias de los sujetos incluidos.
"""
import numpy as np

from procesamiento import config


def para_lep(ep):
    """Copia para medir el LEP: pasa-bajos de 30 Hz (la banda del LEP) y linea de base de -1 a 0 s."""
    return (ep.copy().filter(None, config.PASA_BAJOS_LEP_HZ, picks=["eeg", "eog"], verbose="ERROR")
            .apply_baseline(config.BASELINE, verbose="ERROR"))

def promedio_ponderado(x, t):
    """Promedio inhomogeneo (caso 2): cada epoca pesa 1 / varianza de su ruido.

    x: epocas x canales x tiempo. El ruido de cada epoca se estima en el pre-estimulo (t < 0), donde no hay
    respuesta. Los pesos se calculan por canal y suman 1. Si todas las epocas tienen el mismo ruido, da lo
    mismo que el promedio comun; si alguna es mas ruidosa, pesa menos.
    """
    var = x[:, :, t < 0].var(axis=2)                           # epocas x canales
    w = (1 / var) / (1 / var).sum(axis=0, keepdims=True)
    return (w[:, :, None] * x).sum(axis=0)                     # canales x tiempo

def snr_par_impar(x, t, tramo, roi):
    """SNR (dB) del promedio de la ROI, con subpromedios de epocas pares e impares.

    La semidiferencia (pares - impares) / 2 cancela la respuesta (esta igual en los dos) y deja un ruido de la
    misma varianza que el que queda en el promedio completo. SNR = potencia del promedio / potencia de esa
    semidiferencia, dentro de `tramo` (mascara booleana sobre t; la zona del LEP). roi: indices de canales.
    """
    roi_de = lambda y: promedio_ponderado(y, t)[roi].mean(axis=0)     # promedio de la ROI
    senal = roi_de(x)
    ruido = (roi_de(x[0::2]) - roi_de(x[1::2])) / 2
    return 10 * np.log10((senal[tramo] ** 2).mean() / (ruido[tramo] ** 2).mean())