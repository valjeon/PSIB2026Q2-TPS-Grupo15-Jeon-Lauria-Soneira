"""Analisis previo sobre los registros crudos: PSD por sesion y N2-P2 en cada canal.

Se usa de forma independiente del resto del pipeline para poder visualizar la señal cruda y decidir qué pasos hay que incluir en el pipeline.
Se producen 3 salidas por sujeto en derived_mine/analisis/sub-XXX/:
    psd_por_sesion.png    PSD cruda (media entre canales EEG), una curva por sesion: justifica los filtros
    n2p2_por_canal.png    N2-P2 de cada canal (mapa y barras de adelante hacia atras): justifica los electrodos
    n2p2_por_canal.csv    los valores de la figura

El N2-P2 se mide sobre el promedio de los cuatro niveles juntos, con la
senal continua filtrada 1-30 Hz antes de cortar las épocas y sobre la referencia promedio sin los frontopolares.

Uso desde proyecto/:
    python -m procesamiento.analisis sub-001 (reemplazar con el sujeto que se quiera analizar)
"""
import argparse
import re
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from procesamiento import config #Se importan las constantes que quedan definidas en config

FMAX_PSD = 250.0                                       # hasta donde se calcula la PSD cruda
MEDIA_PICO_S = 0.02                                    # se promedia +-20 ms alrededor de cada latencia
NIVELES = list(config.NIVELES)                         

def listar_sesiones(sub, raiz=None):
    #Se abren las rutas .vhdr de las sesiones de un sujeto ordenadas ses-1..ses-6
    raiz = Path(config.RAIZ_DATOS if raiz is None else raiz)
    vhdrs = sorted(raiz.glob(f"{sub}/ses-*/eeg/*_eeg.vhdr"),
                   key=lambda p: int(re.search(r"ses-(\d+)", p.as_posix()).group(1)))
    if not vhdrs:
        raise FileNotFoundError(f"No hay registros de {sub} en {raiz}")
    return vhdrs 

def cargar_sesion(vhdr):
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    raw.drop_channels([c for c in config.CANALES_DESCARTADOS if c in raw.ch_names]) 
    #Se dejan los canales que son descartados por los autores del dataset
    if config.CANAL_EOG in raw.ch_names:
        raw.set_channel_types({config.CANAL_EOG: "eog"}) #se guarda el canal de eog junto al crudo de la señal
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR") #se establece que tipo de montaje se usó en la adquisición
    return raw #devuelve la señal cruda correspondiente al archivo en la ruta vhdr


def pulsos_laser(raw):
    ev, _ = mne.events_from_annotations(raw, verbose="ERROR") # se convierten las anotaciones del curdo en una matriz de eventos
    pulsos = [[ev[k - 1, 0], 0, config.CODIGOS_LASER[ev[k - 1, 2]]] for k in range(1, len(ev))
              #recorre los eventos buscando una secuencia de 2 eventos adyacentes
              if ev[k, 2] == config.CODIGO_CONFIRMACION and ev[k - 1, 2] in config.CODIGOS_LASER] 
                #para cada coincidencia genera un nuevo evento
    return np.array(pulsos, dtype=int).reshape(-1, 3) #se convierte la lista en un array y el reshape sirve por si no se encuentra ningún pulso válido para que no quede un array vacío


def psd_media(raw):
    #Se calcula la PSD de Welch (ventanas de 4 s: 0.25 Hz de resolucion), media entre canales EEG, en uV^2/Hz
    fmax = min(FMAX_PSD, raw.info["sfreq"] / 2)
    psd = raw.compute_psd(method="welch", fmin=0, fmax=fmax, n_fft=int(4 * raw.info["sfreq"]),
                          picks="eeg", verbose="ERROR")
    datos, freqs = psd.get_data(return_freqs=True)
    return freqs, datos.mean(axis=0) * 1e12


def fig_psd(freqs, por_sesion, titulo):
    #Se grafican las PSD de cada sesion del sujeto en una misma figura
    fig, ax = plt.subplots(figsize=(8, 4.6))
    m = freqs >= 0.5
    ax.axvspan(config.PASA_ALTOS_HZ, config.PASA_BAJOS_HZ, color="#e5f4e4", alpha=0.6, zorder=0) 
    #Se marca con color la zona del pasabandas que se aplicaría en el preprosamiento
    for f in (config.PASA_ALTOS_HZ, config.NOTCH_HZ, config.PASA_BAJOS_HZ): 
        #Se marcan con una línea punteada las frecuencias importantes
        ax.axvline(f, color="k", ls="--", lw=0.8)
    for ses, p in por_sesion.items():
        ax.semilogx(freqs[m], 10 * np.log10(p[m]), lw=1.1, label=ses)
    ax.set_xlim(0.3, freqs.max())
    ax.grid(True, which="both", color="0.9", lw=0.6)
    ax.set_xlabel("frecuencia (Hz)")
    ax.set_ylabel("densidad espectral (dB re 1 uV²/Hz)")
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, title="sesion")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def epocas_lep(raw, ev):
   #Para hacer un gráfico del promedio de las épocas, primer se filtra el crudo con un pasabandas de 1-30Hz
   #Se corta la señal  con linea de base -1 a 0 s
    filt = raw.copy().filter(config.PASA_ALTOS_HZ, config.PASA_BAJOS_LEP_HZ, picks=["eeg", "eog"],
                             method="fir", fir_design="firwin", verbose="ERROR")
        #se crea una copia del crudo a la que se le aplican los filtros a ambos el eeg y el eog
    ep = mne.Epochs(filt, ev, event_id=config.NIVELES, tmin=config.TMIN, tmax=config.TMAX,
                    baseline=config.BASELINE, reject=None, preload=True, on_missing="ignore", verbose="ERROR") 
                    #se guardan las épocas
    return ep.resample(config.SFREQ_GUARDADO, verbose="ERROR") #se devuelve como salida las épocas remuestreadas


def promedio_colapsado(ep):
    #Promedio global de las respuestas EEG
    #combinando los promedios de los distintos niveles experimentales
    eeg = ep.copy().pick("eeg") #se toman solo los canales de eeg
    eeg.set_eeg_reference([c for c in eeg.ch_names if c not in config.CANALES_SIN_CRITERIO_EEG], verbose="ERROR")
    #se aplica la referencia promedio con determinados electrodos, excluyendo a los frontopolares
    niveles = [n for n in NIVELES if len(eeg[n])] #se arma una lista con los niveles que tienen épocas
    datos = np.mean([eeg[n].average().get_data() for n in niveles], axis=0) * 1e6 #cálculo del promedio
    return datos, eeg.info, eeg.times #se devuelve una matriz con los canales x tiempo, información del eeg y el vector temporal de las muestras


def n2p2_por_canal(datos, nombres, t):
    #Se calculan las latencias de N2 (minimo) y P2 (maximo) en Cz, y en cada canal la media en +-20 ms alrededor de esas latencias: N2-P2 = P2 - N2
    lat = {}
    for comp, (a, b), f in (("N2", config.VENTANA_N2, np.argmin), ("P2", config.VENTANA_P2, np.argmax)):
        m = (t >= a) & (t <= b)
        lat[comp] = float(t[m][f(x[m])])
    # Se busca en cada ventana un max y un min que se son el N2 y el P2
    media = lambda tc: datos[:, np.abs(t - tc) <= MEDIA_PICO_S].mean(axis=1) #se calculan las amplitudes promedio alrededor de los picos 
    n2, p2 = media(lat["N2"]), media(lat["P2"]) #Las amplitudes se calculan individualmente para todos los electrodos
    tabla = pd.DataFrame({"canal": nombres, "N2_uV": n2.round(2), "P2_uV": p2.round(2), "N2P2_uV": (p2 - n2).round(2)})
    #se arma una tabla con los resultados para cada canal
    return tabla, lat #devuelve la tabla de amplitudes y el diccionario con las latencias


def fig_n2p2(tabla, info, lat, titulo):
    #Mapa topográfico y barras (de adelante hacia atras) del N2-P2 de cada canal
    con_pos = [i for i, ch in enumerate(info["chs"]) if np.isfinite(ch["loc"][:3]).all() and np.any(ch["loc"][:3])]
    #se guardan las posiciones de los electrodos 
    vals = tabla["N2P2_uV"].to_numpy(dtype=float) #se guardan los valores de las amplitudes de los potenciales en un array
    y = np.array([info["chs"][i]["loc"][1] if i in set(con_pos) else -np.inf for i in range(len(vals))])
    orden = np.argsort(-y, kind="stable") #se ordenan los canales de anterior a posterior

    #se arma la figura
    fig, (axm, axb) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 2.6]})
    lim = np.abs(vals[con_pos]).max()
    im, _ = mne.viz.plot_topomap(vals[con_pos], mne.pick_info(info, con_pos), axes=axm, show=False,
                                 cmap="RdBu_r", vlim=(-lim, lim), contours=0, sensors=True)
    cb = fig.colorbar(im, ax=axm, fraction=0.04, pad=0.02, shrink=0.8, location="bottom")
    cb.set_label("N2-P2 (uV)")
    cb.outline.set_visible(False)
    axb.bar(np.arange(len(orden)), vals[orden], color="#2a78d6", width=0.7)
    axb.set_xticks(np.arange(len(orden)))
    axb.set_xticklabels(tabla["canal"].to_numpy()[orden], rotation=90, fontsize=7.5)
    axb.axhline(0, color="k", lw=0.4)
    axb.set_ylabel("N2-P2 (uV)")
    axb.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{titulo}  (N2 {lat['N2'] * 1000:.0f} ms, P2 {lat['P2'] * 1000:.0f} ms en {config.CANAL_QC}; "
                 "barras de adelante hacia atras)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def analizar_sujeto(sub, raiz=None, carpeta=None):
    #Recorre las sesiones una vez: PSD de cada una y epocas 1-30 Hz. Guarda las dos figuras y la tabla.
    psd, epocas, freqs = {}, [], None
    for vhdr in listar_sesiones(sub, raiz): #se recorren las sesiones de cada sujeto
        ses = re.search(r"(ses-\d+)", vhdr.as_posix()).group(1) #se extrae el identificador de sesion
        raw = cargar_sesion(vhdr) #se cargan los archivos crudos
        freqs, psd[ses] = psd_media(raw) #se calcula la PSD
        ev = pulsos_laser(raw) #se extraen las epocas
        if len(ev):
            epocas.append(epocas_lep(raw, ev))
        del raw #una vez que se terminó la extracción se elimina el crudo y se libera memoria
    if not epocas:
        raise RuntimeError(f"{sub}: ninguna sesion tuvo pulsos laser")
    ep = epocas[0] if len(epocas) == 1 else mne.concatenate_epochs(epocas, verbose="ERROR") #se juntan todas las epocas en un objeto
    datos, info, t = promedio_colapsado(ep) #se calcula el promedio 
    tabla, lat = n2p2_por_canal(datos, info["ch_names"], t) #se determinan las latencias de N2 y P2 en Cz y se calculan las amplitudes promedio de ambas componentes en cada canal

    destino = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "analisis" / sub
    destino.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino / "n2p2_por_canal.csv", index=False)
    for nombre, fig in (("psd_por_sesion", fig_psd(freqs, psd, f"{sub}: PSD cruda, media entre canales EEG "
                                                               "(verde: 1-100 Hz; lineas: 1, 50 y 100 Hz)")),
                        ("n2p2_por_canal", fig_n2p2(tabla, info, lat, f"{sub}: N2-P2 por canal, "
                                                                      f"{len(ep)} epocas, los 4 niveles juntos"))):
        fig.savefig(destino / f"{nombre}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    return tabla, lat, len(ep), destino


def main(argv=None):
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description="PSD por sesion y N2-P2 por canal, sobre los registros crudos")
    p.add_argument("sujetos", nargs="+")
    for sub in p.parse_args(argv).sujetos: #se procesan todos los sujetos ingresados
        tabla, lat, n, destino = analizar_sujeto(sub)
        top = tabla.sort_values("N2P2_uV", ascending=False).head(5) #se identifican los 5 canales con mayor amplitud
        print(f"{sub}: {n} epocas; N2 {lat['N2'] * 1000:.0f} ms, P2 {lat['P2'] * 1000:.0f} ms; mayor N2-P2: "
              + ", ".join(f"{c} {v:.1f}" for c, v in zip(top["canal"], top["N2P2_uV"])) + f" uV  -> {destino}")


if __name__ == "__main__":
    main()
