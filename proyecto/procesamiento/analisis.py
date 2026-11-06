# hacemos un análisis de la señal cruda para ver qué necesitamos incluir en el preprocesamiento
"""
Uso desde proyecto/:
    python -m procesamiento.analisis sub-004 sub-005
"""
import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from procesamiento import config

# Lectura
CANALES_DESCARTADOS = ["TP9", "TP10", "HEOR"]          # los descartan los autores
CANAL_EOG = "VEO"
CODIGOS_LASER = {32: 1, 33: 2, 34: 3, 35: 4}           # trigger S 3x -> nivel E1..E4
CODIGO_CONFIRMACION = 100                              # S 100 confirma el disparo
# PSD
FMAX_PSD = 250.0                                       # Hz
BANDA_FILTRO = (1.0, 100.0)                            # Hz; se sombrea en la figura
LINEAS_HZ = (1.0, 50.0, 100.0)
# N2-P2
FILTRO_LEP = (1.0, 30.0)                               # Hz; sobre la senal continua
TMIN, TMAX, BASELINE = -1.0, 2.0, (-1.0, 0.0)          # s; como el paper
SFREQ_EPOCAS = 250                                     # Hz; las epocas se remuestrean para ocupar menos memoria
FRONTOPOLARES = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AF4", "AF8"]   # fuera de la referencia promedio
CANAL_LATENCIAS = "Cz"                                 # donde se buscan los picos
VENTANA_N2, VENTANA_P2 = (0.12, 0.30), (0.25, 0.50)    # s
MEDIA_PICO_S = 0.02                                    # se promedia +-20 ms alrededor de cada latencia



# Lectura de los archivos

def listar_sesiones(sub, raiz=None):
    #guarda las rutas de cada sesión del sujeto
    raiz = Path(config.RAIZ_DATOS if raiz is None else raiz)
    vhdrs = sorted(raiz.glob(f"{sub}/ses-*/eeg/*_eeg.vhdr"),
                   key=lambda p: int(re.search(r"ses-(\d+)", p.as_posix()).group(1)))
    if not vhdrs:
        raise FileNotFoundError(f"No hay registros de {sub} en {raiz}")
    return vhdrs


def cargar_sesion(vhdr):
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    raw.drop_channels([c for c in CANALES_DESCARTADOS if c in raw.ch_names]) 
    #se eliminan los canales que ya son descartados por el paper
    if CANAL_EOG in raw.ch_names:
        raw.set_channel_types({CANAL_EOG: "eog"})
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    return raw


def pulsos_laser(raw):
    ev, _ = mne.events_from_annotations(raw, verbose="ERROR")
    pulsos = [[ev[k - 1, 0], 0, CODIGOS_LASER[ev[k - 1, 2]]] for k in range(1, len(ev))
              if ev[k, 2] == CODIGO_CONFIRMACION and ev[k - 1, 2] in CODIGOS_LASER]
    return np.array(pulsos, dtype=int).reshape(-1, 3)


# 1. PSD

def psd_media(raw):
    # se calcula la PSD de Welch (ventanas de 4 s: 0.25 Hz de resolucion) media entre canales EEG
    fmax = min(FMAX_PSD, raw.info["sfreq"] / 2)
    psd = raw.compute_psd(method="welch", fmin=0, fmax=fmax, n_fft=int(4 * raw.info["sfreq"]),
                          picks="eeg", verbose="ERROR")
    datos, freqs = psd.get_data(return_freqs=True)
    return freqs, datos.mean(axis=0) * 1e12


def fig_psd(freqs, por_sesion, titulo):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    m = freqs >= 0.5
    ax.axvspan(*BANDA_FILTRO, color="#e5f4e4", alpha=0.6, zorder=0)
    for f in LINEAS_HZ:
        ax.axvline(f, color="k", ls="--", lw=0.8)
    for ses, p in por_sesion.items():
        ax.semilogx(freqs[m], 10 * np.log10(p[m]), lw=1.1, label=ses)
    ax.set_xlim(0.3, freqs.max())
    ax.grid(True, which="both", color="0.9", lw=0.6)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Densidad espectral (uV²/Hz)")
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, title="sesion")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# 2. N2-P2 por canal

def epocas_lep(raw, ev):
    # se aplica un filtro pasabandas de 1-30 Hz y se corta la señal de -1 a 2s con línea de base de -1 a 0s
    filt = raw.copy().filter(*FILTRO_LEP, picks=["eeg", "eog"], method="fir", fir_design="firwin", verbose="ERROR")
    ep = mne.Epochs(filt, ev, event_id={"E1": 1, "E2": 2, "E3": 3, "E4": 4}, tmin=TMIN, tmax=TMAX,
                    baseline=BASELINE, reject=None, preload=True, on_missing="ignore", verbose="ERROR")
    return ep.resample(SFREQ_EPOCAS, verbose="ERROR")


def promedio_colapsado(ep):
    #Promedio de los promedios de cada nivel
    eeg = ep.copy().pick("eeg")
    eeg.set_eeg_reference([c for c in eeg.ch_names if c not in FRONTOPOLARES], verbose="ERROR")
    niveles = [n for n in ("E1", "E2", "E3", "E4") if len(eeg[n])]
    datos = np.mean([eeg[n].average().get_data() for n in niveles], axis=0) * 1e6
    return datos, eeg.info, eeg.times


def n2p2_por_canal(datos, nombres, t):
    """Latencias de N2 (minimo) y P2 (maximo) en Cz, y en cada canal la media en +-20 ms alrededor de esas
    latencias: N2-P2 = P2 - N2 """
    x = datos[nombres.index(CANAL_LATENCIAS)]
    lat = {}
    for comp, (a, b), f in (("N2", VENTANA_N2, np.argmin), ("P2", VENTANA_P2, np.argmax)):
        m = (t >= a) & (t <= b)
        lat[comp] = float(t[m][f(x[m])])
    media = lambda tc: datos[:, np.abs(t - tc) <= MEDIA_PICO_S].mean(axis=1)
    n2, p2 = media(lat["N2"]), media(lat["P2"])
    tabla = pd.DataFrame({"canal": nombres, "N2_uV": n2.round(2), "P2_uV": p2.round(2), "N2P2_uV": (p2 - n2).round(2)})
    return tabla, lat


def fig_n2p2(tabla, info, lat, titulo):
    con_pos = [i for i, ch in enumerate(info["chs"]) if np.isfinite(ch["loc"][:3]).all() and np.any(ch["loc"][:3])]
    vals = tabla["N2P2_uV"].to_numpy(dtype=float)
    y = np.array([info["chs"][i]["loc"][1] if i in set(con_pos) else -np.inf for i in range(len(vals))])
    orden = np.argsort(-y, kind="stable")
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
    fig.suptitle(f"{titulo}  (N2 {lat['N2'] * 1000:.0f} ms, P2 {lat['P2'] * 1000:.0f} ms en {CANAL_LATENCIAS}; "
                 "barras de adelante hacia atras)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    return fig



def analizar_sujeto(sub, raiz=None, carpeta=None):
    psd, epocas, freqs = {}, [], None
    for vhdr in listar_sesiones(sub, raiz): #recorre las sesiones del sujeto
        ses = re.search(r"(ses-\d+)", vhdr.as_posix()).group(1)
        raw = cargar_sesion(vhdr) 
        freqs, psd[ses] = psd_media(raw) #guarda los datos del EEG crudo
        ev = pulsos_laser(raw)
        if len(ev):
            epocas.append(epocas_lep(raw, ev))
        del raw
    if not epocas:
        raise RuntimeError(f"{sub}: ninguna sesion tuvo pulsos laser")
    ep = epocas[0] if len(epocas) == 1 else mne.concatenate_epochs(epocas, verbose="ERROR")
    datos, info, t = promedio_colapsado(ep)
    tabla, lat = n2p2_por_canal(datos, info["ch_names"], t)

    destino = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "analisis" / sub
    destino.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino / "n2p2_por_canal.csv", index=False)
    for nombre, fig in (("psd_por_sesion", fig_psd(freqs, psd, f"{sub}: PSD media entre canales EEG "
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
    for sub in p.parse_args(argv).sujetos:
        tabla, lat, n, destino = analizar_sujeto(sub)
        top = tabla.sort_values("N2P2_uV", ascending=False).head(5)
        print(f"{sub}: {n} epocas; N2 {lat['N2'] * 1000:.0f} ms, P2 {lat['P2'] * 1000:.0f} ms; mayor N2-P2: "
              + ", ".join(f"{c} {v:.1f}" for c, v in zip(top["canal"], top["N2P2_uV"])) + f" uV  -> {destino}")


if __name__ == "__main__":
    main()