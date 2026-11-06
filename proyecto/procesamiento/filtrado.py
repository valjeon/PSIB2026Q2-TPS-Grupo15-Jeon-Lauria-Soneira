
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from procesamiento import analisis, config

CANALES = ["eeg", "eog"]     # el VEO se filtra igual que el EEG

def filtrar(raw):
    # Aplica un filtro pasabanda de 1-100 Hz y un notch 48-52 Hz, FIR de fase cero, sobre una copia de la sesion continua
    filt = raw.copy().filter(config.PASA_ALTOS_HZ, config.PASA_BAJOS_HZ, picks=CANALES,
                             method="fir", fir_design="firwin", verbose="ERROR")
    filt.notch_filter(config.NOTCH_HZ, notch_widths=config.NOTCH_ANCHO_HZ, picks=CANALES,
                      method="fir", fir_design="firwin", verbose="ERROR")
    return filt

def leer_y_filtrar(vhdr):
    raw = analisis.cargar_sesion(vhdr)
    ev = analisis.pulsos_laser(raw)
    return filtrar(raw), ev, vhdr.parent.parent.name, raw

def fig_antes_despues(raw, filt, titulo):
    #Hace el gráfico de la PSD de la señal antes y después de haber sido filtrada
    freqs, antes = analisis.psd_media(raw)
    _, despues = analisis.psd_media(filt)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    m = freqs >= 0.5
    ax.axvspan(config.PASA_ALTOS_HZ, config.PASA_BAJOS_HZ, color="#e5f4e4", alpha=0.6, zorder=0)
    for f in (config.PASA_ALTOS_HZ, config.NOTCH_HZ, config.PASA_BAJOS_HZ):
        ax.axvline(f, color="k", ls="--", lw=0.8)
    ax.semilogx(freqs[m], 10 * np.log10(antes[m]), color="#9a9993", lw=1.2, label="crudo")
    ax.semilogx(freqs[m], 10 * np.log10(despues[m]), color="#2a78d6", lw=1.4, label="filtrado")
    ax.set_xlim(0.3, freqs.max())
    ax.grid(True, which="both", color="0.9", lw=0.6)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Densidad espectral (uV²/Hz)")
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def guardar_verificacion(sub, sesion=0, carpeta=None):
    #Filtra una sesion del sujeto y guarda la PSD antes y despues en derived_mine/filtrado/.
    vhdr = analisis.listar_sesiones(sub)[sesion]
    ses = vhdr.parent.parent.name                       # .../sub-004/ses-1/eeg/x.vhdr -> "ses-1"
    raw = analisis.cargar_sesion(vhdr)
    fig = fig_antes_despues(raw, filtrar(raw), f"{sub} {ses}: PSD antes y despues del filtro "
                                               f"({config.PASA_ALTOS_HZ:g}-{config.PASA_BAJOS_HZ:g} Hz, notch "
                                               f"{config.NOTCH_HZ - config.NOTCH_ANCHO_HZ / 2:g}-{config.NOTCH_HZ + config.NOTCH_ANCHO_HZ / 2:g} Hz)")
    destino = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "filtrado"
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{sub}_{ses}_psd_filtro.png"
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return ruta