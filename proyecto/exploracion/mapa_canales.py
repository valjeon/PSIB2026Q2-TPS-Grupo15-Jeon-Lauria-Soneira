"""Mapa 2D aproximado de los electrodos del registro, con los canales marcados por grupo.

Grupos:
    ROI             canales candidatos para medir el N2-P2 (vertex y alrededores)
    frontopolares   los que el criterio de EEG ignora (config.CANALES_SIN_CRITERIO_EEG): ahi se ven los ojos
    descartados     los que se sacan en datos.py (config.CANALES_DESCARTADOS) y tienen posicion (TP9, TP10)
    resto           el resto de los canales EEG
Los canales sin posicion en el montaje 10-05 (VEO, HEOR) no se pueden dibujar: se listan en consola.

Solo muestra la figura, no guarda nada.

Uso desde proyecto/:
    python -m exploracion.mapa_canales            (sub-010 por defecto)
    python -m exploracion.mapa_canales sub-050
"""
import argparse

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from procesamiento import config, datos

# Candidatos para la ROI: vertex (Cz) y vecinos. No hay FCz en este montaje: FC1/FC2 lo reemplazan.
ROI = ["Cz", "C1", "C2", "FC1", "FC2", "CPz"]

GRUPOS = [  # (nombre, color)
    ("ROI", "tab:red"),
    ("frontopolares (sin criterio EEG)", "tab:orange"),
    ("descartados", "0.75"),
    ("resto", "tab:blue"),
]


def info_del_registro(sub):
    """Info de la primera sesion del sujeto, sin cargar la senal y sin descartar canales."""
    vhdr = datos.listar_sesiones(sub)[0]
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    eog = [c for c in [config.CANAL_EOG, "HEOR", "HEOL"] if c in raw.ch_names]
    raw.set_channel_types({c: "eog" for c in eog})
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    return raw.info


def sin_posicion(info):
    """Canales (de cualquier tipo) sin posicion valida en el montaje."""
    return [ch["ch_name"] for ch in info["chs"]
            if np.isnan(ch["loc"][:3]).any() or not np.any(ch["loc"][:3])]


def agrupar(nombres_eeg):
    """Lista de listas de indices (sobre nombres_eeg), en el orden de GRUPOS."""
    frontales = set(config.CANALES_SIN_CRITERIO_EEG)
    descartados = set(config.CANALES_DESCARTADOS)

    def grupo(c):
        if c in ROI:
            return 0
        if c in frontales:
            return 1
        if c in descartados:
            return 2
        return 3

    g = [grupo(c) for c in nombres_eeg]
    return [[i for i, x in enumerate(g) if x == k] for k in range(len(GRUPOS))]


def graficar(info, sub):
    eeg = mne.pick_info(info, mne.pick_types(info, eeg=True, exclude=sin_posicion(info)))
    grupos = agrupar(eeg.ch_names)
    # plot_sensors colorea cada grupo con el cmap evaluado en linspace(0, 1, n_grupos)
    cmap = ListedColormap([color for _, color in GRUPOS])
    fig, ax = plt.subplots(figsize=(7, 7))
    mne.viz.plot_sensors(eeg, kind="topomap", show_names=True, ch_groups=grupos, cmap=cmap,
                         axes=ax, pointsize=60, linewidth=0, show=False)
    for t in ax.texts:   # nombres mas legibles
        t.set_fontsize(8)
    marcas = [Line2D([], [], marker="o", ls="", color=color, label=f"{nombre} ({len(idx)})")
              for (nombre, color), idx in zip(GRUPOS, grupos)]
    ax.legend(handles=marcas, loc="upper left", bbox_to_anchor=(-0.05, 1.08), frameon=False, fontsize=8)
    ax.set_title(f"{sub}: electrodos ({len(eeg.ch_names)} EEG con posicion)", pad=30)
    fig.tight_layout()
    return fig


def resumen(info):
    """Imprime los grupos y avisa de nombres de config que no estan en el registro."""
    nombres = set(info.ch_names)
    print("Canales en el registro:", len(info.ch_names))
    print("Sin posicion en el montaje (no se dibujan):", sin_posicion(info))
    print("EOG:", [info.ch_names[i] for i in mne.pick_types(info, eeg=False, eog=True)])
    for nombre, lista in (("ROI", ROI), ("frontopolares", config.CANALES_SIN_CRITERIO_EEG),
                          ("descartados", config.CANALES_DESCARTADOS)):
        faltan = [c for c in lista if c not in nombres]
        print(f"{nombre}: {[c for c in lista if c in nombres]}" + (f"   (no estan: {faltan})" if faltan else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description="Mapa 2D de los electrodos con los canales marcados")
    p.add_argument("sujeto", nargs="?", default="sub-010")
    a = p.parse_args(argv)
    info = info_del_registro(a.sujeto)
    resumen(info)
    graficar(info, a.sujeto)
    plt.show()


if __name__ == "__main__":
    main()