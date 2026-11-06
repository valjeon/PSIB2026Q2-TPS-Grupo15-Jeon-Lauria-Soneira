"""Extrae epocas de pulso laser de un sujeto de ds005293 (experimento 6), con limpieza de parpadeos.

Version 2: agrega ICA (los autores usan infomax extendido; aqui fastica, mucho mas rapido) y rechazo de epocas por amplitud.

Etapas por sujeto:
  1. Leer las 6 sesiones del crudo; quitar TP9, TP10 y HEOR; VEO pasa a canal de ojo (EOG).
  2. Filtrar: pasaaltos 1 Hz, pasabajos 100 Hz, notch 48-52 Hz (FIR, fase cero).
  3. Bajar a 250 Hz (alcanza para 100 Hz de gamma).
  4. ICA (fastica; requiere scikit-learn) sobre las 6 sesiones juntas; marcar como parpadeo los componentes que
     correlacionan con VEO; sacarlos de cada sesion.
  5. Regla de los autores: S 32..S 35 seguido inmediatamente de S100 = pulso laser (E1..E4).
  6. Epocas de -1 a +2 s, baseline -1..0 s, rechazo si algun canal EEG supera RECHAZO_UV pico a pico.

Uso (desde la raiz del repo, con el venv activo):
    python procesamiento/extract_laser_epochs_v2.py sub-001
Sin probar sobre los datos reales: revisar la salida (cuantos componentes se sacan, cuantas epocas se rechazan).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import mne
import numpy as np

LASER_CODES = {32: 1, 33: 2, 34: 3, 35: 4}   # codigo del trigger -> nivel E1..E4
DROP = ["TP9", "TP10", "HEOR"]                # canales que los autores descartan
FS = 250.0                                    # frecuencia de muestreo despues de filtrar
RECHAZO_UV = 150.0                            # umbral pico a pico (uV); ajustar mirando drop_log
N_COMP = 25                                   # componentes de la ICA
FRONTALES = ["Fp1", "Fp2", "Fpz", "AF3", "AF4", "AF7", "AF8"]
RAZON_FRONTAL_MIN = 3.0   # un parpadeo pesa mucho mas en los canales frontales que en el resto

def cargar_sesion(vhdr):
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    raw.drop_channels([c for c in DROP if c in raw.ch_names])
    if "VEO" in raw.ch_names:
        raw.set_channel_types({"VEO": "eog"})
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")  # posiciones, para los mapas de la ICA
    raw.filter(1.0, 100.0, verbose="ERROR")
    raw.filter(52.0, 48.0, l_trans_bandwidth=1.0, h_trans_bandwidth=1.0, verbose="ERROR")
    raw.resample(FS, verbose="ERROR")
    return raw


def laser_events(events):
    """[[muestra, 0, nivel]] para cada S3x seguido inmediatamente de S100."""
    out = []
    for i in range(1, len(events)):
        if events[i, 2] == 100 and events[i - 1, 2] in LASER_CODES:
            out.append([events[i - 1, 0], 0, LASER_CODES[events[i - 1, 2]]])
    return np.array(out, dtype=int).reshape(-1, 3)


def quitar_parpadeos(raws, tag, out_dir):
    """Ajusta una ICA con todas las sesiones y la aplica a cada una (in place)."""
    todo = mne.concatenate_raws([r.copy() for r in raws], verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=N_COMP, method="fastica",
                                random_state=97,
                                max_iter="auto")
    ica.fit(todo, decim=3, verbose="ERROR")
    candidatos, _ = ica.find_bads_eog(todo, ch_name="VEO", verbose="ERROR")
    # Un candidato (correlaciona con VEO) solo se saca si ademas su mapa es frontal.
    # Asi evitamos sacar componentes difusos que pueden llevar senal cerebral.
    mezcla = np.abs(ica.get_components())          # canales x componentes
    es_front = np.array([c in FRONTALES for c in ica.ch_names])
    ica.exclude = []
    for c in candidatos:
        razon = mezcla[es_front, c].mean() / mezcla[~es_front, c].mean()
        sacar = razon >= RAZON_FRONTAL_MIN
        print(f"  componente {int(c)}: razon frontal/resto = {razon:.1f} -> "
              f"{'se saca' if sacar else 'se conserva (no parece parpadeo)'}")
        if sacar:
            ica.exclude.append(int(c))
    print(f"ICA: {len(ica.exclude)} componentes sacados {ica.exclude} (candidatos por VEO: {[int(c) for c in candidatos]})")
    if ica.exclude:
        try:  # el grafico es opcional: si falla, el procesamiento sigue
            figs = ica.plot_components(picks=ica.exclude, show=False)
            figs = figs if isinstance(figs, list) else [figs]
            figs[0].savefig(Path(out_dir) / f"{tag}_ica_componentes_quitados.png", dpi=120)
        except Exception as e:
            print("No se pudo graficar los componentes quitados:", e)
    for r in raws:
        ica.apply(r, verbose="ERROR")
    return ica


def epocar(raw, reject_uv=RECHAZO_UV):
    ev, _ = mne.events_from_annotations(raw, verbose="ERROR")
    le = laser_events(ev)
    if len(le) == 0:
        return None, 0
    ep = mne.Epochs(raw, le, event_id={"E1": 1, "E2": 2, "E3": 3, "E4": 4},
                    tmin=-1.0, tmax=2.0, baseline=(-1.0, 0.0), preload=True,
                    reject=dict(eeg=reject_uv * 1e-6), verbose="ERROR")
    return ep, len(le)


def main(sub, root="data/ds005293"):
    out = Path("data/derived_mine")
    out.mkdir(parents=True, exist_ok=True)
    raws, nombres = [], []
    for ses in sorted(Path(root, sub).glob("ses-*")):
        vhdr = next(ses.glob("eeg/*_eeg.vhdr"))
        print("leyendo", ses.name)
        raws.append(cargar_sesion(vhdr))
        nombres.append(ses.name)
    if not raws:
        print("No hay sesiones en", Path(root, sub))
        return

    quitar_parpadeos(raws, sub, out)

    lista, total, quedan = [], 0, 0
    for nombre, raw in zip(nombres, raws):
        ep, n = epocar(raw)
        if ep is None:
            continue
        print(f"{nombre}: {n} pulsos laser -> {len(ep)} epocas despues del rechazo")
        total += n
        quedan += len(ep)
        lista.append(ep)
    epochs = mne.concatenate_epochs(lista, verbose="ERROR")
    print(f"TOTAL: {total} pulsos, {quedan} epocas conservadas "
          f"({total - quedan} rechazadas por amplitud > {RECHAZO_UV:.0f} uV)")
    print({k: len(epochs[k]) for k in epochs.event_id if len(epochs[k])})
    epochs.save(out / f"{sub}_laser_ica-epo.fif", overwrite=True)


if __name__ == "__main__":
    main(sys.argv[1])
