"""Extrae las epocas de pulso laser de un sujeto de ds005293 (experimento 6).

Regla de los autores (code/preprocessing_95.m): un evento S 32, S 33, S 34 o S 35
seguido inmediatamente por S100 es un pulso laser (L1..L4 = intensidades E1..E4).
Los demas codigos (S 8..S 20, S 64..S 67, S128..S131) son otras modalidades.

Uso (desde la raiz del repo, con el venv activo):
    python src/extract_laser_epochs.py sub-001
Sin probar sobre los datos completos: revisar la salida antes de escalar.
"""
import sys
from pathlib import Path

import mne
import numpy as np

LASER_CODES = {32: 1, 33: 2, 34: 3, 35: 4}  # codigo del trigger -> nivel E1..E4
DROP = ["TP9", "TP10", "HEOR"]               # canales que los autores descartan


def laser_events(events):
    """Devuelve [[muestra, 0, nivel]] para cada S3x seguido inmediatamente de S100."""
    out = []
    for i in range(1, len(events)):
        if events[i, 2] == 100 and events[i - 1, 2] in LASER_CODES:
            out.append([events[i - 1, 0], 0, LASER_CODES[events[i - 1, 2]]])
    return np.array(out, dtype=int)


def main(sub, root="data/ds005293"):
    all_ev, all_raws = [], []
    for ses in sorted(Path(root, sub).glob("ses-*")):
        vhdr = next(ses.glob("eeg/*_eeg.vhdr"))
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
        raw.drop_channels([c for c in DROP if c in raw.ch_names])
        ev, _ = mne.events_from_annotations(raw, verbose="ERROR")
        le = laser_events(ev)
        print(f"{ses.name}: {len(le)} pulsos laser", np.bincount(le[:, 2], minlength=5)[1:] if len(le) else "")
        if len(le) == 0:
            continue
        # Filtros como los autores: pasaaltos 1 Hz, pasabajos 100 Hz, notch 48-52 Hz (FIR)
        raw.filter(1.0, 100.0, verbose="ERROR")
        raw.filter(52.0, 48.0, l_trans_bandwidth=1.0, h_trans_bandwidth=1.0, verbose="ERROR")
        ep = mne.Epochs(raw, le, event_id={"E1": 1, "E2": 2, "E3": 3, "E4": 4},
                        tmin=-1.0, tmax=2.0, baseline=(-1.0, 0.0), preload=True,
                        verbose="ERROR")
        all_raws.append(ep)
    if not all_raws:
        print("No se encontraron pulsos laser.")
        return
    epochs = mne.concatenate_epochs(all_raws)
    print(epochs)
    out = Path("data/derived_mine")
    out.mkdir(parents=True, exist_ok=True)
    epochs.save(out / f"{sub}_laser-epo.fif", overwrite=True)


if __name__ == "__main__":
    main(sys.argv[1])
