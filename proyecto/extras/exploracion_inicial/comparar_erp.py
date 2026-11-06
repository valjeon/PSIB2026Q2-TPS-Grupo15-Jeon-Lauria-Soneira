"""Compara el potencial evocado en Cz de NUESTRO pipeline (desde el crudo) con el de los autores.

Uso (desde la raiz del repo, con el venv activo; el canal es opcional, por defecto Cz):
    python procesamiento/comparar_erp.py sub-001 Fp1 laser_ica
(argumentos: sujeto, canal, sufijo del archivo; "laser" = sin ICA, "laser_ica" = con ICA)
Requiere haber corrido antes extract_laser_epochs.py para ese sujeto.
Sin probar sobre los datos reales: revisar la salida.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

sub = sys.argv[1] if len(sys.argv) > 1 else "sub-001"
canal = sys.argv[2] if len(sys.argv) > 2 else "Fp1"  # ej. Cz, Fp1, AF3
sufijo = sys.argv[3] if len(sys.argv) > 3 else "laser"  # "laser" = sin ICA, "laser_ica" = con ICA

# Nuestras epocas: 1-100 Hz, notch 48-52 Hz, referencia a la nariz, sin ICA
ours = mne.read_epochs(f"data/derived_mine/{sub}_{sufijo}-epo.fif", verbose="ERROR")
# Epocas de los autores: post-ICA, referencia al promedio, pasabajos 30 Hz
theirs = mne.read_epochs_eeglab(
    f"data/ds005293/derivatives/rerefer/{sub}_95ByBP.set", verbose="ERROR")
theirs.apply_baseline((-1.0, 0.0), verbose="ERROR")
# v2 baja a 250 Hz y los derivados de los autores estan a 1000 Hz: los igualamos
# (ya vienen con pasabajos de 30 Hz, asi que reducir la frecuencia no pierde nada)
if theirs.info["sfreq"] != ours.info["sfreq"]:
    theirs.resample(ours.info["sfreq"], verbose="ERROR")

# Para comparar en igualdad de condiciones, llevamos las nuestras a lo que hicieron ellos:
# referencia al promedio y pasabajos 30 Hz (solo para esta comparacion)
ours_cmp = ours.copy().set_eeg_reference("average", projection=False, verbose="ERROR")
ours_cmp.filter(None, 30.0, verbose="ERROR")
ours_cmp.apply_baseline((-1.0, 0.0), verbose="ERROR")

print("canales nuestros:", len(ours.ch_names), "| de los autores:", len(theirs.ch_names))
print("epocas nuestras:", len(ours), "| de los autores:", len(theirs))

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
mask = (ours.times >= 0.0) & (ours.times <= 0.8)
for ax, k in zip(axes, (1, 2, 3, 4)):
    a = ours_cmp[f"E{k}"].average().copy().pick(canal).data[0] * 1e6
    b = theirs[f"L{k}"].average().copy().pick(canal).data[0] * 1e6
    n = min(len(a), len(b))
    r = np.corrcoef(a[:n][mask[:n]], b[:n][mask[:n]])[0, 1]
    ax.plot(ours.times[:n], a[:n], label="nuestro pipeline")
    ax.plot(theirs.times[:n], b[:n], label="autores")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set(title=f"E{k}  (r 0-0.8 s = {r:.2f})", xlabel="Tiempo (s)")
    print(f"E{k}: correlacion de las curvas 0-0.8 s = {r:.2f}")
axes[0].set_ylabel(f"{canal} (uV)")
axes[0].legend()
fig.suptitle(f"{sub}: potencial evocado en {canal}, nuestro pipeline ({sufijo}) vs autores")
out = Path("data/derived_mine")
fig.savefig(out / f"{sub}_comparacion_erp_{canal}_{sufijo}.png", dpi=150, bbox_inches="tight")
plt.show()
