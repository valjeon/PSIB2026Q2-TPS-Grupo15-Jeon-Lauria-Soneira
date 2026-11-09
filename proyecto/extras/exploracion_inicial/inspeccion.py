from pathlib import Path
import mne
import matplotlib.pyplot as plt

ARCHIVO = (
    Path(__file__).resolve().parent.parent
    / "data" / "ds005293"
    / "sub-001" / "ses-1" / "eeg"
    / "sub-001_ses-1_task-95ByBP_eeg.vhdr"
)

raw = mne.io.read_raw_brainvision(str(ARCHIVO), preload=True, verbose="ERROR")

print("Canales:", len(raw.ch_names))
print("Nombres:", raw.ch_names)
print("Tipos:", set(raw.get_channel_types()))
print("Frecuencia de muestreo:", raw.info["sfreq"], "Hz")
print(f"Duración: {raw.times[-1]:.1f} s")
print("Anotaciones (primeras 10):")
print(raw.annotations[:10])

# Espectro sin filtrar
raw.compute_psd(fmax=120).plot()
plt.show()