"""N2-P2 por ensayo en Cz para un sujeto de ds005293 (derivados de los autores).

Uso (desde la raiz del repo, con el venv activo):
    python src/n2p2_por_ensayo.py sub-001
Compara la amplitud N2-P2 contra la intensidad (L1..L4) y contra la calificacion de dolor.
Sin probar sobre los datos reales: revisar la salida antes de escalar a mas sujetos.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

sub = sys.argv[1] if len(sys.argv) > 1 else "sub-001"
path = Path(f"data/ds005293/derivatives/rerefer/{sub}_95ByBP.set")

# Epocas (MNE) y metadatos por ensayo (scipy): los dos salen del mismo .set, mismo orden
ep = mne.read_epochs_eeglab(path, verbose="ERROR")
ep.apply_baseline((-1.0, 0.0), verbose="ERROR")
m = loadmat(path, struct_as_record=False, squeeze_me=True)
ev = m["event"]
tipos = np.array([e.type for e in ev])
rating = np.array([e.rating for e in ev], dtype=float)
potencia = np.array([e.laser_power for e in ev], dtype=float)

# Verifica que el orden coincide con el de las epocas de MNE
inv = {v: k for k, v in ep.event_id.items()}
tipos_mne = np.array([inv[c] for c in ep.events[:, 2]])
assert (tipos == tipos_mne).all(), "El orden de eventos no coincide con las epocas"

nivel = np.array([int(t[1]) for t in tipos])  # L1..L4 -> 1..4

# N2 = minimo entre 100 y 300 ms; P2 = maximo entre 250 y 550 ms (microvoltios, Cz)
cz = ep.get_data(picks="Cz")[:, 0, :] * 1e6
t = ep.times
n2 = cz[:, (t >= 0.10) & (t <= 0.30)].min(axis=1)
p2 = cz[:, (t >= 0.25) & (t <= 0.55)].max(axis=1)
pp = p2 - n2

print(f"{sub}: {len(pp)} ensayos")
for k in (1, 2, 3, 4):
    s = nivel == k
    print(f"L{k}: n={s.sum():2d} | N2-P2 medio = {pp[s].mean():6.1f} uV | "
          f"calificacion media = {rating[s].mean():.2f} | potencia laser = {np.unique(potencia[s])}")
r1, p_1 = spearmanr(nivel, pp)
r2, p_2 = spearmanr(rating, pp)
print(f"Spearman N2-P2 vs intensidad: rho = {r1:.2f} (p = {p_1:.3g})")
print(f"Spearman N2-P2 vs calificacion: rho = {r2:.2f} (p = {p_2:.3g})")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(nivel + np.random.default_rng(0).normal(0, 0.05, len(nivel)), pp, s=18)
ax[0].set(xlabel="Intensidad (L1..L4)", ylabel="N2-P2 en Cz (uV)", xticks=[1, 2, 3, 4])
ax[1].scatter(rating, pp, c=nivel, cmap="viridis", s=18)
ax[1].set(xlabel="Calificacion de dolor (0-10)", ylabel="N2-P2 en Cz (uV)")
fig.suptitle(f"{sub}: N2-P2 por ensayo")
out = Path("data/derived_mine")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"{sub}_n2p2.png", dpi=150, bbox_inches="tight")
plt.show()
