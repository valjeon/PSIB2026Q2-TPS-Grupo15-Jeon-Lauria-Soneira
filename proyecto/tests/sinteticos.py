"""Sujetos sinteticos en formato BrainVision (como ds005293) para los tests.

Cada sujeto: 6 sesiones, 80 pulsos (20 por nivel, orden al azar), triggers S 3x seguidos de S 100.
    - LEP (N2 a 220 ms, P2 a 380 ms) en la ROI, con amplitud proporcional al nivel
    - ruido blanco de 4 uV en todos los canales y 20 uV de red a 50 Hz
    - parpadeos (300 uV en el VEO a +0.3 s, la mitad en Fp1/Fp2) en los ensayos PARPADEOS
    - O2 muy ruidoso (canal malo propio); un salto de 500 uV en Pz en el ensayo GROSERO
"""
import numpy as np

CANALES_EEG = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F3", "Fz", "F4", "FC1", "FC2", "C3", "C1", "Cz", "C2", "C4",
               "CP1", "CPz", "CP2", "P3", "Pz", "P4", "O1", "Oz", "O2", "T7", "T8"]
CANALES = CANALES_EEG + ["TP9", "TP10", "VEO", "HEOR"]
PESO_LEP = {"Cz": 1.0, "FC1": 0.8, "FC2": 0.8, "C1": 0.8, "C2": 0.8, "CP1": 0.7, "CPz": 0.8, "CP2": 0.7,
            "Fz": 0.5, "C3": 0.4, "C4": 0.4, "Pz": 0.4}
FS = 500
CODIGOS = {1: 32, 2: 33, 3: 34, 4: 35}
PARPADEOS = set(range(5, 80, 9))           # 9 ensayos con parpadeo
GROSERO = 40
TAREA = "95ByBP"


def forma_lep(t):
    return -np.exp(-((t - 0.22) / 0.03) ** 2) + 1.5 * np.exp(-((t - 0.38) / 0.05) ** 2)


def escribir_sujeto(raiz, sub, semilla=0, amplitud_uV=5.0):
    import pybv
    rng = np.random.default_rng(semilla)
    niveles = rng.permutation(np.repeat([1, 2, 3, 4], 20))
    ensayo = 0
    for n, grupo in enumerate(np.array_split(niveles, 6), start=1):
        dur = 4 * len(grupo) + 6
        t = np.arange(dur * FS) / FS
        x = rng.normal(0, 4e-6, (len(CANALES), len(t)))
        x += 20e-6 * np.sin(2 * np.pi * 50 * t)
        x[CANALES.index("O2")] += rng.normal(0, 60e-6, len(t))
        ann = []
        for i, nivel in enumerate(grupo):
            m = int((3 + 4 * i) * FS)
            ann += [[m, CODIGOS[nivel]], [m + 2, 100]]
            tt = t - m / FS
            cerca = np.abs(tt) < 1.5
            for c, w in PESO_LEP.items():
                x[CANALES.index(c), cerca] += nivel * amplitud_uV * 1e-6 * w * forma_lep(tt[cerca])
            if ensayo in PARPADEOS:
                b = 300e-6 * np.exp(-((tt[cerca] - 0.3) / 0.08) ** 2)
                x[CANALES.index("VEO"), cerca] += b
                for c in ("Fp1", "Fp2"):
                    x[CANALES.index(c), cerca] += 0.5 * b
            if ensayo == GROSERO:
                x[CANALES.index("Pz"), cerca] += 500e-6 * np.exp(-((tt[cerca] - 0.5) / 0.1) ** 2)
            ensayo += 1
        d = raiz / sub / f"ses-{n}" / "eeg"
        d.mkdir(parents=True, exist_ok=True)
        pybv.write_brainvision(data=x, sfreq=FS, ch_names=CANALES, ref_ch_names=None,
                               fname_base=f"{sub}_ses-{n}_task-{TAREA}_eeg", folder_out=d,
                               events=np.array(ann), unit="µV", overwrite=True)
    return niveles


def escribir_csv_autores(ruta, marcas):
    """marcas: {numero de sujeto: ('epocas 1-based separadas por espacio', 'electrodos')}."""
    filas = [["Experiment 1", "", "", "", "Experiment 6_95 By BP", "", "", ""],
             ["sub_id", "bad epoch", "bad electrode", "bad ica", "sub_id", "bad epoch", "bad electrode", "bad ica"]]
    for num, (ep, el) in marcas.items():
        filas.append(["1", "\\", "\\", "\\", str(num), ep, el, "\\"])
    ruta.write_text("\n".join(",".join(f) for f in filas), encoding="utf-8")
