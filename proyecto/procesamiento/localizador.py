"""Localizador colapsado (Luck & Gaspelin 2017): ROI y latencias de N2/P2 sin mirar la diferencia entre niveles.

Sujetos: los incluidos en la etapa previa (inclusion_previa.csv).
Por sujeto: epocas preprocesadas leidas con limpieza.para_analisis (ref. sin frontales, 30 Hz, linea de base),
    promedio ponderado por 1/varianza del ruido pre-estímulo (inhomogéneo, caso 2) en cada nivel -> ERP colapsado.
Grupo: promedio de los ERP colapsados (cada sujeto pesa igual).

1. Latencias: minimo (N2) y maximo (P2) del gran promedio en VENTANA_N2 / VENTANA_P2, sobre el promedio de
   los canales de la ROI a priori (config.CANALES_ROI).
2. Medida con polaridad por canal: media en +-MEDIA_PICO_S alrededor de P2 menos la de N2. El ojo (positivo
   en la latencia de N2) resta en vez de sumar.
3. ROI de los datos: canales (sin Fp/AF) con medida >= FRACCION_ROI x maximo del gran promedio.
4. Verificacion de grupo: canal maximo por sujeto (sin Fp/AF) y si cae en la ROI a priori; ROI a priori vs
   resto de los canales, Wilcoxon pareado entre sujetos.

Salida en derived_mine/localizador/: resumen por sujeto, medida por canal, gran promedio (-ave.fif) y figuras.

Lo llama pipeline.py en la etapa de grupo (python -m procesamiento.pipeline --grupo).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from procesamiento import config, inclusion, limpieza

# usan getattr: si el valor está en config, usan ese; si no, usan el valor por defecto.
CANALES_ROI = getattr(config, "CANALES_ROI", ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"])
MEDIA_PICO_S = getattr(config, "MEDIA_PICO_S", 0.02) # +-20 ms alrededor de cada pico
FRACCION_ROI = getattr(config, "FRACCION_ROI_DATOS", 0.8) # canales con >= 80% del maximo
FRACCION_VENTANA = getattr(config, "FRACCION_VENTANA", 0.5) # ventanas: ancho a media altura de N2 y P2



# ERP colapsado por sujeto y gran promedio
def promedio_ponderado(x, t):
    # promedia las épocas dándole a cada una un peso 1/varianza de su ruido pre-estímulo con pesos por canal que suman 1
    var = x[:, :, t < 0].var(axis=2) # epocas x canales
    w = (1 / var) / (1 / var).sum(axis=0, keepdims=True)
    return (w[:, :, None] * x).sum(axis=0)


# se calcula el ERP colapsado de UN sujeto
def erp_colapsado(sub):
    v, _ = limpieza.para_analisis(sub) # lee las épocas limpias del sujeto con limpieza.para_analisis: referencia sin frontales, 30 Hz y línea de base.
    v = v.copy().pick("eeg") # selecciona EEG sacando el EOG
    cond = v.metadata["condicion"].to_numpy() # se queda con los valores de E1 a E4 de cada epoca desde la metadata
    promedios = [promedio_ponderado(v.get_data()[cond == c], v.times) for c in config.NIVELES if (cond == c).any()] # hace el promedio ponderado de las epocas
    return mne.EvokedArray(np.mean(promedios, axis=0), v.info, tmin=v.tmin, nave=len(v), # promedio los ERPs
                           # así cada nivel pesa lo mismo, aunque uno tenga menos épocas, como E4 después del rechazo
                           comment=f"{sub} colapsado", verbose="ERROR")


# funcion que recibe los ERPs colapsados uno por sujeto
def gran_promedio(erps):
    datos = np.mean([e.get_data() for e in erps], axis=0) # los promedia para que cada sujeto pese igual sin importar la cantidad de epocas
    return mne.EvokedArray(datos, erps[0].info, tmin=erps[0].tmin, nave=len(erps),
                           comment="gran promedio colapsado", verbose="ERROR") # devuelve el gran promedio


# Latencias y medida por canal
def senal_roi(evoked, canales=None):
    canales = CANALES_ROI if canales is None else canales # saca la señal de los canales de la ROI (los que existan) y los promedia
    return evoked.get_data(picks=[c for c in canales if c in evoked.ch_names]).mean(axis=0) # devuelve una sola curva en el tiempo


# encuentra cuando ocurren N2 y P2
def latencias(evoked):
    t, x = evoked.times, senal_roi(evoked) # toma la curva de la ROI
    out = []
    # dentro de config.VENTANA_N2 busca el mínimo (argmin) porque el N2 es el pico negativo
    # dentro de config.VENTANA_P2 busca el máximo (argmax) porque P2 es el pico positivo
    for (a, b), f in ((config.VENTANA_N2, np.argmin), (config.VENTANA_P2, np.argmax)):
        m = (t >= a) & (t <= b)
        out.append(float(t[m][f(x[m])]))
    return tuple(out) # devuelve las latencias en segundos



# funcion que define la ventana de medición a partir del pico
def ventana_media_altura(evoked, t_pico, fraccion=None):
    fraccion = FRACCION_VENTANA if fraccion is None else fraccion
    t, x = evoked.times, senal_roi(evoked)
    i = int(np.argmin(np.abs(t - t_pico))) # busca el indice i del tiempo mas cercano al pico
    umbral = fraccion * x[i] # calcula el umbral como la mitad de la amplitud del pico
    dentro = x >= umbral if x[i] > 0 else x <= umbral # marca los instantes que superan el umbral con el mismo signo que el pico
    # Desde el pico, camina hacia la izquierda mientras siga dentro, y lo mismo hacia la derecha
    # Así se queda con el tramo continuo que contiene al pico, y no con otros tramos sueltos que también superen el umbral.
    a = b = i
    while a > 0 and dentro[a - 1]:
        a -= 1
    while b < len(t) - 1 and dentro[b + 1]:
        b += 1
    return float(t[a]), float(t[b]) # devuelve el inicio y fin de la ventana en segundos


# funcion que lee localizador/ventanas.csv y devuelve un diccionario con el inicio y fin de N2 y P2
def leer_ventanas(carpeta=None):
    ruta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "localizador" / "ventanas.csv"
    t = pd.read_csv(ruta).set_index("componente")
    return {c: (float(t.loc[c, "inicio_s"]), float(t.loc[c, "fin_s"])) for c in t.index}


# calcula cuánto vale la respuesta N2–P2 para cada canal
def medida_canales(evoked, t_n2, t_p2, ancho=None):
    ancho = MEDIA_PICO_S if ancho is None else ancho
    t, x = evoked.times, evoked.get_data() * 1e6 # pasa a microvolts
    # define media que es el promedio de cada canal en +-20 ms alrededor del instante tc
    media = lambda tc: x[:, (t >= tc - ancho) & (t <= tc + ancho)].mean(axis=1)
    return media(t_p2) - media(t_n2) # devuelve un valor por canal de media en P2 − media en N2
# hace la resta y no la amplitud absoluta porque sino pueden ganas los canales forntales por el pico de los ojos
# si hacemos la resta el valor del LEP da positivo y grande siempre
# si hacemos amplitud absoluta da siempre positiva pero en los ojos daria negativo al hacer la resta
# nos permite diferenciarlos


# devuelve una máscara verdadero/falso que marca qué canales pueden competir para la ROI
# da todos menos los Fp/AF (CANALES_SIN_CRITERIO_EEG) porque están cerca de los ojos
def validos(nombres):
    return ~np.isin(np.array(nombres), config.CANALES_SIN_CRITERIO_EEG)



# Figuras
# graficamos el gran promedio en el tiempo
def fig_erp(ga, t_n2, t_p2, ventanas=()):
    fig, ax = plt.subplots(figsize=(8, 4))
    t = ga.times * 1000
    for c in ga.ch_names:
        ax.plot(t, ga.get_data(picks=c)[0] * 1e6, color="0.85", lw=0.6)
    ax.plot(t, senal_roi(ga) * 1e6, color="tab:red", lw=2, label="ROI a priori (promedio)") # ROI a priori en rojo y más gruesa
    for (a, b) in (config.VENTANA_N2, config.VENTANA_P2):
        ax.axvspan(a * 1000, b * 1000, color="0.93", zorder=0)
    for (a, b) in ventanas:
        ax.axvspan(a * 1000, b * 1000, color="tab:red", alpha=0.15, zorder=0)
    for tc, nombre in ((t_n2, "N2"), (t_p2, "P2")):
        ax.axvline(tc * 1000, color="k", ls="--", lw=0.8)
        ax.text(tc * 1000, 0.97, f" {nombre} {tc * 1000:.0f} ms", transform=ax.get_xaxis_transform(), va="top")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-200, 1000)
    ax.set_xlabel("tiempo desde el pulso (ms)")
    ax.set_ylabel("uV (ref. sin frontales)")
    ax.set_title(f"Gran promedio colapsado ({ga.nave} sujetos); gris: todos los canales")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# dibuja el topomapa de la medida P2 − N2
# dibuja el mapa (rojo = positivo, azul = negativo) y marca con círculos los canales de la ROI a priori
def fig_mapa(ga, medida, roi_datos):
    nombres = np.array(ga.ch_names)
    ok = validos(nombres)
    vmax = medida[ok].max()
    fig, ax = plt.subplots(figsize=(5, 5))
    mascara = np.isin(nombres, CANALES_ROI)
    mne.viz.plot_topomap(medida, ga.info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vmax, vmax),
                         extrapolate="local", mask=mascara,
                         mask_params=dict(marker="o", markersize=7, markerfacecolor="none",
                                          markeredgecolor="k", markeredgewidth=1.5))
    ax.set_title("P2 - N2 por canal (gran promedio)\n"
                 f"circulos: ROI a priori | ROI datos: {' '.join(roi_datos)}", fontsize=9)
    return fig



# main
def correr(sujetos=None, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    salida = carpeta / "localizador" # crea la carpeta localizador
    salida.mkdir(parents=True, exist_ok=True)
    # si no le pasan sujetos usa los incluidos en la etapa previa (inclusion_previa.csv)
    sujetos = sujetos or inclusion.incluidos("previa", carpeta)
    print(f"Localizador con {len(sujetos)} sujetos (referencia sin frontales: {config.REF_SIN_FRONTALES})")

    erps = {}
    for sub in sujetos:
        erps[sub] = erp_colapsado(sub) # calcula el ERP colapsado de cada sujeto
    ga = gran_promedio(list(erps.values())) # calcula el gran promedio
    t_n2, t_p2 = latencias(ga) # busca las latencias de N2 y P2 en el gran promedio
    v_n2, v_p2 = ventana_media_altura(ga, t_n2), ventana_media_altura(ga, t_p2) # Calcula las ventanas alrededor de cada latencia

    nombres = np.array(ga.ch_names)
    ok = validos(nombres)
    en_roi = np.isin(nombres, CANALES_ROI)
    medida = medida_canales(ga, t_n2, t_p2) # calcula la medida P2 − N2 de cada canal en el gran promedio
    maximo = medida[ok].max() # busca el máximo entre los canales válidos
    roi_datos = [n for n, m, v in zip(nombres, medida, ok) if v and m >= FRACCION_ROI * maximo]

    # verificacion x sujeto
    filas = []
    for sub, e in erps.items():
        m = medida_canales(e, t_n2, t_p2) # se calcula la medida N2-P2 por canal
        i_max = int(np.argmax(np.where(ok, m, -np.inf))) # busca su canal máximo entre los válidos
        filas.append({"sujeto": sub, "canal_max": nombres[i_max], "max_en_roi": bool(en_roi[i_max]),
                      "medida_roi_uV": round(float(m[en_roi].mean()), 2),
                      "medida_resto_uV": round(float(m[ok & ~en_roi].mean()), 2)})
    tabla = pd.DataFrame(filas)
    # Anota en la tabla:
    # cuál es ese canal y si cae en la ROI a priori
    # la medida promedio en la ROI
    # la medida promedio en el resto de los canales válidos
    p = wilcoxon(tabla["medida_roi_uV"], tabla["medida_resto_uV"], alternative="greater").pvalue \
        if len(tabla) >= 6 else np.nan 
    # compara ROI vs resto con un test de Wilcoxon pareado

    # guarda:
    # localizador_sujetos.csv: la tabla por sujeto.
    # localizador_canales.csv: la medida de cada canal, si está en cada ROI y si compite, ordenada de mayor a menor.
    # ventanas.csv: latencias y ventanas de N2 y P2 (lo que lee leer_ventanas).
    # gran_promedio_colapsado-ave.fif: el gran promedio.
    # las dos figuras.
    tabla.to_csv(salida / "localizador_sujetos.csv", index=False)
    pd.DataFrame({"canal": nombres, "medida_uV": medida.round(2), "en_roi_a_priori": en_roi,
                  "en_roi_datos": np.isin(nombres, roi_datos), "compite": ok}) \
        .sort_values("medida_uV", ascending=False).to_csv(salida / "localizador_canales.csv", index=False)
    pd.DataFrame({"componente": ["N2", "P2"], "latencia_s": [t_n2, t_p2], "inicio_s": [v_n2[0], v_p2[0]],
                  "fin_s": [v_n2[1], v_p2[1]]}).to_csv(salida / "ventanas.csv", index=False)
    ga.save(salida / "gran_promedio_colapsado-ave.fif", overwrite=True, verbose="ERROR")
    fig_erp(ga, t_n2, t_p2, (v_n2, v_p2)).savefig(salida / "gran_promedio_erp.png", dpi=130)
    fig_mapa(ga, medida, roi_datos).savefig(salida / "mapa_P2_N2.png", dpi=130)
    plt.close("all")

    # informa en pantalla
    orden = np.argsort(-np.where(ok, medida, -np.inf))[:8]
    print(f"\nLatencias (gran promedio, ROI a priori): N2 {t_n2 * 1000:.0f} ms, P2 {t_p2 * 1000:.0f} ms")
    print(f"Ventanas ({FRACCION_VENTANA:.0%} del pico): N2 {v_n2[0] * 1000:.0f}-{v_n2[1] * 1000:.0f} ms, "
          f"P2 {v_p2[0] * 1000:.0f}-{v_p2[1] * 1000:.0f} ms")
    print("Canales con mayor P2-N2 (sin Fp/AF): " + ", ".join(f"{nombres[i]} {medida[i]:.1f}" for i in orden))
    print(f"ROI de los datos (>= {FRACCION_ROI:.0%} del maximo): {' '.join(roi_datos)}")
    print(f"  en comun con la ROI a priori: {' '.join(c for c in roi_datos if c in CANALES_ROI) or '-'}")
    print(f"  solo en datos: {' '.join(c for c in roi_datos if c not in CANALES_ROI) or '-'}"
          f"   |   solo a priori: {' '.join(c for c in CANALES_ROI if c not in roi_datos) or '-'}")
    n_en = int(tabla["max_en_roi"].sum())
    print(f"Sujetos con el maximo dentro de la ROI a priori: {n_en} de {len(tabla)} ({100 * n_en / len(tabla):.0f}%)")
    print(f"ROI a priori vs resto (Wilcoxon pareado, una cola): mediana {tabla['medida_roi_uV'].median():.1f} "
          f"vs {tabla['medida_resto_uV'].median():.1f} uV, p = {p:.2g}")
    fuera = tabla[~tabla["max_en_roi"]]
    if len(fuera):
        print("Maximo fuera de la ROI: " + ", ".join(f"{s} ({c})" for s, c in zip(fuera["sujeto"], fuera["canal_max"])))
    print(f"\nGuardado en {salida}")
    return ga, tabla, roi_datos, (t_n2, t_p2)
