"""Estadistica por ensayo: relacion graduada con la intensidad (Spearman) y discriminacion E1 vs E4 (AUROC).

Datos: medicion.datos_variante(variante): ensayos no rechazados de los sujetos incluidos en la variante.
Rasgos: PRINCIPAL (N2-P2) + SECUNDARIOS + CONTROL (ver abajo).

1. Spearman (todos los niveles): por sujeto, rho entre el nivel (1-4) y el rasgo de cada ensayo.
   Grupo: Wilcoxon de una muestra sobre los rho de los sujetos (bilateral: rho distinto de 0).
   Signo esperado: N2-P2, P2, gamma > 0; N2 (mas negativa) y latencia de N2 (mas corta) < 0.
2. AUROC E1 vs E4 directo, por sujeto: probabilidad de que un ensayo de E4 tenga un valor mayor que uno de E1
   en el sentido esperado (SIGNO), es decir el U de Mann-Whitney normalizado. No se ajusta nada con los datos:
   el sentido se fija a priori, asi que el resultado de cada sujeto ya es el de un sujeto nuevo.
   Grupo: Wilcoxon de una muestra (AUROC > 0.5) y bootstrap de los sujetos (IC 95% del AUROC medio).

Salida en derived_mine/estadistica/<variante>/: spearman_sujetos.csv, auroc_sujetos.csv, resumen.csv, figuras.

Lo llama pipeline.py en la etapa de grupo (python -m procesamiento.pipeline --grupo).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from procesamiento import config, medicion

PRINCIPAL = "N2P2_uV" # rasgo principal del trabajo
SECUNDARIOS = ["N2_uV", "P2_uV", "lat_N2_ms", "gamma_ERS_dB"] # otros rasgos analizados
CONTROL = []
SIGNO = {"N2P2_uV": 1, "N2_uV": -1, "P2_uV": 1, "lat_N2_ms": -1, "gamma_ERS_dB": 1}   # sentido esperado con la intensidad
NIVELES_AUROC = ("E1", "E4") # los 2 niveles que compara AUROC
N_BOOTSTRAP = 2000 # cuantas veces se sortean los sujetos
SEMILLA = 0 # fija el sorteo para que cada corrida de el mismo resultado


# función que coloca una etiqueta a cada rasgo para que aparezca en el resumen
def rol(rasgo):
    return "principal" if rasgo == PRINCIPAL else "control" if rasgo in CONTROL else "secundario"

# Spearman
# calcula para cada sujeot por separado cuanto sigue cada rasgo al nivel de intensidad
# se fija si efectivamente cuando el laser es mas fuerte el LEP sale mas grande o no
def spearman_sujetos(d, rasgos):
    filas = []
    for sub, g in d.groupby("sujeto"): # recorre sujeto suno a uno
        # g son los ensayos de ese sujeto
        nivel = g["condicion"].map(config.NIVELES) # convierte las 4 intensidades en 4 numeros: 1 2 3 4 
        fila = {"sujeto": sub}
        for r in rasgos: # para cada rasgo o valor que medimos marca los ensayos que en efecto tienen ese valor medido
            ok = g[r].notna() # notna es que no este vacio
            fila[r] = spearmanr(nivel[ok], g.loc[ok, r])[0] if ok.sum() > 3 else np.nan
            # calcula el p de spearman entre el nivel y el rasgo usando solo los ensayos con dato
            # si quedan 3 ensayos o menos no calcula nada y pone NaN porque no tiene sentido con pocos datos
        filas.append(fila)
    return pd.DataFrame(filas)



# AUROC
# se fija si alcanza con mirar un solo ensayo para saber si el laser fue debil o fuerte
# AUROC = 1: el de E4 siempre fue más grande. Mirando un solo ensayo alcanza
# AUROC = 0,5: gana la mitad de las veces. El rasgo no sirve para distinguir.
def auroc_sujetos(d, rasgos):
    e = d[d["condicion"].isin(NIVELES_AUROC)] # se queda solo con E1 y E4
    filas = []
    for sub, g in e.groupby("sujeto"): # recorre sujetos de a uno
        y = (g["condicion"] == NIVELES_AUROC[1]).to_numpy().astype(int) # arma la respuesta correcta
        # 1 e E4, 0 es E1
        fila = {"sujeto": sub}
        for r in rasgos:
            ok = g[r].notna().to_numpy() # para cada rasgo marca lso ensayos que tienen el dato
            fila[r] = (roc_auc_score(y[ok], SIGNO.get(r, 1) * g[r].to_numpy()[ok])
                       if len(set(y[ok])) == 2 else np.nan)
            # calcula el AUROC comparando la etiqueta E1 o E4 con el valor del rasgo medido
            # se usa SIGNO porque algunos rasgos bajan con la intensidad (N2) y lo multipicamos por -1 para que se lean todos igual
            # se verifica que el sujeto tenga ensayos de ambas intensidades si no pierde sentido ya que no hay nada que comparar
        filas.append(fila) # guarda el renglon y arma una auroc_sujetos.csv
    return pd.DataFrame(filas)


# Se fija si puede confiar en el resultado del AUROC
# se sortean entre los sujetos que hay armando un grupo inventando de 32 sujetos (pueden repetirse)
# se calcula el promedio y se repite 2000 veces 
# se ordenan los promedios de menor a mayor
# si el intevalo es angosto es un resultado estable y no dependen de los sujetos
def bootstrap_media(valores, n=N_BOOTSTRAP, semilla=SEMILLA):
    v = np.asarray(valores, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(semilla)
    medias = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


# resumen
# junta el p y el AUROC por sujeto y da el resultado del grupo armando resumen.csv
def resumir(rho, auc, rasgos): # recibe las dos tablas (p y AUROC) y recorre los rasgos
    filas = []
    for r in rasgos:
        f = {"rasgo": r, "rol": rol(r)}
        if r in rho: # empieza por spearman tomando los p de ese rasgo
            v = rho[r].dropna()
            f.update(rho_mediana=round(v.median(), 3), rho_n_positivos=f"{int((v > 0).sum())}/{len(v)}",
                     p_spearman=wilcoxon(v).pvalue if len(v) >= 6 else np.nan)
            # rho_mediana es el p típico por grupo
            # rho_n_positivos es la cantidad de sujetos que tuvieron p>o
            # p_spearman hace wilcoxon y prueba si los p son distintos de 0. Solo los calcula si hay al menos 6 sujetos
        if r in auc: # sigue con AUROC
            a = auc[r].dropna()
            ic = bootstrap_media(a) # calcula el intervalo de confianza a partir de los AUROC de ese rasgo
            f.update(auroc_media=round(a.mean(), 3), auroc_ic95=f"[{ic[0]:.3f}; {ic[1]:.3f}]",
                     p_auroc=wilcoxon(a - 0.5, alternative="greater").pvalue if len(a) >= 6 else np.nan)
            # auroc_media es el valor pormedio 
            # auroc_ic95 es el intervalo
            # p_auroc es el wilcoxon y prueba si los AUROC son mayores a 0,5 al azar (menor a 0,5 no tiene sentido)
        filas.append(f)
    return pd.DataFrame(filas)


# graficos
# arma la imagen resultados.png
def figura(rho, auc, rasgos, variante):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5)) # panel a1: izquierod a2: derecho
    for i, r in enumerate(rasgos): # panel izquierdo: spearman
        v = rho[r].dropna()
        a1.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=12, alpha=0.6)
        a1.hlines(v.median(), i - 0.3, i + 0.3, color="k", lw=2)
    a1.axhline(0, color="0.5", lw=0.8)
    a1.set_xticks(range(len(rasgos)), rasgos, rotation=35, ha="right")
    a1.set_ylabel("rho de Spearman (por sujeto)")
    a1.set_title("Relacion con el nivel (E1-E4); linea: mediana")
    cols = list(rasgos)
    for i, r in enumerate(cols): # panel izquierdo: AUROC
        a = auc[r].dropna()
        ic = bootstrap_media(a)
        a2.errorbar(i, a.mean(), yerr=[[a.mean() - ic[0]], [ic[1] - a.mean()]], fmt="o", color="k", capsize=4)
        a2.scatter(np.full(len(a), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(a)), a, s=10,
                   alpha=0.4, color="tab:blue")
    a2.axhline(0.5, color="0.5", lw=0.8, ls="--")
    a2.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    a2.set_ylabel("AUROC E1 vs E4 (por sujeto)")
    a2.set_title("Discriminacion; negro: media e IC 95% bootstrap")
    fig.suptitle(f"Variante {variante}")
    fig.tight_layout()
    return fig


# vemos si el valor de los sujetos coincide con la señal obtenida
# función que marca los ensayos que tienen los dos datos (asignacion de dolor y estimulo)
def _rho(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 5 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan 
    # si hay menos de 5 ensayos o una de las columnas tiene siempre el mismo valor devuelve NaN
    return spearmanr(x[ok], y[ok])[0] # devuelve el p si funciona bien


def cruces_sujetos(d, rasgos):
    filas = []
    for sub, g in d.groupby("sujeto"):
        nivel = g["condicion"].map(config.NIVELES) # hace lo mismo de pasar las intensidades a numeros del 1 al 4
        f = {"sujeto": sub,
             "nivel_vs_VEO": _rho(nivel, g["ptp_veo_uV"]), # analiza si con mas intensidad el VEO se mueve mas
             "nivel_vs_rating": _rho(nivel, g["rating"])} # se fija si con mas intensidad la persona informa mas dolor
        for r in rasgos:                                    
            f[f"rating_vs_{r}"] = _rho(g["rating"], g[r]) # se fija si los que tienen mas dolor tienen mas N2-P2 (si es asi da un p alto)
            # dentro de cada nivel (mismo estimulo) y promediado: si el rasgo sigue a lo que se siente
            f[f"rating_vs_{r}_intra"] = np.nanmean([_rho(h["rating"], h[r]) for _, h in g.groupby("condicion")]
                                                   or [np.nan])
            # se fija si en ensayos con misma intensidad pero distinta validacion el que dolio mas tiene distinto LEP
            # sirve para ver si el LEP sigue al nivel del laser o al dolor de la persona (si es asi deberia ser aprox 0 el p)
        filas.append(f)
    return pd.DataFrame(filas)


# la funcion anterior da un p para cada comparacion
# esta funcion los junta y da el resultado de grupo
def resumir_cruces(t):
    filas = []
    for c in t.columns.drop("sujeto"):
        v = t[c].dropna() # toma los 32 p de una comparacion particular y saca los vacios
        filas.append({"cruce": c, "n": len(v), "rho_mediana": round(v.median(), 3),
                      "n_positivos": f"{int((v > 0).sum())}/{len(v)}",
                      "p_wilcoxon": wilcoxon(v).pvalue if len(v) >= 6 else np.nan})
        # n: cantidad de sujetos x comparacion
        # rho_mediana es el p tipico 
        # n_positivos es cuantos dieron p>0
        # se usa wilcoxon para ver si los p estan corridos para algun lado del cero o si es casualidad (necesita al menos 6 sujetos)
    return pd.DataFrame(filas)


# arma tablas descriptivas para ver los numeros sin estadisticos
# dos archivos:
# 1. para cada nivel (E1 a E4), el dolor buscado, el dolor informado y el valor típico de cada rasgo.
# 2. lo mismo, pero ordenado por la energía real del láser en Joules
def tablas_estimulo(d, rasgos):
    cols = ["rating"] + rasgos
    por_sujeto = d.groupby(["sujeto", "condicion"])[cols].mean() # calcula el promedio de los ensayos para cada combinacion de sujeto y nivel
    # primero de promedian dentrod e cada sujeto para que pesen igual
    por_nivel = por_sujeto.groupby("condicion").median().round(2) # para cada nivel toma la mediana de los 32 sujetos
    por_nivel.insert(0, "rating_esperado", [config.NRS_ESPERADO.get(c) for c in por_nivel.index])
    por_j = d.groupby(["sujeto", "laser_J"])[cols].mean().groupby("laser_J").agg(["mean", "count"]) # promedia los ensayos de cada sujeto en cada energia
    por_j = por_j.loc[:, [(c, "mean") for c in cols] + [("rating", "count")]].round(2) # calcula el promedio entre sujetos y cuantos sujetos hay para cada energia
    por_j.columns = cols + ["n_sujetos"]
    return por_nivel, por_j


# arma figura_cruces.png
def figura_cruces(por_nivel, cruces, rasgos, variante):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(por_nivel)) # panel izquierdo es dolor esperado vs informado
    a1.plot(x, por_nivel["rating_esperado"], "o--", color="0.5", label="buscado")
    a1.plot(x, por_nivel["rating"], "o-", color="k", label="informado (mediana de sujetos)")
    a1.set_xticks(x, por_nivel.index)
    a1.set_ylabel("Puntaje de dolor (NRS 0-10)")
    a1.set_title("Dolor esperado vs informado")
    a1.legend()
    cols = [f"rating_vs_{r}" for r in rasgos] + [f"rating_vs_{r}_intra" for r in rasgos]
    for i, c in enumerate(cols): # panel derecho es dolor informado vs rasgos
        v = cruces[c].dropna()
        a2.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=10, alpha=0.5)
        a2.hlines(v.median(), i - 0.3, i + 0.3, color="k", lw=2)
    a2.axhline(0, color="0.5", lw=0.8)
    a2.axvline(len(rasgos) - 0.5, color="0.8", lw=0.8)
    a2.set_xticks(range(len(cols)), [c.replace("rating_vs_", "") for c in cols], rotation=40, ha="right")
    a2.set_ylabel("rho puntaje-rasgo (por sujeto)")
    a2.set_title("Percepción vs señal: todos los ensayos | dentro de cada nivel")
    fig.suptitle(f"Variante {variante}")
    fig.tight_layout()
    return fig


# función que coordina todo el bloque de las comparaciones
# verifica y llama a las funciones en orden, guarda archivos, imprime
def cruces(d, rasgos, salida, variante):
    if "rating" not in d or d["rating"].notna().sum() == 0:
        print("Sin puntajes por ensayo (falta config.RUTA_RATINGS): no se hacen los cruces")
        return None
    d = d[d["rating"].notna()]                                   # solo ensayos con puntaje
    t = cruces_sujetos(d, rasgos)
    res = resumir_cruces(t)
    por_nivel, por_j = tablas_estimulo(d, rasgos)
    t.to_csv(salida / "cruces_sujetos.csv", index=False)
    res.to_csv(salida / "cruces_resumen.csv", index=False)
    por_nivel.to_csv(salida / "por_nivel.csv")
    por_j.to_csv(salida / "por_energia.csv")
    figura_cruces(por_nivel, t, rasgos, variante).savefig(salida / "cruces.png", dpi=130)
    plt.close("all")
    print(f"\nCruces ({d['sujeto'].nunique()} sujetos, {len(d)} ensayos con puntaje)\n{por_nivel}\n")
    with pd.option_context("display.float_format", lambda x: f"{x:.3g}"):
        print(res.to_string(index=False))
    return res


# ejecución del archivo
# hace todo en orden
def correr(variante=None, carpeta=None): # carga los ensayos de la variante (conVEO, VEO15, VEO20 etc)
    variante = medicion.PRINCIPAL if variante is None else variante
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    d = medicion.datos_variante(variante, carpeta) # carga los ensayos de esa variante (no rechazados)
    rasgos = [r for r in [PRINCIPAL] + SECUNDARIOS + CONTROL if r in d] # arma lista de rasgos a analizar con los que estan en la tabla 
    faltan = [r for r in [PRINCIPAL] + SECUNDARIOS + CONTROL if r not in d]
    if faltan: # avisa, no da error
        print(f"Aviso: faltan rasgos {faltan} (los agrega rasgos_ensayo.py en la etapa de grupo)")
    print(f"Variante {variante}: {d['sujeto'].nunique()} sujetos, {len(d)} ensayos")

    rho = spearman_sujetos(d, rasgos)
    auc = auroc_sujetos(d, rasgos)
    resumen = resumir(rho, auc, rasgos)

    salida = carpeta / "estadistica" / variante
    salida.mkdir(parents=True, exist_ok=True) # no da error si se corre ya ya existe la carpeta
    # guarda las tres tablas
    rho.to_csv(salida / "spearman_sujetos.csv", index=False)
    auc.to_csv(salida / "auroc_sujetos.csv", index=False)
    resumen.to_csv(salida / "resumen.csv", index=False)
    figura(rho, auc, rasgos, variante).savefig(salida / "resultados.png", dpi=130)
    plt.close("all")

    with pd.option_context("display.float_format", lambda x: f"{x:.3g}"):
        print(resumen.to_string(index=False)) # imprime resumen en la terminal
    cruces(d, rasgos, salida, variante) # cruces con la energia y el puntaje de dolor (si esta el CSV)
    print(f"\nGuardado en {salida}")
    return resumen
