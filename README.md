# Potenciales evocados por láser como indicador de la intensidad nociceptiva

PSIB 2026 Q2 · Grupo 15 · Jeon, Lauría, Soneira

Procesamiento de EEG del dataset [OpenNeuro ds005293](https://openneuro.org/datasets/ds005293) (experimento 6: estímulos láser en cuatro niveles de energía, E1 a E4, 20 pulsos por nivel). El objetivo es evaluar si rasgos del potencial evocado por láser (LEP) medidos en cada ensayo permiten seguir y discriminar la intensidad del estímulo.

## Qué hace

Todo se corre desde `procesamiento/pipeline.py`:

| Paso | Módulo | Qué hace |
|---|---|---|
| Análisis previo (aparte) | `analisis.py` | PSD por sesión y N2-P2 por canal, para justificar filtros y electrodos |
| 1. Filtrado | `filtrado.py` | Pasa-banda 1–100 Hz y notch 48–52 Hz sobre la señal continua |
| 2. Épocas | `epocas.py` | Pulsos S32–S35 + S100 → E1–E4; épocas de −1 a 2 s, línea de base −1 a 0 s, 250 Hz |
| 3. Limpieza | `limpieza.py` | Marcas de los autores, canales malos, referencia promedio sin frontopolares, épocas groseras |
| 4. Inclusión previa | `inclusion.py` | Sujetos que entran al localizador |
| 5. Localizador | `localizador.py` | Ventanas de N2 y P2 con el promedio colapsado (los cuatro niveles juntos) |
| 6. Medición | `medicion.py` | N2, P2 y N2-P2 por ensayo en la ROI; rechazo final por ROI y VEO, con variantes de sensibilidad |
| 7. Inclusión final | `inclusion.py` | Sujetos que entran al análisis |
| 8. Ensambles | `ensambles.py` | Promedio ponderado por sujeto y nivel, SNR par/impar, gran promedio |
| 9. Rasgos | `rasgos_ensayo.py` | Latencia de N2 y γ-ERS por ensayo |
| 10. Estadística | `estadistica.py` | Spearman por sujeto y AUROC E1 vs E4 por sujeto, para cada variante; cruce con el puntaje de dolor |

Los parámetros (filtros, ventanas, ROI, umbrales, criterios de inclusión) están en `procesamiento/config.py`.

## Estructura

```
.
├── README.md
├── INSTALACION.md                          instalación paso a paso y problemas frecuentes
├── descargar_sujeto.sh                     descarga de sujetos desde OpenNeuro
├── all experiments_preprocessing.csv       épocas y electrodos malos marcados por los autores
├── participants_singletrial_95ByBP.csv     energía y puntaje de dolor por ensayo (autores)
├── data/                                   (no está en git) dataset y resultados
│   ├── ds005293/                           crudo descargado
│   └── derived_mine/                       todo lo que genera el pipeline
└── proyecto/
    ├── pyproject.toml, requirements.txt
    ├── procesamiento/                      el paquete (un módulo por paso)
    └── tests/                              prueba de punta a punta con sujetos sintéticos
```

Los dos CSV de la raíz vienen del [material suplementario de los autores](https://github.com/XiangyueZhao/supplement-data).

## Cómo correrlo

Instalación: ver [INSTALACION.md](INSTALACION.md). Después, desde `proyecto/` y con el entorno activo:

```bash
bash ../descargar_sujeto.sh sub-001 sub-002            # bajar sujetos (una vez)
python -m procesamiento.analisis sub-001               # análisis previo (opcional)
python -m procesamiento.pipeline --todos               # pipeline completo
```

Otras opciones:

```bash
python -m procesamiento.pipeline sub-004 sub-005       # solo esos sujetos (y después la etapa de grupo)
python -m procesamiento.pipeline --todos --sin-grupo   # solo los pasos por sujeto
python -m procesamiento.pipeline --grupo               # solo la etapa de grupo, con lo ya preprocesado
python -m procesamiento.pipeline sub-004 --rehacer     # volver a extraer las épocas desde el crudo
```

Si las épocas de un sujeto ya están extraídas (`data/derived_mine/sub-XXX/sub-XXX_laser-epo.fif`), se reusan y no hace falta el crudo. Un sujeto con error no frena a los demás.

## Salidas principales (`data/derived_mine/`)

- `preprocesado_resumen.csv`: épocas y canales descartados por sujeto.
- `inclusion_previa.csv`, `inclusion_final.csv`: criterios de inclusión y motivo de exclusión.
- `localizador/`: ventanas de N2 y P2, verificación de la ROI.
- `medicion/rasgos_ensayos.csv`: una fila por ensayo con todos los rasgos y el rechazo de cada variante.
- `ensambles/`: LEP por nivel (gran promedio, topomapas, SNR).
- `estadistica/<variante>/`: `resumen.csv` y `resultados.png` (Spearman y AUROC), cruces con el puntaje de dolor. Variantes: `veo150` (principal), `veo200` y `sinveo`.

## Tests

```bash
python -m pytest tests -q
```

Corren el pipeline completo con seis sujetos sintéticos; no hace falta haber descargado nada.
