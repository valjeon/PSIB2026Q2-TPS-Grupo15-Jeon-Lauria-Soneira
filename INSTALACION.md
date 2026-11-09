# Instalación

## Qué hay que tener
- **Python 3.11 o más nuevo** (`python --version`; en Mac puede ser `python3`).
- **Git**, para clonar y subir cambios.
- **curl**, para bajar los datos. Ya viene en macOS y en Windows 10/11.
- **Espacio en disco:** ~1,4 GB por sujeto descargado, más lo que ocupen los resultados.
- Internet para la descarga (los datos vienen de OpenNeuro, vía S3).

## Pasos (una sola vez)

macOS / Linux:
```bash
git clone https://github.com/valjeon/PSIB2026Q2-TPS-Grupo15-Jeon-Lauria-Soneira.git PSIB_Grupo15_EEG
cd PSIB_Grupo15_EEG
python3 -m venv .venv
source .venv/bin/activate
cd proyecto
pip install -e ".[dev]"
```

Windows (PowerShell):
```powershell
git clone https://github.com/valjeon/PSIB2026Q2-TPS-Grupo15-Jeon-Lauria-Soneira.git PSIB_Grupo15_EEG
cd PSIB_Grupo15_EEG
python -m venv .venv
.venv\Scripts\activate
cd proyecto
pip install -e ".[dev]"
```

`pip install -e ".[dev]"` deja el paquete `procesamiento` importable e instala las dependencias (con `pytest` y `pybv` por el `[dev]`). Hay que correrlo parado en `proyecto/`, que es donde está `pyproject.toml`. La alternativa es `pip install -r requirements.txt`, desde la misma carpeta.

Cada vez que abras una terminal nueva hay que volver a activar el entorno (`source .venv/bin/activate` o `.venv\Scripts\activate`, desde la raíz del repo). Se sabe que está activo porque aparece `(.venv)` al principio de la línea.

## Datos
Desde la raíz del repo, bajar los sujetos que te tocan:
```bash
bash descargar_sujeto.sh sub-001 sub-002
```
Quedan en `data/ds005293/`, y los resultados del pipeline en `data/derived_mine/`. La carpeta `data/` no se sube a git.

Si los datos están en otro lugar, indicá la carpeta del dataset con la variable `PSIB_DATOS`; los resultados se escriben en la carpeta hermana `derived_mine/`:
```bash
export PSIB_DATOS=/ruta/a/data/ds005293
```

## Verificar que todo funciona
Desde `proyecto/`:
```bash
python -m pytest tests -q
```
Deben pasar los 5 tests (usan datos sintéticos, no hace falta haber descargado nada).

Con un sujeto descargado:
```bash
python -m procesamiento.pipeline sub-001 --sin-grupo
```
Debe terminar con `sub-001: 80 extraidas -> ...` y dejar `data/derived_mine/sub-001/sub-001_laser-epo.fif`.

Cómo correr el pipeline completo y qué genera: ver el [README](README.md).

## Problemas frecuentes
- `ModuleNotFoundError: No module named 'mne'` o `'procesamiento'`: el entorno no está activo o falta `pip install -e ".[dev]"`.
- `error: indicar sujetos, --todos o --grupo` con `--todos`, o `No esta data/derived_mine/preprocesado_resumen.csv`: el pipeline no encuentra `data/ds005293`. Correr desde `proyecto/` dentro del repo que tiene la carpeta `data/`, o definir `PSIB_DATOS`.
- `FileNotFoundError` al leer un sujeto: ese sujeto no está descargado ni tiene épocas extraídas en esta computadora.
- `mmap failed: Operation canceled` en git (macOS): el repo está en una carpeta de iCloud con archivos sin descargar; ver la nota de arriba.
- En Windows, si PowerShell bloquea la activación del entorno: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.