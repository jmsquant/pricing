# Derivatives Pricing Library Hesperides

## Integración con curvas

Esta copia se ha extraído de `../LibreriaValoracion.zip`, que se mantiene intacto.
Los argumentos de tipos de la API aceptan también curvas con `discount(t)`.
El constructor de departamento cuantitativo es el paquete independiente
`../libreria_curvas`; instalación y ejemplo completo en `../README.md`.
`engine` elige el método de pricing y no el método de construcción de curvas.
En binomial, `step_years` convierte pasos a la unidad temporal de la curva.

La implementación usa clases y métodos en todas las capas. `HesperidesAPI()` es
la fachada orientada a objetos; los nombres históricos de `hesperides.api`
son alias de sus métodos enlazados, con los mismos argumentos. Los pricers,
motores, validadores y simuladores se componen mediante protocolos e inyección
de dependencias. No hay funciones sueltas, funciones anidadas ni `lambda` en
el código o los tests. Véase la sección SOLID de `../README.md`.

Python library for pricing financial derivatives. Course project for *Asset Pricing Fundamentals* and *Interest Rate Models*.

**We use Python 3.13.** Declare `requires-python = "==3.13.*"` in `pyproject.toml`.

## Goal
You will build your own pricing library incrementally, adding new derivatives and models in each assignment, following good software engineering practices.

## Requirements
- Clean, modular architecture.
- Deliver the source inside **Curvas.zip**, with the BAT launcher and the bundled runtime; all tests must pass.
- Public tests are included in the repo and must pass before submission.
- Private tests are used by us for grading.
- Submissions are cumulative: each one must include all previous functionality.
- Keep `pyproject.toml` in the repo.

---

## Modular Architecture

Build a **modular, maintainable, and extensible** library for derivatives pricing, with clear separation among the following parts.

- **Contracts / Instruments** (what is being priced).
- **Market** (curves, forwards, volatilities, fixings).
- **Models** (dynamics under the pricing measure).
- **Engines** (algorithms: analytical, tree, MC, PDE).
- **Pricers** (orchestration: combines contract + market + model + engine).

---

## Submission Requirements
- Clean and modular architecture (see package structure).
- **Single public interface** for tests: `hesperides/api.py`.
- Public tests are included in the repo.
- Private tests are used by us for grading.
- Distribution in **ZIP** format with `ejecutar.bat`; **all tests must pass**.

---

## Uso desde el ZIP

Extraiga todo `Curvas.zip`. El lanzador `../ejecutar.bat` usa Python y las
dependencias incluidas en `../.runtime`; no instala paquetes ni requiere Internet.
Configure `../ejecutar.py` y aporte los libros de mercado en `../inputs`.

Para ejecutar las pruebas desde la carpeta principal:

```powershell
.\.runtime\python.exe -B -m pytest -q
```

### 1) Single Public Interface: `hesperides.api`
Tests (both public and private) can **ONLY** do the following.

```python
import hesperides.api as hapi
```

Therefore, follow these rules.

- `hesperides/api.py` is the **only public interface**.
- The rest of the package is **internal** and can change as long as the API is not broken.

### 2) Private Tests Philosophy
Private tests typically do the following.

1. Generate inputs (test cases).
2. Call **facade method aliases** from `hesperides.api` with an **exact signature**.
3. Check expected values / properties (bounds, monotonicity, put-call parity…) / references.

If the assignment asks for `get_price_binomial_european_call(...)`, that callable name must exist in `api.py` as a method alias and behave as specified.

In any case, you can also check your code with your own tests, taking examples from the list of exercises in the course or other sources, like Privault's exercises.

---

## API Contract (`hesperides/api.py`)

### Principles
- `api.py` exposes **simple and stable aliases of facade methods**.
- Inputs and outputs as specified in each assignment.


**(A) Assignment-specific methods (mandatory if requested)**  
Example: binomial for European call. 

```python
from hesperides.api import HesperidesAPI

precio = HesperidesAPI().get_price_binomial_european(
    St=100, K=100, T=10, R=0.01, u=1.1, d=0.9, call=True,
)
```


---

## Performance

> **Using loops instead of vectorized NumPy code will be penalized.**

Specific requirements will be defined in each assignment. An example of an unacceptable loop would be iterating over Monte Carlo paths during simulation when NumPy vectorization (using array operations and broadcasting instead of Python loops) is expected; see the NumPy broadcasting guide: https://numpy.org/doc/stable/user/basics.broadcasting.html.

Also follow the best practices explained in *Fundamentos de Python* for NumPy course: specify `dtype` when possible for additional optimization, use views instead of copies when a copy is not needed, and apply vectorization patterns where appropriate.

> **Monte Carlo must be reproducible through seeds.**

This will be detailed in the corresponding assignments.



---


## Distribuci?n y dependencias

La entrega ?nica es `Curvas.zip`, con c?digo, documentaci?n, tests, lanzador BAT
y un entorno Python 3.13 para Windows de 64 bits. Se ejecuta directamente desde
la carpeta extra?da. Los scripts de usuario colocados en la carpeta principal
pueden importar `hesperides.api` usando `.runtime/python.exe`.

Las dependencias de ejecuci?n permanecen declaradas en `pyproject.toml`.
`pytest` y sus dependencias se incluyen para poder verificar el c?digo sin
instalar herramientas. Las versiones y licencias est?n dentro de `.runtime`.
No se modifica la metodolog?a financiera ni las semillas de Monte Carlo.
