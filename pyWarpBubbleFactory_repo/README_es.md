# pyWarpBubbleFactory 🚀

`pyWarpBubbleFactory` es un toolkit matemático avanzado de Python, construido sobreponiéndose directamente al motor crudo de Relatividad General tensorial usado en la librería original de *WarpFactory*.

**Origen del Proyecto:** El toolkit *WarpFactory* original fue desarrollado por Applied Physics y lanzado enteramente en MATLAB de forma privativa/cerrada. Nuestra misión fundamental con el ecosistema `pyWarpFactory` es portar, modernizar y expandir completamente todo ese marco de trabajo hacia una **arquitectura en Python 100% libre y de código abierto**.

Mientras que los paquetes anteriores (como la raíz `StaticWarpBubbles`) se limitaban exclusivamente a la integración estática de curvas con simetría esférica 1D, este nuevo ecosistema está enfocado 100% en el campo de **Espaciotiempos Paramétricos 4D**. Aquí los teóricos pueden introducir conceptos puramente abstractos al tensor $g_{\mu\nu}$ mediante código nativo, y usar algoritmos para **buscar matemáticamente configuraciones viables de Relatividad**.

---

## 🛠️ Características Principales

1. **Ansätze Topológicos Dinámicos**: Prescinde de calcular símbolos de Christoffel a mano por meses. Instancia familias paramétricas (`WarpShellAnsatz`, `CustomAnsatz`) inyectando funciones espaciales abstractas usando NumPy al Vuelo.
2. **Evaluación de Energía Rigurosa**: Pasa todos tus arreglos topológicos por el motor nativo exacto de `pyWarpFactory` para medir en Julios cuánto violaste las Condiciones Normales y Fuertes de Energía (NEC / WEC).
3. **Motor Optimizador Matemático Autónomo**: Destaca el sistema `ExoticMinimizer` adosado a `scipy.optimize`. Ajusta *iterativamente* y a ciegas la forma de tu burbuja (buscando un mínimo en pozos n-dimensionales usando métodos de `Powell`, etc.), sorteando los colapsos matemáticos o singularidades del modelo automáticamente para reducir a 0 el factor de Materia Exótica.

---

## 📂 Arquitectura Interna

- **`pywarpbubblefactory/ansatze/`**: Wrappers para conectar tu métrica física al colisionador.
   - `warp_shell.py`: Burbuja warp tradicional usando el perfil Alcubierre-shell.
   - `custom.py`: Modo libre. Le pasas tu propia función `(T, X, Y, Z)` a las coordenadas del grid y listo.
- **`pywarpbubblefactory/engine.py`**: El canal estabilizador con el motor de validación GR, configurado para amortiguar y recuperar tensores corrompidos numéricamente.
- **`pywarpbubblefactory/optimizer.py`**: El corazón del marco. Ejecuta los solvers de `SciPy` evaluando decenas de configuraciones, manipulándolas y ajustando tus variables hacia estados de energía positiva pura de forma dinámica.

---

## 🚀 Guía de Uso

Cuentas con un laboratorio base de pruebas en las carpetas `examples/`.

### 1. Escaneo en Grilla Manual (*Scanner*)
Evalúa vectorialmente decenas de anchos y tamaños para trazar una gráfica cruda sobre dónde y cómo colapsa la física:
```bash
python examples/search_warpshell.py
```

### 2. Optimización Matemática Guiada (*Minimizer*)
Abandona la prueba de fuerza bruta. Inyéctale tu Ansätze y deja que tu procesador busque a velocidades algorítmicas extremas en qué métricas exactas desciende el cálculo de energía negativa artificial:
```bash
# Usa derivativa Nelder-Mead sobre funciones Shell base
python examples/optimize_warpshell.py

# Utiliza métodos numéricos profundos tipo 'Powell' en formas abstractas libres
python examples/optimize_custom.py

# Despliega métricas avanzadas (VdB) junto al Dashboard Interactivo (UI de Tkinter)
python examples/explore_vdb_lapse.py
```
