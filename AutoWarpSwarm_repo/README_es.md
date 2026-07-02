# AutoWarp Swarm 🌌

AutoWarp Swarm es un marco de inteligencia artificial autónomo y escalable, diseñado para descubrir y optimizar iterativamente topologías de métricas Warp factibles (Alcubierre, Van Den Broeck, etc.) que minimicen o eliminen la necesidad de Materia Exótica (energía negativa).

Este proyecto cambia el paradigma de la simulación numérica en Relatividad General: abandona las estimaciones lineales y el "ensayo y error" humano, para adentrarse en un entorno de IA Multi-Agente acoplado a un algoritmo heurístico multi-núcleo (PyGAD).

---

## 🏗️ Arquitectura

El ecosistema está estrictamente desacoplado en dos hemisferios funcionales: **El Cerebro** (Inteligencia) y **El Músculo** (Cálculo Físico).

### 🧠 1. El Cerebro (Sistema LLM Multi-Agente)
Ubicado en `brain/agents.py`, esta capa actúa como la fuerza de tracción analítica y creativa, simulando un laboratorio de físicos teóricos orquestado por LLMs (configurado por defecto con **Google Gemini 2.5 Flash**):
- **Agente Teórico (Theorist)**: Lee las restricciones físicas y los datos residuales WEC/NEC para inventar conceptualmente nuevas estructuras para el tensor métrico $g_{\mu\nu}$.
- **Agente Programador (Coder)**: Toma el concepto analítico y lo traduce implícitamente en arrays de Python puro (Numpy) de alto desempeño (generando el script dinámico `latest_ansatz.py`).
- **Agente Crítico (Critic)**: Funciona como un revisor de pares externo. Interpreta los límites en los que el colisionador detonó la geometría y redacta un reporte para que el Teórico recalibre las ecuaciones en la siguiente iteración.

### 🦾 2. El Músculo (Algoritmos Genéticos y Multiprocesamiento)
Ubicado en `muscle/`, este entorno prueba brutalmente las topologías alucinadas por la IA a máxima velocidad.
- **`parallel_engine.py`**: Un envoltorio (*wrapper*) concurrente y seguro para el evaluador principal de tensores de Einstein de `pyWarpFactory`. Satura y aprovecha el 100% de los hilos de procesamiento (Threads) de tu CPU.
- **`swarm_optimizer.py`**: Se enlaza con `PyGAD` (Algoritmo Genético). En lugar de probar una variante secuencial a la vez, genera nubes de parámetros simultáneos aleatorios, los evalúa en paralelo, cruza las métricas "sobrevivientes" (las que requieren menos masa exótica gravitacional), y las muta iterativamente hasta dar con un valle absoluto aceptable.

---

## 🚀 Guía de Inicio Rápido (Manual de Usuario)

El framework es altamente portable. Funciona transparentemente ya sea que corras Windows localmente en tu computadora o en una instancia virtual de Google Cloud / Google Colab.

### Requisitos Previos

Asegúrate de instalar los siguientes componentes vitales:
```bash
pip install google-genai pygad
```

### 🔑 Paso 1: Conectar el Córtex LLM
AutoWarp Swarm necesita acceso a los modelos de lenguaje para escribir matemáticas. Obtén una Clave de API (*API Key*) del portal Google AI Studio.

**Si ejecutas de forma local (Windows/Linux/Mac):**
Establece la variable de entorno en tu consola antes de ejecutar:
```powershell
# En Windows (CMD o Powershell)
set GEMINI_API_KEY="AIzaSyTuLlaveFalsaAca..."

# En Linux / MacOS (Terminal)
export GEMINI_API_KEY="AIzaSyTuLlaveFalsaAca..."
```

**Si ejecutas en un Entorno Web (Google Colab):**
Simplemente agrega tu clave en el menú de "Secretos" (El ícono de la llave a la izquierda del Colab) poniéndole como nombre `GEMINI_API_KEY`. El sistema `llm_wrapper.py` reconocerá Colab y sustraerá la bóveda de forma segura y automática.

### 🏃‍♂️ Paso 2: Lanzar el Ciclo
Navega a la carpeta principal del proyecto a través de la consola y enciende el bucle maestro infinito:
```bash
python main_loop.py
```

### ¿Qué sucederá a continuación?
Verás a la consola anunciar el inicio de la `ÉPOCA 1`:
1. El **Teórico** reportará haber ideado una métrica en sus pensamientos.
2. El **Programador** arrojará a tu disco rígido el archivo de la matriz geométrica real en `latest_ansatz.py`.
3. Aparecerá el logo **[⚡ SWARM]**, indicando que la PC acaba de dispararle esa geometría a todos los núcleos físicos de tu microprocesador a la vez.
4. El colisionador entregará (en Julios) el piso mínimo de Energía Negativa/Exótica de esa estructura particular en nuestro Universo.
5. El **Crítico** generará un resumen forense instruyendo las fallas del tejido y lanzará instantáneamente la `ÉPOCA 2`.
6. El laboratorio trabajará en infinito hasta que presiones `Ctrl+C` para detenerlo.
