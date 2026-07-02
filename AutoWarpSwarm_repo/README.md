# AutoWarp Swarm 🌌

AutoWarp Swarm is an autonomous, scalable artificial intelligence framework designed to iteratively discover and optimize physically viable warp drive metric topologies (Alcubierre, Van Den Broeck, etc.) that minimize or eliminate the need for Exotic Matter (negative energy). 

This project shifts the paradigm of General Relativity numerical simulation away from manual human parameter-guessing into an automated, multi-agent AI environment paired with a multi-core heuristic algorithm (PyGAD).

---

## 🏗️ Architecture

The system is strictly decoupled into two functional hemispheres: **The Brain** (Intelligence) and **The Muscle** (Computation).

### 🧠 1. The Brain (LLM Multi-Agent System)
Located in `brain/agents.py`, this layer acts as the creative and analytical driving force, simulating a laboratory of theoretical physicists orchestrated natively by an LLM (defaulted to **Google Gemini 2.5 Flash**):
- **The Theorist Agent**: Reads physics constraints and raw WEC/NEC penalty data to conceptually invent new structures for the metric tensor $g_{\mu\nu}$.
- **The Coder Agent**: Takes the theoretical concept and seamlessly translates the mathematics into optimized, purely Numpy-based Python arrays (generating the `latest_ansatz.py` file dynamically).
- **The Critic Agent**: Serves as a peer-reviewer. Reads the tensor-collision limits from the genetic swarm and compresses mathematical failure points into readable feedback to steer the Theorist on the next epoch.

### 🦾 2. The Muscle (Genetic Algorithms & Multiprocessing)
Located in `muscle/`, this layer tests the hallucinated topologies aggressively fast.
- **`parallel_engine.py`**: A thread-safe concurrent wrapper around the core `pyWarpFactory` Einstein-tensor evaluators. It saturates 100% of available CPU processor threads.
- **`swarm_optimizer.py`**: Interlaces with `PyGAD` (Genetic Algorithms). Instead of testing one parameter sequentially, it spans a massive randomized population array of geometries, evaluates them in parallel, crosses the surviving metrics ("the ones requiring the least negative mass"), and mutates them iteratively until an absolute minima is found.

---

## 🚀 Quick Start (User Manual)

The framework is highly portable. It works seamlessly whether you are running natively on a Windows Machine or deep inside a Google Cloud / Google Colab instance.

### Prerequisites

Ensure the following core components are installed:
```bash
pip install google-genai pygad
```

### 🔑 Step 1: Set Up your LLM Cortex
AutoWarp Swarm needs language models to generate math. Get an API Key from Google AI Studio. 

**If running locally (Windows/Linux/Mac):**
Set the environment variable in your terminal before execution:
```powershell
# Windows
set GEMINI_API_KEY="AIzaSyYourKeyHere..."

# Linux / Mac
export GEMINI_API_KEY="AIzaSyYourKeyHere..."
```

**If running in Google Colab:**
Simply add your key under the internal "Secrets" lock icon on the left sidebar and label it `GEMINI_API_KEY`. The `llm_wrapper.py` script automatically parses Colab vaults securely.

### 🏃‍♂️ Step 2: Ignite the Flow
Navigate to the root project directory and execute the master infinite loop:
```bash
python main_loop.py
```

### What to Expect:
Once launched, the console will track `EPOCH 1`.
1. You will see the **Theorist** log an idea.
2. The **Coder** will generate the `latest_ansatz.py` tensor grid.
3. The Terminal will spawn a **[⚡ SWARM]** log indicating the system is sending the geometry into the genetic multiprocessing matrix across all your CPUs.
4. The system outputs the minimum Exotic energy found (in Joules).
5. The **Critic** generates a physics summary and immediately re-triggers `EPOCH 2` adjusting the math.
6. The process runs infinitely until stopped manually.
