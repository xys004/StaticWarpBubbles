# pyWarpBubbleFactory 🚀

`pyWarpBubbleFactory` is an advanced Python toolkit built on top of the exact tensor GR engine used in *WarpFactory*. 

**Project Origin:** The original *WarpFactory* toolkit was developed by Applied Physics and released entirely in MATLAB. Our overarching mission with the `pyWarpFactory` ecosystem is to completely port, modernize, and expand this framework into a **100% free, open-source Python architecture**.

While previous toolkits (like our original `StaticWarpBubbles`) were confined to 1D spherically symmetric static integration, this repository shifts focus entirely toward **4D Parametric Spacetimes**. It allows researchers to seamlessly inject arbitrarily curved abstract spatial vectors alongside the capability to **mathematically and autonomously search** for physical configurations via non-linear heuristic solvers.

---

## 🛠️ Key Features

1. **Topological Ansätze**: Stop dealing with raw Christoffel calculations. Instantiate generic parameter-based geometries (`WarpShellAnsatz`, `CustomAnsatz`) wrapping pure 4D Numpy matrix formulations.
2. **Abstract Gridding**: Feed customized topologies instantly into the powerful `pyWarpFactory` Einstein-tensor backends for rigorous verification of standard energy conditions (WEC, NEC, null rays).
3. **Autonomous Optimization Engine**: Features an `ExoticMinimizer` linked to `scipy.optimize`. Automatically iterates your custom topologies using derivative-free approaches (`Powell` / `Nelder-Mead`) to seek dimensional modifications that actively push the spacetime toward zero exotic matter.

---

## 📂 Project Architecture

- **`pywarpbubblefactory/ansatze/`**: Contains the wrappers to dictate your metric structures.
   - `warp_shell.py`: Standard natively integrated Alcubierre shell behavior.
   - `custom.py`: Absolute sandbox. Pass any scalar function `(t, x, y, z)` mapped to $g_{\mu\nu}$.
- **`pywarpbubblefactory/engine.py`**: A robust connector passing boundaries strictly into the strict GR validation engines to capture raw physical penalties (joules).
- **`pywarpbubblefactory/optimizer.py`**: The "Collider." Embeds a SciPy loop that evaluates and manipulates grid physics blindly, handling internal tensor mathematical singularities (NaN gradients) natively and driving parameters specifically to positive energy states.

---

## 🚀 Usage 

Example scripts are available under `examples/` offering a ready-to-use playground.

### 1. Manual Grid Scanning
To brute-force a specific dimension and visualize where boundaries collapse against Energy Violations:
```bash
python examples/search_warpshell.py
```

### 2. Autonomous SciPy Minimization
To input an abstract topography and command your CPU to mathematically search for the parameter configuration yielding the least negative energy requirements:
```bash
# Using Nelder-Mead on standard shells
python examples/optimize_warpshell.py

# Using the Powell method on generic dynamic shapes
python examples/optimize_custom.py

# Live UI / Graphic Dashboard of the iteration
python examples/explore_vdb_lapse.py
```
