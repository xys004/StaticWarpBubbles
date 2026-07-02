import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from pywarpbubblefactory import CustomAnsatz
from pywarpbubblefactory.optimizer import ExoticMinimizer
from pywarpbubblefactory.engine import Engine

def pg_lapse_metric(T, X, Y, Z, A=0.5, w=1.0, lapse_drop=0.0, lapse_w=1.0):
    shape = (4, 4) + X.shape
    g = np.zeros(shape)
    
    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-12
    
    # 1. Shift vector magnitude (Esférico / Radial)
    beta_mag = A * np.exp(-(R/w)**2)
    
    # 2. Lapse function non-trivial. 
    # Lejos de la burbuja es 1 (tiempo plano). En el centro cae un porcentaje lapse_drop.
    alpha = 1.0 - lapse_drop * np.exp(-(R/lapse_w)**2)
    
    # Proyecciones cartesianas del vector shift radial
    beta_x = beta_mag * (X / R)
    beta_y = beta_mag * (Y / R)
    beta_z = beta_mag * (Z / R)
    
    # Construcción de la matriz covariante g_μν
    g[0, 0] = - (alpha**2) + (beta_mag**2)
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    
    # Componentes cruzados (tx, ty, tz)
    g[0, 1] = g[1, 0] = beta_x
    g[0, 2] = g[2, 0] = beta_y
    g[0, 3] = g[3, 0] = beta_z
    
    return g

# Instanciamos el ansatz con una grilla 3D de alta definición
size = (1, 30, 30, 30)
scale = (1.0, 0.2, 0.2, 0.2)
ansatz = CustomAnsatz(func=pg_lapse_metric, grid_size=size, grid_scale=scale)

# Fijamos el shift vector para que sí o sí haya expansión/contracción (A=0.5)
fixed = {
    'A': 0.5,
    'w': 1.0
}

# Dejamos que el optimizador juegue con la forma en la que pasa el tiempo
initial_guess = {
    'lapse_drop': 0.1,  # Tiempo 10% más lento en el núcleo
    'lapse_w': 1.0      # El "pozo" temporal tiene el mismo ancho que la burbuja
}

bounds = {
    'lapse_drop': (0.0, 0.99), # No permitimos tiempo invertido
    'lapse_w': (0.1, 5.0)
}

if __name__ == "__main__":
    print("========== EXPERIMENTO 1: BURBUJA PG CON LAPSE NO TRIVIAL ==========\n")
    
    # 1. Primero evaluamos el caso clásico trivial (Alcubierre/PG original con alpha = 1 constante)
    print(">>> 1. Calculando costo de Energía Negativa para Ansatz Trivial (lapse_drop = 0.0)")
    baseline_params = fixed.copy()
    baseline_params['lapse_drop'] = 0.0
    baseline_params['lapse_w'] = 1.0
    
    baseline_res = Engine().evaluate(ansatz, baseline_params)
    base_exotic = baseline_res.exotic_matter
    print(f"    Energía Exótica (Penalización): {base_exotic:.3e}\n")
    
    # 2. Encendemos el colisionador con la IA de optimización matemática
    print(">>> 2. Encendiendo Colisionador SciPy... Minimizando materia exótica modificando el tiempo.")
    minimizer = ExoticMinimizer(method='Powell', maxiter=15, verbose=True)
    res = minimizer.optimize(ansatz, initial_guess, bounds, fixed)
    
    opt_exotic = res['exotic_matter_min']
    mejora = 100 * (base_exotic - opt_exotic) / base_exotic if base_exotic != 0 else 0
    
    print("\n========== RESULTADO DEL EXPERIMENTO ==========")
    print(f"Masa Exótica Inicial (Trivial):  {base_exotic:.3e}")
    print(f"Masa Exótica Optimizada:         {opt_exotic:.3e}")
    print(f"Reducción: {mejora:.2f}% de materia exótica eliminada.")
    print(f"Mejores parámetros encontrados por SciPy:\n  {res['best_params']}")
