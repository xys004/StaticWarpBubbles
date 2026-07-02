import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from pywarpbubblefactory import CustomAnsatz
from pywarpbubblefactory.optimizer import ExoticMinimizer
from pywarpbubblefactory.engine import Engine

def vdb_lapse_metric(T, X, Y, Z, A=1.0, b=1.0, B_inner=10.0, lapse_drop=0.5):
    shape = (4, 4) + X.shape
    g = np.zeros(shape)
    
    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-12
    
    # 1. Función analítica de β(R) para Single Shell evaluada a partir de ρ = A*exp(-b*r)
    # Integral C(r) = exp(-br) * (r^2/b + 2r/b^2 + 2/b^3)
    def C_func(r_val):
        return np.exp(-b * r_val) * ((r_val**2)/b + (2*r_val)/(b**2) + 2/(b**3))
    
    r_crit = 2.0 / b
    C_crit = C_func(r_crit)
    
    beta2 = np.zeros_like(R)
    mask = R > r_crit
    beta2[mask] = (8 * np.pi * A / (R[mask]**2)) * (C_crit - C_func(R[mask]))
    
    beta_mag = np.sqrt(np.maximum(beta2, 0.0))
    
    # Perfil Single Shell (sólo para afectar B y Lapse donde la materia y la curvatura operan)
    # Podemos usar la misma máscara o un decaimiento suave para acompañar a beta
    # Aquí aproximaremos la influencia topológica a la región R > r_crit
    f_R = np.exp(-b * np.maximum(R - r_crit, 0.0))
    
    # 2. Vector Shift Radial (Simetría Esférica Estática, sin velocidad "v")
    beta_x = beta_mag * (X / R)
    beta_y = beta_mag * (Y / R)
    beta_z = beta_mag * (Z / R)
    
    # 3. Factor Conformal Van Den Broeck B(R)
    B = 1.0 + (B_inner - 1.0) * f_R
    B2 = B**2
    
    # 4. Lapse Function
    alpha = 1.0 - lapse_drop * f_R
    
    # 5. Geometría Espacial Expandida
    g[1, 1] = B2
    g[2, 2] = B2
    g[3, 3] = B2
    
    # 6. Flujo Radial (Shift cruzado)
    g[0, 1] = g[1, 0] = - B2 * beta_x
    g[0, 2] = g[2, 0] = - B2 * beta_y
    g[0, 3] = g[3, 0] = - B2 * beta_z
    
    # 7. Coordenada Temporal Completa (g_tt = -alpha^2 + \gamma_{ij} \beta^i \beta^j)
    # Aquí \gamma_{ij} \beta^i \beta^j = B^2 (beta_x^2 + beta_y^2 + beta_z^2) = B^2 * beta_mag^2
    g[0, 0] = - (alpha**2) + B2 * (beta_mag**2)
    
    return g

# Instanciamos la grilla
size = (1, 30, 30, 30)
scale = (1.0, 0.2, 0.2, 0.2)
ansatz = CustomAnsatz(func=vdb_lapse_metric, grid_size=size, grid_scale=scale)

# Fijamos los parámetros del Single Shell Estático original (A, b)
fixed = {
    'A': 1.0,
    'b': 1.5
}

# Iniciamos el Optimizador con una suposición
# B_inner inicial: el interior de la burbuja es 10 veces más grande que el exterior
# lapse_drop inicial: el tiempo es 50% más lento adentro (0.5)
initial_guess = {
    'B_inner': 5.0,
    'lapse_drop': 0.5
}

bounds = {
    'B_inner': (1.0, 50.0),    # No permitimos implosiones (B<1) ni burbujas gigantes colapsantes
    'lapse_drop': (0.0, 0.99)  # El tiempo siempre avanza hacia adelante (0.0 a 0.99)
}

if __name__ == "__main__":
    print("========== EXPERIMENTO TAO: VDB + LAPSE + SINGLE SHELL ==========\n")
    
    # Evaluamos la base Alcubierre Pura sin VdB ni Lapse (B=1, lapse=0)
    print(">>> Evaluando Base Alcubierre (v=0.5, B=1.0, Lapse=0.0)")
    baseline_params = fixed.copy()
    baseline_params['B_inner'] = 1.0
    baseline_params['lapse_drop'] = 0.0
    
    base_res = Engine().evaluate(ansatz, baseline_params)
    base_exotic = base_res.exotic_matter
    print(f"    Energía Exótica Alcubierre: {base_exotic:.3e}\n")
    
    print(">>> Iniciando Motor SciPy (Reconfigurando Geometría Van Den Broeck/Lapse)...")
    minimizer = ExoticMinimizer(method='Powell', maxiter=10, verbose=True, use_gui=True)
    res = minimizer.optimize(ansatz, initial_guess, bounds, fixed)
    
    if res.get('aborted', False):
        print("\n[!] OPTIMIZACIÓN ABORTADA POR EL USUARIO DESDE EL UI.")
    else:
        opt_exotic = res.get('exotic_matter_min', 0)
        mejora = 100 * (base_exotic - opt_exotic) / base_exotic if base_exotic != 0 else 0
        
        print("\n========== REPORTE DE GEOMETRÍA ==========")
        print(f"Falla Base:    {base_exotic:.3e}")
        print(f"Mejor Resultado VdB+Lapse: {opt_exotic:.3e}")
        print(f"Reducción: {mejora:.2f}%")
        print(f"Parámetros Óptimos: {res.get('best_params', 'N/A')}")
