"""
pywarpbubblefactory/results.py
==============================
Data structures to hold analysis results.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class ECResult:
    """
    Holds the output of an Energy Conditions evaluation.
    """
    NEC_map: np.ndarray      # Minimum Null Energy across null vectors at each grid point
    WEC_map: np.ndarray      # Weak energy condition map
    rho_map: np.ndarray      # Energy density (T^{00} in Eulerian frame)
    T_euler: np.ndarray      # Full stress-energy tensor in Eulerian frame
    
    # Aggregated boolean checks (ignoring numeric noise below -1e-10)
    NEC_ok: bool
    WEC_ok: bool
    
    # Integral of violated regions: the "amount" of exotic matter
    exotic_matter: float
    
    # Overall sum of energy density (rough mass approximation)
    total_energy: float
