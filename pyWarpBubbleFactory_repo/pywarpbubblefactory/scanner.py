"""
pywarpbubblefactory/scanner.py
==============================
Sweep through parameter space arrays to evaluate the ansatz at many points.
Because 3D tensor computations are expensive, we use a simple loop (could be parallelized).
"""

import itertools
import pandas as pd
from typing import Dict, List, Any
from .engine import Engine
from .ansatze.base import Ansatz

def scan(ansatz: Ansatz, param_grid: Dict[str, List[Any]], engine: Engine = None) -> pd.DataFrame:
    """
    Perform a grid search over param_grid.
    Returns a Pandas DataFrame with parameters and scalar EC results.
    """
    if engine is None:
        engine = Engine()
        
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    # Generate all combinations
    combinations = list(itertools.product(*values))
    
    rows = []
    print(f"Scanning {len(combinations)} configurations for Ansatz '{ansatz.name}'...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"[{i+1}/{len(combinations)}] Evaluating {params}...", end='', flush=True)
        
        try:
            res = engine.evaluate(ansatz, params)
            print(f" Done. Valid={res.WEC_ok and res.NEC_ok}")
            
            # Record scalars
            row = copy_params(params)
            row['NEC_ok'] = res.NEC_ok
            row['WEC_ok'] = res.WEC_ok
            row['Valid'] = res.NEC_ok and res.WEC_ok
            row['ExoticMatter'] = res.exotic_matter
            row['TotalEnergy']  = res.total_energy
            rows.append(row)
            
        except Exception as e:
            print(f" FAILED: {e}")
            row = copy_params(params)
            row['Valid'] = False
            row['Error'] = str(e)
            rows.append(row)
            
    return pd.DataFrame(rows)

def copy_params(p):
    return {k: v for k, v in p.items()}
