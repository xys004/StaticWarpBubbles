"""
parallel_engine.py
==================
Evaluates a massive batch of metric parameters in parallel using ProcessPoolExecutor.
"""

import os
import sys
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def _worker_evaluate(ansatz_config, params):
    """
    Isolated worker function to execute pyWarpFactory operations concurrently.
    We import pyWarpFactory internally to avoid fork corruption on Windows.
    """
    import sys
    # Connect paths
    sys.path.append(ansatz_config['engine_path'])
    from pywarpbubblefactory.engine import Engine
    from pywarpbubblefactory.ansatze.custom import CustomAnsatz
    import __main__ # Required to pull the dynamically generated metric function
    
    # Reload function dynamically
    # Expects the dynamic function to be stored in the module 'latest_ansatz'
    import importlib.util
    spec = importlib.util.spec_from_file_location("latest_ansatz", ansatz_config['func_path'])
    ansatz_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ansatz_module)
    
    func = getattr(ansatz_module, ansatz_config['func_name'])
    
    ansatz = CustomAnsatz(func=func, grid_size=ansatz_config['grid_size'], grid_scale=ansatz_config['grid_scale'])
    
    try:
        res = Engine().evaluate(ansatz, params)
        exotic = res.exotic_matter
        if np.isnan(exotic):
            return 1e50
        return exotic
    except Exception as e:
        # Heavily penalize any function crash
        return 1e50

class ParallelEngine:
    def __init__(self, engine_path, workers):
        self.engine_path = engine_path
        self.workers = workers
        
    def evaluate_batch(self, ansatz_config, parameters_list):
        """
        Takes a list of parameter dictionaries and evaluates their metric topologies 
        in parallel across all valid CPU cores.
        """
        results = []
        # Bundle config with the engine path for the isolated worker
        ansatz_config['engine_path'] = self.engine_path
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Create a future for each parameter set
            futures = [executor.submit(_worker_evaluate, ansatz_config, p) for p in parameters_list]
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append(1e50)
                    
        return results
