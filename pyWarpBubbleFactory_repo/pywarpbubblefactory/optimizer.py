"""
pywarpbubblefactory/optimizer.py
================================
Heuristic optimization engine using SciPy to find parameters that minimize exotic matter.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize
import threading
from .engine import Engine
from .ansatze.base import Ansatz

class StopOptimizationException(Exception):
    pass

class ExoticMinimizer:
    def __init__(self, engine: Engine = None, method: str = 'Nelder-Mead', maxiter: int = 50, verbose: bool = True, use_gui: bool = False):
        self.engine = engine if engine is not None else Engine()
        self.method = method
        self.maxiter = maxiter
        self.verbose = verbose
        self.use_gui = use_gui
        self.gui = None
        
        self.current_best = float('inf')
        
        # History tracking
        self.history = []

    def _objective(self, x: np.ndarray, 
                   var_names: List[str], 
                   fixed_params: Dict[str, Any], 
                   ansatz: Ansatz) -> float:
        """
        The objective function to minimize: Exotic matter.
        """
        # Reconstruct full parameters
        params = fixed_params.copy()
        for k, v in zip(var_names, x):
            params[k] = v
            
        # Check abort early
        if self.use_gui and self.gui and self.gui.check_abort():
            raise StopOptimizationException("User Aborted")
            
        try:
            res = self.engine.evaluate(ansatz, params)
            obj_val = res.exotic_matter
            
            if np.isnan(obj_val):
                msg = f"Eval: {params} -> Exotic: NaN [✗] -> Penalizing"
                if self.verbose: print(msg)
                if self.gui: self.gui.add_log(msg)
                self.history.append({'params': params, 'exotic': np.nan, 'valid': False})
                return 1e50
            
            valid_str = "✓" if res.NEC_ok and res.WEC_ok else "✗"
            msg = f"Eval: {params} -> Exotic: {obj_val:.3e} [{valid_str}]"
            if self.verbose: print(msg)
            if self.gui:
                self.gui.add_log(msg)
                # Update best score natively
                if obj_val < self.current_best:
                    self.current_best = obj_val
                    self.gui.update_best(obj_val, params)
                
            self.history.append({
                'params': params,
                'exotic': obj_val,
                'valid': res.NEC_ok and res.WEC_ok
            })
            
            return obj_val
            
        except Exception as e:
            if self.verbose:
                print(f"Eval: {params} -> ERROR: {e}")
            # Penalize severely on physical crash to force optimizer away
            return 1e50 

    def optimize(self, ansatz: Ansatz, 
                 initial_guess: Dict[str, float], 
                 bounds: Dict[str, Tuple[float, float]], 
                 fixed_params: Dict[str, Any]) -> dict:
        """
        Run the minimization procedure.
        
        initial_guess: Dict of parameter names to their starting float values.
        bounds: Dict of parameter names to (min, max) bounds.
        fixed_params: Dict of parameters to hold constant.
        """
        self.history = []
        var_names = list(initial_guess.keys())
        x0 = np.array([initial_guess[k] for k in var_names])
        
        bnds = [bounds[k] for k in var_names] if bounds else None
        
        print(f"\n--- Starting Optimization on {ansatz.name} ---")
        print(f"Method: {self.method}")
        print(f"Variables: {var_names}")
        print(f"Initial guess: {x0}")
        print(f"Fixed: {fixed_params}\n")
        
        self.current_best = float('inf')
        
        def _optimization_worker():
            try:
                if self.gui: self.gui.update_status("Status: RUNNING OPTIMIZATION...")
                
                opt_res = minimize(
                    self._objective, 
                    x0, 
                    args=(var_names, fixed_params, ansatz),
                    method=self.method,
                    bounds=bnds,
                    options={'maxiter': self.maxiter, 'disp': self.verbose}
                )
                
                # Format final return
                best_params = fixed_params.copy()
                for k, v in zip(var_names, opt_res.x):
                    best_params[k] = v
                    
                final_res = {
                    'best_params': best_params,
                    'exotic_matter_min': opt_res.fun,
                    'success': opt_res.success,
                    'message': opt_res.message,
                    'iterations': opt_res.nit,
                    'history': self.history
                }
                
                if self.gui:
                    self.gui.on_finish(f"SUCCESS: {opt_res.message}\nMinimum found: {opt_res.fun:.3e}")
                
                return final_res
                
            except StopOptimizationException:
                msg = "ABORTED BY USER. See best params found so far."
                if self.verbose: print(msg)
                if self.gui: self.gui.on_finish(msg)
                return {'aborted': True, 'history': self.history}
            
            except Exception as e:
                msg = f"ERROR IN OPTIMIZATION: {e}"
                if self.verbose: print(msg)
                if self.gui: self.gui.on_finish(msg)
                return {'aborted': True, 'error': str(e), 'history': self.history}

        if self.use_gui:
            from .gui import OptimizerDashboard
            self.gui = OptimizerDashboard(title=f"Warp Factory - {ansatz.name} Optimization")
            self.gui.add_log(f"Method: {self.method} | Variables: {var_names}")
            
            # Start scipy loop in background thread
            t = threading.Thread(target=_optimization_worker, daemon=True)
            t.start()
            
            # This blocks the main thread until the user closes the window
            self.gui.run()
            
            return {'history': self.history, 'aborted': self.gui.check_abort()}
        else:
            return _optimization_worker()
