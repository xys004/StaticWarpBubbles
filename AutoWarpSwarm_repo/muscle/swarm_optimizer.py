"""
swarm_optimizer.py
==================
A Genetic Algorithm orchestrator leveraging PyGAD to find minimums concurrently.
"""

import sys
import os
import pygad
import numpy as np

# Load parallel engine
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from muscle.parallel_engine import ParallelEngine

class SwarmOptimizer:
    def __init__(self, parallel_engine: ParallelEngine, bounds: dict, fixed: dict, pop_size=20, generations=5):
        self.parallel_engine = parallel_engine
        self.bounds = bounds
        self.fixed = fixed
        self.pop_size = pop_size
        self.generations = generations
        
        self.var_names = list(self.bounds.keys())
        self.best_history = []
        
        # We need to map pygad genes to PyWarpFactory Dictionaries
        self._ansatz_config = None

    def _fitness_func(self, ga_instance, solutions, solutions_indices):
        """
        Batch fitness function called by PyGAD passing the entire population.
        """
        dict_list = []
        for solution in solutions:
            params = self.fixed.copy()
            for k, v in zip(self.var_names, solution):
                params[k] = v
            dict_list.append(params)
            
        pens = self.parallel_engine.evaluate_batch(self._ansatz_config, dict_list)
        fitnesses = [1.0 / (abs(p) + 1e-10) for p in pens]
        
        best_pen = min(pens)
        best_idx = pens.index(best_pen)
        self.best_history.append((best_pen, dict_list[best_idx]))
        
        return fitnesses

    def run_swarm(self, ansatz_config):
        self._ansatz_config = ansatz_config
        
        gene_space = []
        for k in self.var_names:
            gene_space.append({'low': self.bounds[k][0], 'high': self.bounds[k][1]})
        
        initial_pop = []
        for _ in range(self.pop_size):
            individual = [np.random.uniform(self.bounds[k][0], self.bounds[k][1]) for k in self.var_names]
            initial_pop.append(individual)
            
        ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=max(2, self.pop_size // 3),
            fitness_func=self._fitness_func,
            fitness_batch_size=self.pop_size,
            initial_population=initial_pop,
            gene_space=gene_space,
            mutation_type="random",
            mutation_percent_genes=20,
            suppress_warnings=True
        )
        
        print("\n[⚡ SWARM] Firing multithreading genetic algorithm...")
        ga_instance.run()
        
        # Retrieve absolute best
        best_so_far = sorted(self.best_history, key=lambda x: x[0])[0]
        
        return {
            'best_exotic_matter': best_so_far[0],
            'best_params': best_so_far[1],
            'history': self.best_history
        }
