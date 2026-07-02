"""
main_loop.py
============
The master infinite loop connecting the Swarm Engine to the Multi-Agent LLMs.
"""

import sys
import os

# Ensure project structure is accessible
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from config import SWARM_WORKERS, PYWARP_ENGINE_PATH
from muscle.parallel_engine import ParallelEngine
from muscle.swarm_optimizer import SwarmOptimizer
from brain.agents import TheoristAgent, CoderAgent, CriticAgent

def main():
    print(f"==================================================")
    print(f" 🚀 AUTO-WARP SWARM ENGINE INITIATED")
    print(f" LLM Agent Pipeline    -> ACTIVE")
    print(f" PyGAD Multiprocessing -> ACTIVE ({SWARM_WORKERS} cores)")
    print(f"==================================================\n")
    
    # Initialize Core Engines
    engine = ParallelEngine(engine_path=PYWARP_ENGINE_PATH, workers=SWARM_WORKERS)
    
    theorist = TheoristAgent()
    coder = CoderAgent()
    critic = CriticAgent()
    
    # Starting loop constraints
    ansatz_grid = {
        'grid_size': (1, 20, 20, 20),
        'grid_scale': (1.0, 0.2, 0.2, 0.2),
        'func_path': os.path.join(PROJECT_ROOT, 'latest_ansatz.py'),
        'func_name': 'latest_ansatz'
    }
    
    feedback = "Let's start our search. Please propose a completely novel spherically-symmetric topological warp metric using off-diagonal constraints."
    
    epoch = 0
    while True:
        epoch += 1
        print(f"\n[=========== EPOCH {epoch} ===========]")
        
        # 1. Brain: Ideation
        idea = theorist.propose_idea(feedback)
        
        # 2. Brain: Translation
        python_code = coder.write_code(idea)
        
        if not python_code.strip():
            print("[!] API rate limit hit or empty response. Sleeping 60 seconds to reset quotas...")
            import time
            time.sleep(60)
            continue
        
        # Save the code so the Swarm Parallel Workers can import it safely
        with open(ansatz_grid['func_path'], "w", encoding="utf-8") as f:
            f.write("import numpy as np\n\n")
            f.write(python_code)
        print(f"💾 [System] Generated Ansatz saved to {ansatz_grid['func_path']}.")
        
        # We define bounded arbitrary variables for the LLM to play with
        bounds = {
            'param1': (-10.0, 10.0),
            'param2': (-10.0, 10.0)
        }
        fixed = {}
        
        # 3. Muscle: Swarm evaluation
        optimizer = SwarmOptimizer(parallel_engine=engine, bounds=bounds, fixed=fixed, pop_size=10, generations=2)
        swarm_results = optimizer.run_swarm(ansatz_grid)
        
        best_exotic = swarm_results['best_exotic_matter']
        best_params = swarm_results['best_params']
        print(f"🧬 [System] Best configuration found: {best_exotic:.3e} J")
        
        # 4. Phase 5: Observable Physics Extraction
        if best_exotic < 1e-10:
            print(f"\n🎉 [Eureka] Zero Exotic Matter Found! Compiling visual physics charts...")
            
            # Reprocess the single best finding on the main thread to get the full T_euler
            import sys
            sys.path.append(PYWARP_ENGINE_PATH)
            from pywarpbubblefactory.engine import Engine
            from pywarpbubblefactory.ansatze.custom import CustomAnsatz
            from pywarpbubblefactory.visualizer import Visualizer
            import importlib.util
            
            # Reload module
            spec = importlib.util.spec_from_file_location("latest_ansatz", ansatz_grid['func_path'])
            ansatz_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ansatz_mod)
            
            best_ansatz = CustomAnsatz(func=getattr(ansatz_mod, ansatz_grid['func_name']), 
                                       grid_size=ansatz_grid['grid_size'], grid_scale=ansatz_grid['grid_scale'])
            
            try:
                final_result = Engine().evaluate(best_ansatz, best_params)
                print(f"  --> Processing ADM 3+1 tensors...")
                plot_dir = os.path.join(PROJECT_ROOT, "eureka_plots")
                vis = Visualizer(save_dir=plot_dir)
                vis.render_analysis(final_result)
                
                # ------ PHASE 6 GCP UPLOAD HOOK ----- 
                import gcp_sync
                bucket = os.getenv("GCP_BUCKET_NAME", "autowarp-swarm-data")
                gcp_sync.upload_folder_to_gcs(bucket, plot_dir, "latest_eureka_plots")
                gcp_sync.upload_file_to_gcs(bucket, ansatz_grid['func_path'], "latest_eureka_ansatz.py")
                # ------------------------------------
                
                print("\n🛑 Shutting down loop globally. Mission Accomplished.")
                sys.exit(0)
            except Exception as e:
                print(f"[!] Extractor failed to chart: {e}")
                
        # 5. Brain: Diagnostics
        feedback = critic.critique(swarm_results)
        print(f"\n📝 [Feedback for Epoch {epoch}]:\n{feedback}")
        
        # Throttling to protect the Free Tier API (Max 15 requests/minute).
        # We make 3 requests per epoch. Let's rest 20 seconds.
        print("\n⏳ Sleeping 20s to prevent API Rate Limiting...")
        import time
        time.sleep(20)

if __name__ == '__main__':
    main()
