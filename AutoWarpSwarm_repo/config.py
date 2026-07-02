import os
import multiprocessing

# --- LLM configuration ---
# The active provider. 'gemini' or 'local' (e.g. LMStudio via openai library proxy)
LLM_PROVIDER = os.getenv("AUTOWARP_LLM", "gemini")
GEMINI_MODEL = "gemini-2.5-flash"  # Flash is fast and cheap, ideal for repetitive iterating

# --- Cluster Configuration ---
# Detects available cores to scale the Genetic Algorithm Swarm automatically
AVAILABLE_CORES = multiprocessing.cpu_count()

# Safety cap for multiprocessing: Reserve 1 core for the OS and 1 core for the LLM agents
SWARM_WORKERS = max(1, AVAILABLE_CORES - 2)

# --- Path Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Fallback local path to the PyWarpBubbleFactory engine
PYWARP_ENGINE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '../pyWarpBubbleFactory_repo'))
