"""
agents.py
=========
The AI Brains. LLM definitions for Theorist, Coder, and Critic.
"""

import sys
import os
import re
from .llm_wrapper import call_llm

class TheoristAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a cutting edge Theoretical Physicist."
            "Your job is to invent novel 4D Topological Ansatze for General Relativity Warp Metrics."
            "You will receive feedback from the Swarm Optimizer."
            "You must respond with a purely theoretical conceptual explanation of the next formula shape to try."
        )
        
    def propose_idea(self, past_failures_feedback: str) -> str:
        prompt = (
            "Propose a new layout for the metric g_mu_nu components depending on grid (X, Y, Z)."
            "Explain specifically how the Shift vector (beta) and Lapse function (alpha) or Space Expansion (B) should behave functionally.\n\n"
            f"Here is the feedback from our Swarm Collider runs:\n{past_failures_feedback}\n\n"
            "What should we try next?"
        )
        print("\n🧠 [Theorist] Thinking about a new physics approach...")
        return call_llm(self.system_prompt, prompt)

class CoderAgent:
    def __init__(self):
        self.system_prompt = (
            "You are an expert Python Tensor Programmer."
            "Your job is to read theoretical physics ideas from the Theorist and write EXACTLY ONE Python function "
            "that constructs the 4D metric array for pyWarpFactory."
            "It must return an array of shape (4, 4) + X.shape."
            "Your python code must be enclosed in ```python\n...``` blocks."
            "Only use numpy (np)."
            "Provide the function signature: `def latest_ansatz(X, Y, Z, **params):`"
        )
        
    def write_code(self, theorist_idea: str) -> str:
        prompt = (
            f"Here is the theoretical idea:\n{theorist_idea}\n\n"
            "Write the python function `latest_ansatz(X, Y, Z, param1=1.0, ...):`.\n"
            "It must accept a 3D numpy meshgrid and float parameters to be optimized."
            "It must return `g`, an array of shape (4, 4, M, N, P)."
        )
        print("\n👨‍💻 [Coder] Translating physics into tensor calculus code...")
        
        raw_response = call_llm(self.system_prompt, prompt)
        
        # Extract the python code block
        match = re.search(r'```python\n(.*?)```', raw_response, re.DOTALL)
        if match:
            return match.group(1)
        else:
            # Fallback if LLM forgets code blocks
            return raw_response

class CriticAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a Senior Peer Reviewer in General Relativity."
            "You receive raw Exotic Matter totals in Joules and parameter sets from a Genetic Algorithm."
            "Your job is to summarize this data into a 2-paragraph highly technical complain report for the Theorist."
            "You must point out why the parameters failed to reach 0 exotic matter."
        )
        
    def critique(self, swarm_results: dict) -> str:
        prompt = (
            f"Critique these results.\n"
            f"Best Exotic Matter Found: {swarm_results['best_exotic_matter']:.3e} Joules\n"
            f"Params evaluated: {swarm_results['best_params']}\n"
            "Tell the Theorist what to fix structurally for the next iteration."
        )
        print("\n🕵️‍♂️ [Critic] Analyzing Swarm penalty logs...")
        return call_llm(self.system_prompt, prompt)
