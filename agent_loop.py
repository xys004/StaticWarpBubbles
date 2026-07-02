import sys
import subprocess
import json
from warp_mcts import add_node, update_node

conjectures = [
    {
        "conjecture": "Exponential profile (Baseline). High energy concentration at center.",
        "code": "def rho_func(r): return 0.05 * np.exp(-1.0 * r)"
    },
    {
        "conjecture": "Gaussian profile. Faster decay, potentially less total energy if scaled properly.",
        "code": "def rho_func(r): return 0.05 * np.exp(-(r/1.5)**2)"
    },
    {
        "conjecture": "Fermi-Dirac (smoothed top-hat). Constant energy density inside the bubble, rapidly dropping to zero at the wall.",
        "code": "def rho_func(r): return 0.05 / (np.exp((r - 2.0) * 5.0) + 1.0)"
    },
    {
        "conjecture": "Algebraic decay (Inverse polynomial). Slower decay, could have issues with total energy mass if not decaying fast enough.",
        "code": "def rho_func(r): return 0.05 / (1.0 + (r/1.5)**4)"
    },
    {
        "conjecture": "Gaussian ring with a central core. Testing if an off-center density peak violates NEC or improves beta.",
        "code": "def rho_func(r): return 0.05 * np.exp(-(r/1.5)**2) + 0.02 * np.exp(-((r-2.0)/0.5)**2)"
    }
]

for item in conjectures:
    print(f"\\n--- Evaluating: {item['conjecture']}")
    # 1. Add node
    node_id = add_node("root", item["conjecture"], item["code"])
    
    # 2. Evaluate via cluster_eval.py
    cmd = [sys.executable, "cluster_eval.py", item["code"]]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    output = res.stdout.strip()
    try:
        res_json = json.loads(output)
        if res_json.get("valid"):
            # Reward is 1 / M_tilde (so lower mass = higher reward).
            # Scaled by 10 for convenience.
            reward = 10.0 / res_json["M_tilde"]
        else:
            reward = -1.0
            
        update_node(node_id, res_json, reward)
        print(f"Result: {res_json}")
        print(f"Reward assigned: {reward:.4f}")
    except Exception as e:
        print(f"Error parsing output: {output}")
        update_node(node_id, {"error": str(e), "raw": output}, -1.0)
