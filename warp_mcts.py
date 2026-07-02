import json
import os
import math

TREE_FILE = os.path.join(os.path.dirname(__file__), "workspace", "warp_tree.json")

def load_tree():
    if not os.path.exists(TREE_FILE):
        return None
    with open(TREE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_tree(tree_data):
    os.makedirs(os.path.dirname(TREE_FILE), exist_ok=True)
    with open(TREE_FILE, "w", encoding="utf-8") as f:
        json.dump(tree_data, f, indent=2, ensure_ascii=False)

def init_tree():
    tree = {
        "problem": "Minimize total energy (M_tilde) for a warp bubble rho(r) subject to WEC, NEC, and beta_max < 1.",
        "nodes": {
            "root": {
                "id": "root",
                "parent": None,
                "visits": 1,
                "value": 0.0,
                "conjecture": "Baseline",
                "code": "",
                "status": "EVALUATED",
                "result_log": "",
                "children": []
            }
        }
    }
    save_tree(tree)
    print("MCTS Tree initialized.")

def add_node(parent_id, conjecture, code):
    tree = load_tree()
    import uuid
    new_id = f"node_{uuid.uuid4().hex[:8]}"
    tree["nodes"][new_id] = {
        "id": new_id,
        "parent": parent_id,
        "visits": 0,
        "value": 0.0,
        "conjecture": conjecture,
        "code": code,
        "status": "PENDING",
        "result_log": "",
        "children": []
    }
    tree["nodes"][parent_id]["children"].append(new_id)
    save_tree(tree)
    return new_id

def update_node(node_id, result_json, reward):
    tree = load_tree()
    node = tree["nodes"][node_id]
    node["status"] = "EVALUATED"
    node["result_log"] = json.dumps(result_json)
    
    curr_id = node_id
    while curr_id:
        curr = tree["nodes"][curr_id]
        curr["visits"] += 1
        curr["value"] += reward
        curr_id = curr["parent"]
        
    save_tree(tree)

if __name__ == "__main__":
    init_tree()
