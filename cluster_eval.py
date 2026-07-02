import numpy as np
import json
import sys

def evaluate_rho(rho_code_str, r_max=10.0, N=4000, L_meter=1.0):
    r_arr = np.linspace(0.0, r_max, N)
    
    # execute the custom rho_code_str to define 'rho_func(r)'
    local_env = {"np": np}
    exec(rho_code_str, local_env)
    rho_func = local_env.get("rho_func")
    
    rho = rho_func(r_arr)
    
    # beta
    dr = r_arr[1] - r_arr[0]
    ig = rho * r_arr**2
    integral = np.zeros_like(r_arr)
    integral[1:] = np.cumsum(0.5*(ig[:-1]+ig[1:])*dr)
    with np.errstate(divide='ignore', invalid='ignore'):
        bs = np.where(r_arr > 0, 8*np.pi*integral/r_arr**2, 0.0)
    beta = np.sqrt(np.maximum(bs, 0.0))
    beta_max = float(np.max(beta))
    
    # conditions
    drho = np.gradient(rho, r_arr)
    nec = -0.5 * r_arr * drho
    nec_ok = bool(np.all(nec[1:] >= -1e-10))
    wec_ok = bool(np.all(rho >= -1e-10))
    hor_ok = beta_max < 1.0
    valid = nec_ok and wec_ok and hor_ok
    
    # mass and energy
    M_tilde = float(np.trapz(rho * r_arr**2, r_arr))
    E_J = 4 * np.pi * M_tilde * L_meter * 1.2135e44
    
    return {
        "beta_max": beta_max,
        "nec_ok": nec_ok,
        "wec_ok": wec_ok,
        "hor_ok": hor_ok,
        "valid": valid,
        "M_tilde": M_tilde,
        "E_J": E_J
    }

if __name__ == "__main__":
    code = sys.argv[1]
    try:
        res = evaluate_rho(code)
        print(json.dumps(res))
    except Exception as e:
        print(json.dumps({"error": str(e), "valid": False}))
