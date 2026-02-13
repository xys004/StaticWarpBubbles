import numpy as np

def profile_single_shell(r, a=1.0, b=1.0):
    """
    Piecewise-exponential (Single Shell) profile.
    Ref: Eq. (16) in Bolívar-Abellán-Vasilev (2025).
    
    rho(r) = 0                  for r < 2/b
    rho(r) = a * exp(-b*r)      for r >= 2/b
    """
    r_crit = 2.0 / b
    val = np.zeros_like(r)
    mask = r >= r_crit
    val[mask] = a * np.exp(-b * r[mask])
    return val

def profile_double_shell(r, A=1.0, b=1.0, R=0.5):
    """
    Exponential/Power law decay (Double Shell) profile.
    Ref: Eq. (23) in Bolívar-Abellán-Vasilev (2025).
    
    rho(r) = 0                          for r < R
    rho(r) = A * exp(-b(r-R)) / r^2     for R <= r <= 2/b
    rho(r) = 0                          for r > 2/b
    """
    r_outer = 2.0 / b
    val = np.zeros_like(r)
    
    # Region II: R <= r <= 2/b
    mask = (r >= R) & (r <= r_outer)
    
    # Avoid division by zero if R=0
    # We only evaluate where mask is true, so if R>0, r_safe>0.
    r_safe = r[mask]
    val[mask] = (A * np.exp(-b * (r_safe - R))) / (r_safe**2)
    
    return val
