"""
examples/search.py
===================
Script de búsqueda de soluciones para la burbuja de warp estática.

CÓMO USARLO
-----------
1. Edita la lista CANDIDATES (abajo) con los perfiles que quieras probar.
2. Corre:  python examples/search.py
3. Revisa la tabla de resultados en la terminal y los plots en examples/.

CICLO DE BÚSQUEDA
-----------------
Para encontrar una "solución válida" necesitás:

   ✓ WEC : ρ(r) ≥ 0  para todo r
   ✓ NEC : ρ'(r) ≤ 0 para todo r  (densidad no-creciente)
   ✓ β_max < 1        (sin horizonte)

Con los perfiles de esta librería (Exponencial, Gaussiano, Sech²)
los dos primeros están garantizados ANALÍTICAMENTE.
El único parámetro libre a ajustar es β_max.

PARÁMETROS CLAVE
----------------
  A          : amplitud de ρ(r) — controla cuánta energía tiene la burbuja
  b / sigma  : escala de caída — controla el tamaño de la burbuja
  L_meter    : escala física en metros (1 código-unit = L_meter metros)
  target_beta: β_max deseado (0.3 = conservador, 0.8 = cercano al horizonte)

REGLAS DE ORO
-------------
  Perfil Exponencial:   A_crit = b  / (8π·C_exp)   →  A ∝ b   (lineal)
  Perfil Gaussiano:     A_crit = 1  / (8π·σ·C_gau) →  A ∝ 1/σ
  Perfil Sech²:         A_crit = b  / (8π·C_sech)  →  A ∝ b   (lineal)

  Para un β_max dado, A se calcula automáticamente con A_for_beta().
  La energía escala como:  E [J] = 4π · M̃ · L [m] · 1.21×10⁴⁴ J/m

Variables de iteración sugeridas
---------------------------------
  Subir b (o bajar σ)  → burbuja más compacta, menos energía total
  Subir target_beta    → más curvatura (más efecto gravitacional)
  Cambiar perfil       → diferente distribución de ρ(r)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from static_bubbles.profiles import (
    profile_exponential, profile_gaussian, profile_sech_squared,
    profile_power_law,
)
from static_bubbles.units import (
    A_for_beta, beta_max as analytic_beta_max,
    M_exponential, M_gaussian, M_sech2,
    total_energy_J, format_energy, no_horizon_A,
)

OUT = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
#  ▶  EDITAR ESTA LISTA PARA AGREGAR CANDIDATOS
#
#  Campos requeridos:
#    name        : nombre descriptivo (aparece en la tabla y plots)
#    kind        : 'exponential' | 'gaussian' | 'sech2'
#    func        : función del perfil (de static_bubbles.profiles)
#    scale_key   : nombre del parámetro de escala ('b' o 'sigma')
#    scale_val   : valor del parámetro de escala
#    target_beta : β_max deseado (0 < valor < 1)
#    L_meter     : escala física — 1 código-unit de r = L_meter metros
#                  Ejemplos: 1.0 = 1 métro, 1e3 = 1 km, 1e-2 = 1 cm
#
#  El campo 'A' se calcula automáticamente de target_beta.
#  Si querés forzar un A específico en vez de usar target_beta, agrega:
#    fixed_A=<valor>   (y target_beta se ignora)
# ═══════════════════════════════════════════════════════════════════════════════

CANDIDATES = [

    # ── Exponencial: burbuja compacta (b alto) ────────────────────────────────
    dict(name="Exp b=1.0 β_max=0.3", kind='exponential',
         func=profile_exponential, scale_key='b', scale_val=1.0,
         target_beta=0.3, L_meter=1.0),

    dict(name="Exp b=1.5 β_max=0.5", kind='exponential',
         func=profile_exponential, scale_key='b', scale_val=1.5,
         target_beta=0.5, L_meter=1.0),

    dict(name="Exp b=2.0 β_max=0.8", kind='exponential',
         func=profile_exponential, scale_key='b', scale_val=2.0,
         target_beta=0.8, L_meter=1.0),

    dict(name="Exp b=3.0 β_max=0.5", kind='exponential',
         func=profile_exponential, scale_key='b', scale_val=3.0,
         target_beta=0.5, L_meter=1.0),

    # ── Gaussiano: caída más rápida, menor energía total ──────────────────────
    dict(name="Gau σ=1.0 β_max=0.5", kind='gaussian',
         func=profile_gaussian, scale_key='sigma', scale_val=1.0,
         target_beta=0.5, L_meter=1.0),

    dict(name="Gau σ=1.5 β_max=0.5", kind='gaussian',
         func=profile_gaussian, scale_key='sigma', scale_val=1.5,
         target_beta=0.5, L_meter=1.0),

    dict(name="Gau σ=0.5 β_max=0.8", kind='gaussian',
         func=profile_gaussian, scale_key='sigma', scale_val=0.5,
         target_beta=0.8, L_meter=1.0),

    # ── Sech²: comportamiento solitónico ──────────────────────────────────────
    dict(name="Sech² b=1.0 β_max=0.5", kind='sech2',
         func=profile_sech_squared, scale_key='b', scale_val=1.0,
         target_beta=0.5, L_meter=1.0),

    dict(name="Sech² b=2.0 β_max=0.5", kind='sech2',
         func=profile_sech_squared, scale_key='b', scale_val=2.0,
         target_beta=0.5, L_meter=1.0),

    # ── Agregar tus propios candidatos acá ───────────────────────────────────
    # Opción A: usar un perfil predefinido con parámetros propios
    # dict(name="Mi exp", kind='exponential',
    #      func=profile_exponential, scale_key='b', scale_val=4.0,
    #      target_beta=0.6, L_meter=10.0),
    #
    # Opción B: definir tu propia ρ(r) directamente como función Python.
    # Reglas:
    #   - r es un numpy array
    #   - ρ(r) debe ser ≥ 0 y no-creciente para satisfacer WEC + NEC
    #   - usar kind='custom' y fixed_A=1.0 (A ya está absorbida en tu función)
    #
    # Ejemplos de formas funcionales arbitrarias:
    #
    # Combinación exponencial + gaussiana:
    # dict(name="Exp+Gau custom", kind='custom', L_meter=1.0,
    #      func=lambda r, **k: 0.05*np.exp(-1.5*r) + 0.02*np.exp(-(r/2.0)**2),
    #      scale_key='_', scale_val=1.0, fixed_A=1.0),
    #
    # Power-law con cutoff exponencial:
    # dict(name="Power×Exp", kind='custom', L_meter=1.0,
    #      func=lambda r, **k: 0.1 * np.exp(-0.5*r) / (1 + r**2),
    #      scale_key='_', scale_val=1.0, fixed_A=1.0),
    #
    # Cualquier función decreciente que se te ocurra:
    # dict(name="Custom libre", kind='custom', L_meter=1.0,
    #      func=lambda r, **k: 0.08 / (1 + r**4),
    #      scale_key='_', scale_val=1.0, fixed_A=1.0),

]  # ← fin de CANDIDATES


# ═══════════════════════════════════════════════════════════════════════════════
#  Motor de análisis — no necesitás editar esto
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_beta_arr(r_arr, rho_arr):
    """β(r) via integración numérica acumulada."""
    dr = r_arr[1] - r_arr[0]
    ig = rho_arr * r_arr**2
    integral = np.zeros_like(r_arr)
    integral[1:] = np.cumsum(0.5*(ig[:-1]+ig[1:])*dr)
    with np.errstate(divide='ignore', invalid='ignore'):
        bs = np.where(r_arr > 0, 8*np.pi*integral/r_arr**2, 0.0)
    return np.sqrt(np.maximum(bs, 0.0))


def _mass_tilde_numerical(rho_arr, r_arr):
    """M̃ = ∫ρ r² dr por cuadratura."""
    return float(np.trapz(rho_arr * r_arr**2, r_arr))


def _mass_tilde(kind, A, scale, rho_arr=None, r_arr=None):
    if kind == 'exponential':
        return M_exponential(A, scale)
    elif kind == 'gaussian':
        return M_gaussian(A, scale)
    elif kind == 'sech2':
        return M_sech2(A, scale)
    elif kind == 'custom':
        return _mass_tilde_numerical(rho_arr, r_arr)
    else:
        raise ValueError(kind)


def analyse_candidate(cand, r_arr):
    """Analiza un candidato y devuelve un dict con todos los resultados."""
    kind      = cand['kind']
    scale_key = cand['scale_key']
    scale_val = cand['scale_val']
    L_meter   = cand['L_meter']

    is_custom = (kind == 'custom')

    if is_custom:
        # ρ(r) completamente definida por el usuario; A ya está dentro
        A           = cand.get('fixed_A', 1.0)
        target_beta = None
        rho         = cand['func'](r_arr)
        beta_max_ana= None
        A_crit      = None
    else:
        if 'fixed_A' in cand:
            A           = cand['fixed_A']
            target_beta = None
        else:
            target_beta = cand['target_beta']
            A           = A_for_beta(kind, target_beta, scale_val)
        kwargs = {'A': A, scale_key: scale_val}
        rho    = cand['func'](r_arr, **kwargs)
        beta_max_ana = analytic_beta_max(kind, A, scale_val)
        A_crit       = no_horizon_A(kind, scale_val)

    beta = _compute_beta_arr(r_arr, rho)
    beta_max_num = float(np.max(beta))

    drho  = np.gradient(rho, r_arr)
    nec   = -0.5 * r_arr * drho
    wec   = rho

    nec_ok = bool(np.all(nec[1:] >= -1e-10))
    wec_ok = bool(np.all(wec    >= -1e-10))
    hor_ok = beta_max_num < 1.0

    M_tild = _mass_tilde(kind, A, scale_val, rho_arr=rho, r_arr=r_arr)
    E_J    = total_energy_J(M_tild, L_meter)

    return dict(
        name        = cand['name'],
        kind        = kind,
        A           = A,
        A_crit      = A_crit,
        scale_key   = scale_key,
        scale_val   = scale_val,
        target_beta = target_beta,
        beta_max    = beta_max_num,
        beta_max_ana= beta_max_ana,
        nec_ok      = nec_ok,
        wec_ok      = wec_ok,
        hor_ok      = hor_ok,
        valid       = nec_ok and wec_ok and hor_ok,
        M_tilde     = M_tild,
        E_J         = E_J,
        L_meter     = L_meter,
        rho         = rho,
        beta        = beta,
        nec         = nec,
        r_arr       = r_arr,
    )


# ── Tabla de resultados ───────────────────────────────────────────────────────

def print_table(results):
    print("\n" + "═"*100)
    print("  RESULTADOS DE BÚSQUEDA — StaticWarpBubbles")
    print("═"*100)
    hdr = (f"{'#':>2}  {'Nombre':<28} {'A':>8} {'A_crit':>8} "
           f"{'β_max':>6}  {'NEC':>4} {'WEC':>4} {'Hor':>4}  "
           f"{'M̃':>8}  {'Energía (L=1 m)'}")
    print(hdr)
    print("─"*100)

    for i, r in enumerate(results):
        ok_str   = lambda b: "✓" if b else "✗"
        valid_mk = "  ← VÁLIDO ✓✓✓" if r['valid'] else ""
        print(
            f"{i+1:>2}  {r['name']:<28} {r['A']:>8.4f} {r['A_crit']:>8.4f} "
            f"{r['beta_max']:>6.3f}  {ok_str(r['nec_ok']):>4} "
            f"{ok_str(r['wec_ok']):>4} {ok_str(r['hor_ok']):>4}  "
            f"{r['M_tilde']:>8.4f}  {format_energy(r['E_J'])}{valid_mk}"
        )

    print("─"*100)
    n_valid = sum(1 for r in results if r['valid'])
    print(f"\n  Total candidatos: {len(results)}   Válidos: {n_valid}\n")
    print("  A < A_crit  →  β_max < 1  (sin horizonte)")
    print("  Energía calculada para L=L_meter (ver cada candidato para L)")
    print("═"*100 + "\n")


# ── Plot de comparación ───────────────────────────────────────────────────────

BG, PANEL, BORDER, FG = '#0d1117', '#161b22', '#30363d', '#e6edf3'
COLORS = ['#58a6ff','#3fb950','#f0883e','#d2a8ff','#f85149',
          '#79c0ff','#56d364','#ffa657','#e3b341','#ff7b72']

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': PANEL, 'axes.edgecolor': BORDER,
    'axes.labelcolor': FG, 'xtick.color': FG, 'ytick.color': FG,
    'text.color': FG, 'grid.color': BORDER, 'grid.alpha': 0.35, 'axes.grid': True,
    'legend.facecolor': '#21262d', 'legend.edgecolor': BORDER,
    'legend.labelcolor': FG, 'font.size': 9, 'lines.linewidth': 1.8,
})


def plot_results(results):
    fig, (ax_rho, ax_beta, ax_nec) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("StaticWarpBubbles — Resultados de búsqueda", fontsize=13)

    for i, r in enumerate(results):
        col  = COLORS[i % len(COLORS)]
        lbl  = r['name'] + (' ✓' if r['valid'] else ' ✗')
        ls   = '-' if r['valid'] else '--'
        alpha = 1.0 if r['valid'] else 0.45

        ax_rho.plot(r['r_arr'], r['rho'],  color=col, ls=ls, alpha=alpha, label=lbl)
        ax_beta.plot(r['r_arr'], r['beta'], color=col, ls=ls, alpha=alpha, label=lbl)
        ax_nec.plot(r['r_arr'], r['nec'],  color=col, ls=ls, alpha=alpha, label=lbl)

    ax_beta.axhline(1.0, color='#f85149', lw=1.5, ls=':', label='β=1 (horizonte)')
    ax_nec.axhline(0.0,  color='white',   lw=0.8, ls='--')

    for ax, ylabel, title in [
        (ax_rho, 'ρ(r)',                    'Perfil de densidad'),
        (ax_beta, 'β(r)',                   'Vector shift  (β < 1 = sin horizonte)'),
        (ax_nec,  'ρ + p⊥ = −r/2 · ρ\'',  'NEC⊥  (≥ 0 = satisfecha)'),
    ]:
        ax.set_xlabel('r  [unidades código]')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=1 if len(results) <= 6 else 2)

    plt.tight_layout()
    out = os.path.join(OUT, 'search_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓  search_results.png")


def plot_energy_comparison(results):
    """Barras horizontales de energía para L=1 m."""
    valid   = [r for r in results if r['valid']]
    invalid = [r for r in results if not r['valid']]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(results) + 2)))
    fig.suptitle("Energía requerida (L=1 m) por candidato válido", fontsize=12)

    labels, energies, colors = [], [], []
    for r in (valid + invalid):
        labels.append(r['name'])
        energies.append(r['E_J'])
        colors.append('#3fb950' if r['valid'] else '#f85149')

    bars = ax.barh(labels, energies, color=colors, alpha=0.75, edgecolor=BORDER)
    ax.set_xscale('log')
    ax.set_xlabel('Energía total E [J]  (L = 1 m)')

    # Referencias
    from static_bubbles.units import J_earth, J_sun, J_moon
    for val, lbl in [(J_moon,'M☽c²'),(J_earth,'M⊕c²'),(J_sun,'M☉c²')]:
        ax.axvline(val, color='#e3b341', lw=0.9, ls='--', alpha=0.7)
        ax.text(val*1.05, -0.5, lbl, color='#e3b341', fontsize=7, va='top', rotation=90)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#3fb950', label='Válido (NEC✓WEC✓β<1)'),
                        Patch(color='#f85149', label='Inválido (β≥1)')], fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT, 'search_energy.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓  search_energy.png")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    r_arr = np.linspace(0.0, 10.0, 4000)

    print(f"\nAnalizando {len(CANDIDATES)} candidatos …\n")
    results = [analyse_candidate(c, r_arr) for c in CANDIDATES]

    print_table(results)
    plot_results(results)
    plot_energy_comparison(results)

    # Resumen de los válidos
    valid = [r for r in results if r['valid']]
    if valid:
        print("  Candidatos válidos detallados:")
        for r in valid:
            Acrit = r['A_crit']
            margin = (Acrit - r['A']) / Acrit * 100
            print(f"    • {r['name']}")
            print(f"        A={r['A']:.5f}  A_crit={Acrit:.5f}  margen={margin:.1f}%")
            print(f"        β_max(num)={r['beta_max']:.4f}  β_max(analítico)={r['beta_max_ana']:.4f}")
            print(f"        M̃={r['M_tilde']:.5f}  E(L={r['L_meter']}m) = {format_energy(r['E_J'])}")
            print()
    else:
        print("  ⚠  Ningún candidato válido. Reducí A o target_beta.")

    print("✅  Listo. Revisá search_results.png y search_energy.png en examples/\n")
