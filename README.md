# StaticWarpBubbles

Toolkit para analizar métricas de burbujas warp **estáticas y esféricamente simétricas** usando Python.

---

## La métrica

```
ds² = −dt² + (dr − β(r) dt)² + r² dΩ²
```

La función `β(r)` (vector shift) se calcula integrando la densidad de energía `ρ(r)`:

```
β²(r) = (8π / r²) ∫₀ʳ ρ(s) s² ds
```

**Tu trabajo como investigador**: proponer una `ρ(r)`, e integrar para ver qué métrica resulta.

---

## Criterios de solución válida

| Condición | Expresión | Significado |
|-----------|-----------|-------------|
| **WEC** | `ρ(r) ≥ 0` | Densidad de energía positiva |
| **NEC** | `ρ'(r) ≤ 0` | Densidad no-creciente |
| **Sin horizonte** | `β_max < 1` | No se forma superficie atrapante |

> Con los perfiles monotónicos de esta librería **WEC y NEC son automáticos**.  
> Solo necesitás controlar `β_max < 1`.

---

## Flujo de trabajo (sin el agente)

### Paso 1 — Editar `examples/search.py`

Abrí el archivo y modificá la lista `CANDIDATES`. Podés usar:

**Opción A — Perfil predefinido con tus parámetros:**
```python
dict(name="Mi exponencial",
     kind='exponential',
     func=profile_exponential,
     scale_key='b', scale_val=2.0,
     target_beta=0.6,      # β_max que querés
     L_meter=100.0),       # 1 código-unit = 100 metros
```

**Opción B — Tu propia `ρ(r)` arbitraria:**
```python
dict(name="Mi rho custom",
     kind='custom',
     func=lambda r: 0.05 * np.exp(-1.5*r) + 0.02 * np.exp(-(r/2.0)**2),
     scale_key='_', scale_val=1.0,
     fixed_A=1.0,          # A ya está absorbida en tu función
     L_meter=1.0),
```

> **Regla física**: para que WEC + NEC se satisfagan, `ρ(r)` debe ser:
> - `≥ 0` para todo `r`
> - Decreciente (o al menos no creciente) en `r`

### Paso 2 — Correr el script

```bash
cd C:\Users\Nelson\OneDrive\Documents\StaticWarpBubbles
python examples/search.py
```

### Paso 3 — Interpretar los resultados

La terminal muestra una tabla:

```
#  Nombre              A       A_crit   β_max   NEC  WEC  Hor    M̃     Energía
1  Mi perfil        0.1234   0.3511   0.500    ✓    ✓    ✓   0.052  147 M⊕c²  ← VÁLIDO ✓✓✓
```

- `A_crit`: máximo A permitido para no tener horizonte (superar = β_max ≥ 1)
- `margen`: qué tan lejos estás del límite (mayor = más seguro)
- Plots guardados en `examples/search_results.png` y `examples/search_energy.png`

### Paso 4 — Iterar

| Si querés… | Ajustá… |
|------------|---------|
| Menos energía | Aumentar `b` (ó bajar `sigma`) |
| Más curvatura (β_max mayor) | Aumentar `target_beta` |
| Burbuja más compacta | Aumentar `b` (escala de caída) |
| Cambiar forma del perfil | Cambiar `kind` o definir `ρ(r)` custom |
| Unidades físicas | Cambiar `L_meter` (tamaño físico de la burbuja) |

---

## Reglas de escala analíticas

Estas fórmulas te dicen exactamente qué A usar sin escanear toda la grilla:

| Perfil | β_max | A_crit (= β_max=1) |
|--------|-------|---------------------|
| Exponencial `A·e^{-br}` | `√(8π·A·C/b)` | `b / (8π·C_exp)` |
| Gaussiano `A·e^{-(r/σ)²}` | `√(8π·A·σ·C)` | `1 / (8π·σ·C_gau)` |
| Sech² `A/cosh²(br)` | `√(8π·A·C/b)` | `b / (8π·C_sech)` |

Donde `C_exp ≈ 0.170`, `C_gau ≈ 0.190`, `C_sech ≈ 0.200`.

Para obtener un `target_beta` específico: `A = target_beta² × A_crit`.

---

## Energía física

```
E [J] = 4π · M̃ · L [m] · 1.2135×10⁴⁴ J/m

donde:
  M̃     = ∫₀^∞ ρ(r) r² dr    (integral numérica, sin unidades)
  L [m] = escala física: 1 código-unit de r = L metros
```

La energía escala **linealmente** con el tamaño físico de la burbuja.

---

## Scripts disponibles

| Script | Descripción |
|--------|-------------|
| `examples/search.py` | **Punto de entrada principal** — tu ρ(r), tus parámetros |
| `examples/demo_symbolic.py` | Análisis simbólico (sympy) de los perfiles predefinidos |
| `examples/demo_scanner.py` | Mapa 2D completo del espacio de parámetros |
| `examples/demo_physical.py` | Conversión a unidades SI + gráficos de energía |
| `examples/demo.py` | Ejemplos del paper original (Single/Double Shell) |

---

## Estructura del paquete

```
static_bubbles/
├── profiles.py    # Perfiles predefinidos de ρ(r)
├── generator.py   # Integración de β(r), construcción de la métrica 3D
├── analyzer.py    # Verificación numérica de NEC/WEC/DEC/SEC
├── symbolic.py    # Derivación simbólica con sympy
├── scanner.py     # Scan 2D vectorizado del espacio (A, escala)
└── units.py       # Conversión a SI: energía, densidad, escalas físicas
```

---

## Instalación

```bash
# Requisitos
pip install numpy scipy matplotlib sympy

# Opcional: pyWarpFactory para métricas 3D completas
cd C:\Users\Nelson\OneDrive\Documents\pyWarpFactory
pip install -e .
```

---

## Referencia

Bolívar, Abellán, Vasilev — *"Static spherically-symmetric warp bubble metrics"* (2025)
