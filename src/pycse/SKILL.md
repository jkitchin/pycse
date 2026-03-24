---
name: pycse
description: "Python computations in science and engineering (pycse) - helps with scientific computing tasks including nonlinear regression, uncertainty quantification, design of experiments (DOE), Latin hypercube sampling, surface response modeling, and neural network-based UQ with DPOSE. Use when working with numerical optimization, data fitting, experimental design, or uncertainty analysis."
---

# pycse - Python Computations in Science and Engineering

## Core Capabilities

### 1. Nonlinear Regression and Curve Fitting
- `nlinfit`: Nonlinear least squares fitting with uncertainty quantification
- `regress`: Linear regression with statistics
- Parameter uncertainty estimation and confidence intervals

### 2. Design of Experiments (DOE)
- **Latin Hypercube Sampling (LHC)**: Space-filling designs for efficient parameter exploration
- **Surface Response Modeling**: Polynomial response surfaces for experimental data

### 3. Uncertainty Quantification with DPOSE
- **DPOSE** (Direct Propagation of Shallow Ensembles): Neural network ensemble for UQ
- Per-sample heteroscedastic uncertainty estimates
- Trained using CRPS (recommended) or NLL loss

### 4. Numerical Methods
- Root finding, optimization, integration, differentiation
- ODE solvers with event detection (`odelay`)

## Quick Start Examples

### Nonlinear Regression with Uncertainty

```python
import numpy as np
from pycse import nlinfit

def model(x, a, b): return a * np.exp(b * x)

x = np.array([0, 1, 2, 3, 4])
y = np.array([1.0, 2.7, 7.4, 20.1, 54.6])

pars, pint, se = nlinfit(model, [1, 1], x, y)
# pars: fitted parameters, pint: confidence intervals, se: standard errors
# ✓ Check: se values should be small relative to pars
# ✓ Check: pint should not span zero for significant parameters
```

### Latin Hypercube Design

```python
from pycse.sklearn.lhc import LatinSquare

factors = {'Temperature': [20, 40, 60], 'Pressure': [1, 2, 3]}
ls = LatinSquare(factors)
design = ls.design()
# ✓ Check: len(design) == max(len(v) for v in factors.values())
```

### Surface Response

```python
from pycse.sklearn.surface_response import SurfaceResponse

sr = SurfaceResponse(
    inputs=['red', 'green', 'blue'],
    outputs=['intensity'],
    bounds=[[0, 1], [0, 1], [0, 1]]
)
design = sr.design()
# ... run experiments ...
sr.set_output(results)
sr.fit()
# ✓ Check: sr.score() for R² — should be > 0.8 for a useful model
# ✓ Check: sr.anova() for statistical significance of terms
```

### DPOSE (Neural Network UQ)

```python
from pycse.sklearn.dpose import DPOSE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ('scaler', StandardScaler()),  # Always scale inputs for DPOSE
    ('dpose', DPOSE(
        layers=(n_features, 50, 32),
        loss_type='crps',
        activation='tanh',
        maxiter=500
    ))
])

model.fit(X_train, y_train)
# ✓ Check: model.score(X_test, y_test) for convergence

y_pred, y_std = model.named_steps['dpose'].predict(
    model.named_steps['scaler'].transform(X_test),
    return_std=True
)
# ✓ Check: y_std values — large std indicates extrapolation or data gaps
```

## Workflows

### 1. Fitting Data with Uncertainty

1. Define model function with signature `f(x, *params)`
2. Call `nlinfit(model, x0, x, y)` with initial parameter guesses
3. **Validate**: Check standard errors are small relative to parameter values
4. **Validate**: Confidence intervals should not span zero for significant parameters
5. If non-convergence: try different initial guesses or rescale data

### 2. Designing Experiments

1. Define factors and their levels as a dictionary
2. Create `LatinSquare` or `SurfaceResponse` design
3. **Validate**: Verify design matrix has correct dimensions
4. Run experiments according to the design
5. Fit response surface and check R² and ANOVA significance

### 3. Uncertainty Quantification with DPOSE

1. Scale training data with `StandardScaler` (required for convergence)
2. Create DPOSE with `loss_type='crps'` (more robust than NLL)
3. Train and **validate**: check `model.score()` on held-out data
4. Get predictions with `return_std=True`
5. **Validate**: Large `y_std` indicates extrapolation — flag to user
6. For full ensemble output, use `predict_ensemble()` to get all members

## Common Pitfalls

- **DPOSE without scaling**: Always use `StandardScaler` — DPOSE will not converge on unscaled data
- **NLL loss instability**: Use `loss_type='crps'` unless NLL is specifically needed
- **nlinfit non-convergence**: Try log-transforming data or providing better initial guesses
- **Singular design matrix**: Ensure factors have sufficient distinct levels

## MCP Server

Install for Claude Desktop integration:
```bash
pycse mcp install
```

Provides tools for: DOE design, function docs, DPOSE examples, Python doc search.

## CLI

```bash
pycse launch   # Launch Jupyter Lab in Docker
pycse pull     # Update Docker image
pycse rm       # Remove stuck container
```
