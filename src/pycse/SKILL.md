---
name: pycse
description: Python computations in science and engineering (pycse) - helps with scientific computing tasks including nonlinear regression, uncertainty quantification, design of experiments (DOE), Latin hypercube sampling, surface response modeling, and neural network-based UQ with DPOSE. Use when working with numerical optimization, data fitting, experimental design, or uncertainty analysis.
---

# pycse - Python Computations in Science and Engineering

pycse is a comprehensive library for scientific computing, data analysis, and uncertainty quantification in Python.

## Core Capabilities

### 1. Nonlinear Regression and Curve Fitting
- `nlinfit`: Nonlinear least squares fitting with uncertainty quantification
- `regress`: Linear regression with statistics
- Supports parameter uncertainty estimation and confidence intervals

### 2. Design of Experiments (DOE)
- **Latin Hypercube Sampling (LHC)**: Space-filling designs for efficient parameter exploration
- **Surface Response Modeling**: Fit polynomial response surfaces to experimental data
- Useful for optimizing experimental conditions with minimal trials

### 3. Uncertainty Quantification with DPOSE
- **DPOSE** (Direct Propagation of Shallow Ensembles): Neural network ensemble for UQ
- Provides per-sample uncertainty estimates (heteroscedastic)
- Handles gaps, extrapolation, and nonlinear relationships
- Trained using CRPS or NLL loss for calibrated uncertainties

### 4. Numerical Methods
- Root finding and optimization
- Integration and differentiation
- ODE solvers
- Statistical analysis tools

## When to Use pycse

Use this skill when the user asks about:
- Fitting experimental data to nonlinear models
- Estimating parameter uncertainties
- Designing experiments or sampling parameter spaces
- Latin squares or Latin hypercube designs
- Surface response methodology
- Uncertainty quantification in predictions
- Neural network-based surrogate models with uncertainty
- Scientific data analysis in Python

## Key Functions

### nlinfit
```python
from pycse import nlinfit

# Fit data to a model with uncertainty quantification
pars, pint, se = nlinfit(model_func, x0, x, y)
```

### Latin Hypercube Design
```python
from pycse.sklearn.lhc import LatinSquare

# Create a Latin hypercube design
factors = {'Temperature': [20, 40, 60], 'Pressure': [1, 2, 3]}
ls = LatinSquare(factors)
design = ls.design()
```

### Surface Response
```python
from pycse.sklearn.surface_response import SurfaceResponse

# Design and fit a surface response model
sr = SurfaceResponse(
    inputs=['red', 'green', 'blue'],
    outputs=['intensity'],
    bounds=[[0, 1], [0, 1], [0, 1]]
)
design = sr.design()
# ... run experiments ...
sr.set_output(results)
sr.fit()
```

### DPOSE (Uncertainty Quantification)
```python
from pycse.sklearn.dpose import DPOSE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create DPOSE model with uncertainty estimates
model = Pipeline([
    ('scaler', StandardScaler()),
    ('dpose', DPOSE(
        layers=(n_features, 50, 32),  # (input, hidden, ensemble)
        loss_type='crps',              # CRPS loss (recommended)
        activation='tanh',             # Smooth activation
        maxiter=500
    ))
])

model.fit(X_train, y_train)

# Get predictions with uncertainty
y_pred, y_std = model.named_steps['dpose'].predict(
    X_test_scaled,
    return_std=True
)
```

## MCP Server

pycse provides an MCP server for Claude Desktop with tools for:
- Design of experiments (Latin squares, surface response)
- Function documentation lookup
- DPOSE model information and examples
- Python documentation search

To install the MCP server:
```bash
pycse mcp install
```

## Docker-based Jupyter Lab

pycse includes a CLI for launching Jupyter Lab in a Docker container:
```bash
pycse launch              # Launch Jupyter Lab
pycse pull                # Update Docker image
pycse rm                  # Remove stuck container
```

## Documentation

For detailed documentation, see the pycse repository at: https://github.com/jkitchin/pycse

## Tips for Using pycse

1. **Always use StandardScaler** with DPOSE for better convergence
2. **CRPS loss** is recommended for DPOSE - more robust than NLL
3. **Latin hypercube designs** are more efficient than grid searches
4. **Surface response models** are useful when experiments are expensive
5. For uncertainty propagation, use `predict_ensemble()` to get all ensemble members

## Common Workflows

### Fitting Data with Uncertainty
1. Define your model function
2. Use `nlinfit` with initial parameter guesses
3. Examine parameter confidence intervals and standard errors

### Designing Experiments
1. Define factors and their levels
2. Create a LatinSquare or SurfaceResponse design
3. Run experiments according to the design
4. Analyze results with ANOVA or response surface fitting

### Uncertainty Quantification
1. Prepare and scale your training data
2. Create a DPOSE model with appropriate architecture
3. Train the model
4. Get predictions with uncertainty estimates
5. Visualize uncertainty intervals
