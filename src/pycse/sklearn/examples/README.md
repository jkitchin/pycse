# DPOSE Examples

This directory contains example scripts demonstrating DPOSE functionality.

## Quick Examples

### 1. Optimizer Comparison (`optimizer_examples.py`)

Comprehensive comparison of all available optimizers:
- BFGS (default)
- L-BFGS
- Adam
- SGD
- Muon (state-of-the-art)
- Gradient Descent

**Run:**
```bash
python optimizer_examples.py
```

**Output:** Performance table comparing MAE, uncertainty, and calibration across optimizers.

### 2. Muon Optimizer Tests (`test_muon.py`)

Detailed benchmarks for the Muon optimizer:
- Default parameters
- Custom parameters
- Performance comparison

**Run:**
```bash
python test_muon.py
```

### 3. Quick Optimizer Test (`test_optimizers.py`)

Fast smoke test of all optimizers (100 iterations each):

**Run:**
```bash
python test_optimizers.py
```

## Diagnostic Examples

### 4. DPOSE Dataset Demonstration (`demo_dpose_datasets.py`)

Complete demonstration with:
- Multiple datasets
- CRPS loss
- Visualization
- Metrics

**Run:**
```bash
python demo_dpose_datasets.py
```

### 5. Ensemble Collapse Diagnosis (`diagnose_collapse.py`)

Diagnostic tool for investigating ensemble collapse issues.

### 6. Initialization Tests (`test_init.py`)

Tests different initialization strategies.

### 7. Parameter Tests (`test_params.py`)

Tests various hyperparameter configurations.

### 8. Fixed DPOSE Tests (`test_dpose_fixed.py`)

Validation tests for DPOSE fixes.

## Usage Patterns

### Basic Training

```python
from pycse.sklearn.dpose import DPOSE

# Create model
model = DPOSE(layers=(1, 20, 32), loss_type='crps')

# Train
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

# Predict
y_pred, y_std = model.predict(X_test, return_std=True)
```

### Using Different Optimizers

```python
# BFGS (default)
model = DPOSE(layers=(1, 20, 32))
model.fit(X, y)

# Muon (best performance)
model = DPOSE(layers=(1, 20, 32), optimizer='muon')
model.fit(X, y, learning_rate=0.02)

# Adam (deep networks)
model = DPOSE(layers=(1, 20, 32), optimizer='adam')
model.fit(X, y, learning_rate=1e-3)
```

### Two-Stage NLL Training

```python
model = DPOSE(layers=(1, 20, 32), loss_type='nll', optimizer='muon')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val,
          pretrain_maxiter=500,    # MSE pre-training
          maxiter=1000,            # NLL fine-tuning
          learning_rate=0.02)
```

## Expected Output

All examples produce:
1. **Training progress** - Convergence information
2. **Performance metrics** - MAE, RMSE, calibration α
3. **Comparison tables** - Side-by-side optimizer/parameter comparisons
4. **Diagnostic info** - Warnings about collapse, calibration issues

## Tips

- Start with `optimizer_examples.py` to see all optimizers in action
- Use `test_muon.py` to see the state-of-the-art optimizer
- Check `demo_dpose_datasets.py` for complete workflow
- Run diagnostic scripts if encountering issues

## Dependencies

All examples require:
- `jax`
- `jaxopt`
- `optax`
- `flax`
- `scikit-learn`
- `matplotlib`
- `numpy`

## Running All Examples

```bash
# Quick tests (fast)
python test_optimizers.py

# Comprehensive (slower but informative)
python optimizer_examples.py
python test_muon.py
python demo_dpose_datasets.py
```

## Troubleshooting

If examples fail:
1. Check dependencies are installed
2. Verify JAX/Optax versions (Optax >= 0.2.6 for Muon)
3. See parent directory docs for detailed guides
