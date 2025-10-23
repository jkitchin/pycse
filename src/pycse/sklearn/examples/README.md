# Uncertainty Quantification Examples

This directory contains Jupyter notebooks demonstrating uncertainty quantification methods for regression:
- **NNBR** (Neural Network + Bayesian Ridge)
- **NNGMM** (Neural Network + Gaussian Mixture Model)
- **DPOSE** (Direct Propagation of Shallow Ensembles)
- **MAPIE** (Model Agnostic Prediction Interval Estimator)

## Quick Start

### 1. UQ_Comparison_Demo.ipynb ⭐ **START HERE**

**Comprehensive comparison of all four UQ methods:**
- Side-by-side comparison of MAPIE, NNBR, NNGMM, and DPOSE
- Heteroscedastic noise demonstration
- Coverage analysis
- Multiple datasets comparison
- Best practices and recommendations

**When to use each method:**
- **MAPIE**: Guaranteed coverage (safety-critical applications)
- **NNBR**: Fast, calibrated uncertainties (general purpose)
- **NNGMM**: Multimodal uncertainty (complex patterns)
- **DPOSE**: State-of-the-art ensemble UQ (research/production)

---

## Individual Method Demos

### 2. NNBR_Demo.ipynb

**Neural Network + Bayesian Ridge Regression**

Demonstrates:
- Fast sklearn-based uncertainty quantification
- Post-hoc calibration
- Heteroscedastic noise handling
- Adaptive uncertainty intervals

**Key Features:**
- Pure sklearn workflow
- Low computational cost
- Gaussian uncertainty assumption
- Excellent for most regression problems

### 3. NNGMM_Demo.ipynb

**Neural Network + Gaussian Mixture Model**

Demonstrates:
- Multimodal uncertainty estimation
- GMM-based sampling for flexible UQ
- Complex noise patterns
- Post-hoc calibration

**Key Features:**
- Captures non-Gaussian uncertainty
- Flexible uncertainty representation
- sklearn + gmr libraries
- Good for complex distributions

### 4. DPOSE_Demo.ipynb

**Direct Propagation of Shallow Ensembles**

Demonstrates:
- State-of-the-art ensemble UQ (Kellner & Ceriotti 2024)
- CRPS loss for robust training
- Multiple optimizers (BFGS, Adam, Muon)
- Uncertainty propagation

**Key Features:**
- JAX/Flax framework
- Shallow ensemble (efficient)
- Automatic differentiation
- Best accuracy + calibration

---

## Method Comparison

| Method | Type | Coverage | Heteroscedastic | Framework | Speed |
|--------|------|----------|-----------------|-----------|-------|
| **MAPIE** | Conformal | ✅ Guaranteed | ❌ Uniform | sklearn | 🟢 Fast |
| **NNBR** | Bayesian | ⚠️ Asymptotic | ✅ Adaptive | sklearn | 🟢 Fast |
| **NNGMM** | GMM-based | ⚠️ Asymptotic | ✅ Adaptive | sklearn+gmr | 🟡 Medium |
| **DPOSE** | Ensemble | ⚠️ Asymptotic | ✅ Adaptive | JAX/Flax | 🟡 Medium |

---

## Usage Patterns

### Basic Workflow (All Methods)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data (always include calibration set!)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.2)

# Always scale!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cal = scaler.transform(X_cal)
X_test = scaler.transform(X_test)
```

### MAPIE (Conformal Prediction)

```python
from mapie.regression import SplitConformalRegressor
from sklearn.neural_network import MLPRegressor

# Fit base model
base_model = MLPRegressor(hidden_layer_sizes=(50, 50))
base_model.fit(X_train, y_train)

# Calibrate
mapie = SplitConformalRegressor(estimator=base_model, confidence_level=0.95, prefit=True)
mapie.conformalize(X_cal, y_cal)

# Predict with intervals
y_pred, y_intervals = mapie.predict_interval(X_test)
```

### NNBR (Bayesian Ridge)

```python
from pycse.sklearn.nnbr import NeuralNetworkBLR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

nn = MLPRegressor(hidden_layer_sizes=(50, 50))
br = BayesianRidge()
nnbr = NeuralNetworkBLR(nn, br)
nnbr.fit(X_train, y_train, val_X=X_cal, val_y=y_cal)

# Predict with uncertainties
y_pred, y_std = nnbr.predict(X_test, return_std=True)
```

### NNGMM (Gaussian Mixture Model)

```python
from pycse.sklearn.nngmm import NeuralNetworkGMM
from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(hidden_layer_sizes=(50, 50))
nngmm = NeuralNetworkGMM(nn, n_components=1, n_samples=500)
nngmm.fit(X_train, y_train, val_X=X_cal, val_y=y_cal)

# Predict with uncertainties
y_pred, y_std = nngmm.predict(X_test, return_std=True)
```

### DPOSE (Shallow Ensemble)

```python
from pycse.sklearn.dpose import DPOSE

# CRPS loss (recommended)
dpose = DPOSE(
    layers=(n_features, 50, 32),  # (input, hidden, ensemble_size)
    loss_type='crps',
    optimizer='bfgs'
)
dpose.fit(X_train, y_train, val_X=X_cal, val_y=y_cal)

# Predict with uncertainties
y_pred, y_std = dpose.predict(X_test, return_std=True)

# Uncertainty propagation (unique to DPOSE)
ensemble_preds = dpose.predict_ensemble(X_test)  # (n_samples, n_ensemble)
z_ensemble = f(ensemble_preds)  # Apply any function
z_mean = z_ensemble.mean(axis=1)
z_std = z_ensemble.std(axis=1)
```

---

## Dependencies

### Core (All Methods)
```bash
pip install numpy scikit-learn matplotlib
```

### MAPIE
```bash
pip install mapie
```

### NNGMM
```bash
pip install gmr
```

### DPOSE
```bash
pip install jax jaxopt optax flax
```

---

## Practical Recommendations

1. **Start with UQ_Comparison_Demo.ipynb** to understand all methods
2. **Use MAPIE** for guaranteed coverage (safety-critical apps)
3. **Use NNBR** for most regression tasks (fast + adaptive sklearn)
4. **Use NNGMM** for multimodal uncertainty or complex patterns
5. **Use DPOSE** for state-of-the-art ensemble UQ with JAX
6. **Always use calibration data** for NNBR, NNGMM, and DPOSE
7. **Check coverage empirically** on test data

---

## Key Findings

From the comparison notebook:

- **Best Coverage**: MAPIE (guaranteed by construction)
- **Best Heteroscedasticity Adaptation**: DPOSE, NNBR (tie)
- **Best for Multimodal UQ**: NNGMM
- **Fastest Training**: MAPIE, NNBR (sklearn)
- **Most Flexible**: DPOSE (JAX autodiff + ensemble propagation)
- **Easiest to Use**: MAPIE (one-line sklearn)

---

## Troubleshooting

### Common Issues

1. **Uncertainties too small/large**: Check calibration data is representative
2. **Poor coverage**: Use more calibration samples or adjust confidence level
3. **NNGMM collapse**: Increase `n_components` or reduce neural network training
4. **DPOSE errors**: Ensure data is scaled and activation is specified

### Getting Help

- Check the individual notebooks for detailed examples
- See parent directory for API documentation
- Review the comparison notebook for best practices

---

## References

- **DPOSE**: Kellner, M., & Ceriotti, M. (2024). Uncertainty quantification by direct propagation of shallow ensembles. *Machine Learning: Science and Technology*, 5(3), 035006.
- **MAPIE**: Taquet, V., et al. (2022). MAPIE: Model-Agnostic Prediction Interval Estimator.
- **Bayesian Ridge**: Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. *Journal of Machine Learning Research*, 1, 211-244.
- **GMM**: Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
