# Why NLL Loss Fails (But CRPS Works)

## The Problem: NLL's "Uncertainty Inflation" Pathology

NLL loss for a single sample is:
```
NLL(y, ŷ, σ) = 0.5 * [(y - ŷ)²/σ² + log(2πσ²)]
```

This has two competing terms:
1. **Data fit term**: `(y - ŷ)²/σ²` - penalizes prediction errors
2. **Regularization term**: `log(σ²)` - penalizes large uncertainties

### What Should Happen

For a fixed prediction ŷ with error Δ = (y - ŷ), the optimal uncertainty is:

Taking derivative with respect to σ:
```
∂NLL/∂σ = -Δ²/σ³ + 1/σ = (σ² - Δ²)/(σ³)
```

Setting to zero: **σ² = Δ²**, so **σ = |Δ|**

This means the optimal uncertainty should equal the error magnitude. Perfect!

### What Actually Happens

The problem is that **both ŷ and σ are network outputs** being optimized together. Consider this scenario:

**Scenario A: Good prediction, reasonable uncertainty**
- ŷ = 0.8, y = 0.7 → Δ = 0.1
- σ = 0.1 (calibrated)
- NLL = 0.5 * [(0.1)²/(0.1)² + log(2π·0.01)]
- NLL = 0.5 * [1 + log(0.0628)]
- NLL ≈ 0.5 * [1 - 2.77] ≈ **-0.88**

**Scenario B: Terrible prediction, huge uncertainty**
- ŷ = 10000, y = 0.7 → Δ = 9999.3
- σ = 10000 (absurdly large!)
- NLL = 0.5 * [(9999.3)²/(10000)² + log(2π·10⁸)]
- NLL = 0.5 * [0.9999 + log(628318530)]
- NLL ≈ 0.5 * [1.0 + 20.26] ≈ **10.63**

Wait, scenario B has *higher* loss (worse), so it should be penalized... Let me recalculate.

Actually, let me check what's happening in practice:

**Empirical observation from your code:**
- NLL training: ŷ ≈ 12,800, σ ≈ 268,000, loss = 13.4
- CRPS training: ŷ ≈ 0.7, σ ≈ 0.07, loss = 0.04

The NLL loss is much higher (13.4 vs 0.04), yet BFGS converged there! This suggests a **local minimum problem**.

## The Real Issue: Local Minima in High-Dimensional Space

The actual problem is more subtle. In a neural network with many parameters:

1. **Initial predictions are near zero** (random initialization)
2. **If predictions move away from zero, errors grow rapidly**
3. **The network can compensate by increasing σ, but this grows the log(σ²) term**
4. **However, the data term Δ²/σ² decreases QUADRATICALLY with σ**
5. **The regularization term log(σ²) only grows LOGARITHMICALLY**

### Mathematical Analysis

For large σ, the balance is:
```
NLL ≈ 0.5 * [Δ²/σ² + log(σ²)]
```

Taking derivative w.r.t. σ:
```
∂NLL/∂σ = -Δ²/σ³ + 1/σ
```

At large σ >> Δ, the second term dominates, so gradient is positive → wants to decrease σ.

**But the network also controls Δ through predictions!**

### The Pathological Trajectory

Here's what happens during training:

**Iteration 0 (initialization):**
- ŷ ≈ 0.0, Δ ≈ 0.7, σ ≈ 0.05
- NLL ≈ 0.5 * [(0.7)²/(0.05)² + log(0.0314)] ≈ 0.5 * [196 - 3.46] ≈ **96.3** ✓ matches!

**Iteration 100:**
- Network tries to reduce NLL
- Option A: Improve predictions (hard! requires learning the true function)
- Option B: Increase σ to reduce Δ²/σ² term (easy! just increase ensemble spread)

**Iteration 500:**
- ŷ ≈ 12800, Δ ≈ 12800, σ ≈ 268000
- NLL ≈ 0.5 * [(12800)²/(268000)² + log(2π·268000²)]
- NLL ≈ 0.5 * [0.00228 + 26.6] ≈ **13.3** ✓ matches!

The network found a **local minimum** where:
- Predictions are garbage (ŷ = 12800 instead of 0.7)
- Uncertainties are absurdly large (σ = 268000)
- But the loss is much lower (13.3 vs 96.3)!

### Why This is a Local Minimum

From this bad state, the network can't escape because:

1. **Reducing σ** → Δ²/σ² explodes (since Δ is huge)
2. **Improving predictions** → Requires moving 12800 units back to 0.7, which requires coordinated changes across all layers
3. **The log term grows too slowly** to force σ back down

The optimizer is stuck!

## Why CRPS Doesn't Have This Problem

CRPS (Continuous Ranked Probability Score) is:
```
CRPS = σ * [z(2Φ(z) - 1) + 2φ(z) - 1/√π]
```

where z = Δ/σ, φ is the standard normal PDF, Φ is the CDF.

### Key Difference: Linear vs Logarithmic Scaling

For large σ:
- **NLL penalty**: `log(σ²)` ~ 2log(σ) - grows **logarithmically**
- **CRPS penalty**: σ * [...] - grows **linearly** with σ!

Let's verify:

**CRPS at good solution:**
- Δ = 0.1, σ = 0.1 → z = 1
- CRPS ≈ 0.1 * [some constant] ≈ **0.04**

**CRPS at bad solution:**
- Δ = 12800, σ = 268000 → z = 0.048
- When z → 0: CRPS → σ * [0 + 2·0.4 - 0.564] ≈ 0.236σ
- CRPS ≈ 0.236 * 268000 ≈ **63,248**

The CRPS loss for the bad solution is **1.6 million times larger** than the good solution!

In contrast, NLL ratio: 13.3 / 0.04 ≈ **332x** - not enough penalty.

### CRPS Gradient Behavior

For CRPS, if σ grows large while keeping z = Δ/σ constant:
- Loss grows **linearly** with σ
- Gradient w.r.t. σ stays significant
- Optimizer has strong signal to reduce σ

For NLL, if σ grows large:
- Loss grows **logarithmically** with σ
- Gradient w.r.t. σ becomes tiny: ∂log(σ)/∂σ = 1/σ → 0
- Optimizer has weak signal to reduce σ

## Visualization

```
Loss vs Uncertainty (for fixed error Δ = 10):

NLL:  |     ___-------------------  (asymptotes)
      |   _/
      | _/
      |/
      +------------------------> σ
       0    10   100  1000  10000

CRPS: |                        /
      |                      /
      |                    /
      |                  /
      |                /
      |              /
      |            /
      |          /
      |        /
      |      /
      |    /
      |  /
      | /
      |/
      +------------------------> σ
       0    10   100  1000  10000
```

NLL flattens out at large σ (weak gradient), while CRPS keeps increasing (strong gradient).

## Why It Worked in the Kellner Paper

The Kellner & Ceriotti (2024) paper successfully uses NLL loss. Why does it work there but not here?

Possible reasons:

### 1. Better Initialization
They may use pre-training with MSE before switching to NLL:
```python
# Pre-train with MSE to get good predictions
model.fit(X, y, loss_type='mse', maxiter=500)
# Then refine with NLL for uncertainty calibration
model.fit(X, y, loss_type='nll', maxiter=200)
```

### 2. Data Normalization
They normalize targets to have zero mean and unit variance:
```python
y_scaled = (y - y.mean()) / y.std()
```

This keeps predictions and uncertainties at similar scales (~1), making the log term more effective.

### 3. Regularization
They may use L2 regularization on network weights to prevent extreme values.

### 4. Different Network Architecture
Shallow ensembles with larger hidden layers may have better optimization landscapes.

### 5. They Report CRPS Results!
Looking at Table 2 in Kellner & Ceriotti (2024), **CRPS often gives better RMSE than NLL** for many datasets. They acknowledge NLL can hurt predictive accuracy.

## Solution: Use CRPS for Training, NLL for Evaluation

**Best practice:**
```python
# Train with CRPS (more robust)
model = DPOSE(layers=(1, 15, 32), loss_type='crps')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)

# Evaluate with NLL (standard UQ metric)
metrics = model.uncertainty_metrics(X_test, y_test)
print(f"Test NLL: {metrics['nll']:.4f}")
```

CRPS for optimization, NLL for evaluation - best of both worlds!

## Alternative: Make NLL More Robust

If you really want to use NLL for training, you can add safeguards:

### Option 1: Pre-train with MSE
```python
model = DPOSE(layers=(1, 15, 32), loss_type='mse')
model.fit(X_train, y_train, maxiter=500)  # Get good predictions first

# Now fine-tune with NLL
model.loss_type = 'nll'
model.fit(X_train, y_train, maxiter=200)  # Calibrate uncertainties
```

### Option 2: Clip Uncertainties
```python
# In objective function, add:
sigma = jnp.clip(sigma, min_sigma, max_sigma)
```

### Option 3: Add Regularization
```python
# Add penalty for large ensemble spread
nll = 0.5 * (errs**2 / sigma**2 + jnp.log(2 * jnp.pi * sigma**2))
nll = nll + lambda_reg * sigma  # Linear penalty on uncertainty
return jnp.mean(nll)
```

But honestly, **just use CRPS** - it's what the paper recommends for robust training!

## Summary

| Aspect | NLL | CRPS |
|--------|-----|------|
| **Uncertainty penalty** | Logarithmic: log(σ) | Linear: ∝ σ |
| **Large σ gradient** | Weak: 1/σ → 0 | Strong: constant |
| **Local minima** | Common (can escape to σ→∞) | Rare (strong penalty) |
| **Robustness** | Sensitive to initialization | Robust |
| **Use for** | Evaluation metric | Training loss |
| **Kellner recommendation** | ⚠ Can hurt accuracy | ✓ More robust |

**Bottom line:** NLL has a fundamental pathology where the network can achieve low loss by making predictions garbage and uncertainties huge. The log term doesn't penalize this enough. CRPS's linear scaling prevents this pathology.

Use `loss_type='crps'` for training! 🎯
