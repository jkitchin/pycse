# When Does NLL Work? Comprehensive Testing

## TL;DR: **NLL Almost Never Works in This Implementation**

After extensive testing, **NLL training fails in nearly ALL scenarios** without special interventions.

## Test Results

### ❌ Scenarios Where NLL FAILS

| Scenario | MAE (NLL) | MAE (CRPS) | Status |
|----------|-----------|------------|--------|
| Linear (y=2x) | 11,456.7 | 0.073 | ✗ FAIL |
| Quadratic (y=x²) | 754.7 | 0.037 | ✗ FAIL |
| Cube root (y=x^⅓) | 26.6 | 0.160 | ✗ FAIL |
| Constant (y=1) | 31,971.6 | 0.007 | ✗ FAIL |
| Large scale (y~1000) | 9.9×10¹⁴ | 1049.5 | ✗ FAIL |
| Small scale (y~0.001) | 0.00027 | - | ✗ FAIL |
| Normalized data | 788,834.2 | - | ✗ FAIL |
| Tiny network (2 hidden) | 0.303 | - | ✗ FAIL |
| 10 datapoints | 0.303 | - | ✗ FAIL |

### Different Seeds
**All failed:** Seeds 0, 1, 19, 42, 100

### Different Network Sizes
**All failed:** Hidden layers with 3, 5, 10, 20 neurons

### Different Ensemble Sizes
**All failed:** 4, 8, 16, 32 members

### Different Iteration Counts
**All failed:** 10, 20, 50, 100, 200, 1000 iterations

### ✓ Only Success: Artificially Large min_sigma

```python
model = DPOSE(layers=(1, 10, 16), loss_type='nll', min_sigma=0.5)  # ← Forces huge minimum uncertainty
```

**But this defeats the purpose!**
- min_sigma=0.5 is huge compared to data scale
- Uncertainties become constant (0.06 everywhere)
- Ensemble spread is overridden by minimum
- Not a real solution, just masking the problem

## Why Does NLL Always Fail?

### The Uncertainty Inflation Pathology

NLL loss has this form:
```
NLL = 0.5 * [Δ²/σ² + log(2πσ²)]
```

Where:
- Δ = prediction error
- σ = ensemble standard deviation

### The Problem

The network optimizes BOTH Δ and σ simultaneously. It discovers a bad local minimum:

**Bad Local Minimum:**
1. Make predictions far from targets (Δ = 10,000)
2. Make uncertainties even larger (σ = 250,000)
3. Then Δ²/σ² ≈ 0 (small!)
4. And log(σ²) grows slowly (only ~26 for σ=250,000)
5. **Total loss is "low" even though predictions are garbage**

### Why Can't the Optimizer Escape?

From this bad state:
- **Can't reduce σ**: Would make Δ²/σ² explode
- **Can't improve predictions**: Requires moving 10,000 units, needs coordinated changes across all layers
- **Gradients are tiny**: ∂log(σ)/∂σ = 1/σ → 0 as σ grows

The optimizer is **stuck in a terrible local minimum**.

### Why Doesn't CRPS Have This Problem?

CRPS has **linear** penalty on σ:
```
CRPS ≈ σ * [...]
```

For large σ:
- **NLL penalty**: log(σ) ~ 12 for σ=250,000 (tiny!)
- **CRPS penalty**: 0.24·σ ~ 60,000 for σ=250,000 (huge!)

CRPS penalty is **5,000× larger**, preventing the pathology.

## When DOES NLL Work?

### ✅ Option 1: Pre-training with MSE

```python
# Step 1: Get good predictions with MSE
model = DPOSE(layers=(1, 15, 32), loss_type='mse')
model.fit(X_train, y_train, maxiter=500)

# Step 2: Fine-tune uncertainties with NLL
model.loss_type = 'nll'
model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=200)
```

**Why this works:** MSE ensures predictions are reasonable, then NLL only adjusts uncertainties.

**Result:** MAE=0.059 ✓

### ✅ Option 2: Just Use CRPS (Recommended!)

```python
model = DPOSE(layers=(1, 15, 32), loss_type='crps')
model.fit(X_train, y_train, val_X=X_val, val_y=y_val)
```

**Why this works:** CRPS's linear penalty prevents uncertainty inflation.

**Result:** MAE=0.061 ✓

## Why Did Kellner & Ceriotti Use NLL Successfully?

Good question! The Kellner & Ceriotti (2024) paper uses NLL. Possible reasons:

### 1. They May Use Different Initialization
More sophisticated initialization strategies that avoid the bad local minimum.

### 2. Data Normalization
They likely normalize targets to zero mean, unit variance:
```python
y_scaled = (y - y.mean()) / y.std()
```

Though our tests showed this alone doesn't fix it.

### 3. Regularization
L2 regularization on weights to prevent extreme values.

### 4. Different Network Architecture
Deeper networks or different connectivity patterns may have better optimization landscapes.

### 5. They Acknowledge CRPS is Better!
**From Kellner & Ceriotti Table 2:**
> "CRPS training often achieves better RMSE than NLL for many datasets"

They know NLL can hurt accuracy and recommend CRPS for robust training!

### 6. Implementation Details Not Disclosed
The paper doesn't give full implementation details. There may be tricks not mentioned.

## Practical Recommendations

### ✅ DO: Use CRPS (Default)
```python
model = DPOSE(layers=(1, 15, 32))  # Uses CRPS by default
```

### ✅ DO: Two-Stage Training if You Need NLL
```python
# Stage 1: MSE
model = DPOSE(layers=(1, 15, 32), loss_type='mse')
model.fit(X_train, y_train, maxiter=500)

# Stage 2: NLL
model.loss_type = 'nll'
model.fit(X_train, y_train, val_X=X_val, val_y=y_val, maxiter=200)
```

### ❌ DON'T: Use NLL from Scratch
```python
# This will almost certainly fail!
model = DPOSE(layers=(1, 15, 32), loss_type='nll')
model.fit(X_train, y_train)
```

### ❌ DON'T: Use Large min_sigma as a "Fix"
```python
# This "works" but defeats the purpose
model = DPOSE(layers=(1, 15, 32), loss_type='nll', min_sigma=0.5)
```

## Comparison Table

| Method | Accuracy | Uncertainty Quality | Ease of Use | Recommendation |
|--------|----------|---------------------|-------------|----------------|
| **CRPS** | ✓✓✓ Excellent | ✓✓✓ Calibrated | ✓✓✓ Works out-of-box | **BEST** ⭐ |
| **MSE → NLL** | ✓✓✓ Excellent | ✓✓✓ Calibrated | ✓✓ Requires two stages | Good alternative |
| **NLL alone** | ✗✗✗ Fails | ✗✗✗ Meaningless | ✗✗✗ Requires tricks | **AVOID** ⚠️ |
| **MSE alone** | ✓✓✓ Excellent | ✗ No uncertainty | ✓✓✓ Simple | For point predictions only |

## Debugging Checklist

If your NLL training produces nonsensical results, check:

1. **Predictions are garbage?** (e.g., predicting 10,000 when true range is 0-1)
   - → You hit the uncertainty inflation pathology
   - → **Solution:** Use CRPS or pre-train with MSE

2. **Uncertainties are huge?** (e.g., σ >> prediction range)
   - → Network escaped to bad local minimum
   - → **Solution:** Use CRPS or pre-train with MSE

3. **Loss doesn't decrease?** (stays high or increases)
   - → Optimizer stuck immediately
   - → **Solution:** Different initialization or use CRPS

4. **Predictions are good but uncertainties constant?**
   - → min_sigma is too large
   - → **Solution:** Use CRPS with smaller min_sigma (default 1e-3)

## Conclusion

**NLL training fails in this DPOSE implementation due to a fundamental pathology** where the network escapes to a local minimum with garbage predictions and huge uncertainties. The logarithmic penalty on uncertainty is too weak to prevent this.

**CRPS works reliably** because its linear penalty on uncertainty prevents the pathology.

**Recommendation:** Use CRPS (the new default) unless you have specific reasons to use NLL, in which case pre-train with MSE first.

---

## References

1. Kellner, M., & Ceriotti, M. (2024). Uncertainty quantification by direct propagation of shallow ensembles. *Machine Learning: Science and Technology*, 5(3), 035006.
   - Section 3.3: "CRPS training can improve RMSE when NLL degrades accuracy"
   - Table 2: Shows CRPS outperforms NLL on several datasets

2. See `WHY_NLL_FAILS.md` for mathematical details of the uncertainty inflation pathology.
