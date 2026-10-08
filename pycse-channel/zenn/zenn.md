---
marp: true
theme: default
paginate: true
size: 16:9
title: "ZENN: Zentropy-Enhanced Neural Networks"
author: John Kitchin
math: mathjax
style: |
  :root {
    --bg: #1a1a19; --panel: #232321; --ink: #f2f1ec; --ink-2: #c4c2ba;
    --muted: #8d8b83; --accent: #d95926; --blue: #3987e5;
  }
  section {
    background: var(--bg); color: var(--ink);
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 26px; padding: 50px 70px;
  }
  section::after { color: var(--muted); font-size: 16px; }
  h1 { color: var(--ink); font-size: 52px; margin-bottom: 0.2em; }
  h2 { color: var(--ink); font-size: 38px; margin: 0 0 0.5em 0; }
  h2 em { color: var(--accent); font-style: normal; }
  strong { color: var(--accent); }
  a { color: #7fb0f0; }
  code { background: var(--panel); color: var(--ink); font-size: 20px; }
  pre { background: var(--panel) !important; border-radius: 10px; border: 1px solid #33332f; }
  pre code { font-size: 21px; line-height: 1.45; }
  .hljs-keyword, .hljs-built_in { color: #e0864f; }
  .hljs-string { color: #9fc97a; }
  .hljs-comment { color: #8d8b83; font-style: italic; }
  .hljs-number { color: #b9a6f5; }
  .muted { color: var(--muted); }
  .small { font-size: 20px; }
  .tiny { font-size: 16px; color: var(--muted); }
  table { font-size: 21px; border-collapse: collapse; margin: 0 auto; }
  th { background: var(--panel) !important; color: var(--ink-2); border-bottom: 2px solid #44443f; }
  td, th { border: none !important; padding: 8px 18px !important; background: transparent !important; color: var(--ink); }
  table tr { background: transparent !important; }
  tr:nth-child(even) td { background: #20201e !important; }
  .z { color: var(--accent); font-weight: 600; }
  .n { color: var(--blue); font-weight: 600; }

  /* full-bleed interactive slides */
  section.widget { padding: 22px 34px 22px 34px; justify-content: flex-start; }
  section.widget h2 { font-size: 30px; margin-bottom: 4px; }
  section.widget iframe { width: 100%; height: 478px; border: 0; border-radius: 10px; background: var(--bg); }

  /* title slide */
  section.title { justify-content: center; }
  section.title h1 { font-size: 68px; }
  section.title h1 span { color: var(--accent); }
  section.title p { color: var(--ink-2); }

  /* two columns */
  .cols { display: flex; gap: 46px; align-items: flex-start; }
  .cols > div { flex: 1; }
  .card { background: var(--panel); border: 1px solid #33332f; border-radius: 12px; padding: 18px 24px; }
  .card h3 { margin: 0 0 8px 0; font-size: 26px; }

  section iframe.tlframe { width: 100%; height: 330px; border: 0; }

  /* every slide ends with one takeaway */
  section { padding-bottom: 150px; }
  section.widget { padding-bottom: 118px; }
  .takeaway {
    position: absolute; left: 70px; right: 70px; bottom: 36px;
    background: #2b1f18; border-left: 6px solid var(--accent); border-radius: 8px;
    padding: 12px 22px; font-size: 23px; line-height: 1.35; color: var(--ink);
  }
  .takeaway::before {
    content: "Takeaway"; display: block; font-size: 14px; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase; color: var(--accent); margin-bottom: 2px;
  }
  section.widget .takeaway { left: 34px; right: 34px; bottom: 18px; padding: 9px 20px; font-size: 21px; }
  .watch { font-size: 19px; color: var(--ink-2); margin: -2px 0 6px 0; }
  .watch b { color: var(--ink); }

  /* mixture-of-regimes sketch for the problem slide */
  .regimes { display: flex; gap: 16px; margin-top: 10px; }
  .regimes div { flex: 1; text-align: center; padding: 14px; border-radius: 12px; background: var(--panel);
    border-top: 5px solid; font-size: 21px; }
---

<!-- _class: title -->
<!-- _paginate: false -->

# <span>ZENN</span>: neural networks that think in free energy

Zentropy-Enhanced Neural Networks, and what they can do that a regular NN can't

**pycse** · `pycse.sklearn.zenn`

<div class="takeaway">ZENN is a neural network whose output is a thermodynamic free energy, built from several competing "configurations".</div>

---

## The problem: one dataset, *hidden regimes*

Real data often mixes several underlying "states": phases of a material, sources of a dataset, operating modes of a process.

<div class="regimes">
<div style="border-color:#199e70">ordered, left well</div>
<div style="border-color:#c98500">ordered, right well</div>
<div style="border-color:#9085e9">disordered, high entropy</div>
</div>

<br>

A regular neural network sees only `(x, T) → y`. Nothing tells it that regimes exist.

<div class="takeaway">When data hides several regimes, a plain NN has to memorize how they trade off, so it fails where it has no data.</div>

---

## Where ZENN came from

<iframe class="tlframe" src="widgets/timeline.html"></iframe>

<div class="takeaway">ZENN is a materials-thermodynamics idea (zentropy, Z.-K. Liu's group at Penn State) turned into a neural network architecture.</div>

---

## Zentropy in two equations

A system is a mixture of **configurations** $k$, each with its own energy $E_k$ and entropy $S_k$.

$$
F_k = E_k - T S_k, \qquad p_k = \frac{e^{-F_k/k_BT}}{Z}, \qquad Z = \sum_k e^{-F_k/k_BT}
$$

$$
S = \underbrace{\sum_k p_k S_k}_{\text{inside each configuration}} \; \underbrace{-\, k_B \sum_k p_k \ln p_k}_{\text{mixing among configurations}}
\qquad\Rightarrow\qquad F = -k_BT \ln Z
$$

<div class="takeaway">The total free energy is a temperature-weighted blend of the configurations' free energies: T decides which one is in charge.</div>

---

<!-- _class: widget -->

## How zentropy works

<p class="watch"><b>Watch:</b> press Play. As T rises, the purple high-entropy configuration drops and takes over the middle, and the white total goes from two wells to one.</p>

<iframe src="widgets/zentropy.html"></iframe>

<div class="takeaway">A phase transition emerges from simple pieces: temperature alone switches which configuration wins.</div>

---

## From theory to network

ZENN keeps the zentropy *structure* and lets neural networks fill in the pieces:

<div class="cols">
<div class="card">
<h3>Learned</h3>

$E_k(x, T)$ and $S_k(x, T) \ge 0$, one small MLP each, for $k = 1 \dots K$

</div>
<div class="card">
<h3>Fixed by physics</h3>

$F_k = E_k - TS_k$, Boltzmann weights $p_k$, and the zentropy total $F$

</div>
</div>

<div class="takeaway">ZENN learns only each configuration's E and S; the physics does the mixing, so the output behaves like a real free energy you can differentiate.</div>

---

<!-- _class: widget -->

## Inside ZENN

<p class="watch"><b>Watch:</b> press Next to follow one input through each stage, then drag T and see the weights p<sub>k</sub> shift. Numbers are the true system from our example.</p>

<iframe src="widgets/architecture.html"></iframe>

<div class="takeaway">Temperature acts twice in ZENN (in F<sub>k</sub> = E<sub>k</sub> − T S<sub>k</sub> and in the softmax). A plain NN treats it as just another input.</div>

---

## Two estimators, sklearn-style

<div class="cols">
<div class="card">
<h3><code>ZENNRegressor</code></h3>

Energy landscapes $F(x, T)$: derivatives via autodiff, equilibria, critical points.

</div>
<div class="card">
<h3><code>ZENNClassifier</code></h3>

Multi-source data: **learnable temperatures** act as hidden data modes (cross-zentropy loss).

</div>
</div>

<p class="small muted">Paper results: gains over cross-entropy on CIFAR-10/100, AG News, BBC News; Fe₃Pt energy landscape from DFT.</p>

<div class="takeaway">Use the regressor for energy landscapes and the classifier for data pooled from different sources. Today we use the regressor.</div>

---

## pycse's ZENN vs the published code

<div class="small">

| | Published code (paper) | `pycse.sklearn.zenn` |
|---|---|---|
| Form | PyTorch scripts, one per experiment | JAX sklearn estimators |
| $S_k \ge 0$ via | net² | softplus(net) |
| Regression loss | Jensen-Shannon divergence | **NLL (default)**, MSE, hybrid, JS |
| Uncertainty | none | aleatoric + epistemic, intervals, calibration |
| Extras | | critical points, KAN backbone, OOD scores |

</div>

<p class="small">With NLL, the noise variance is σ² = Σ p<sub>k</sub>S<sub>k</sub>: the entropy networks double as a <b>noise model</b>, a reinterpretation not in the paper. Our example uses <code>loss_type="mse"</code>.</p>

<div class="takeaway">Same zentropy core, packaged as reusable sklearn models, with an NLL loss that adds calibrated uncertainty the paper doesn't have.</div>

---

## Using it in pycse

```python
from pycse.sklearn.zenn import ZENNRegressor

zenn = ZENNRegressor(n_configs=4, hidden_dims=(16, 16), loss_type="mse",
                     learning_rate=5e-3, max_epochs=8000)

zenn.fit(X, F, T=T)                       # T = temperature of each sample

zenn.predict(X_new, T=1.5)                # any temperature, even unseen ones
zenn.compute_derivatives(X_new, T=0.5, order=2)    # exact, via JAX autodiff
zenn.get_configuration_probabilities(X_new, T=0.5) # which p_k is in charge
```

<div class="takeaway">It is a standard sklearn estimator. The one difference: temperature is passed separately from the features.</div>

---

## The test: an order/disorder free energy

<div class="cols">
<div>

**Truth:** 3 configurations (left well, right well, disordered). Double well below $T_c = 0.642$, single well above.

**Data:** 25 points at each of $T = 0.3, 0.4, 0.5, 0.8, 1.0$, noise σ = 0.02.

</div>
<div>

<span class="z">ZENN</span>: K = 4 configurations, (16, 16) MLPs

<span class="n">Regular NN</span>: sklearn `MLPRegressor`, inputs (x, T), 64×64 tanh

</div>
</div>

<div class="takeaway">Both models get the same data at 5 temperatures. Then we test them at temperatures neither has seen.</div>

---

<!-- _class: widget -->

## Same data, different answers

<p class="watch"><b>Watch:</b> press Play. White dots appear at training temperatures. Once the orange marker leaves the training range (shaded strip), compare each model to the dashed truth.</p>

<iframe src="widgets/compare.html"></iframe>

<div class="takeaway">Inside the data both models fit well. Beyond it, ZENN stays near the truth while the plain NN drifts away.</div>

---

<!-- _class: widget -->

## Derivatives reveal the physics

<p class="watch"><b>Watch:</b> the curvature at x = 0 vs T. Below zero means a barrier (two wells); where it crosses zero is T<sub>c</sub>. Press Draw to replay.</p>

<iframe src="widgets/curvature.html"></iframe>

<div class="takeaway">ZENN captures the steep low-T curvature; the plain NN flattens it. Both place T<sub>c</sub> roughly right.</div>

---

## Is it a fluke? Four random seeds

<div class="small">

| metric (mean of 4 seeds) | <span class="z">ZENN</span> | <span class="n">regular NN</span> | ZENN better in |
|---|---:|---:|:---:|
| F RMSE, interpolation (T 0.55 to 0.75) | 0.011 | 0.011 | 3 / 4 |
| F RMSE, colder than data (T 0.15 to 0.25) | **0.025** | 0.039 | 4 / 4 |
| F RMSE, hotter than data (T 1.2 to 2.0) | **0.17** | 0.38 | 4 / 4 |
| curvature error at x = 0, cold | **5.3** | 11.6 | 4 / 4 |
| error in $T_c$ | 0.040 | **0.020** | 2 / 4 |

</div>

<div class="takeaway">The extrapolation advantage holds for every seed. Inside the data there is no clear winner, and the plain NN found T<sub>c</sub> slightly better.</div>

---

<!-- _class: widget -->

## You can look inside

<p class="watch"><b>Watch:</b> left is the true weight of each configuration, right is what ZENN learned without being told. Press Play to sweep T.</p>

<iframe src="widgets/learned.html"></iframe>

<div class="takeaway">ZENN discovered the left and right wells on its own, but folded the disordered state into them: interpretable, not exact.</div>

---

## Honest caveats

<div class="cols">
<div>

**Cost.** About 10 to 25 s to train vs about 1 s for the MLP here.

**Not identifiable.** Learned configurations are a lens, not ground truth.

</div>
<div>

**Helps, does not guarantee.** E<sub>k</sub> and S<sub>k</sub> still see T, so hot extrapolation is better but not perfect.

</div>
</div>

<div class="takeaway">Reach for ZENN when your data really is a temperature-controlled mixture of states, not as a general NN replacement.</div>

---

## Summary

ZENN builds **zentropy** into a neural network: learn $E_k$ and $S_k$, let thermodynamics do the mixing.

`pip install pycse` → `from pycse.sklearn.zenn import ZENNRegressor`

<p class="tiny">
S. Wang, S.-L. Shang, Z.-K. Liu, W. Hao, PNAS 123, e2511227122 (2026). doi:10.1073/pnas.2511227122 · arXiv:2505.09851<br>
Z.-K. Liu, Y. Wang, S.-L. Shang, J. Phase Equilib. Diffus. 43, 598 (2022). doi:10.1007/s11669-022-00942-z<br>
Z.-K. Liu, Y. Wang, S.-L. Shang, Sci. Rep. 4, 7043 (2014). doi:10.1038/srep07043<br>
Z.-K. Liu, <i>Zentropy: Theory and Fundamentals</i>, Jenny Stanford (2024). doi:10.1201/9781032692401<br>
Authors' reference code: github.com/WilliamMoriaty/ZENN · example: pycse-channel/zenn/experiment.py
</p>

<div class="takeaway">Put the physics in the architecture and you get better extrapolation and internals you can interpret.</div>
