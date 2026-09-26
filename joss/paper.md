---
title: 'pycse: Python Computations in Science and Engineering with Uncertainty Quantification'
tags:
  - Python
  - scientific computing
  - uncertainty quantification
  - regression analysis
  - machine learning
  - design of experiments
authors:
  - name: John R. Kitchin
    orcid: 0000-0003-2625-9232
    affiliation: 1
    corresponding: true
affiliations:
  - name: Department of Chemical Engineering, Carnegie Mellon University, Pittsburgh, PA 15213, USA
    index: 1
date: 10 January 2026
bibliography: paper.bib
---

# Summary

`pycse` is a Python library for scientific and engineering computations with a focus on regression analysis and uncertainty quantification. Unlike general-purpose scientific computing libraries where uncertainty is often an afterthought, `pycse` treats uncertainty as a first-class citizen—every regression function returns parameter estimates alongside confidence intervals and standard errors. The library combines classical statistical methods with modern machine learning techniques, all accessible through a consistent, scikit-learn compatible API.

# Statement of Need

Uncertainty quantification is essential in scientific research, yet existing tools make it surprisingly difficult. While `scipy.optimize.curve_fit` returns parameter covariance, users must manually compute confidence intervals—a process prone to errors and inconsistent practices. Modern uncertainty quantification methods like ensemble-based approaches and conformal prediction exist but are scattered across different packages without a unified interface.

`pycse` addresses these gaps by providing:

1. **Built-in uncertainty**: Core regression functions (`nlinfit`, `regress`, `polyfit`) automatically return confidence intervals and prediction bands using proper statistical methods.

2. **Modern UQ methods**: Advanced techniques including Direct Propagation of Shallow Ensembles (DPOSE) [@kellner2024dpose], Kolmogorov-Arnold Networks with Last-Layer Prediction Rigidity [@liu2024kan; @bigi2024llpr], and neural network-Bayesian Ridge hybrids—all with sklearn-compatible APIs.

3. **Design of experiments**: Latin hypercube sampling and response surface methodology for efficient experimental planning [@mckay1979lhs].

4. **Educational foundation**: Extensive examples and Jupyter notebooks developed over a decade of teaching computational methods to chemical engineering students at Carnegie Mellon University.

The target audience includes scientists and engineers who need rigorous uncertainty quantification, graduate students learning computational methods, and researchers building machine learning pipelines that require reliable uncertainty estimates.

# State of the Field

Several Python packages address aspects of scientific computing and uncertainty quantification. `scipy` [@virtanen2020scipy] provides optimization and fitting but requires manual uncertainty calculations. `statsmodels` [@seabold2010statsmodels] offers statistical modeling with confidence intervals but lacks modern ML-based UQ methods. `scikit-learn` [@pedregosa2011sklearn] provides excellent ML infrastructure but minimal native uncertainty quantification. The `uncertainties` package handles error propagation but not regression or ML-based methods.

`pycse` differentiates itself by unifying classical statistical methods and modern ML-based uncertainty quantification under a consistent sklearn-compatible interface. This enables researchers to seamlessly move from simple polynomial fits with prediction intervals to neural network ensembles with calibrated uncertainty estimates, all within familiar sklearn pipelines.

# Software Design

`pycse` employs a two-tier architecture:

**Core module**: Simple, dependency-light functions (`nlinfit`, `regress`, `polyfit`, `ivp`) for common scientific computing tasks. These require only NumPy and SciPy.

**sklearn submodule**: Advanced ML estimators following sklearn conventions (`BaseEstimator`, `RegressorMixin`). This enables pipeline integration, cross-validation, and hyperparameter tuning with `GridSearchCV`. Methods include:

- `DPOSE`: JAX-based shallow ensembles for efficient uncertainty propagation
- `KAN`/`KANLLPR`: Kolmogorov-Arnold Networks with uncertainty
- `NNBR`: Neural network feature extraction with Bayesian Ridge regression
- `SurfaceResponse`: Response surface methodology for design of experiments

The advanced methods leverage JAX [@jax2018github] for GPU acceleration and automatic differentiation, while keeping these as optional dependencies to maintain a lightweight core installation.

# Research Impact

`pycse` has been used in chemical engineering education at Carnegie Mellon University for over a decade. The DPOSE uncertainty quantification method implemented in `pycse` has been applied to graph neural networks for materials modeling [@vinchurkar2025uq]. Related work on differentiable programming for uncertainty quantification [@alves2025mapping] and the broader paradigm shift in chemical engineering modeling [@kitchin2025paradigm] demonstrates the research context that `pycse` supports.

The software is actively maintained with 817 tests, continuous integration, and comprehensive documentation including a Jupyter Book with worked examples covering topics from basic regression to advanced machine learning.

# AI Usage Disclosure

The `pycse` project was developed without AI assistance from its inception in February 2013 through September 2025, comprising 798 commits of original work including all core regression functions, the foundational architecture, and educational materials.

Beginning in October 2025, Claude (Anthropic's AI assistant) was used as a collaborative coding tool to accelerate development of new features and improve software quality. Claude contributed to approximately 98 commits (as of January 2026), including: implementation of new sklearn-compatible estimators (JAXPeriodicRegressor, JAXMonotonicRegressor, JAXICNNRegressor, KANLLPR, NFlowsRegressor), bug fixes in existing code, expansion of the test suite, documentation improvements, and CI/CD pipeline optimization. All AI-assisted contributions were reviewed and approved by the author before merging.

This paper was drafted with assistance from Claude Code, with the author providing direction, reviewing content, and making editorial decisions.

# Acknowledgements

This work was supported by [funding sources to be added]. The author thanks the students and researchers who have contributed to and used `pycse` over the years.

# References
