# Test Suite Optimization Plan

## Executive Summary

The pycse test suite currently takes ~40 minutes to complete in CI, approaching timeout limits. This plan outlines a comprehensive strategy to reduce test execution time to under 15 minutes while maintaining code quality and test coverage.

## Current State Analysis

### Test Suite Composition
- **Total tests**: 817 tests across 23 test files
- **Test types**: Heavy focus on ML/neural network tests using JAX
- **Current timeout**: 40 minutes (increased from 20 minutes)
- **Parallelization**: Already using pytest-xdist with `-n auto` (2 cores on ubuntu-latest)
- **Smart testing**: pytest-testmon for running affected tests on PRs
- **Optimizations already attempted**:
  - Reduced epochs from 500 to 5-10 in most tests
  - Reduced dataset sizes (e.g., 150 → 30 samples)
  - Reduced network sizes (e.g., (32,32) → (8,8))
  - Relaxed accuracy assertions
  - Reduced maxiter from 50 to 10

### Key Bottlenecks
1. **Neural network training**: Each test trains models from scratch (JAX, KAN, ICNN, etc.)
2. **JAX compilation overhead**: JIT compilation on first run for each test
3. **Test granularity**: Many integration-style tests that could be unit tests
4. **Lack of test categorization**: @pytest.mark.slow exists but isn't being used effectively
5. **Sequential expensive operations**: Tests don't share compiled models or trained states
6. **Python version matrix**: Testing on both 3.12 and 3.13 doubles execution time

## Optimization Strategy

### Phase 1: Immediate Wins (High Impact, Low Effort)

#### 1.1 Split Test Suite by Speed
**Goal**: Run fast tests on every commit, slow tests less frequently

**Implementation**:
```yaml
# .github/workflows/pycse-tests-fast.yaml
jobs:
  fast-tests:
    name: Fast Tests (Python ${{ matrix.python-version }})
    timeout-minutes: 10
    steps:
      - name: Run fast tests only
        run: |
          pytest src/pycse/tests -v -n auto -m "not slow" --durations=20
```

```yaml
# .github/workflows/pycse-tests-slow.yaml
on:
  push:
    branches: [master]
  schedule:
    - cron: '0 2 * * *'  # Nightly at 2 AM

jobs:
  slow-tests:
    name: Slow Tests (Python ${{ matrix.python-version }})
    timeout-minutes: 40
    steps:
      - name: Run slow tests only
        run: |
          pytest src/pycse/tests -v -n auto -m "slow" --durations=20
```

**Action items**:
- [ ] Mark all ML/training tests with `@pytest.mark.slow`
- [ ] Create separate workflows for fast/slow tests
- [ ] Configure fast tests to run on all pushes/PRs
- [ ] Configure slow tests to run only on master pushes and nightly

**Expected impact**: Reduce PR test time from 40min to ~10min

#### 1.2 Reduce Python Version Matrix
**Goal**: Test fewer Python versions in CI

**Current**: Testing 3.12 and 3.13
**Proposed**: Test only 3.12 on PRs, test both 3.12 and 3.13 on master

**Implementation**:
```yaml
strategy:
  matrix:
    os: [ubuntu-latest]
    python-version: ${{ github.event_name == 'pull_request' && ['3.12'] || ['3.12', '3.13'] }}
```

**Expected impact**: Reduce PR test time by 50% (if both versions run same tests)

#### 1.3 Improve Test Parallelization
**Goal**: Better distribute tests across workers

**Current**: Using `-n auto` (2 cores on ubuntu-latest)
**Proposed**: Use pytest-split for better load balancing

**Implementation**:
```yaml
strategy:
  matrix:
    split: [1, 2, 3, 4]  # 4 parallel jobs
steps:
  - name: Run tests (split ${{ matrix.split }}/4)
    run: |
      pytest src/pycse/tests -v --splits 4 --group ${{ matrix.split }}
```

**Action items**:
- [ ] Add pytest-split to test dependencies
- [ ] Configure matrix strategy with splits
- [ ] Test and tune number of splits (2-4 optimal)

**Expected impact**: Reduce test time by 30-50% depending on load balancing

### Phase 2: Test Suite Refactoring (Medium Impact, Medium Effort)

#### 2.1 Create Shared Fixtures for Trained Models
**Goal**: Avoid retraining models in every test

**Current issue**: Each test trains its own model from scratch
**Proposed**: Use session-scoped fixtures for pre-trained models

**Implementation example**:
```python
# conftest.py
@pytest.fixture(scope="session")
def trained_kan_model():
    """Pre-trained KAN model for fast predictions."""
    X, y = generate_standard_dataset()
    model = KAN(layers=(1, 5, 1), grid_size=3)
    model.fit(X, y, maxiter=50)
    return model, X, y

# test file
def test_prediction(trained_kan_model):
    model, X, y = trained_kan_model
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
```

**Action items**:
- [ ] Identify tests that only need prediction (not training)
- [ ] Create session-scoped fixtures for common trained models
- [ ] Refactor prediction/inference tests to use shared models

**Expected impact**: Reduce time for prediction tests by 90%

#### 2.2 Reduce Test Redundancy
**Goal**: Eliminate duplicate/overlapping tests

**Approach**:
- Combine similar tests using parametrization
- Remove tests that validate library functionality (not pycse code)
- Focus on interface/integration tests rather than testing every parameter combo

**Example refactoring**:
```python
# Before: 3 separate tests
def test_grid_size_3(self, data):
    model = KAN(layers=(1, 3, 1), grid_size=3)
    model.fit(X, y, maxiter=10)
    assert model.predict(X).shape == (len(X),)

def test_grid_size_5(self, data):
    model = KAN(layers=(1, 3, 1), grid_size=5)
    model.fit(X, y, maxiter=10)
    assert model.predict(X).shape == (len(X),)

# After: 1 parametrized test
@pytest.mark.parametrize("grid_size", [3, 5, 7])
def test_grid_sizes(self, data, grid_size):
    model = KAN(layers=(1, 3, 1), grid_size=grid_size)
    model.fit(X, y, maxiter=10)
    assert model.predict(X).shape == (len(X),)
```

**Action items**:
- [ ] Audit test suite for redundancy
- [ ] Combine similar tests using parametrization
- [ ] Remove tests that don't add unique coverage

**Expected impact**: Reduce total test count by 10-20%

#### 2.3 JAX Compilation Caching
**Goal**: Cache JIT compiled functions across tests

**Approach**:
- Use persistent compilation cache
- Share compiled functions via fixtures
- Pre-compile common operations

**Implementation**:
```yaml
# In GitHub Actions
env:
  JAX_COMPILATION_CACHE_DIR: /tmp/jax-cache
  XLA_FLAGS: "--xla_gpu_cuda_data_dir=/usr/local/cuda"

- name: Cache JAX compilations
  uses: actions/cache@v4
  with:
    path: /tmp/jax-cache
    key: jax-cache-${{ runner.os }}-${{ hashFiles('src/pycse/sklearn/*.py') }}
```

**Action items**:
- [ ] Enable JAX persistent cache in CI
- [ ] Configure cache directory and invalidation strategy
- [ ] Monitor cache hit rates

**Expected impact**: Reduce JAX test time by 20-30%

### Phase 3: Infrastructure Changes (High Impact, High Effort)

#### 3.1 Separate Test Categories into Jobs
**Goal**: Run different test categories in parallel jobs

**Proposed structure**:
```yaml
jobs:
  unit-tests:
    name: Unit Tests
    timeout-minutes: 5
    steps:
      - run: pytest src/pycse/tests -v -n auto -m "not slow and not integration"

  integration-tests:
    name: Integration Tests
    timeout-minutes: 10
    steps:
      - run: pytest src/pycse/tests -v -n auto -m "integration and not slow"

  sklearn-ml-tests:
    name: Sklearn ML Tests
    timeout-minutes: 20
    steps:
      - run: pytest src/pycse/tests/test_sklearn_*.py -v -n auto -m "not slow"

  slow-tests:
    name: Slow Tests (master only)
    if: github.event_name == 'push' && github.ref == 'refs/heads/master'
    timeout-minutes: 30
    steps:
      - run: pytest src/pycse/tests -v -n auto -m "slow"
```

**Action items**:
- [ ] Add pytest markers: unit, integration, ml
- [ ] Tag all tests with appropriate markers
- [ ] Create multi-job workflow
- [ ] Configure job dependencies (if needed)

**Expected impact**: Tests run in parallel, total wall-clock time ~15-20min

#### 3.2 Use Larger GitHub Runners
**Goal**: More CPU cores for parallelization

**Current**: ubuntu-latest (2 cores)
**Proposed**: ubuntu-latest-4-cores or ubuntu-latest-8-cores

**Cost-benefit**:
- 4-core: 2x cost, potential 1.5-2x speedup
- 8-core: 4x cost, potential 2-3x speedup (diminishing returns)

**Implementation**:
```yaml
jobs:
  test:
    runs-on: ubuntu-latest-4-cores  # GitHub Teams/Enterprise
    # OR for free tier:
    runs-on: ubuntu-latest
    steps:
      - run: pytest src/pycse/tests -v -n 4  # Explicit core count
```

**Expected impact**: 1.5-2x speedup for $$$

#### 3.3 Use Test Result Caching
**Goal**: Skip tests that haven't changed

**Note**: pytest-testmon already implemented for PRs
**Enhancement**: Extend to all branches

**Current**:
```yaml
- name: Run tests with pytest (affected tests only)
  if: github.event_name == 'pull_request'
  run: pytest src/pycse/tests -v --testmon -n auto
```

**Proposed**:
```yaml
- name: Run tests with pytest (affected tests only)
  # Run testmon for all pushes except master
  if: github.ref != 'refs/heads/master'
  run: pytest src/pycse/tests -v --testmon -n auto

- name: Run tests with pytest (full suite)
  # Only run full suite on master
  if: github.ref == 'refs/heads/master'
  run: pytest src/pycse/tests -v --cov=src/pycse -n auto
```

**Expected impact**: 50-70% reduction in branch test time

### Phase 4: Long-term Improvements (Medium Impact, High Effort)

#### 4.1 Mock Expensive Operations in Unit Tests
**Goal**: Replace real training with mocks where appropriate

**Approach**:
- Unit tests should test interfaces, not convergence
- Mock training for tests that check error handling, I/O, etc.
- Keep integration tests for actual training

**Example**:
```python
def test_fit_invalid_shape(self, mocker):
    # No need to actually train, just test error handling
    model = KAN(layers=(1, 5, 1))
    X_invalid = np.array([1, 2, 3])  # 1D instead of 2D
    with pytest.raises(ValueError, match="Expected 2D array"):
        model.fit(X_invalid, y)
```

#### 4.2 Continuous Benchmarking
**Goal**: Track test execution time over time

**Implementation**:
- Use pytest-benchmark for performance regression tests
- Track test durations in CI artifacts
- Alert on significant slowdowns

#### 4.3 Consider Test Sampling
**Goal**: Run subset of tests on PRs, full suite on master

**Approach**:
- Select representative tests for each module
- Run 30% of tests on PR (sampled to cover all modules)
- Run 100% on master

## Implementation Priority

### Week 1: Quick Wins
1. Mark slow tests with `@pytest.mark.slow`
2. Create fast-tests workflow (run on PRs, exclude slow tests)
3. Move slow tests to master-only or nightly
4. Reduce Python version matrix for PRs

**Expected outcome**: PR test time drops to ~10-15 minutes

### Week 2: Parallelization
1. Implement pytest-split for better load balancing
2. OR implement multi-job parallel strategy
3. Set up JAX compilation caching

**Expected outcome**: PR test time drops to ~7-10 minutes

### Week 3: Test Refactoring
1. Create shared fixtures for trained models
2. Refactor prediction tests to use shared models
3. Audit and remove redundant tests
4. Combine similar tests with parametrization

**Expected outcome**: Total test count reduced 10-20%, time reduced further

### Week 4: Monitoring
1. Add test duration tracking
2. Set up alerts for slow tests
3. Document test categorization strategy
4. Create guidelines for adding new tests

## Success Metrics

- **Primary goal**: PR tests complete in < 15 minutes
- **Stretch goal**: PR tests complete in < 10 minutes
- **Maintain**:
  - Code coverage > 80%
  - No false negatives (tests still catch bugs)
  - Test suite remains maintainable

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Skipping slow tests misses bugs | High | Run slow tests nightly and on master |
| Shared fixtures hide test dependencies | Medium | Carefully scope fixtures, document dependencies |
| Over-optimization reduces coverage | High | Monitor coverage metrics, keep integration tests |
| JAX cache invalidation issues | Low | Use hash of source files for cache keys |
| Parallel tests cause flakiness | Medium | Ensure test isolation, use proper fixtures |

## Monitoring and Maintenance

1. **Weekly review**: Check test duration trends
2. **Monthly audit**: Review slow tests, look for optimization opportunities
3. **Quarterly planning**: Reassess test strategy as codebase evolves

## Next Steps

1. **Review this plan** with stakeholders
2. **Prioritize phases** based on urgency and resources
3. **Create implementation tickets** for each action item
4. **Start with Phase 1** (immediate wins)
