"""Comprehensive tests for LatinSquare class.

This module tests the LatinSquare class for Latin Square design of
experiments with ANOVA analysis.
"""

import numpy as np
import pandas as pd
import pytest

from pycse.sklearn.lhc import LatinSquare


class TestLatinSquareInitialization:
    """Test LatinSquare initialization and validation."""

    def test_basic_initialization(self):
        """Test basic initialization with valid inputs."""
        vars_dict = {
            "Temperature": [100, 150, 200],
            "Pressure": [1, 2, 3],
            "Catalyst": ["A", "B", "C"],
        }
        ls = LatinSquare(vars=vars_dict)

        assert ls.vars == vars_dict
        assert ls.labels == ["Temperature", "Pressure", "Catalyst"]

    def test_initialization_with_4_levels(self):
        """Test initialization with 4 levels per factor."""
        vars_dict = {
            "Row": [1, 2, 3, 4],
            "Col": ["A", "B", "C", "D"],
            "Treatment": ["W", "X", "Y", "Z"],
        }
        ls = LatinSquare(vars=vars_dict)

        assert len(ls.labels) == 3
        for levels in ls.vars.values():
            assert len(levels) == 4

    def test_initialization_requires_vars(self):
        """Test that vars parameter is required."""
        with pytest.raises(ValueError, match="vars parameter is required"):
            LatinSquare(vars=None)

    def test_initialization_vars_must_be_dict(self):
        """Test that vars must be a dictionary."""
        with pytest.raises(TypeError, match="vars must be a dictionary"):
            LatinSquare(vars=[1, 2, 3])

        with pytest.raises(TypeError, match="vars must be a dictionary"):
            LatinSquare(vars="not a dict")

    def test_initialization_requires_3_factors(self):
        """Test that exactly 3 factors are required."""
        # Too few factors
        with pytest.raises(ValueError, match="exactly 3 factors"):
            LatinSquare(vars={"A": [1, 2], "B": [3, 4]})

        # Too many factors
        with pytest.raises(ValueError, match="exactly 3 factors"):
            LatinSquare(vars={"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]})

    def test_initialization_equal_level_counts(self):
        """Test that all factors must have equal numbers of levels."""
        with pytest.raises(ValueError, match="same number of levels"):
            LatinSquare(
                vars={
                    "A": [1, 2, 3],
                    "B": [10, 20],  # Only 2 levels
                    "C": ["X", "Y", "Z"],
                }
            )

    def test_initialization_minimum_levels(self):
        """Test that factors must have at least 2 levels."""
        with pytest.raises(ValueError, match="at least 2 levels"):
            LatinSquare(
                vars={
                    "A": [1],  # Only 1 level
                    "B": [10],
                    "C": ["X"],
                }
            )

    def test_seed_attribute(self):
        """Test that seed is set correctly."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        assert ls.seed == 42
        assert LatinSquare.seed == 42


class TestLatinSquareDesign:
    """Test design() method."""

    def test_design_basic(self):
        """Test basic design generation."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        # Should have 9 experiments (3x3)
        assert len(design) == 9
        assert list(design.columns) == ["A", "B", "C"]

    def test_design_without_shuffle(self):
        """Test that design without shuffle is deterministic."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        design1 = ls.design(shuffle=False)
        design2 = ls.design(shuffle=False)

        pd.testing.assert_frame_equal(design1, design2)

    def test_design_with_shuffle(self):
        """Test that design with shuffle randomizes order."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        # Successive shuffles draw from the instance's private generator
        design1 = ls.design(shuffle=True)
        design2 = ls.design(shuffle=True)

        # Should have different order
        # (may occasionally fail by chance, but very unlikely)
        assert not design1.equals(design2)

    def test_design_latin_square_property(self):
        """Test that design satisfies Latin Square property."""
        vars_dict = {"Row": [1, 2, 3], "Col": ["A", "B", "C"], "Treatment": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        # Each treatment should appear exactly once per row
        for row_val in [1, 2, 3]:
            row_data = design[design["Row"] == row_val]
            assert set(row_data["Treatment"]) == {"X", "Y", "Z"}

        # Each treatment should appear exactly once per column
        for col_val in ["A", "B", "C"]:
            col_data = design[design["Col"] == col_val]
            assert set(col_data["Treatment"]) == {"X", "Y", "Z"}

    def test_design_4_levels(self):
        """Test design with 4 levels."""
        vars_dict = {
            "Row": [1, 2, 3, 4],
            "Col": ["A", "B", "C", "D"],
            "Treatment": ["W", "X", "Y", "Z"],
        }
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        # Should have 16 experiments (4x4)
        assert len(design) == 16

    def test_design_all_combinations(self):
        """Test that design includes all row×column combinations."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        # Check all combinations exist
        for a_val in [1, 2, 3]:
            for b_val in [10, 20, 30]:
                matches = design[(design["A"] == a_val) & (design["B"] == b_val)]
                assert len(matches) == 1  # Exactly one match


class TestLatinSquarePivot:
    """Test pivot() method."""

    def test_pivot_creates_square(self):
        """Test that pivot creates a square table."""
        vars_dict = {"Row": [1, 2, 3], "Col": ["A", "B", "C"], "Treatment": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        pivot = ls.pivot()

        assert pivot.shape == (3, 3)

    def test_pivot_shows_treatments(self):
        """Test that pivot table shows treatment assignments."""
        vars_dict = {"Row": [1, 2, 3], "Col": ["A", "B", "C"], "Treatment": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        pivot = ls.pivot()

        # Pivot should show treatments
        all_values = pivot.values.flatten()
        assert set(all_values) == {"X", "Y", "Z"}


class TestLatinSquareFit:
    """Test fit() method."""

    def test_fit_basic(self):
        """Test basic fitting."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        # Create simple response data
        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        results = ls.fit(design, y)

        assert results is not None
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 9

    def test_fit_computes_effects(self):
        """Test that fit computes all effects."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        # Check effects dictionary exists
        assert hasattr(ls, "effects")
        assert "avg" in ls.effects
        assert "A" in ls.effects
        assert "B" in ls.effects
        assert "C" in ls.effects

    def test_fit_stores_results(self):
        """Test that fit stores results."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        assert hasattr(ls, "results")
        assert hasattr(ls, "y")
        assert ls.y == "Response"

    def test_fit_includes_effect_columns(self):
        """Test that results include effect columns."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        # Check effect columns exist
        assert "A_effect" in ls.results.columns
        assert "B_effect" in ls.results.columns
        assert "C_effect" in ls.results.columns
        assert "residuals" in ls.results.columns
        assert "avg" in ls.results.columns

    def test_fit_requires_dataframe(self):
        """Test that X must be a DataFrame."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        y = pd.Series([1, 2, 3], name="Y")

        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            ls.fit([[1, 10, "X"]], y)

    def test_fit_requires_series(self):
        """Test that y must be a Series."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        with pytest.raises(TypeError, match="y must be a pandas Series"):
            ls.fit(design, [1, 2, 3])

    def test_fit_requires_named_series(self):
        """Test that y must have a name."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62])  # No name

        with pytest.raises(AttributeError, match="y must have a name"):
            ls.fit(design, y)

    def test_fit_requires_matching_columns(self):
        """Test that X must have the required columns."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        # Design with wrong columns
        design = pd.DataFrame(
            {"Wrong1": [1, 2, 3], "Wrong2": [10, 20, 30], "Wrong3": ["X", "Y", "Z"]}
        )
        y = pd.Series([1, 2, 3], name="Y")

        with pytest.raises(ValueError, match="missing required columns"):
            ls.fit(design, y)

    def test_fit_requires_matching_lengths(self):
        """Test that X and y must have the same length."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([1, 2, 3], name="Y")  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            ls.fit(design, y)


class TestLatinSquareANOVA:
    """Test anova() method."""

    def test_anova_basic(self):
        """Test basic ANOVA computation."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        anova_table = ls.anova()

        assert isinstance(anova_table, pd.DataFrame)
        assert len(anova_table) == 4  # Three effects plus residuals

    def test_anova_column_names(self):
        """Test that ANOVA table has correct columns."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        anova_table = ls.anova()

        assert "Response effect" in anova_table.columns
        assert any("F-score" in col for col in anova_table.columns)
        assert "Significant" in anova_table.columns

    def test_anova_requires_fit(self):
        """Test that anova() requires fit() to be called first."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        with pytest.raises(AttributeError, match="Must call fit.*before anova"):
            ls.anova()


class TestLatinSquarePredict:
    """Test predict() method."""

    def test_predict_basic(self):
        """Test basic prediction."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        prediction = ls.predict([2, 20, "Y"])

        assert isinstance(prediction, (int, float, np.number))

    def test_predict_returns_reasonable_value(self):
        """Test that prediction is in reasonable range."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        prediction = ls.predict([2, 20, "Y"])

        # Prediction should be within range of observed values
        assert y.min() <= prediction <= y.max() * 1.2  # Allow some extrapolation

    def test_predict_requires_fit(self):
        """Test that predict() requires fit() to be called first."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        with pytest.raises(AttributeError, match="Must call fit.*before predict"):
            ls.predict([1, 10, "X"])

    def test_predict_requires_3_args(self):
        """Test that predict() requires exactly 3 arguments."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        with pytest.raises(ValueError, match="exactly 3 arguments"):
            ls.predict([1, 10])  # Only 2 args

        with pytest.raises(ValueError, match="exactly 3 arguments"):
            ls.predict([1, 10, "X", "Extra"])  # Too many args

    def test_predict_requires_known_levels(self):
        """Test that predict() requires known factor levels."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        with pytest.raises(ValueError, match="Unknown level"):
            ls.predict([999, 20, "Y"])  # Unknown level for A


class TestLatinSquareStringMethods:
    """Test __repr__ and __str__ methods."""

    def test_repr_before_fit(self):
        """Test __repr__ before fitting."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        repr_str = repr(ls)

        assert "LatinSquare" in repr_str
        assert "factors=3" in repr_str
        assert "levels=3" in repr_str
        assert "experiments=9" in repr_str

    def test_repr_after_fit(self):
        """Test __repr__ after fitting."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        repr_str = repr(ls)

        assert "fitted" in repr_str
        assert "9 observations" in repr_str

    def test_str_before_fit(self):
        """Test __str__ before fitting."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        str_str = str(ls)

        assert "Latin Square Design" in str_str
        assert "'A', 'B', 'C'" in str_str
        assert "not fitted" in str_str

    def test_str_after_fit(self):
        """Test __str__ after fitting."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        str_str = str(ls)

        assert "fitted" in str_str


class TestLatinSquareIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self):
        """Test complete workflow from design to prediction."""
        # Create design
        vars_dict = {
            "Temperature": [100, 150, 200],
            "Pressure": [1, 2, 3],
            "Catalyst": ["A", "B", "C"],
        }
        ls = LatinSquare(vars=vars_dict)

        # Generate design
        design = ls.design()
        assert len(design) == 9

        # Simulate experimental data
        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Yield")

        # Fit model
        results = ls.fit(design, y)
        assert results is not None

        # Run ANOVA
        anova = ls.anova()
        assert len(anova) == 4  # Three effects plus residuals

        # Make prediction
        prediction = ls.predict([150, 2, "B"])
        assert isinstance(prediction, (int, float, np.number))

    def test_workflow_with_shuffled_design(self):
        """Test workflow with shuffled design."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)

        # Shuffled design
        design = ls.design(shuffle=True)

        y = pd.Series([45, 52, 58, 50, 55, 60, 53, 58, 62], name="Response")
        ls.fit(design, y)

        # Should still work
        prediction = ls.predict([2, 20, "Y"])
        assert prediction is not None

    def test_4_level_complete_workflow(self):
        """Test complete workflow with 4 levels."""
        vars_dict = {
            "Row": [1, 2, 3, 4],
            "Col": ["A", "B", "C", "D"],
            "Treatment": ["W", "X", "Y", "Z"],
        }
        ls = LatinSquare(vars=vars_dict)

        design = ls.design()
        assert len(design) == 16

        # Create response data
        y = pd.Series(range(45, 61), name="Response")

        results = ls.fit(design, y)
        assert len(results) == 16

        anova = ls.anova()
        assert len(anova) == 4  # Three effects plus residuals

        prediction = ls.predict([2, "B", "X"])
        assert prediction is not None


class TestLatinSquareEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_minimum_2_levels(self):
        """Test with minimum 2 levels."""
        vars_dict = {"A": [1, 2], "B": [10, 20], "C": ["X", "Y"]}
        ls = LatinSquare(vars=vars_dict)

        design = ls.design()
        assert len(design) == 4  # 2x2

    def test_mixed_types(self):
        """Test with mixed data types."""
        vars_dict = {
            "Numeric": [1, 2, 3],
            "String": ["Low", "Med", "High"],
            "Mixed": [1.5, "B", None],
        }
        ls = LatinSquare(vars=vars_dict)

        design = ls.design()
        assert len(design) == 9

    def test_fit_with_zero_effects(self):
        """Test fitting with constant response (zero effects)."""
        vars_dict = {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["X", "Y", "Z"]}
        ls = LatinSquare(vars=vars_dict)
        design = ls.design()

        # Constant response
        y = pd.Series([50] * 9, name="Response")
        ls.fit(design, y)

        # All effects should be zero
        assert ls.effects["A"].abs().max() < 1e-10
        assert ls.effects["B"].abs().max() < 1e-10
        assert ls.effects["C"].abs().max() < 1e-10


class TestLatinSquareRegressions:
    """Regression tests for RNG handling and the ANOVA table."""

    VARS = {"A": [1, 2, 3, 4], "B": [10, 20, 30, 40], "C": ["W", "X", "Y", "Z"]}

    def _data(self):
        ls = LatinSquare(vars=self.VARS)
        design = ls.design()
        rng = np.random.default_rng(1)
        y = pd.Series(rng.normal(size=16) + np.repeat([0.0, 1.0, 2.0, 3.0], 4), name="Response")
        ls.fit(design, y)
        return ls, design, y

    def test_init_does_not_touch_global_rng(self):
        """__init__ used to call np.random.seed(42)."""
        np.random.seed(0)
        expected = np.random.rand(3)
        np.random.seed(0)
        ls = LatinSquare(vars=self.VARS)
        ls.design(shuffle=True)
        np.testing.assert_array_equal(np.random.rand(3), expected)

    def test_shuffle_reproducible(self):
        """Same random_state gives the same shuffled design; default is reproducible."""
        d1 = LatinSquare(vars=self.VARS).design(shuffle=True)
        d2 = LatinSquare(vars=self.VARS).design(shuffle=True)
        pd.testing.assert_frame_equal(d1, d2)
        d3 = LatinSquare(vars=self.VARS, random_state=7).design(shuffle=True)
        d4 = LatinSquare(vars=self.VARS, random_state=7).design(shuffle=True)
        pd.testing.assert_frame_equal(d3, d4)

    def test_shuffled_design_is_latin_square(self):
        """Shuffling keeps the Latin square property (also for ndarray levels)."""
        vars_dict = {k: np.array(v) for k, v in self.VARS.items()}
        design = LatinSquare(vars=vars_dict, random_state=3).design(shuffle=True)
        assert (design.groupby("A")["C"].nunique() == 4).all()
        assert (design.groupby("B")["C"].nunique() == 4).all()

    def test_anova_numeric_dtypes_and_pvalues(self):
        """anova() used to return an object-dtype table without p-values."""
        ls, _, _ = self._data()
        table = ls.anova()
        assert "p-value" in table.columns
        for col in ["df", "sum_sq", "mean_sq", "p-value"]:
            assert pd.api.types.is_numeric_dtype(table[col])
        fcol = next(c for c in table.columns if c.startswith("F-score"))
        assert pd.api.types.is_float_dtype(table[fcol])
        assert table["Significant"].dtype == bool
        assert list(table["df"]) == [3, 3, 3, 6]
        assert np.isnan(table[fcol].iloc[-1])

    def test_anova_matches_regression_f_test(self):
        """F and p agree with an extra-sum-of-squares test on a dummy-coded OLS fit."""
        from scipy import stats

        ls, design, y = self._data()
        table = ls.anova()
        fcol = next(c for c in table.columns if c.startswith("F-score"))

        X = pd.get_dummies(design.astype(str), drop_first=True).astype(float)
        X.insert(0, "const", 1.0)

        def rss(M):
            beta = np.linalg.lstsq(M, y.values, rcond=None)[0]
            r = y.values - M @ beta
            return r @ r

        rss_full = rss(X.values)
        df_resid = len(y) - X.shape[1]
        assert df_resid == 6
        for i, factor in enumerate("ABC"):
            reduced = X[[c for c in X.columns if not c.startswith(f"{factor}_")]].values
            F = ((rss(reduced) - rss_full) / 3) / (rss_full / df_resid)
            assert table[fcol].iloc[i] == pytest.approx(F)
            assert table["p-value"].iloc[i] == pytest.approx(stats.f.sf(F, 3, df_resid))
        assert table["sum_sq"].iloc[-1] == pytest.approx(rss_full)
        assert list(table["Significant"]) == [True, False, False, False]
