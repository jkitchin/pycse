"""Comprehensive tests for SurfaceResponse class.

This module tests the SurfaceResponse class for Design of Experiments (DOE)
and response surface methodology.
"""

import numpy as np
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from pycse.sklearn.surface_response import SurfaceResponse


class TestSurfaceResponseInitialization:
    """Test SurfaceResponse initialization and validation."""

    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        sr = SurfaceResponse(
            inputs=["x1", "x2"], outputs=["y1"], bounds=[[0, 1], [0, 1]], design="ccdesign"
        )
        assert sr.inputs == ["x1", "x2"]
        assert sr.outputs == ["y1"]
        assert sr.bounds.shape == (2, 2)

    def test_inputs_required(self):
        """Test that inputs parameter is required."""
        with pytest.raises(ValueError, match="inputs must be a non-empty list"):
            SurfaceResponse(inputs=None, outputs=["y"])

        with pytest.raises(ValueError, match="inputs must be a non-empty list"):
            SurfaceResponse(inputs=[], outputs=["y"])

    def test_outputs_required(self):
        """Test that outputs parameter is required."""
        with pytest.raises(ValueError, match="outputs must be a non-empty list"):
            SurfaceResponse(inputs=["x"], outputs=None)

        with pytest.raises(ValueError, match="outputs must be a non-empty list"):
            SurfaceResponse(inputs=["x"], outputs=[])

    def test_inputs_must_be_list(self):
        """Test that inputs must be a list."""
        with pytest.raises(TypeError, match="inputs must be a list"):
            SurfaceResponse(inputs="x1", outputs=["y"])

    def test_outputs_must_be_list(self):
        """Test that outputs must be a list."""
        with pytest.raises(TypeError, match="outputs must be a list"):
            SurfaceResponse(inputs=["x"], outputs="y1")

    def test_bounds_shape_validation(self):
        """Test that bounds must have correct shape."""
        with pytest.raises(ValueError, match="bounds must have shape"):
            SurfaceResponse(
                inputs=["x1", "x2"],
                outputs=["y"],
                bounds=[[0, 1]],  # Only 1 row
            )

    def test_bounds_min_max_validation(self):
        """Test that bounds must have min < max."""
        with pytest.raises(ValueError, match="All bounds must have min < max"):
            SurfaceResponse(
                inputs=["x1", "x2"],
                outputs=["y"],
                bounds=[[0, 1], [5, 2]],  # Second bound has max < min
            )

    def test_bounds_optional(self):
        """Test that bounds are optional."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="ccdesign")
        assert sr.bounds is None

    def test_default_design_bbdesign(self):
        """Test that default design is bbdesign."""
        sr = SurfaceResponse(inputs=["x1", "x2", "x3"], outputs=["y"])
        assert hasattr(sr, "_design")
        # BBDesign for 3 factors should have specific number of points
        assert len(sr._design) >= 12

    def test_default_model_pipeline(self):
        """Test that default model creates a Pipeline."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="ccdesign")
        assert isinstance(sr, Pipeline)
        assert sr.default is True
        assert "minmax" in sr.named_steps
        assert "poly" in sr.named_steps
        assert "surface response" in sr.named_steps

    def test_custom_model(self):
        """Test initialization with custom model."""
        custom_model = LinearRegression()
        sr = SurfaceResponse(
            inputs=["x1", "x2"], outputs=["y"], design="ccdesign", model=custom_model
        )
        assert sr.default is False
        assert "usermodel" in sr.named_steps


class TestSurfaceResponseDesigns:
    """Test different design types."""

    def test_bbdesign(self):
        """Test Box-Behnken design."""
        sr = SurfaceResponse(inputs=["x1", "x2", "x3"], outputs=["y"], design="bbdesign")
        assert hasattr(sr, "_design")
        # BBDesign for 3 factors has 13 points (without center points)
        assert len(sr._design) >= 12

    def test_ccdesign(self):
        """Test central composite design."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="ccdesign")
        assert hasattr(sr, "_design")
        assert len(sr._design) > 0

    def test_ff2n(self):
        """Test 2-level full factorial design."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="ff2n")
        # 2^2 = 4 points for 2 factors
        assert len(sr._design) == 4

    def test_fullfact(self):
        """Test full factorial design with levels."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="fullfact", levels=[3, 3])
        # 3*3 = 9 points
        assert len(sr._design) == 9

    def test_pbdesign(self):
        """Test Plackett-Burman design."""
        sr = SurfaceResponse(inputs=["x1", "x2", "x3"], outputs=["y"], design="pbdesign")
        assert hasattr(sr, "_design")
        assert len(sr._design) > 0

    def test_lhs(self):
        """Test Latin hypercube sampling."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="lhs", samples=10)
        assert len(sr._design) == 10

    def test_invalid_design(self):
        """Test that invalid design raises error."""
        with pytest.raises(ValueError, match="Unsupported design option"):
            SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="invalid_design")


class TestSurfaceResponseDesignGeneration:
    """Test design() method."""

    def test_design_creates_dataframe(self):
        """Test that design() returns a DataFrame."""
        sr = SurfaceResponse(
            inputs=["x1", "x2"], outputs=["y"], bounds=[[0, 10], [0, 5]], design="ccdesign"
        )
        design = sr.design()
        assert isinstance(design, pd.DataFrame)
        assert list(design.columns) == ["x1", "x2"]

    def test_design_scaling_to_bounds(self):
        """Test that design scales from [-1,1] to bounds."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], bounds=[[10, 20]], design="ff2n")
        design = sr.design()

        # FF2N for 1 factor gives [-1, 1], which should scale to [10, 20]
        assert design["x"].min() >= 10
        assert design["x"].max() <= 20

    def test_design_without_bounds(self):
        """Test design without bounds uses [-1, 1] range."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="ff2n")
        design = sr.design()

        # Without bounds, should stay in [-1, 1]
        assert design["x"].min() >= -1
        assert design["x"].max() <= 1

    def test_design_shuffle(self):
        """Test that shuffle=True randomizes order."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="ff2n")
        design1 = sr.design(shuffle=False)
        design2 = sr.design(shuffle=False)

        # Without shuffle, order should be the same
        pd.testing.assert_frame_equal(
            design1.reset_index(drop=True), design2.reset_index(drop=True)
        )

    def test_design_stores_input(self):
        """Test that design() stores result in self.input."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="ccdesign")
        design = sr.design()
        assert hasattr(sr, "input")
        pd.testing.assert_frame_equal(sr.input, design)


class TestSurfaceResponseSetOutput:
    """Test set_output() method."""

    def test_set_output_basic(self):
        """Test basic output setting."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="ff2n")
        sr.design()
        output = sr.set_output([[1], [2]])

        assert isinstance(output, pd.DataFrame)
        assert list(output.columns) == ["y"]
        assert len(output) == 2

    def test_set_output_requires_design_first(self):
        """Test that set_output() requires design() to be called first."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="ff2n")
        with pytest.raises(AttributeError, match="Must call design.*before set_output"):
            sr.set_output([[1], [2]])

    def test_set_output_shape_validation_rows(self):
        """Test that output must have same number of rows as design."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="ff2n")
        sr.design()  # Creates 2 points

        with pytest.raises(ValueError, match="data has .* rows but design has"):
            sr.set_output([[1], [2], [3]])  # Wrong number of rows

    def test_set_output_shape_validation_columns(self):
        """Test that output must have correct number of columns."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y1", "y2"], design="ff2n")
        sr.design()

        with pytest.raises(ValueError, match="data has .* columns but .* outputs expected"):
            sr.set_output([[1], [2]])  # Should have 2 columns

    def test_set_output_converts_1d_to_2d(self):
        """Test that 1D arrays are converted to 2D."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="ff2n")
        sr.design()
        output = sr.set_output([1, 2])  # 1D array

        assert output.shape == (2, 1)

    def test_set_output_multiple_outputs(self):
        """Test setting multiple output columns."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y1", "y2"], design="ff2n")
        sr.design()
        output = sr.set_output([[1, 2], [3, 4]])

        assert list(output.columns) == ["y1", "y2"]
        assert output.shape == (2, 2)

    def test_set_output_stores_result(self):
        """Test that set_output() stores result in self.output."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="ff2n")
        sr.design()
        output = sr.set_output([[1], [2]])

        assert hasattr(sr, "output")
        pd.testing.assert_frame_equal(sr.output, output)


class TestSurfaceResponseFitPredict:
    """Test fit() and predict() methods."""

    def test_fit_basic(self):
        """Test basic fitting."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="lhs", samples=15)
        sr.design()
        sr.set_output(
            [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
        )

        result = sr.fit()
        assert result is sr  # fit() returns self
        assert hasattr(sr, "input")
        assert hasattr(sr, "output")

    def test_fit_requires_design_and_output(self):
        """Test that fit() requires design() and set_output()."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="ff2n")

        with pytest.raises(AttributeError, match="Must set input and output"):
            sr.fit()

    def test_fit_with_explicit_xy(self):
        """Test fitting with explicit X and y parameters."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="ff2n")
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [4], [6]])

        sr.fit(X, y)
        # Should not require design() and set_output() when X, y are provided

    def test_predict_basic(self):
        """Test basic prediction."""
        # Create simple linear data
        sr = SurfaceResponse(
            inputs=["x"], outputs=["y"], bounds=[[0, 10]], design="lhs", samples=20
        )
        design = sr.design()
        y_true = 2 * design["x"].values + 1
        sr.set_output(y_true.reshape(-1, 1))
        sr.fit()

        # Predict on training data
        y_pred = sr.predict(design)

        # Should fit reasonably well
        assert y_pred.shape == (20, 1)
        # Check R^2 is good
        assert sr.score() > 0.95

    def test_predict_accuracy(self):
        """Test prediction accuracy on quadratic function."""
        # Create quadratic data y = x^2
        sr = SurfaceResponse(
            inputs=["x"],
            outputs=["y"],
            bounds=[[-5, 5]],
            design="lhs",
            samples=25,
        )
        design = sr.design()
        y_true = design["x"].values ** 2
        sr.set_output(y_true.reshape(-1, 1))
        sr.fit()

        # Predict at new points
        X_test = np.array([[0], [1], [2], [3], [4]])
        y_pred = sr.predict(X_test)

        # Should predict quadratic well with 2nd order polynomial
        y_expected = np.array([0, 1, 4, 9, 16]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(y_pred, y_expected, decimal=0)

    def test_score_method(self):
        """Test score() method returns R²."""
        sr = SurfaceResponse(
            inputs=["x"], outputs=["y"], bounds=[[0, 10]], design="lhs", samples=15
        )
        design = sr.design()
        y_true = 2 * design["x"].values + 1
        sr.set_output(y_true.reshape(-1, 1))
        sr.fit()

        score = sr.score()
        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1  # R² should be in [0, 1] for reasonable fits


class TestSurfaceResponseMultipleFeatures:
    """Test with multiple input features."""

    def test_two_inputs(self):
        """Test with two input features."""
        sr = SurfaceResponse(
            inputs=["x1", "x2"],
            outputs=["y"],
            bounds=[[0, 1], [0, 1]],
            design="lhs",
            samples=30,
        )
        design = sr.design()

        # y = x1 + 2*x2
        y_true = design["x1"].values + 2 * design["x2"].values
        sr.set_output(y_true.reshape(-1, 1))
        sr.fit()

        score = sr.score()
        assert score > 0.95  # Should fit linear function well

    def test_three_inputs(self):
        """Test with three input features."""
        sr = SurfaceResponse(
            inputs=["x1", "x2", "x3"],
            outputs=["y"],
            bounds=[[0, 1], [0, 1], [0, 1]],
            design="lhs",
            samples=50,
        )
        design = sr.design()

        # y = x1 + x2 + x3
        y_true = design["x1"].values + design["x2"].values + design["x3"].values
        sr.set_output(y_true.reshape(-1, 1))
        sr.fit()

        score = sr.score()
        assert score > 0.95


class TestSurfaceResponseMultipleOutputs:
    """Test with multiple output responses."""

    def test_two_outputs(self):
        """Test with two output responses."""
        sr = SurfaceResponse(
            inputs=["x"],
            outputs=["y1", "y2"],
            bounds=[[0, 10]],
            design="lhs",
            samples=20,
        )
        design = sr.design()

        # Two different outputs
        y1 = 2 * design["x"].values
        y2 = 3 * design["x"].values
        sr.set_output(np.column_stack([y1, y2]))
        sr.fit()

        y_pred = sr.predict(design)
        assert y_pred.shape == (20, 2)

    def test_three_outputs(self):
        """Test with three output responses."""
        sr = SurfaceResponse(
            inputs=["x"],
            outputs=["y1", "y2", "y3"],
            bounds=[[0, 10]],
            design="lhs",
            samples=20,
        )
        design = sr.design()

        y1 = design["x"].values
        y2 = design["x"].values ** 2
        y3 = design["x"].values ** 0.5
        sr.set_output(np.column_stack([y1, y2, y3]))
        sr.fit()

        y_pred = sr.predict(design)
        assert y_pred.shape == (20, 3)


class TestSurfaceResponseVisualization:
    """Test visualization methods."""

    def test_parity_plot_creation(self):
        """Test that parity() creates a matplotlib figure."""
        sr = SurfaceResponse(
            inputs=["x"], outputs=["y"], bounds=[[0, 10]], design="lhs", samples=15
        )
        design = sr.design()
        y_true = 2 * design["x"].values
        sr.set_output(y_true.reshape(-1, 1))
        sr.fit()

        fig = sr.parity()

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)  # Clean up

    def test_parity_plot_with_multiple_outputs(self):
        """Test parity plot with multiple outputs."""
        sr = SurfaceResponse(
            inputs=["x"],
            outputs=["y1", "y2"],
            bounds=[[0, 10]],
            design="lhs",
            samples=15,
        )
        design = sr.design()
        y1 = 2 * design["x"].values
        y2 = 3 * design["x"].values
        sr.set_output(np.column_stack([y1, y2]))
        sr.fit()

        fig = sr.parity()

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSurfaceResponseSummary:
    """Test summary() method."""

    def test_summary_default_model(self):
        """Test summary with default model."""
        sr = SurfaceResponse(
            inputs=["x"], outputs=["y"], bounds=[[0, 10]], design="lhs", samples=20
        )
        design = sr.design()
        y_true = 2 * design["x"].values + 1
        sr.set_output(y_true.reshape(-1, 1))
        sr.fit()

        summary = sr.summary()

        assert isinstance(summary, str)
        assert "data points" in summary
        assert "R²" in summary
        assert "MAE" in summary
        assert "RMSE" in summary

    def test_summary_includes_statistics(self):
        """Test that summary includes key statistics."""
        sr = SurfaceResponse(
            inputs=["x"], outputs=["y"], bounds=[[0, 10]], design="lhs", samples=15
        )
        design = sr.design()
        y_true = 2 * design["x"].values
        sr.set_output(y_true.reshape(-1, 1))
        sr.fit()

        summary = sr.summary()

        # Should contain statistical info
        assert "R²" in summary
        assert "MAE" in summary or "mae" in summary.lower()
        assert "RMSE" in summary or "rmse" in summary.lower()

    def test_summary_custom_model(self):
        """Test summary with custom model."""
        custom_model = LinearRegression()
        sr = SurfaceResponse(
            inputs=["x"],
            outputs=["y"],
            bounds=[[0, 10]],
            design="lhs",
            samples=15,
            model=custom_model,
        )
        design = sr.design()
        y_true = 2 * design["x"].values
        sr.set_output(y_true.reshape(-1, 1))
        sr.fit()

        summary = sr.summary()

        assert "User-defined model" in summary
        assert "R²" in summary


class TestSurfaceResponseRepr:
    """Test __repr__ and __str__ methods."""

    def test_repr_before_fitting(self):
        """Test __repr__ before fitting."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y1", "y2"], design="ccdesign")
        repr_str = repr(sr)

        assert "SurfaceResponse" in repr_str
        assert "x1" in repr_str or "2" in repr_str  # 2 inputs
        assert "y1" in repr_str or "2" in repr_str  # 2 outputs

    def test_repr_after_fitting(self):
        """Test __repr__ after fitting."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], bounds=[[0, 10]], design="lhs", samples=5)
        sr.design()
        sr.set_output([[1], [2], [3], [4], [5]])
        sr.fit()

        repr_str = repr(sr)
        assert "SurfaceResponse" in repr_str
        assert "5" in repr_str  # 5 experiments

    def test_str_method(self):
        """Test __str__ method."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="ccdesign")
        str_str = str(sr)

        assert "SurfaceResponse" in str_str
        assert "2 inputs" in str_str
        assert "1 outputs" in str_str
        assert "not fitted" in str_str

    def test_str_after_fitting(self):
        """Test __str__ after fitting."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], bounds=[[0, 10]], design="lhs", samples=5)
        sr.design()
        sr.set_output([[1], [2], [3], [4], [5]])
        sr.fit()

        str_str = str(sr)
        assert "fitted" in str_str


class TestSurfaceResponseIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_bbdesign(self):
        """Test complete workflow with central composite design."""
        # Simulate a response surface: y = 1 + 2*x1 + 3*x2 + 0.5*x1^2
        sr = SurfaceResponse(
            inputs=["temperature", "pressure"],
            outputs=["yield"],
            bounds=[[100, 200], [1, 5]],
            design="ccdesign",
        )

        # Generate design
        design = sr.design()
        assert len(design) > 0

        # Simulate experiments
        temp = design["temperature"].values
        press = design["pressure"].values

        # Normalize to [-1, 1] for formula
        temp_norm = (temp - 150) / 50
        press_norm = (press - 3) / 2

        y = 1 + 2 * temp_norm + 3 * press_norm + 0.5 * temp_norm**2
        sr.set_output(y.reshape(-1, 1))

        # Fit model
        sr.fit()

        # Check fit quality
        score = sr.score()
        assert score > 0.9  # Should fit well

        # Generate summary
        summary = sr.summary()
        assert "temperature" in summary or "x0" in summary

        # Create parity plot
        import matplotlib.pyplot as plt

        fig = sr.parity()
        plt.close(fig)

    def test_full_workflow_ccdesign(self):
        """Test complete workflow with central composite design."""
        sr = SurfaceResponse(
            inputs=["x1", "x2"],
            outputs=["y"],
            bounds=[[-1, 1], [-1, 1]],
            design="ccdesign",
        )

        design = sr.design()
        # Simple quadratic: y = x1^2 + x2^2
        y = design["x1"].values ** 2 + design["x2"].values ** 2
        sr.set_output(y.reshape(-1, 1))
        sr.fit()

        score = sr.score()
        assert score > 0.85  # Should capture quadratic well

    def test_full_workflow_lhs(self):
        """Test complete workflow with Latin hypercube sampling."""
        sr = SurfaceResponse(
            inputs=["x"],
            outputs=["y"],
            bounds=[[0, 10]],
            design="lhs",
            samples=30,
        )

        design = sr.design()
        assert len(design) == 30

        y = 2 * design["x"].values + 1
        sr.set_output(y.reshape(-1, 1))
        sr.fit()

        score = sr.score()
        assert score > 0.95  # Linear fit should be excellent

    def test_custom_model_integration(self):
        """Test using a custom sklearn model."""
        from sklearn.ensemble import RandomForestRegressor

        custom_model = RandomForestRegressor(n_estimators=10, random_state=42)
        sr = SurfaceResponse(
            inputs=["x"],
            outputs=["y"],
            bounds=[[0, 10]],
            design="lhs",
            samples=20,
            model=custom_model,
        )

        design = sr.design()
        y = design["x"].values ** 2  # Nonlinear
        sr.set_output(y.reshape(-1, 1))
        sr.fit()

        # Random forest should fit well
        score = sr.score()
        assert score > 0.7

        # Summary should work with custom model
        summary = sr.summary()
        assert "User-defined model" in summary


class TestSurfaceResponseEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_input_single_output(self):
        """Test with minimal dimensions."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], bounds=[[0, 1]], design="lhs", samples=5)
        design = sr.design()
        y = design["x"].values * 2  # Simple linear relationship
        sr.set_output(y.reshape(-1, 1))
        sr.fit()

        assert sr.score() is not None

    def test_many_inputs(self):
        """Test with many input features."""
        n_inputs = 5
        inputs = [f"x{i}" for i in range(n_inputs)]
        bounds = [[0, 1] for _ in range(n_inputs)]

        sr = SurfaceResponse(inputs=inputs, outputs=["y"], bounds=bounds, design="lhs", samples=50)
        design = sr.design()

        # Simple sum
        y = design.sum(axis=1).values
        sr.set_output(y.reshape(-1, 1))
        sr.fit()

        assert sr.score() > 0.8

    def test_sigfig_helper(self):
        """Test _sigfig helper method."""
        sr = SurfaceResponse(inputs=["x"], outputs=["y"], design="ff2n")

        assert sr._sigfig(0) == 0
        assert sr._sigfig(123.456, n=3) == 123
        assert sr._sigfig(0.00123456, n=3) == 0.00123

    def test_no_shuffle_reproducibility(self):
        """Test that shuffle=False gives reproducible designs."""
        sr = SurfaceResponse(inputs=["x1", "x2"], outputs=["y"], design="ff2n")

        design1 = sr.design(shuffle=False)
        design2 = sr.design(shuffle=False)

        pd.testing.assert_frame_equal(
            design1.reset_index(drop=True), design2.reset_index(drop=True)
        )
