"""
ZENN Classifier for classification tasks.

Implements cross-zentropy loss with learnable temperatures for
heterogeneous data classification.
"""

from typing import Optional, Dict, Any, Union
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, validate_data
from sklearn.preprocessing import LabelEncoder

from pycse.sklearn.zenn.estimators.base import ZENNBase
from pycse.sklearn.zenn.losses.cross_zentropy import (
    cross_zentropy_loss,
    cross_zentropy_loss_with_temperature_posterior,
    compute_accuracy_with_temperature_marginalization,
)
from pycse.sklearn.zenn.temperature.learnable import (
    compute_temperature_posterior,
    marginalize_over_temperatures,
)
from pycse.sklearn.zenn.utils.thermodynamics import compute_configuration_probabilities


class ZENNClassifier(ClassifierMixin, ZENNBase):
    """
    Zentropy-Enhanced Neural Network Classifier.

    A thermodynamics-inspired classifier that uses zentropy theory to
    handle heterogeneous, multi-source data. Outperforms standard
    cross-entropy by learning latent temperature modes that capture
    data heterogeneity.

    Parameters
    ----------
    n_temperatures : int, default=4
        Number of learnable temperature modes. Higher values can capture
        more diverse data sources but may overfit.

    kb : float, default=1.0
        Boltzmann constant scaling.

    gamma : float, default=100.0
        Entropy fluctuation scale. Higher values reduce entropy influence.

    learning_rate : float, default=0.01
        Learning rate for network parameters.

    max_epochs : int, default=100
        Maximum training epochs.

    batch_size : int, default=32
        Mini-batch size.

    hidden_dims : tuple, default=(64, 64)
        Hidden layer dimensions for E and S networks.

    omega_train : float, default=5.0
        Temperature posterior sharpness during training.
        Higher values emphasize worst-case robustness.

    omega_test : float, default=1.0
        Temperature posterior sharpness during inference.

    random_state : int or None, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.

    n_classes_ : int
        Number of classes.

    temperatures_ : ndarray
        Learned temperature values.

    Examples
    --------
    >>> from zenn import ZENNClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_classes=5)
    >>> clf = ZENNClassifier(n_temperatures=3, max_epochs=50)
    >>> clf.fit(X, y)
    >>> predictions = clf.predict(X)
    >>> probabilities = clf.predict_proba(X)
    """

    _is_classifier = True
    _estimator_type = "classifier"

    def __init__(
        self,
        n_temperatures: int = 4,
        kb: float = 1.0,
        gamma: float = 100.0,
        learning_rate: float = 0.01,
        temperature_lr: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 32,
        hidden_dims: tuple = (64, 64),
        early_stopping: bool = False,
        patience: int = 10,
        validation_fraction: float = 0.1,
        omega_train: float = 5.0,
        omega_test: float = 1.0,
        label_smoothing: float = 0.0,
        activation: str = "tanh",
        network_type: str = "mlp",
        degree: int = 3,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_configs=1,  # Will be set to n_classes during fit
            hidden_dims=hidden_dims,
            n_temperatures=n_temperatures,
            kb=kb,
            gamma=gamma,
            learning_rate=learning_rate,
            temperature_lr=temperature_lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            patience=patience,
            validation_fraction=validation_fraction,
            convexity_lambda=0.0,  # Not used for classification
            omega_train=omega_train,
            omega_test=omega_test,
            activation=activation,
            network_type=network_type,
            degree=degree,
            random_state=random_state,
            verbose=verbose,
        )
        self.label_smoothing = label_smoothing

    def _validate_data(self, X, y):
        """Validate and encode labels."""
        X, y = validate_data(self, X, y, reset=True)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Convert to one-hot encoding
        y_onehot = np.zeros((len(y_encoded), self.n_classes_))
        y_onehot[np.arange(len(y_encoded)), y_encoded] = 1

        return X, y_onehot

    def _get_n_outputs(self, y: jnp.ndarray) -> int:
        """Get number of classes."""
        return y.shape[1]

    def _compute_loss(
        self,
        params: Dict[str, Any],
        X: jnp.ndarray,
        y: jnp.ndarray,
        temperatures: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute cross-zentropy loss with temperature posterior."""
        batch_size = X.shape[0]
        n_temps = temperatures.shape[0]

        def cz_at_temp(T):
            T_arr = jnp.full((batch_size,), T)
            outputs = self.model_.apply(params, X, T_arr)
            return cross_zentropy_loss(
                outputs["E"],
                outputs["S"],
                y,
                T,
                self.kb,
                self.gamma,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )

        # Compute loss at each temperature
        cz_losses = jax.vmap(cz_at_temp)(temperatures)  # (n_temps, batch_size)

        # Compute temperature posterior
        q = compute_temperature_posterior(cz_losses, self.omega_train)

        # Weighted average loss
        weighted_loss = jnp.sum(q * cz_losses, axis=0)
        return jnp.mean(weighted_loss)

    def _predict_impl(self, X: jnp.ndarray) -> jnp.ndarray:
        """Internal prediction implementation using lowest temperature."""
        batch_size = X.shape[0]

        # Use the lowest temperature for prediction
        # Lower temperatures give sharper, more discriminative predictions
        T = self.temperatures_[0]  # Lowest temperature
        T_arr = jnp.full((batch_size,), T)
        outputs = self.model_.apply(self.params_, X, T_arr)
        p, _ = compute_configuration_probabilities(
            outputs["E"], outputs["S"], T, self.kb, self.gamma
        )
        return p

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = jnp.array(X, dtype=jnp.float32)

        probs = self._predict_impl(X)
        predictions = jnp.argmax(probs, axis=-1)

        return self._label_encoder.inverse_transform(np.array(predictions))

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = jnp.array(X, dtype=jnp.float32)

        probs = self._predict_impl(X)
        return np.array(probs)

    def predict_with_temperature(
        self,
        X,
        T: float,
    ) -> np.ndarray:
        """
        Predict class labels at a specific temperature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        T : float
            Temperature value.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        T_arr = jnp.full((X.shape[0],), T)
        outputs = self.model_.apply(self.params_, X, T_arr)
        p, _ = compute_configuration_probabilities(
            outputs["E"], outputs["S"], T, self.kb, self.gamma
        )
        predictions = jnp.argmax(p, axis=-1)

        return self._label_encoder.inverse_transform(np.array(predictions))

    def get_temperature_posterior(
        self,
        X,
        y,
    ) -> np.ndarray:
        """
        Get the temperature posterior q(T|x,y) for labeled samples.

        This reveals which temperature mode each sample likely belongs to,
        useful for analyzing data heterogeneity.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        q : ndarray of shape (n_temperatures, n_samples)
            Temperature posterior probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        # Encode labels
        y_encoded = self._label_encoder.transform(y)
        y_onehot = np.zeros((len(y_encoded), self.n_classes_))
        y_onehot[np.arange(len(y_encoded)), y_encoded] = 1
        y_onehot = jnp.array(y_onehot)

        batch_size = X.shape[0]

        def cz_at_temp(T):
            T_arr = jnp.full((batch_size,), T)
            outputs = self.model_.apply(self.params_, X, T_arr)
            return cross_zentropy_loss(
                outputs["E"],
                outputs["S"],
                y_onehot,
                T,
                self.kb,
                self.gamma,
                reduction="none",
            )

        cz_losses = jax.vmap(cz_at_temp)(self.temperatures_)
        q = compute_temperature_posterior(cz_losses, self.omega_test)

        return np.array(q)

    def score(self, X, y) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def predict_with_uncertainty(
        self,
        X,
        return_decomposition: bool = False,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Predict class labels with uncertainty estimates.

        This method returns predictions along with multiple uncertainty
        measures derived from ZENN's thermodynamic framework.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        return_decomposition : bool, default=False
            If True, return full uncertainty decomposition.
            If False, return simplified output.

        Returns
        -------
        dict with:
            - 'predictions': Predicted class labels
            - 'probabilities': Class probabilities
            - 'confidence': Max probability (higher = more confident)
            - 'predictive_entropy': Entropy over classes (aleatoric)
            - 'config_entropy': Entropy over configs (epistemic)
            If return_decomposition=True, also includes:
            - 'epistemic': Configuration disagreement
            - 'aleatoric': Expected entropy from S networks
            - 'total_uncertainty': Combined uncertainty
        """
        from pycse.sklearn.zenn.analysis.uncertainty import (
            configuration_entropy,
            predictive_entropy,
            confidence_score,
            epistemic_uncertainty,
            aleatoric_uncertainty,
        )

        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        # Get predictions and probabilities
        probs = self._predict_impl(X)
        predictions = jnp.argmax(probs, axis=-1)
        pred_labels = self._label_encoder.inverse_transform(np.array(predictions))

        # Basic uncertainty measures
        conf = confidence_score(probs)
        H_pred = predictive_entropy(probs)
        H_config = configuration_entropy(probs)

        result = {
            'predictions': pred_labels,
            'probabilities': np.array(probs),
            'confidence': np.array(conf),
            'predictive_entropy': np.array(H_pred),
            'config_entropy': np.array(H_config),
        }

        if return_decomposition:
            # Get full energy landscape for decomposition
            T = self.temperatures_[0]
            T_arr = jnp.full((X.shape[0],), T)
            outputs = self.model_.apply(self.params_, X, T_arr)

            epist = epistemic_uncertainty(outputs['E'], probs)
            aleat = aleatoric_uncertainty(outputs['S'], probs)
            total = jnp.sqrt(epist ** 2 + aleat ** 2)

            result['epistemic'] = np.array(epist)
            result['aleatoric'] = np.array(aleat)
            result['total_uncertainty'] = np.array(total)

        return result

    def get_epistemic_uncertainty(self, X) -> np.ndarray:
        """
        Get epistemic uncertainty (configuration disagreement).

        High epistemic uncertainty means the model's configurations
        disagree about the prediction - more data might help.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        uncertainty : ndarray of shape (n_samples,)
            Epistemic uncertainty for each sample.
        """
        from pycse.sklearn.zenn.analysis.uncertainty import configuration_entropy

        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        probs = self._predict_impl(X)
        H = configuration_entropy(probs)
        return np.array(H)

    def get_aleatoric_uncertainty(self, X) -> np.ndarray:
        """
        Get aleatoric uncertainty (inherent data noise).

        This is derived from the learned entropy networks S(k).
        High aleatoric uncertainty means inherent noise in the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        uncertainty : ndarray of shape (n_samples,)
            Aleatoric uncertainty for each sample.
        """
        from pycse.sklearn.zenn.analysis.uncertainty import aleatoric_uncertainty

        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        T = self.temperatures_[0]
        T_arr = jnp.full((X.shape[0],), T)
        outputs = self.model_.apply(self.params_, X, T_arr)
        probs = self._predict_impl(X)

        U = aleatoric_uncertainty(outputs['S'], probs)
        return np.array(U)

    def get_uncertainty_decomposition(self, X) -> Dict[str, np.ndarray]:
        """
        Get full uncertainty decomposition into epistemic and aleatoric.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        dict with:
            - 'epistemic': Configuration disagreement
            - 'aleatoric': Expected entropy from S networks
            - 'total': Combined uncertainty (sqrt of sum of squares)
        """
        from pycse.sklearn.zenn.analysis.uncertainty import total_uncertainty

        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        T = self.temperatures_[0]
        T_arr = jnp.full((X.shape[0],), T)
        outputs = self.model_.apply(self.params_, X, T_arr)
        probs = self._predict_impl(X)

        decomp = total_uncertainty(outputs['E'], outputs['S'], probs)
        return {k: np.array(v) for k, v in decomp.items()}

    def get_confidence(self, X) -> np.ndarray:
        """
        Get prediction confidence (max class probability).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        confidence : ndarray of shape (n_samples,)
            Confidence scores (0 to 1).
        """
        check_is_fitted(self)
        X = check_array(X)
        X = jnp.array(X, dtype=jnp.float32)

        probs = self._predict_impl(X)
        return np.array(jnp.max(probs, axis=-1))

    def calibration_curve(
        self,
        X,
        y,
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Compute calibration curve for the classifier.

        A well-calibrated classifier has confidence matching accuracy:
        samples with 80% confidence should be correct 80% of the time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels.

        n_bins : int, default=10
            Number of bins for calibration curve.

        Returns
        -------
        dict with:
            - 'ece': Expected Calibration Error (lower = better)
            - 'mce': Maximum Calibration Error
            - 'bin_confidences': Mean confidence per bin
            - 'bin_accuracies': Accuracy per bin
            - 'bin_counts': Number of samples per bin
        """
        from pycse.sklearn.zenn.analysis.uncertainty import uncertainty_calibration

        predictions = self.predict(X)
        confidences = self.get_confidence(X)

        return uncertainty_calibration(predictions, confidences, y, n_bins)

    def get_calibration_metrics(
        self,
        X,
        y,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Compute summary calibration metrics for the classifier.

        Use this to check if confidence matches accuracy.
        Ideal: ECE ≈ 0, coverage at each confidence level matches accuracy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        n_bins : int, default=10
            Number of bins for calibration.

        Returns
        -------
        dict with:
            - 'accuracy': Overall accuracy
            - 'mean_confidence': Mean prediction confidence
            - 'ece': Expected Calibration Error (lower = better calibrated)
            - 'mce': Maximum Calibration Error
            - 'overconfidence': mean_confidence - accuracy (positive = overconfident)
        """
        calib = self.calibration_curve(X, y, n_bins)

        predictions = self.predict(X)
        confidences = self.get_confidence(X)
        accuracy = np.mean(predictions == y)
        mean_confidence = np.mean(confidences)

        return {
            'accuracy': float(accuracy),
            'mean_confidence': float(mean_confidence),
            'ece': float(calib['ece']),
            'mce': float(calib['mce']),
            'overconfidence': float(mean_confidence - accuracy),
        }

    def selective_predict(
        self,
        X,
        coverage: float = 0.9,
    ) -> Dict[str, np.ndarray]:
        """
        Selective prediction: only predict on most confident samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        coverage : float, default=0.9
            Fraction of samples to keep (most confident).

        Returns
        -------
        dict with:
            - 'predictions': Predictions for kept samples (None for rejected)
            - 'kept_mask': Boolean mask of kept samples
            - 'confidences': Confidence for all samples
            - 'threshold': Confidence threshold used
        """
        check_is_fitted(self)

        predictions = self.predict(X)
        confidences = self.get_confidence(X)

        n_keep = int(coverage * len(X))
        threshold = np.sort(confidences)[-n_keep] if n_keep > 0 else 1.0

        kept_mask = confidences >= threshold

        # Create output with None for rejected samples
        selective_predictions = np.where(
            kept_mask, predictions, np.array([None] * len(X))
        )

        return {
            'predictions': selective_predictions,
            'kept_mask': kept_mask,
            'confidences': confidences,
            'threshold': threshold,
        }
