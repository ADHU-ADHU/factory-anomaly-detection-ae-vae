"""Normal-only latent fitting and calibration; larger scores indicate anomalies."""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.mixture import GaussianMixture


def _matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite matrix of latent vectors")
    return values


class LatentScorer:
    def __init__(self, method="mahalanobis", components=2, seed=42):
        if method not in ("mahalanobis", "gmm"):
            raise ValueError("Latent method must be mahalanobis or gmm")
        self.method = method
        self.components = components
        self.seed = seed
        self.model = None

    def fit(self, normal_latents):
        normal_latents = _matrix(normal_latents)
        if len(normal_latents) < 2:
            raise ValueError("At least two normal training examples are required")
        if self.method == "mahalanobis":
            self.model = LedoitWolf().fit(normal_latents)
            # Retain a positive floor even for an entirely constant training set.
            covariance = self.model.covariance_
            floor = max(float(np.trace(covariance) / covariance.shape[0]) * 1e-6, 1e-8)
            self.covariance = covariance + floor * np.eye(covariance.shape[0])
        else:
            if not 1 <= self.components <= len(normal_latents):
                raise ValueError("GMM components must be between 1 and the normal training count")
            self.model = GaussianMixture(
                n_components=self.components, covariance_type="full", reg_covar=1e-5,
                random_state=self.seed,
            ).fit(normal_latents)
        return self

    def score(self, latents):
        if self.model is None:
            raise ValueError("Fit the scorer on normal training data first")
        latents = _matrix(latents)
        if self.method == "gmm":
            return -self.model.score_samples(latents)
        delta = latents - self.model.location_
        solved = np.linalg.solve(self.covariance, delta.T).T
        return np.sqrt(np.maximum(np.einsum("ij,ij->i", delta, solved), 0))


def calibrate_threshold(normal_scores, quantile=0.99):
    scores = np.asarray(normal_scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Calibration requires nonempty finite normal scores")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    return float(np.quantile(scores, quantile))


def evaluation_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires matching score/label vectors containing both 0 and 1")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Evaluation scores and threshold must be finite")
    predictions = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "threshold": float(threshold), "normal_count": int(sum(labels == 0)),
        "anomaly_count": int(sum(labels == 1)),
    }
