"""Discrete-time monthly default hazard modeling.

The model uses a sklearn preprocessing pipeline so numeric and categorical
features are treated consistently in fitting and projection. Loan age advances
one month in every forecast step, while other supplied covariates remain fixed.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Self, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .survival import ArrayLike, PDCurves, conditional_hazards_to_pd


MODEL_ARTIFACT_TYPE = "ifrs9_ecl.discrete_time_hazard"
MODEL_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class HazardFitDiagnostics:
    """Weighted sample composition and solver convergence evidence."""

    observations: int
    events: int
    non_events: int
    sample_weight_provided: bool
    weight_sum: float
    weighted_events: float
    weighted_non_events: float
    transformed_features: int
    solver: str
    max_iter: int
    n_iter: tuple[int, ...]
    converged: bool

    def to_dict(self) -> dict[str, int | float | bool | str | tuple[int, ...]]:
        """Return JSON-ready fit diagnostics."""
        return asdict(self)


class DiscreteTimeHazardModel:
    """Monthly logistic default hazard model with mixed feature preprocessing.

    Parameters are intentionally stored as plain values and model artifacts use
    a versioned payload. This makes the feature contract visible at load time
    rather than relying on an unlabelled pickled estimator.
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str] = (),
        *,
        age_col: str | None = "loan_age_months",
        C: float = 1.0,
        max_iter: int = 1_000,
        solver: str = "lbfgs",
        tol: float = 1e-6,
        class_weight: Mapping[int, float] | str | None = None,
        random_state: int | None = 1729,
    ) -> None:
        numeric = tuple(numeric_features)
        categorical = tuple(categorical_features)
        if not numeric and not categorical:
            raise ValueError("at least one model feature is required")
        all_features = numeric + categorical
        if any(not isinstance(feature, str) or not feature for feature in all_features):
            raise ValueError("feature names must be nonempty strings")
        if len(set(all_features)) != len(all_features):
            raise ValueError("numeric and categorical feature names must be unique")
        if age_col is not None and age_col not in numeric:
            raise ValueError("age_col must be included in numeric_features")
        if not np.isfinite(float(C)) or float(C) <= 0.0:
            raise ValueError("C must be positive and finite")
        if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(float(tol)) or float(tol) <= 0.0:
            raise ValueError("tol must be positive and finite")

        self.numeric_features = numeric
        self.categorical_features = categorical
        self.age_col = age_col
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.solver = str(solver)
        self.tol = float(tol)
        self.class_weight = (
            dict(class_weight) if isinstance(class_weight, Mapping) else class_weight
        )
        self.random_state = random_state
        self.pipeline_: Pipeline | None = None
        self.fit_diagnostics_: HazardFitDiagnostics | None = None

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Ordered columns consumed by the fitted pipeline."""
        return self.numeric_features + self.categorical_features

    @property
    def converged_(self) -> bool:
        """Whether every fitted logistic subproblem stopped below ``max_iter``."""
        self._require_fitted()
        assert self.fit_diagnostics_ is not None
        return self.fit_diagnostics_.converged

    @property
    def n_iter_(self) -> tuple[int, ...]:
        """Actual solver iterations reported by sklearn."""
        self._require_fitted()
        assert self.fit_diagnostics_ is not None
        return self.fit_diagnostics_.n_iter

    def _require_fitted(self) -> None:
        if self.pipeline_ is None or self.fit_diagnostics_ is None:
            raise RuntimeError("hazard model has not been fitted")

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        duplicate_columns = set(X.columns[X.columns.duplicated()].tolist())
        duplicated_required = sorted(duplicate_columns.intersection(self.feature_columns))
        if duplicated_required:
            raise ValueError(f"X contains duplicate model columns: {duplicated_required}")
        missing = [feature for feature in self.feature_columns if feature not in X.columns]
        if missing:
            raise KeyError(f"X is missing model features: {missing}")
        selected = X.loc[:, self.feature_columns].copy()
        for feature in self.categorical_features:
            selected[feature] = selected[feature].map(
                lambda value: np.nan if pd.isna(value) else str(value)
            )
        return selected

    @staticmethod
    def _binary_target(y: ArrayLike, *, length: int) -> np.ndarray:
        try:
            target = np.asarray(y, dtype="float64")
        except (TypeError, ValueError) as exc:
            raise ValueError("y must contain numeric zero and one values") from exc
        if target.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(target) != length:
            raise ValueError("X and y must contain the same number of observations")
        if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
            raise ValueError("y must contain only finite zero and one values")
        parsed = target.astype("int8")
        if len(np.unique(parsed)) != 2:
            raise ValueError("y must contain both event and non-event observations")
        return parsed

    @staticmethod
    def _sample_weights(
        sample_weight: ArrayLike | None,
        *,
        length: int,
    ) -> np.ndarray:
        if sample_weight is None:
            return np.ones(length, dtype="float64")
        try:
            weights = np.asarray(sample_weight, dtype="float64")
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_weight must be numeric") from exc
        if weights.ndim != 1 or len(weights) != length:
            raise ValueError("sample_weight must have one value per observation")
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("sample_weight must be finite and nonnegative")
        if float(weights.sum()) <= 0.0:
            raise ValueError("sample_weight must have a positive total")
        return weights

    def _build_pipeline(self) -> Pipeline:
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if self.numeric_features:
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median", keep_empty_features=True),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(
                ("numeric", numeric_pipeline, list(self.numeric_features))
            )
        if self.categorical_features:
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="__MISSING__",
                            keep_empty_features=True,
                        ),
                    ),
                    (
                        "one_hot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    ),
                ]
            )
            transformers.append(
                ("categorical", categorical_pipeline, list(self.categorical_features))
            )
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        classifier = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
            tol=self.tol,
        )
        return Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", classifier)]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Fit the monthly hazard model using positional sample weights.

        Passing ``sample_weight`` is recommended for sampled risk sets. The
        weights are forwarded explicitly to the logistic estimator, not to the
        preprocessing steps. sklearn convergence warnings are not suppressed.
        """
        features = self._select_features(X)
        if features.empty:
            raise ValueError("at least one training observation is required")
        target = self._binary_target(y, length=len(features))
        weights = self._sample_weights(sample_weight, length=len(features))
        weighted_events = float(weights[target == 1].sum())
        weighted_non_events = float(weights[target == 0].sum())
        if weighted_events <= 0.0 or weighted_non_events <= 0.0:
            raise ValueError("sample_weight must leave positive weight in both target classes")

        pipeline = self._build_pipeline()
        pipeline.fit(features, target, classifier__sample_weight=weights)
        classifier = pipeline.named_steps["classifier"]
        n_iter = tuple(int(value) for value in classifier.n_iter_.reshape(-1))
        converged = bool(n_iter) and all(value < classifier.max_iter for value in n_iter)
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed_features = len(preprocessor.get_feature_names_out())

        self.pipeline_ = pipeline
        self.fit_diagnostics_ = HazardFitDiagnostics(
            observations=len(features),
            events=int(target.sum()),
            non_events=int(len(target) - target.sum()),
            sample_weight_provided=sample_weight is not None,
            weight_sum=float(weights.sum()),
            weighted_events=weighted_events,
            weighted_non_events=weighted_non_events,
            transformed_features=transformed_features,
            solver=classifier.solver,
            max_iter=classifier.max_iter,
            n_iter=n_iter,
            converged=converged,
        )
        return self

    def check_convergence(self, *, raise_on_failure: bool = False) -> bool:
        """Return fitted convergence status, optionally failing if the cap was hit."""
        converged = self.converged_
        if not converged and raise_on_failure:
            assert self.fit_diagnostics_ is not None
            raise RuntimeError(
                "logistic hazard model did not converge: "
                f"n_iter={self.fit_diagnostics_.n_iter}, "
                f"max_iter={self.fit_diagnostics_.max_iter}"
            )
        return converged

    def predict_hazard(self, X: pd.DataFrame) -> np.ndarray:
        """Predict conditional monthly default probabilities."""
        self._require_fitted()
        features = self._select_features(X)
        if features.empty:
            return np.empty(0, dtype="float64")
        assert self.pipeline_ is not None
        probabilities = self.pipeline_.predict_proba(features)
        classifier = self.pipeline_.named_steps["classifier"]
        event_positions = np.flatnonzero(classifier.classes_ == 1)
        if len(event_positions) != 1:
            raise AssertionError("fitted classifier does not have event class one")
        hazards = probabilities[:, int(event_positions[0])]
        return np.clip(np.asarray(hazards, dtype="float64"), 0.0, 1.0)

    def predict_hazard_curve(
        self,
        X: pd.DataFrame,
        horizon_months: int,
        *,
        age_col: str | None = None,
    ) -> np.ndarray:
        """Project hazards while advancing each loan's age by one per month.

        Forecast month one uses current age plus one. Every other supplied
        feature remains at its value in ``X`` for all forecast months.
        """
        self._require_fitted()
        if (
            isinstance(horizon_months, bool)
            or not isinstance(horizon_months, Integral)
            or horizon_months < 1
        ):
            raise ValueError("horizon_months must be a positive integer")
        projection_age_col = self.age_col if age_col is None else age_col
        if projection_age_col is None:
            raise ValueError("an age_col is required for monthly projection")
        if projection_age_col not in self.numeric_features:
            raise ValueError("projection age_col must be a fitted numeric feature")

        features = self._select_features(X)
        if features.empty:
            return np.empty((0, int(horizon_months)), dtype="float64")
        ages = pd.to_numeric(features[projection_age_col], errors="coerce").to_numpy(
            dtype="float64"
        )
        if not np.isfinite(ages).all() or (ages < 0.0).any():
            raise ValueError("loan ages must be finite and nonnegative for projection")

        horizon = int(horizon_months)
        row_positions = np.repeat(np.arange(len(features)), horizon)
        expanded = features.iloc[row_positions].reset_index(drop=True)
        offsets = np.tile(np.arange(1, horizon + 1, dtype="float64"), len(features))
        expanded[projection_age_col] = np.repeat(ages, horizon) + offsets
        hazards = self.predict_hazard(expanded)
        return hazards.reshape(len(features), horizon)

    def predict_pd_curves(
        self,
        X: pd.DataFrame,
        horizon_months: int,
        *,
        age_col: str | None = None,
    ) -> PDCurves:
        """Project conditional, marginal, cumulative, and survival curves."""
        hazards = self.predict_hazard_curve(
            X,
            horizon_months,
            age_col=age_col,
        )
        return conditional_hazards_to_pd(hazards, axis=1)

    def serialization_metadata(self) -> dict[str, Any]:
        """Return the versioned, non-estimator portion of a model artifact."""
        self._require_fitted()
        assert self.fit_diagnostics_ is not None
        return {
            "artifact_type": MODEL_ARTIFACT_TYPE,
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "model_config": {
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
                "age_col": self.age_col,
                "C": self.C,
                "max_iter": self.max_iter,
                "solver": self.solver,
                "tol": self.tol,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
            },
            "fit_diagnostics": self.fit_diagnostics_.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        """Atomically save a fitted model in a versioned pickle payload."""
        self._require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.serialization_metadata()
        assert self.pipeline_ is not None
        payload = {**metadata, "pipeline": self.pipeline_}

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                pickle.dump(payload, handle, protocol=5)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a model artifact created by ``save`` from a trusted path."""
        source = Path(path)
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("invalid hazard model artifact payload")
        if payload.get("artifact_type") != MODEL_ARTIFACT_TYPE:
            raise ValueError("artifact is not a discrete-time hazard model")
        if payload.get("artifact_version") != MODEL_ARTIFACT_VERSION:
            raise ValueError(
                "unsupported hazard model artifact version: "
                f"{payload.get('artifact_version')!r}"
            )
        config = payload.get("model_config")
        diagnostics_payload = payload.get("fit_diagnostics")
        pipeline = payload.get("pipeline")
        if not isinstance(config, dict) or not isinstance(diagnostics_payload, dict):
            raise ValueError("hazard model artifact metadata is incomplete")
        if not isinstance(pipeline, Pipeline):
            raise ValueError("hazard model artifact does not contain a fitted pipeline")

        model = cls(**config)
        try:
            diagnostics = HazardFitDiagnostics(**diagnostics_payload)
        except TypeError as exc:
            raise ValueError("hazard model fit diagnostics are invalid") from exc
        model.pipeline_ = pipeline
        model.fit_diagnostics_ = diagnostics
        return model
