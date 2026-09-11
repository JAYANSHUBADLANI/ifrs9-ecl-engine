"""Per-loan explanation of the hazard model's monthly default probability.

`staging.py` already records an auditable primary trigger for a stage move,
and `pd_model.py` already reports fit level diagnostics and coefficients
implicitly through the fitted pipeline, but neither answers the question a
reviewer actually asks about one loan: why is this specific loan's monthly
PD what it is, in terms of its own features.

The hazard model is a logistic regression on top of a fixed preprocessing
pipeline, numeric features standardized, categorical features one hot
encoded. For a linear model, the exact Shapley decomposition of the log
odds has a closed form and needs no sampling or approximation: each
transformed feature's contribution is its coefficient times its deviation
from a reference value, and the contributions plus the reference's own log
odds sum to exactly the loan's log odds. This module computes that closed
form directly against the fitted pipeline rather than depending on the
`shap` package's general purpose, sampling based `LinearExplainer`, which
would spend approximation for something this model already has an exact
answer for.

The reference value matters as much here as it does for the scorecard's
adverse action reasons in `reasons.py`, and for the same reason: a
contribution only means something relative to a stated baseline. The
convention here matches that module's default, population mean, so a
contribution reads as "this many log odds above or below a typical loan in
the reference set" rather than against an arbitrary all zero or best case
point. For a standardized numeric feature the training mean is exactly
zero by construction, so no reference values are needed to get that part
of the decomposition exactly right. For a one hot categorical feature the
reference is the reference set's prevalence of that category, which is not
zero and does have to be supplied, so a background frame is a required
argument here rather than optional the way it could be for numeric only
model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pd_model import DiscreteTimeHazardModel


@dataclass(frozen=True)
class FeatureContribution:
    """One original feature's contribution to a loan's log odds of default.

    `transformed_columns` is kept because a one hot categorical feature can
    touch more than one transformed column, and collapsing to the original
    feature name is a presentation choice a reader of this contribution
    should be able to check rather than take on faith.
    """

    feature: str
    contribution: float
    transformed_columns: tuple[str, ...]


@dataclass(frozen=True)
class HazardExplanation:
    """The exact decomposition of one loan's monthly hazard into log odds terms.

    `reference_log_odds` plus every `contributions` entry's `contribution`
    sums to `log_odds` by construction; `check_additivity` verifies this
    rather than assuming it, since a decomposition that does not add up to
    the actual prediction is not a decomposition of that prediction.
    """

    log_odds: float
    hazard: float
    reference_log_odds: float
    reference_hazard: float
    contributions: tuple[FeatureContribution, ...]

    def check_additivity(self, atol: float = 1e-8) -> bool:
        total = self.reference_log_odds + sum(
            c.contribution for c in self.contributions
        )
        return bool(np.isclose(total, self.log_odds, atol=atol))

    def ranked(self) -> tuple[FeatureContribution, ...]:
        """Contributions ordered by absolute size, largest driver first."""
        return tuple(sorted(self.contributions, key=lambda c: -abs(c.contribution)))

    def as_dict(self) -> dict[str, object]:
        return {
            "log_odds": self.log_odds,
            "hazard": self.hazard,
            "reference_log_odds": self.reference_log_odds,
            "reference_hazard": self.reference_hazard,
            "contributions": [
                {
                    "feature": c.feature,
                    "contribution": c.contribution,
                    "transformed_columns": list(c.transformed_columns),
                }
                for c in self.ranked()
            ],
        }


def _sigmoid(log_odds: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-log_odds))


def _original_feature_of(transformed_name: str, model: DiscreteTimeHazardModel) -> str:
    """Map a ColumnTransformer output name back to the model's own feature name.

    `verbose_feature_names_out=True` in the fitted preprocessor prefixes
    every column with its transformer name and an underscore, for example
    `numeric__loan_age_months` or `categorical__loan_purpose_auto`. A
    numeric feature's transformed name is that feature name unchanged
    beyond the prefix; a one hot categorical feature's transformed name has
    the category value appended after its own underscore, so matching is
    done by which fitted feature name the transformed name starts with,
    longest match first so a feature name that is a prefix of another
    fitted feature name cannot steal its columns.
    """
    for prefix in ("numeric__", "categorical__"):
        if transformed_name.startswith(prefix):
            remainder = transformed_name[len(prefix) :]
            candidates = [
                f
                for f in model.feature_columns
                if remainder == f or remainder.startswith(f + "_")
            ]
            if candidates:
                return max(candidates, key=len)
            raise ValueError(
                f"transformed column {transformed_name!r} does not map to any fitted feature; "
                "the preprocessor and the model's feature_columns have gone out of step."
            )
    raise ValueError(f"unrecognized transformed column name {transformed_name!r}")


def _reference_transformed_row(
    model: DiscreteTimeHazardModel, background: pd.DataFrame
) -> np.ndarray:
    """The reference point in transformed feature space: the background set's mean.

    Exact for numeric features regardless of what background is passed,
    since StandardScaler centers on the fit time training mean and that is
    zero in transformed space by construction. For one hot categorical
    features this is the background set's prevalence per category, which
    does depend on what background is passed, so the same background frame
    used to fit or intended to represent typical loans should be supplied
    here, the same discipline `reasons.py` applies to its population mean
    basis.
    """
    model._require_fitted()
    preprocessor = model.pipeline_.named_steps["preprocessor"]  # type: ignore[union-attr]
    selected = model._select_features(background)
    transformed = preprocessor.transform(selected)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype="float64").mean(axis=0)


def explain_hazard(
    model: DiscreteTimeHazardModel,
    X: pd.DataFrame,
    background: pd.DataFrame,
) -> list[HazardExplanation]:
    """Exact per loan decomposition of the monthly hazard's log odds.

    `X` is one or more loans to explain. `background` sets the reference
    point every contribution is measured against; pass the training set,
    or a representative sample of currently in force loans, not an
    arbitrary handful of rows, since the reference's categorical
    prevalences are exactly what "contribution" is relative to.
    """
    model._require_fitted()
    if len(background) == 0:
        raise ValueError("background must contain at least one loan")

    preprocessor = model.pipeline_.named_steps["preprocessor"]  # type: ignore[union-attr]
    classifier = model.pipeline_.named_steps["classifier"]  # type: ignore[union-attr]
    coefficients = np.asarray(classifier.coef_, dtype="float64").reshape(-1)
    intercept = float(np.asarray(classifier.intercept_, dtype="float64").reshape(-1)[0])
    transformed_names = list(preprocessor.get_feature_names_out())
    if len(transformed_names) != len(coefficients):
        raise AssertionError(
            f"{len(transformed_names)} transformed columns but {len(coefficients)} "
            "coefficients; the fitted pipeline is not internally consistent."
        )

    feature_of_column = [
        _original_feature_of(name, model) for name in transformed_names
    ]
    reference_row = _reference_transformed_row(model, background)
    reference_log_odds = intercept + float(np.dot(coefficients, reference_row))
    reference_hazard = float(_sigmoid(np.array([reference_log_odds]))[0])

    selected = model._select_features(X)
    transformed = preprocessor.transform(selected)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype="float64")

    explanations: list[HazardExplanation] = []
    for row in transformed:
        per_column_contribution = coefficients * (row - reference_row)
        by_feature: dict[str, float] = {}
        columns_by_feature: dict[str, list[str]] = {}
        for name, feature, value in zip(
            transformed_names, feature_of_column, per_column_contribution
        ):
            by_feature[feature] = by_feature.get(feature, 0.0) + float(value)
            columns_by_feature.setdefault(feature, []).append(name)

        log_odds = reference_log_odds + sum(by_feature.values())
        hazard = float(_sigmoid(np.array([log_odds]))[0])
        contributions = tuple(
            FeatureContribution(
                feature=feature,
                contribution=by_feature[feature],
                transformed_columns=tuple(columns_by_feature[feature]),
            )
            for feature in model.feature_columns
        )
        explanations.append(
            HazardExplanation(
                log_odds=float(log_odds),
                hazard=hazard,
                reference_log_odds=float(reference_log_odds),
                reference_hazard=reference_hazard,
                contributions=contributions,
            )
        )
    return explanations
