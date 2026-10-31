"""
Model promotion policy for TerraMind ML.

This module evaluates whether a candidate rainfall model satisfies
the initial research-stage promotion requirements.

It does not register, deploy, or replace any model.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionPolicy:
    """Minimum validation requirements for a model candidate."""

    min_validation_pr_auc: float = 0.08
    min_validation_f1: float = 0.15


@dataclass(frozen=True)
class PromotionDecision:
    """Result of evaluating a candidate model."""

    approved: bool
    reasons: tuple[str, ...]


def evaluate_candidate(
    validation_pr_auc: float,
    validation_f1: float,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """
    Evaluate a candidate using validation metrics only.

    Invalid or missing metrics cause rejection.
    """

    if policy is None:
        policy = PromotionPolicy()

    metrics = {
        "validation_pr_auc": validation_pr_auc,
        "validation_f1": validation_f1,
    }

    reasons = []

    for metric_name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            reasons.append(f"{metric_name} must be numeric.")
        elif not math.isfinite(value):
            reasons.append(f"{metric_name} must be finite.")
        elif not 0.0 <= value <= 1.0:
            reasons.append(f"{metric_name} must be between 0 and 1.")

    if reasons:
        return PromotionDecision(
            approved=False,
            reasons=tuple(reasons),
        )

    if validation_pr_auc < policy.min_validation_pr_auc:
        reasons.append(
            f"Validation PR-AUC {validation_pr_auc:.4f} is below "
            f"the minimum {policy.min_validation_pr_auc:.4f}."
        )

    if validation_f1 < policy.min_validation_f1:
        reasons.append(
            f"Validation F1 {validation_f1:.4f} is below "
            f"the minimum {policy.min_validation_f1:.4f}."
        )

    if reasons:
        return PromotionDecision(
            approved=False,
            reasons=tuple(reasons),
        )

    return PromotionDecision(
        approved=True,
        reasons=("Candidate meets all configured validation requirements.",),
    )