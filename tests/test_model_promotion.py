import math

from src.models.model_promotion import (
    PromotionPolicy,
    evaluate_candidate,
)


def test_candidate_meeting_requirements_is_approved():
    decision = evaluate_candidate(
        validation_pr_auc=0.1019,
        validation_f1=0.1714,
    )

    assert decision.approved is True


def test_candidate_with_low_pr_auc_is_rejected():
    decision = evaluate_candidate(
        validation_pr_auc=0.0499,
        validation_f1=0.1641,
    )

    assert decision.approved is False
    assert any("PR-AUC" in reason for reason in decision.reasons)


def test_candidate_with_low_f1_is_rejected():
    decision = evaluate_candidate(
        validation_pr_auc=0.1019,
        validation_f1=0.10,
    )

    assert decision.approved is False
    assert any("F1" in reason for reason in decision.reasons)


def test_missing_or_non_finite_metric_is_rejected():
    decision = evaluate_candidate(
        validation_pr_auc=math.nan,
        validation_f1=0.17,
    )

    assert decision.approved is False


def test_custom_policy_is_respected():
    policy = PromotionPolicy(
        min_validation_pr_auc=0.11,
        min_validation_f1=0.15,
    )

    decision = evaluate_candidate(
        validation_pr_auc=0.1019,
        validation_f1=0.1714,
        policy=policy,
    )

    assert decision.approved is False