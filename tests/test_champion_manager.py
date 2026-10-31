from src.models.champion_manager import ChampionManager


def test_manager_starts_without_champion():
    manager = ChampionManager()

    assert manager.champion is None
    assert manager.history == []


def test_approved_candidate_becomes_champion():
    manager = ChampionManager()

    result = manager.promote(
        model_name="logistic_regression",
        run_id="run_lr",
        validation_pr_auc=0.1019,
        validation_f1=0.1714,
    )

    assert result["promoted"] is True
    assert manager.champion["model_name"] == "logistic_regression"
    assert manager.champion["run_id"] == "run_lr"


def test_rejected_candidate_does_not_replace_champion():
    manager = ChampionManager()

    manager.promote(
        model_name="logistic_regression",
        run_id="run_lr",
        validation_pr_auc=0.1019,
        validation_f1=0.1714,
    )

    result = manager.promote(
        model_name="random_forest",
        run_id="run_rf",
        validation_pr_auc=0.0691,
        validation_f1=0.1641,
    )

    assert result["promoted"] is False
    assert manager.champion["model_name"] == "logistic_regression"
    assert manager.champion["run_id"] == "run_lr"


def test_promotion_history_records_success_and_rejection():
    manager = ChampionManager()

    manager.promote(
        model_name="logistic_regression",
        run_id="run_lr",
        validation_pr_auc=0.1019,
        validation_f1=0.1714,
    )

    manager.promote(
        model_name="random_forest",
        run_id="run_rf",
        validation_pr_auc=0.0691,
        validation_f1=0.1641,
    )

    history = manager.history

    assert len(history) == 2
    assert history[0]["status"] == "promoted"
    assert history[1]["status"] == "rejected"
    assert history[1]["previous_champion"]["model_name"] == "logistic_regression"


def test_history_property_returns_a_copy():
    manager = ChampionManager()

    history = manager.history
    history.append({"status": "external_change"})

    assert manager.history == []

def test_rollback_restores_previous_champion():
    manager = ChampionManager()

    manager.promote(
        "logistic_regression",
        "run_lr",
        0.1019,
        0.1714,
    )

    manager.promote(
        "approved_candidate",
        "run_new",
        0.12,
        0.18,
    )

    result = manager.rollback()

    assert result["rolled_back"] is True
    assert manager.champion["model_name"] == "logistic_regression"
    assert manager.champion["run_id"] == "run_lr"
    assert len(manager.history) == 3


def test_rollback_without_previous_champion_fails_safely():
    manager = ChampionManager()

    manager.promote(
        "logistic_regression",
        "run_lr",
        0.1019,
        0.1714,
    )

    result = manager.rollback()

    assert result["rolled_back"] is False
    assert manager.champion["model_name"] == "logistic_regression"
    assert len(manager.history) == 1