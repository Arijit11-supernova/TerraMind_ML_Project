"""
Champion-candidate model management for TerraMind ML.

This module manages model selection, promotion, rollback,
and promotion history.

It does not deploy models or serve predictions.
"""

from datetime import datetime, timezone

from src.models.model_promotion import evaluate_candidate


class ChampionManager:
    """Manage the currently selected model and promotion history."""

    def __init__(self):
        self._champion = None
        self._history = []

    @property
    def champion(self):
        """Return the currently selected champion."""
        return self._champion

    @property
    def history(self):
        """Return a copy of the promotion history."""
        return list(self._history)

    def promote(
        self,
        model_name,
        run_id,
        validation_pr_auc,
        validation_f1,
    ):
        """Promote a model only when it passes the validation policy."""

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string.")

        decision = evaluate_candidate(
            validation_pr_auc=validation_pr_auc,
            validation_f1=validation_f1,
        )

        previous_champion = (
            self._champion.copy() if self._champion is not None else None
        )

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "promotion",
            "model_name": model_name,
            "run_id": run_id,
            "previous_champion": previous_champion,
            "approved": decision.approved,
            "reasons": decision.reasons,
        }

        if not decision.approved:
            event["status"] = "rejected"
            self._history.append(event)

            return {
                "promoted": False,
                "champion": (
                    self._champion.copy()
                    if self._champion is not None
                    else None
                ),
                "reasons": decision.reasons,
            }

        self._champion = {
            "model_name": model_name,
            "run_id": run_id,
            "validation_pr_auc": validation_pr_auc,
            "validation_f1": validation_f1,
        }

        event["status"] = "promoted"
        event["new_champion"] = self._champion.copy()
        self._history.append(event)

        return {
            "promoted": True,
            "champion": self._champion.copy(),
            "reasons": decision.reasons,
        }

    def rollback(self, run_id=None):
        """
        Restore a previously promoted champion.

        If run_id is provided, restore that specific previously
        promoted model. Otherwise, restore the most recent earlier
        champion.

        Rejected candidates and rollback events are never treated
        as successful promotion events.
        """

        current_run_id = (
            self._champion["run_id"]
            if self._champion is not None
            else None
        )

        previous_champions = []

        for event in self._history:
            if event["action"] != "promotion":
                continue

            if event["status"] != "promoted":
                continue

            champion = event.get("new_champion")

            if champion is None:
                continue

            if champion["run_id"] == current_run_id:
                continue

            if run_id is not None and champion["run_id"] != run_id:
                continue

            previous_champions.append(champion)

        if not previous_champions:
            return {
                "rolled_back": False,
                "champion": (
                    self._champion.copy()
                    if self._champion is not None
                    else None
                ),
                "reason": (
                    "No previously promoted champion is available "
                    "for rollback."
                ),
            }

        target = previous_champions[-1].copy()

        old_champion = (
            self._champion.copy() if self._champion is not None else None
        )

        self._champion = target.copy()

        self._history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "rollback",
                "status": "rolled_back",
                "previous_champion": old_champion,
                "new_champion": target.copy(),
            }
        )

        return {
            "rolled_back": True,
            "champion": target.copy(),
            "reason": "Restored a previously promoted champion.",
        }