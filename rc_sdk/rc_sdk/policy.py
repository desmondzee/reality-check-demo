"""Policy base class — subclass this and swap in your model.

Schema v2: 6 arm `joint_delta` values (|Δq| ≤ 0.05 rad) plus gripper in [0, 1].
"""

from typing import Any, Dict


class Policy:
    """Your model behind the reality-check container contract (v0).

    Lifecycle:
      - `load()` runs in a background thread at server start; /readyz flips to
        ready when it returns. Load weights, build your model, warm up here.
      - `act(obs)` is called per step once ready.

    act() must be a PURE FUNCTION of the request envelope: same envelope for
    the same (episode_id, step) => valid action (deterministic recompute is
    fine). No external side effects; no assumed call rate — /act may be
    retried byte-identically by the runner.
    """

    def load(self) -> None:
        """Load weights / initialize the model. Runs once, in the background.

        Weights come from the PUBLIC Hugging Face repo named in
        manifest.json -> weights.repo_id / weights.revision. Do NOT commit
        weights to this repo.
        """
        raise NotImplementedError("implement load() to fetch/prepare your weights")

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """One observation in, one action dict out (schema v2: 6 joint_delta + gripper).

        `obs` is the parsed /act request envelope: {episode_id, step, task_id,
        images?, proprio?}. Return the ACTION dict only — the server echoes
        episode_id/step for you.
        """
        raise NotImplementedError("implement act() to run your model")
