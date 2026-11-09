"""Example policy — replace this with YOUR model. Clearly disposable.

Schema v2: 6 arm joint_delta values (rad, |v| ≤ 0.05) + gripper in [0, 1]
(0 closed, 1 open). Idle zeros are a valid action; swap act() for your model.
"""

import time
from typing import Any, Dict

from rc_sdk.policy import Policy


class ExamplePolicy(Policy):
    """Trivial deterministic policy: does nothing fancy, proves the pipeline."""

    def load(self) -> None:
        """Pretend to load weights. In real use: fetch the HF repo named in
        manifest.json -> weights.repo_id/revision, build your model, warm up.
        Runs in a background thread; /readyz flips ready when this returns."""
        time.sleep(0.5)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "joint_delta",
            "values": [0.0] * 6,
            "gripper": 0.0,
        }
