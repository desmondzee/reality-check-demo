"""rc_sdk — the reality-check policy SDK (contract v0, stdlib only).

Implements the HTTP container contract from docs/platform.md §6.1 so teams
write a tiny Policy subclass instead of raw HTTP handling.

Contract rules encoded here:
- GET /healthz -> 200 {} immediately on start (LIVENESS ONLY, before weights load).
- GET /readyz  -> 200 {"ready": bool, "contract": "v0", "schema_version": 2,
                       "capabilities": {"chunking": false, "caching": false}}
                  ready flips true after Policy.load() finishes (background thread).
- POST /act    -> request envelope {episode_id, step, task_id, images?, proprio?};
                  response echoes episode_id and step VERBATIM plus "action":
                  {"episode_id": ..., "step": ..., "action": <Policy.act result>}.
                  Malformed requests (missing episode_id/step, bad JSON) -> 400.
                  episode_id/step also echoed as headers X-RC-Episode-Id / X-RC-Step.

PURE-FUNCTION REQUIREMENT (/act): the same request envelope for the same
(episode_id, step) must yield a valid action — deterministic recompute suffices.
No external side effects; no assumed call rate (calls may arrive at any rate,
or be retried byte-identically by the runner). This holds for the framework;
the team's act() should honor it too.

OBSERVATION/ACTION SCHEMA v2: 6 arm joint_delta values + gripper. Shape lives in
Policy.act; this shim only reports schema_version on /readyz.
"""

from .policy import Policy
from .server import PolicyServer, main

__all__ = ["Policy", "PolicyServer", "main"]
__version__ = "0.1.0"
