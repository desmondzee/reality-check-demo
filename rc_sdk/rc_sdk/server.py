"""PolicyServer — HTTP shim implementing contract v0 (docs/platform.md §6.1).

Stdlib only (http.server / json). Run with: python -m rc_sdk
[--manifest manifest.json] [--policy your.module:YourPolicy]
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from .policy import Policy

CONTRACT = "v0"
SCHEMA_VERSION = 2  # observation/action schema v2 (UR12e, 6 arm joints)
CAPABILITIES = {"chunking": False, "caching": False}


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_policy(spec: str) -> Policy:
    """'package.module:ClassName' -> instance."""
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"--policy must be 'module:Class', got {spec!r}")
    import importlib

    cls = getattr(importlib.import_module(module_name), class_name)
    return cls()


class _Handler(BaseHTTPRequestHandler):
    server: "PolicyServer"

    def log_message(self, fmt: str, *args: Any) -> None:  # runner captures stdout
        print(f"[rc_sdk] {self.address_string()} {fmt % args}", flush=True)

    # -- helpers ---------------------------------------------------------
    def _send_json(self, status: int, body: Dict[str, Any],
                   extra_headers: Optional[Dict[str, str]] = None) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        if self.path == "/healthz":
            # LIVENESS ONLY: must respond within 5 s of process start,
            # BEFORE weights load. Never gate on readiness here.
            self._send_json(200, {})
        elif self.path == "/readyz":
            self._send_json(200, {
                "ready": self.server.ready.is_set(),
                "contract": CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "capabilities": dict(CAPABILITIES),
            })
        else:
            self._send_json(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/act":
            self._send_json(404, {"error": f"no route {self.path}"})
            return
        if not self.server.ready.is_set():
            self._send_json(503, {"error": "not ready; weights still loading"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            envelope = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        ok, err, echoed = self._validate_envelope(envelope)
        if not ok:
            self._send_json(400, {"error": err})
            return
        episode_id, step = echoed
        try:
            action = self.server.policy.act(envelope)
        except Exception as exc:  # surface a readable failure, keep server up
            self._send_json(500, {"error": f"policy act() failed: {exc}"})
            return
        # CONTRACT: response echoes episode_id and step VERBATIM; headers
        # X-RC-Episode-Id / X-RC-Step ride along for dedupe/logging.
        self._send_json(200, {"episode_id": episode_id, "step": step, "action": action},
                        {"X-RC-Episode-Id": str(episode_id), "X-RC-Step": str(step)})

    @staticmethod
    def _validate_envelope(envelope: Any) -> Tuple[bool, Optional[str], Tuple[Any, Any]]:
        if not isinstance(envelope, dict):
            return False, "request body must be a JSON object", (None, None)
        if "episode_id" not in envelope:
            return False, "missing required field: episode_id", (None, None)
        if "step" not in envelope:
            return False, "missing required field: step", (None, None)
        # Schema v2 (6 joint_delta + gripper); images / proprio are optional and echoed as-is.
        return True, None, (envelope["episode_id"], envelope["step"])


class PolicyServer(ThreadingHTTPServer):
    """HTTPServer + background load thread; ready flips after load() returns."""

    daemon_threads = True

    def __init__(self, port: int, policy: Policy) -> None:
        super().__init__(("0.0.0.0", port), _Handler)
        self.policy = policy
        self.ready = threading.Event()
        self._load_thread = threading.Thread(target=self._load, daemon=True)

    def _load(self) -> None:
        try:
            self.policy.load()
        except Exception as exc:
            print(f"[rc_sdk] load() FAILED: {exc}", flush=True)
            raise
        self.ready.set()
        print("[rc_sdk] ready", flush=True)

    def start_load(self) -> None:
        self._load_thread.start()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(prog="rc_sdk")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--policy", default="policy.example:ExamplePolicy",
                        help="'module:Class' of your Policy subclass")
    parser.add_argument("--port", type=int, default=None,
                        help="overrides manifest.json port")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    port = args.port if args.port is not None else int(manifest["port"])
    policy = import_policy(args.policy)
    server = PolicyServer(port, policy)
    server.start_load()  # healthz answers immediately; readyz flips after load
    print(f"[rc_sdk] serving contract {CONTRACT} on port {port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
