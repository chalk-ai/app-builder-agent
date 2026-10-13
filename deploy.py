#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the app builder demo (server.py + pipeline.py + index.html) as a Chalk ScalingGroup.

The server creates build sandboxes with its own workload identity (`chalk_identity=True`), so
no client secret is baked into the image. Build sandboxes get no identity: their only
credential is a placeholder for the router key, swapped in at egress.

Requirements on the target environment:
  - the Chalk router is enabled (`chalk router status`)
  - a secret holds a router API key (`chalk router api-key create`), CHALK_ROUTER_API_KEY by default

Credentials for the deploy itself come from `chalk config` in the current directory (or the
CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--name app-builder-agent]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_FILES = ["server.py", "pipeline.py", "builder_agent.py", "index.html"]


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="app-builder-agent")
    # Defaults to the environment in `chalk config` (or CHALK_ENVIRONMENT_ID).
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--model", default="anthropic/claude-sonnet-5-5")
    parser.add_argument("--key-secret", default="CHALK_ROUTER_API_KEY")
    parser.add_argument("--max-concurrent", default="2")
    parser.add_argument("--max-per-hour", default="12")
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    args = parser.parse_args()

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ScalingGroup

    image = Image.debian_slim(python_version="3.12").pip_install(
        ["chalkcompute==2.13.5", "fastapi==0.116.1", "uvicorn[standard]==0.35.0"]
    )
    # The app files total more than the 32 KiB inline `copy` budget, so they ride a volume
    # mounted at /app when the replica starts. Nothing at build time needs them.
    for name in APP_FILES:
        image = image.add_local_file(str(HERE / name), f"/app/{name}", strategy="volume")
    image = image.workdir("/app")

    sg = ScalingGroup(
        image=image,
        name=args.name,
        cpu="1",
        memory="2Gi",
        env={
            # Not secrets: which API the workload identity talks to, and where the router lives.
            "CHALK_API_SERVER": os.environ["CHALK_API_SERVER"],
            "CHALK_ENVIRONMENT_ID": os.environ["CHALK_ENVIRONMENT_ID"],
            "PORT": "8080",
            "BUILD_MODEL": args.model,
            "BUILD_KEY_SECRET": args.key_secret,
            "BUILD_MAX_CONCURRENT": args.max_concurrent,
            "BUILD_MAX_PER_HOUR": args.max_per_hour,
        },
        entrypoint=["python", "/app/server.py"],
        port=8080,
        chalk_identity=True,
        # Jobs, the replay cache and the rate limits live in the replica, and each event stream
        # must reach the replica running its build, so this stays at one.
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=1200.0, ready_timeout=args.ready_timeout)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
