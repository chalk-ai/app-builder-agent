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

Credentials for the deploy itself come from your existing `chalk login` (or the
CHALK_* env vars):
    ./deploy.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_FILES = ["server.py", "pipeline.py", "builder_agent.py", "index.html"]


def configure_chalk() -> None:
    """Resolve deployment routing settings using the SDK's ambient login."""
    from chalkcompute import ConnectClient

    client = ConnectClient()
    try:
        client._ensure_auth()
        # Only routing settings are needed by the deployed app; credentials stay in the SDK.
        os.environ["CHALK_API_SERVER"] = client._api_server
        os.environ["CHALK_ENVIRONMENT_ID"] = client._env_id
    finally:
        client.close()


def main() -> None:
    configure_chalk()
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
        name='app-builder-agent',
        cpu="1",
        memory="2Gi",
        env={
            # Not secrets: which API the workload identity talks to, and where the router lives.
            "CHALK_API_SERVER": os.environ["CHALK_API_SERVER"],
            "CHALK_ENVIRONMENT_ID": os.environ["CHALK_ENVIRONMENT_ID"],
            "PORT": "8080",
            "BUILD_MODEL": 'anthropic/claude-sonnet-5-5',
            "BUILD_KEY_SECRET": 'CHALK_ROUTER_API_KEY',
            "BUILD_MAX_CONCURRENT": '2',
            "BUILD_MAX_PER_HOUR": '12',
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
        sg.deploy(build_timeout=1200.0, ready_timeout=900.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
