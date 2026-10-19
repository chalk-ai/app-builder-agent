#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Build an app from a description in a Chalk sandbox, from the terminal.

    ./build_cli.py "A pomodoro timer with a task list" [--out ./out]

The environment needs the Chalk router enabled and a secret holding a router API key
(`chalk router api-key create`, stored as CHALK_ROUTER_API_KEY by default). The app lands in
<out>/app and the QA recording in <out>/qa.mp4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def configure_chalk(env_id: str | None) -> None:
    """Resolve deployment routing settings using the SDK's ambient login."""
    from chalkcompute import ConnectClient

    if env_id:
        os.environ["CHALK_ENVIRONMENT"] = env_id
    client = ConnectClient()
    try:
        client._ensure_auth()
        # Only routing settings are needed by the deployed app; credentials stay in the SDK.
        os.environ["CHALK_API_SERVER"] = client._api_server
        os.environ["CHALK_ENVIRONMENT_ID"] = client._env_id
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--out", default="out")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--model", default="anthropic/claude-sonnet-5-5")
    parser.add_argument("--key-secret", default="CHALK_ROUTER_API_KEY")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    configure_chalk(args.env_id)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline import BuildError, RouterConfig, clean_prompt, run_build

    router = RouterConfig(
        base_url=os.environ["CHALK_API_SERVER"].rstrip("/") + "/v1/router",
        environment_id=os.environ["CHALK_ENVIRONMENT_ID"],
        key_secret=args.key_secret,
        model=args.model,
    )
    try:
        prompt = clean_prompt(args.prompt)
    except BuildError as e:
        sys.exit(str(e))
    for event in run_build(prompt, router, args.timeout, Path(args.out)):
        kind = event.get("t")
        if kind == "text":
            print(event["text"], end="", flush=True)
        elif kind == "text_end":
            print(flush=True)
        elif kind == "tool_use":
            print(f"  → {event['name']} {json.dumps(event['input'])[:160]}", flush=True)
        elif kind == "tool_result" and event.get("is_error"):
            print(f"    ✗ {event['preview'][:200]}", flush=True)
        elif kind == "screenshot":
            print(f"  [screenshot: {event['label']}]", flush=True)
        elif kind in ("text", "log"):
            continue
        else:
            print(f"[{kind}] {json.dumps({k: v for k, v in event.items() if k != 't'})[:300]}", flush=True)


if __name__ == "__main__":
    main()
