"""The build pipeline: description -> sandbox -> Claude Agent SDK -> app + QA recording.

Each step yields plain dict events, so the same code drives the web server (which relays them
to the browser) and build_cli.py (which prints them).
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from chalkcompute import ContainerSecurityPolicy, Image, KernelPolicy, NetworkPolicy, Sandbox, Secret

HERE = Path(__file__).resolve().parent
AGENT_SDK_VERSION = "0.2.165"
PLAYWRIGHT_VERSION = "1.57.0"
WORKSPACE = "/workspace"
MAX_PROMPT_CHARS = 1200


class BuildError(Exception):
    """A problem the visitor can act on (empty or oversized prompt)."""


def agent_image() -> Image:
    # Chromium and its system libraries come from Playwright's own installer, so the browser
    # matches the Python package exactly. ffmpeg cuts and encodes the QA recording.
    return (
        Image.debian_slim(python_version="3.12")
        .apt_install(["ffmpeg"])
        .pip_install([f"claude-agent-sdk=={AGENT_SDK_VERSION}", f"playwright=={PLAYWRIGHT_VERSION}"])
        .run_commands("python -m playwright install --with-deps chromium")
        .add_local_file(str(HERE / "builder_agent.py"), "/opt/builder/builder_agent.py", strategy="copy")
    )


@dataclass
class RouterConfig:
    base_url: str  # e.g. https://<api host>/v1/router
    environment_id: str
    key_secret: str  # name of the Chalk secret holding the router API key
    model: str

    @property
    def host(self) -> str:
        host = urlsplit(self.base_url).hostname
        if host is None:
            raise ValueError(f"router base URL has no host: {self.base_url!r}")
        return host


def make_sandbox(router: RouterConfig, lifetime_secs: int) -> Sandbox:
    """A gVisor sandbox whose only reachable host is the Chalk router.

    With a network policy in place, egress is an allowlist. add_credentials puts the router
    host on it and sets CHALK_ROUTER_API_KEY to a placeholder; the egress proxy swaps in the
    real key on requests to that host, so the key never exists inside the sandbox. The app the
    agent builds is served on loopback inside the sandbox, which needs no allowlist entry.
    """
    return Sandbox(
        image=agent_image(),
        name=f"app-builder-{os.urandom(4).hex()}",
        cpu="2",
        memory="4Gi",
        env={
            "ANTHROPIC_BASE_URL": router.base_url,
            "ANTHROPIC_CUSTOM_HEADERS": f"X-Chalk-Env-Id: {router.environment_id}",
            "BUILD_MODEL": router.model,
            "BUILD_WORKSPACE": WORKSPACE,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            # The agent runs as root inside a disposable sandbox; this lets it skip permission
            # prompts there.
            "IS_SANDBOX": "1",
        },
        network_policy=NetworkPolicy(allowed_hosts=[]),
        security_policy=ContainerSecurityPolicy(kernel_policy=KernelPolicy.RESTRICTED),
        lifetime=f"{lifetime_secs}s",
    ).add_credentials(router.host, Secret.from_chalk_env(router.key_secret))


PROBE = (
    "import urllib.request\n"
    + "try:\n"
    + "    urllib.request.urlopen('https://registry.npmjs.org', timeout=5)\n"
    + "    print('reachable')\n"
    + "except Exception as e:\n"
    + "    print('blocked:', type(e).__name__)\n"
)


def clean_prompt(prompt: str) -> str:
    prompt = " ".join(prompt.split())
    if len(prompt) < 8:
        raise BuildError("Describe the app in a sentence or two.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise BuildError(f"Keep the description under {MAX_PROMPT_CHARS} characters.")
    return prompt


def extract_outputs(blob: bytes, dest: Path) -> dict[str, Any]:
    """Unpacks out.tar into dest, refusing any member that would land outside it."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    files: list[dict[str, Any]] = []
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not member.isfile() or not target.is_relative_to(root):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target.write_bytes(extracted.read())
            if member.name.startswith("app/"):
                files.append({"path": member.name.removeprefix("app/"), "bytes": member.size})
    meta = json.loads((dest / "qa.json").read_text()) if (dest / "qa.json").exists() else {}
    return {"files": sorted(files, key=lambda f: f["path"]), **meta}


def run_build(prompt: str, router: RouterConfig, timeout_secs: int, dest: Path) -> Iterator[dict[str, Any]]:
    """Runs every step, yielding events for the viewer, and leaves the outputs in dest."""
    yield {"t": "step", "step": "sandbox", "status": "running"}
    t0 = time.monotonic()
    sandbox = make_sandbox(router, timeout_secs + 180)
    sandbox.run()
    try:
        # File uploads do not create parent directories.
        sandbox.exec("mkdir", "-p", WORKSPACE, timeout_secs=20)
        sandbox.fs.write_bytes(f"{WORKSPACE}/prompt.txt", prompt.encode())
        probe = sandbox.exec("python", "-c", PROBE, timeout_secs=20).stdout_text.strip()
        yield {
            "t": "sandbox",
            "id": sandbox.id,
            "seconds": round(time.monotonic() - t0, 1),
            "egress_allowlist": [router.host],
            "probe": f"registry.npmjs.org → {probe}",
            "kernel": "gVisor (restricted)",
            "browser": f"Chromium (Playwright {PLAYWRIGHT_VERSION}), recorded",
        }
        yield {"t": "step", "step": "sandbox", "status": "done"}

        yield {"t": "step", "step": "build", "status": "running"}
        yield {"t": "router", "base_url": router.base_url, "model": router.model}
        stdout_buf = ""
        phase = "build"
        for event in sandbox.exec_stream(
            "python", "/opt/builder/builder_agent.py", timeout_secs=timeout_secs, workdir=WORKSPACE
        ):
            if event.stdout:
                stdout_buf += event.stdout
                while "\n" in stdout_buf:
                    line, stdout_buf = stdout_buf.split("\n", 1)
                    if not line.strip():
                        continue
                    try:
                        agent_event = json.loads(line)
                    except json.JSONDecodeError:
                        yield {"t": "log", "text": line[:2000]}
                        continue
                    # The first browser action marks the switch from building to QA.
                    if (
                        phase == "build"
                        and agent_event.get("t") == "tool_use"
                        and str(agent_event.get("name", "")).startswith("mcp__browser__")
                    ):
                        phase = "qa"
                        yield {"t": "step", "step": "build", "status": "done"}
                        yield {"t": "step", "step": "qa", "status": "running"}
                    yield agent_event
            if event.stderr:
                yield {"t": "log", "text": event.stderr[-2000:]}
            if event.is_error:
                yield {"t": "error", "message": f"exec error: {event.error_message}"}
            if event.exit_code is not None:
                yield {"t": "exit", "code": event.exit_code}
        if phase == "build":
            yield {"t": "step", "step": "build", "status": "done"}
        yield {"t": "step", "step": "qa", "status": "done"}

        yield {"t": "step", "step": "publish", "status": "running"}
        try:
            blob = sandbox.fs.read_bytes(f"{WORKSPACE}/out.tar", timeout_secs=60)
        except Exception as e:  # noqa: BLE001 - surfaced to the viewer
            yield {"t": "error", "message": f"no build output: {type(e).__name__}"}
            return
        outputs = extract_outputs(blob, dest)
        yield {"t": "outputs", **outputs, "has_video": (dest / "qa.mp4").exists()}
        yield {"t": "step", "step": "publish", "status": "done"}
    finally:
        try:
            sandbox.terminate()
        except Exception:  # noqa: BLE001 - the sandbox lifetime reaps it anyway
            pass
