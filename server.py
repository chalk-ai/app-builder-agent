"""Web server for the app builder agent demo.

GET  /                           the demo page (index.html); ?embed=1 hides the page header
GET  /api/examples               featured app descriptions
POST /api/builds {"prompt"}      starts a build, or replays a cached one; returns {"id", "run"}
GET  /api/builds/{id}/events     server-sent events for that build
GET  /runs/{run}/app/{path}      the built app, served as an opaque-origin sandbox
GET  /runs/{run}/qa.mp4          the recording of the agent's browser QA session

Each fresh build runs pipeline.run_build in a worker thread. Finished builds are cached by a
hash of the description, in memory, on local disk and on a Chalk volume, so a repeat request
replays the recorded events at a brisk pace instead of spending another agent run.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from chalkcompute import Sandbox, Volume, VolumeFileNotFoundError
from pipeline import BuildError, RouterConfig, clean_prompt, run_build

PORT = int(os.environ.get("PORT", "8080"))
INDEX_HTML = Path(__file__).with_name("index.html")
RUNS_DIR = Path(os.environ.get("BUILD_RUNS_DIR", "/tmp/runs"))
MAX_CONCURRENT = int(os.environ.get("BUILD_MAX_CONCURRENT", "2"))
MAX_FRESH_PER_HOUR = int(os.environ.get("BUILD_MAX_PER_HOUR", "12"))
BUILD_TIMEOUT = int(os.environ.get("BUILD_TIMEOUT", "900"))
CACHE_VOLUME = os.environ.get("BUILD_CACHE_VOLUME", "app-builder-demo-cache")
# Replays keep the recorded order but never sit idle longer than this between events.
REPLAY_MAX_GAP_S = 0.6
POLL_S = 0.1
# The built app is untrusted script. As a CSP sandbox without allow-same-origin it runs in an
# opaque origin, so it cannot read this site's cookies or storage even if opened directly.
APP_CSP = "sandbox allow-scripts allow-forms allow-modals allow-popups; default-src 'self' 'unsafe-inline' data: blob:"
EXAMPLES = [
    {
        "label": "Habit tracker",
        "prompt": "A habit tracker: add habits, check them off for today, see a 7-day streak grid "
        + "for each habit, and delete habits.",
    },
    {
        "label": "Split the bill",
        "prompt": "A bill splitter for a group dinner: enter the total, tip percentage and number of "
        + "people, optionally mark who had drinks (they pay an extra amount), and show what each "
        + "person owes.",
    },
    {
        "label": "Pomodoro + tasks",
        "prompt": "A pomodoro timer with start, pause and reset, a 25/5 minute work/break cycle, "
        + "and a small task list where I can pick the task I'm focusing on.",
    },
]

log = logging.getLogger("app-builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def router_config() -> RouterConfig:
    api = os.environ["CHALK_API_SERVER"].rstrip("/")
    return RouterConfig(
        base_url=os.environ.get("BUILD_ROUTER_BASE_URL", f"{api}/v1/router"),
        environment_id=os.environ["CHALK_ENVIRONMENT_ID"],
        key_secret=os.environ.get("BUILD_KEY_SECRET", "CHALK_ROUTER_API_KEY"),
        model=os.environ.get("BUILD_MODEL", "anthropic/claude-sonnet-5-5"),
    )


def run_key(prompt: str) -> str:
    return hashlib.sha256(prompt.lower().encode()).hexdigest()[:20]


class Job:
    """One build's event log. Readers follow it from the start while it is still growing."""

    def __init__(self, key: str, replay: bool) -> None:
        super().__init__()
        self.id = uuid.uuid4().hex[:12]
        self.key = key
        self.replay = replay
        self.events: list[dict[str, Any]] = []
        self.done = False
        self.started = time.monotonic()

    def push(self, event: dict[str, Any]) -> None:
        event.setdefault("at", round(time.monotonic() - self.started, 2))
        self.events.append(event)

    def finish(self) -> None:
        self.done = True

    async def wait_done(self) -> None:
        while not self.done:
            await asyncio.sleep(POLL_S)


class BuildCache:
    """Finished builds (events + outputs), on local disk and mirrored to a Chalk volume."""

    def __init__(self) -> None:
        super().__init__()
        self.memory: dict[str, list[dict[str, Any]]] = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> list[dict[str, Any]] | None:
        with self.lock:
            if key in self.memory:
                return self.memory[key]
        try:
            vol = Volume(CACHE_VOLUME)
            events = json.loads(vol.read_file(f"builds/{key}/events.json"))
            files = json.loads(vol.read_file(f"builds/{key}/files.json"))
            dest = RUNS_DIR / key
            for rel in files:
                path = dest / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(vol.read_file(f"builds/{key}/{rel}"))
        except (VolumeFileNotFoundError, ValueError):
            return None
        except Exception:  # noqa: BLE001 - a cold cache is fine; log and run fresh
            log.exception("cache read failed for %s", key)
            return None
        with self.lock:
            self.memory[key] = events
        return events

    def put(self, key: str, events: list[dict[str, Any]]) -> None:
        with self.lock:
            self.memory[key] = events
        try:
            vol = Volume(CACHE_VOLUME)
            dest = RUNS_DIR / key
            files = [str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()]
            for rel in files:
                vol.put_file(f"builds/{key}/{rel}", (dest / rel).read_bytes())
            vol.put_file(f"builds/{key}/files.json", json.dumps(files))
            # Written last: its presence marks a complete entry.
            vol.put_file(f"builds/{key}/events.json", json.dumps(events))
        except Exception:  # noqa: BLE001 - the local copy still serves replays
            log.exception("cache write failed for %s", key)


state: dict[str, Any] = {}


def is_complete(events: list[dict[str, Any]]) -> bool:
    return any(e.get("t") == "outputs" for e in events) and not any(e.get("t") == "error" for e in events)


def run_fresh(job: Job, prompt: str, loop: asyncio.AbstractEventLoop) -> None:
    """Worker thread: runs the pipeline and hands each event to the event loop."""
    dest = RUNS_DIR / job.key
    shutil.rmtree(dest, ignore_errors=True)
    try:
        for event in run_build(prompt, router_config(), BUILD_TIMEOUT, dest):
            if event.get("t") == "outputs":
                event["run"] = job.key
            loop.call_soon_threadsafe(job.push, event)
    except BuildError as exc:
        loop.call_soon_threadsafe(job.push, {"t": "error", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - reported to the viewer
        log.exception("build failed for %s", job.key)
        loop.call_soon_threadsafe(job.push, {"t": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        loop.call_soon_threadsafe(job.finish)


async def replay(job: Job, events: list[dict[str, Any]]) -> None:
    last = 0.0
    for event in events:
        at = float(event.get("at", last))
        await asyncio.sleep(min(max(at - last, 0.0), REPLAY_MAX_GAP_S))
        last = at
        job.push({k: v for k, v in event.items() if k != "at"})
    job.finish()


async def watch_fresh(job: Job, slots: asyncio.Semaphore) -> None:
    try:
        await job.wait_done()
        if is_complete(job.events):
            await asyncio.to_thread(state["cache"].put, job.key, job.events)
    finally:
        state["running"].pop(job.key, None)
        slots.release()


async def start_build(raw_prompt: str) -> Job:
    prompt = clean_prompt(raw_prompt)
    key = run_key(prompt)
    if (running := state["running"].get(key)) is not None:
        return running
    if (cached := await asyncio.to_thread(state["cache"].get, key)) is not None:
        job = Job(key, replay=True)
        state["jobs"][job.id] = job
        asyncio.create_task(replay(job, cached))
        return job

    slots: asyncio.Semaphore = state["slots"]
    recent: collections.deque[float] = state["recent"]
    now = time.monotonic()
    while recent and now - recent[0] > 3600:
        recent.popleft()
    if len(recent) >= MAX_FRESH_PER_HOUR:
        raise BuildError("The demo has run its hourly quota of fresh builds. Pick an example; those replay instantly.")
    if slots.locked():
        raise BuildError("Every build slot is busy right now. Try again in a few minutes, or pick an example.")
    await slots.acquire()
    recent.append(now)
    job = Job(key, replay=False)
    job.push({"t": "prompt", "prompt": prompt})
    state["jobs"][job.id] = job
    state["running"][key] = job
    threading.Thread(target=run_fresh, args=(job, prompt, asyncio.get_running_loop()), daemon=True).start()
    asyncio.create_task(watch_fresh(job, slots))
    return job


def reap_leftover_sandboxes() -> None:
    """Terminates build sandboxes a previous replica left behind; their lifetime would end them too."""
    try:
        leftovers = [info for info in Sandbox.list_all() if (info.name or "").startswith("app-builder-")]
    except Exception:  # noqa: BLE001 - best effort
        log.exception("listing sandboxes failed")
        return
    for info in leftovers:
        try:
            Sandbox.from_id(info.id).terminate()
        except Exception:  # noqa: BLE001 - already gone, or terminating on its own
            log.info("skipped leftover sandbox %s (%s)", info.id, info.status)


async def prewarm_examples() -> None:
    """Builds each featured example once, one at a time, so visitors get instant replays."""
    for example in EXAMPLES:
        try:
            job = await start_build(example["prompt"])
            await job.wait_done()
            log.info("example %s ready (%s)", example["label"], "replay" if job.replay else "fresh")
        except Exception:  # noqa: BLE001 - examples fall back to running on demand
            log.exception("prewarm failed for %s", example["label"])


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    state.update(
        jobs={},
        running={},
        cache=BuildCache(),
        slots=asyncio.Semaphore(MAX_CONCURRENT),
        recent=collections.deque(),
    )
    await asyncio.to_thread(reap_leftover_sandboxes)
    prewarm = asyncio.create_task(prewarm_examples()) if os.environ.get("BUILD_PREWARM", "1") == "1" else None
    yield
    if prewarm is not None:
        prewarm.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "running": len(state["running"])})


@app.get("/api/examples")
async def examples() -> JSONResponse:
    return JSONResponse(EXAMPLES)


@app.post("/api/builds")
async def create_build(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        job = await start_build(str(body.get("prompt", ""))[:4000])
    except BuildError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except (ValueError, AttributeError):
        return JSONResponse({"error": 'Send JSON like {"prompt": "A todo list with due dates"}.'}, status_code=400)
    return JSONResponse({"id": job.id, "run": job.key, "replay": job.replay})


@app.get("/api/builds/{job_id}/events")
async def build_events(job_id: str) -> StreamingResponse:
    job: Job | None = state["jobs"].get(job_id)

    async def stream() -> AsyncIterator[str]:
        if job is None:
            yield f"data: {json.dumps({'t': 'error', 'message': 'Unknown build.'})}\n\n"
            return
        yield f"data: {json.dumps({'t': 'hello', 'replay': job.replay, 'run': job.key})}\n\n"
        sent = 0
        idle = 0.0
        while True:
            while sent < len(job.events):
                yield f"data: {json.dumps(job.events[sent])}\n\n"
                sent += 1
                idle = 0.0
            if job.done:
                yield f"data: {json.dumps({'t': 'end'})}\n\n"
                return
            await asyncio.sleep(POLL_S)
            idle += POLL_S
            if idle >= 15:
                yield ": keepalive\n\n"
                idle = 0.0

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def run_file(run: str, rel: str) -> Path:
    if not run.isalnum() or len(run) > 40:
        raise HTTPException(404)
    root = (RUNS_DIR / run).resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404)
    return path


@app.get("/runs/{run}/qa.mp4")
async def qa_video(run: str) -> FileResponse:
    return FileResponse(run_file(run, "qa.mp4"), media_type="video/mp4")


@app.get("/runs/{run}/app/{path:path}")
async def app_file(run: str, path: str) -> FileResponse:
    target = run_file(run, f"app/{path or 'index.html'}")
    media = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(
        target,
        media_type=media,
        headers={"Content-Security-Policy": APP_CSP, "X-Content-Type-Options": "nosniff"},
    )


@app.get("/runs/{run}/source/{path:path}")
async def app_source(run: str, path: str) -> FileResponse:
    """The same files as plain text, for the Code tab."""
    return FileResponse(run_file(run, f"app/{path}"), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
