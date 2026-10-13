"""App builder agent that runs inside a Chalk sandbox.

It turns a description into a small static web app, then QAs that app in a real browser:

    WORKSPACE/app/      the app the agent writes (index.html, styles.css, app.js)
    WORKSPACE/video/    Playwright's raw recording of the QA browser session
    WORKSPACE/out.tar   the app plus qa.mp4 and qa.json, for the server to download

The Claude Agent SDK writes files with its own tools and drives Chromium through a small set of
browser tools defined here. The recording shows a visible cursor, and the idle time while the
model thinks between actions is cut out of the final qa.mp4.

Every event is printed to stdout as one JSON object per line:

    {"t": "text", "text": ...}                      streamed assistant text
    {"t": "tool_use", "id", "name", "input"}        a tool call
    {"t": "tool_result", "id", "is_error", "preview"}
    {"t": "file", "path", "bytes"}                  a file the agent wrote
    {"t": "screenshot", "jpeg_b64", "label"}        what the agent saw in the browser
    {"t": "qa", "report": {...}}                    the agent's QA verdict
    {"t": "video", "seconds", "raw_seconds", "actions"}
    {"t": "result", "turns", "duration_ms", "cost_usd", "usage"}
    {"t": "error", "message"}
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)
from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

WORKSPACE = Path(os.environ.get("BUILD_WORKSPACE", "/workspace"))
APP_DIR = WORKSPACE / "app"
VIDEO_DIR = WORKSPACE / "video"
MODEL = os.environ.get("BUILD_MODEL", "anthropic/claude-sonnet-5-5")
MAX_TURNS = int(os.environ.get("BUILD_MAX_TURNS", "60"))
APP_PORT = 4173
APP_URL = f"http://127.0.0.1:{APP_PORT}"
VIEWPORT = {"width": 1280, "height": 800}
# Seconds of recording kept around each browser action; everything between is the model
# thinking, which the final video drops.
LEAD_S, TAIL_S = 0.4, 1.4
# finish() refuses to end QA below these, so the recording always covers real testing.
MIN_QA_ACTIONS = 12
MIN_QA_SCREENSHOTS = 4
MAX_APP_FILES = 40
MAX_APP_BYTES = 1_500_000

# An arrow cursor and a click ripple drawn into every page, so the recording shows where the
# agent points and clicks. Playwright's video does not include the OS cursor.
CURSOR_SCRIPT = r"""
(() => {
  if (window.__qaCursor) return;
  window.__qaCursor = true;
  const install = () => {
    const c = document.createElement('div');
    c.style.cssText = 'position:fixed;left:0;top:0;width:22px;height:22px;z-index:2147483647;' +
      'pointer-events:none;transform:translate(-100px,-100px);transition:transform 40ms linear;';
    c.innerHTML = '<svg width="22" height="22" viewBox="0 0 22 22"><path d="M2 2 L2 18 L7 13.5 ' +
      'L10.5 21 L13.5 19.6 L10 12.3 L16.5 12.3 Z" fill="#111" stroke="#fff" stroke-width="1.5"/></svg>';
    document.documentElement.appendChild(c);
    document.addEventListener('mousemove', e => {
      c.style.transform = `translate(${e.clientX - 2}px, ${e.clientY - 2}px)`;
    }, true);
    document.addEventListener('mousedown', e => {
      const r = document.createElement('div');
      r.style.cssText = `position:fixed;left:${e.clientX - 18}px;top:${e.clientY - 18}px;width:36px;` +
        'height:36px;border-radius:50%;background:rgba(112,205,150,.45);border:2px solid #279467;' +
        'z-index:2147483646;pointer-events:none;transform:scale(.3);opacity:1;' +
        'transition:transform 450ms ease-out, opacity 450ms ease-out;';
      document.documentElement.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = 'scale(1.4)'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 500);
    }, true);
  };
  if (document.documentElement) install(); else document.addEventListener('DOMContentLoaded', install);
})();
"""


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def tool_error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def tool_ok(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}]}


class BrowserSession:
    """One recorded Chromium session that every browser tool shares."""

    def __init__(self) -> None:
        super().__init__()
        self.pw: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self.started = 0.0
        # (start, end) seconds into the recording for each action, for cutting idle time.
        self.windows: list[tuple[float, float]] = []
        self.console: list[str] = []
        self.screenshots = 0
        self.mouse = (VIEWPORT["width"] / 2, VIEWPORT["height"] / 2)

    async def ensure(self) -> Page:
        if self.page is not None:
            return self.page
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        self.context = await self.browser.new_context(
            viewport=VIEWPORT, record_video_dir=str(VIDEO_DIR), record_video_size=VIEWPORT
        )
        self.started = time.monotonic()
        await self.context.add_init_script(CURSOR_SCRIPT)
        self.page = await self.context.new_page()
        self.page.on(
            "console", lambda m: self.console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None
        )
        self.page.on("pageerror", lambda e: self.console.append(f"pageerror: {e}"))
        return self.page

    def now(self) -> float:
        return time.monotonic() - self.started

    async def glide_to(self, page: Page, x: float, y: float) -> None:
        """Moves the cursor in visible steps so the recording shows where it travels."""
        steps = max(8, int(((x - self.mouse[0]) ** 2 + (y - self.mouse[1]) ** 2) ** 0.5 / 18))
        await page.mouse.move(x, y, steps=min(steps, 40))
        self.mouse = (x, y)

    async def locate(self, page: Page, selector: str | None, text: str | None) -> Any:
        if selector:
            loc = page.locator(selector).first
        elif text:
            loc = page.get_by_text(text, exact=False).first
        else:
            raise ValueError("pass a CSS `selector` or visible `text`")
        await loc.wait_for(state="visible", timeout=4000)
        await loc.scroll_into_view_if_needed(timeout=2000)
        return loc

    async def point_at(self, page: Page, loc: Any) -> None:
        box = await loc.bounding_box()
        if box:
            await self.glide_to(page, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    async def screenshot(self, label: str) -> str:
        page = await self.ensure()
        png = await page.screenshot(type="jpeg", quality=70)
        thumb = await page.screenshot(type="jpeg", quality=55, scale="css", clip={"x": 0, "y": 0, **VIEWPORT})
        emit({"t": "screenshot", "label": label, "jpeg_b64": base64.b64encode(thumb).decode()})
        return base64.b64encode(png).decode()

    async def close(self) -> Path | None:
        if self.context is None:
            return None
        video = self.page.video if self.page else None
        await self.context.close()
        path = Path(await video.path()) if video else None
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()
        return path


def merged_windows(windows: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """Pads each action window, clamps it to the recording, and merges overlaps."""
    out: list[tuple[float, float]] = []
    for start, end in sorted(windows):
        s, e = max(0.0, start - LEAD_S), min(duration, end + TAIL_S)
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def cut_video(raw: Path, windows: list[tuple[float, float]], dest: Path) -> dict[str, Any]:
    """Keeps only the stretches where the agent acted, as browser-playable H.264."""
    duration = probe_duration(raw)
    keep = merged_windows(windows, duration) or [(0.0, duration)]
    parts = "".join(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];" for i, (s, e) in enumerate(keep))
    concat = "".join(f"[v{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=1:a=0[out]"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
            "-filter_complex", parts + concat, "-map", "[out]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-r", "25", "-movflags", "+faststart", str(dest),
        ],
        check=True,
    )  # fmt: skip
    return {
        "seconds": round(sum(e - s for s, e in keep), 1),
        "raw_seconds": round(duration, 1),
        "segments": len(keep),
    }


def build_tools(session: BrowserSession, state: dict[str, Any]) -> Any:
    target_props = {
        "selector": {"type": "string", "description": "CSS selector, e.g. '#add-btn' or 'input[name=title]'."},
        "text": {"type": "string", "description": "Visible text to find instead of a selector."},
    }

    async def act(name: str, body: Any) -> dict[str, Any]:
        page = await session.ensure()
        start = session.now()
        try:
            message = await body(page)
        except Exception as e:  # noqa: BLE001 - reported to the model so it can adapt
            session.windows.append((start, session.now()))
            return tool_error(f"{name} failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
        await page.wait_for_timeout(350)
        session.windows.append((start, session.now()))
        return tool_ok(message)

    @tool(
        "browser_open",
        f"Open a page of the app in the recorded browser. The app is served at {APP_URL}/ from "
        + f"{APP_DIR}; pass a path such as '/' or '/index.html'. Reloads if already open.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    async def browser_open(args: dict[str, Any]) -> dict[str, Any]:
        path = "/" + str(args.get("path", "/")).lstrip("/")

        async def body(page: Page) -> str:
            session.console.clear()
            resp = await page.goto(APP_URL + path, wait_until="load", timeout=15000)
            await session.glide_to(page, VIEWPORT["width"] / 2, VIEWPORT["height"] / 3)
            status = resp.status if resp else "?"
            return f"Opened {path} (HTTP {status}), title {await page.title()!r}."

        return await act("browser_open", body)

    @tool(
        "browser_click",
        "Move the cursor to an element and click it.",
        {"type": "object", "properties": target_props},
    )
    async def browser_click(args: dict[str, Any]) -> dict[str, Any]:
        async def body(page: Page) -> str:
            loc = await session.locate(page, args.get("selector"), args.get("text"))
            await session.point_at(page, loc)
            await page.wait_for_timeout(150)
            await loc.click(timeout=4000)
            return f"Clicked {args.get('selector') or args.get('text')!r}."

        return await act("browser_click", body)

    @tool(
        "browser_type",
        "Click into an input or textarea, replace its contents with `value`, and optionally press "
        + "Enter afterwards.",
        {
            "type": "object",
            "properties": {**target_props, "value": {"type": "string"}, "submit": {"type": "boolean"}},
            "required": ["value"],
        },
    )
    async def browser_type(args: dict[str, Any]) -> dict[str, Any]:
        async def body(page: Page) -> str:
            loc = await session.locate(page, args.get("selector"), args.get("text"))
            await session.point_at(page, loc)
            await loc.click(timeout=4000)
            await loc.fill("")
            await page.keyboard.type(str(args["value"]), delay=35)
            if args.get("submit"):
                await page.keyboard.press("Enter")
            return f"Typed {str(args['value'])[:80]!r}."

        return await act("browser_type", body)

    @tool(
        "browser_press",
        "Press a key or chord in the page, e.g. 'Enter', 'Escape', 'ArrowDown', 'Control+z'.",
        {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    )
    async def browser_press(args: dict[str, Any]) -> dict[str, Any]:
        async def body(page: Page) -> str:
            await page.keyboard.press(str(args["key"]))
            return f"Pressed {args['key']}."

        return await act("browser_press", body)

    @tool(
        "browser_scroll",
        "Scroll the page by `dy` pixels (negative scrolls up).",
        {"type": "object", "properties": {"dy": {"type": "number"}}, "required": ["dy"]},
    )
    async def browser_scroll(args: dict[str, Any]) -> dict[str, Any]:
        async def body(page: Page) -> str:
            await page.mouse.wheel(0, float(args["dy"]))
            return f"Scrolled {args['dy']}px."

        return await act("browser_scroll", body)

    @tool(
        "browser_screenshot",
        "See the current page. Returns a screenshot of the viewport plus any console errors since "
        + "the last page load.",
        {
            "type": "object",
            "properties": {"label": {"type": "string", "description": "The check this screenshot verifies."}},
            "required": ["label"],
        },
    )
    async def browser_screenshot(args: dict[str, Any]) -> dict[str, Any]:
        start = session.now() if session.page else 0.0
        data = await session.screenshot(str(args.get("label") or "screenshot"))
        session.screenshots += 1
        if session.page:
            session.windows.append((start, session.now()))
        errors = "\n".join(session.console[-15:]) or "none"
        return {
            "content": [
                {"type": "image", "data": data, "mimeType": "image/jpeg"},
                {"type": "text", "text": f"Console errors and warnings since load: {errors}"},
            ]
        }

    @tool(
        "browser_read",
        "Read the visible text of the page or of one element, for checking values exactly.",
        {"type": "object", "properties": target_props},
    )
    async def browser_read(args: dict[str, Any]) -> dict[str, Any]:
        page = await session.ensure()
        try:
            if args.get("selector") or args.get("text"):
                loc = await session.locate(page, args.get("selector"), args.get("text"))
                text = await loc.inner_text(timeout=3000)
            else:
                text = await page.inner_text("body", timeout=3000)
        except Exception as e:  # noqa: BLE001
            return tool_error(f"browser_read failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
        return tool_ok(text[:4000])

    @tool(
        "finish",
        "Call once, after QA passes or you have fixed what you could. `checks` lists each behaviour "
        + "you verified in the browser and whether it passed.",
        {
            "type": "object",
            "properties": {
                "app_name": {"type": "string"},
                "summary": {"type": "string", "description": "Two or three sentences on what you built."},
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "passed": {"type": "boolean"},
                            "notes": {"type": "string"},
                        },
                        "required": ["name", "passed"],
                    },
                },
                "fixed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Bugs QA found that you fixed.",
                },
            },
            "required": ["app_name", "summary", "checks"],
        },
    )
    async def finish(args: dict[str, Any]) -> dict[str, Any]:
        if session.page is None:
            return tool_error("QA the app in the browser before finishing.")
        missing = []
        if len(session.windows) < MIN_QA_ACTIONS:
            missing.append(f"{len(session.windows)} browser actions so far, at least {MIN_QA_ACTIONS} needed")
        if session.screenshots < MIN_QA_SCREENSHOTS:
            missing.append(f"{session.screenshots} labeled screenshots so far, at least {MIN_QA_SCREENSHOTS} needed")
        if missing:
            return tool_error(
                "QA is not thorough enough yet ("
                + "; ".join(missing)
                + "). Exercise the features you have not covered, including an edge case, then finish."
            )
        state["report"] = args
        emit({"t": "qa", "report": args})
        return tool_ok("Recorded. You are done; stop here.")

    return create_sdk_mcp_server(
        name="browser",
        version="1.0.0",
        tools=[
            browser_open,
            browser_click,
            browser_type,
            browser_press,
            browser_scroll,
            browser_screenshot,
            browser_read,
            finish,
        ],
    )


BROWSER_TOOLS = [
    "browser_open",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_scroll",
    "browser_screenshot",
    "browser_read",
    "finish",
]


def system_prompt() -> str:
    return f"""You are an app builder. A visitor describes a small web app; you build it, then QA
it in a real browser like a careful human tester, and fix what you find.

Build:
- Write a static app into {APP_DIR}: index.html, styles.css and app.js (more files only if
  they help). Plain HTML, CSS and JavaScript. No frameworks, no build step, no CDNs, fonts or
  images from the internet: the sandbox has no network, and neither will the preview.
- Design it like a polished product, not a code sample. Use a branded header with a small inline
  SVG logo mark and the app name, a layout that uses the 1280x800 viewport well (a main column
  plus a summary or stats panel where it fits), cards with soft shadows and rounded corners,
  one confident accent color over neutral surfaces, the system font stack with a clear type
  scale, hover, focus and active states, and short CSS transitions when items appear or
  change. Empty states get a friendly line and a small inline-SVG illustration.
- Start with a little realistic sample data so the first screen looks alive, as long as the
  visitor can clear or change it.
- It should also work at phone widths.
- Wrap every localStorage or sessionStorage access in try/catch. The preview runs in a
  sandboxed iframe where storage throws, and the app must still work without persistence.
- Give interactive elements ids or stable classes so they are easy to target.

QA:
- The app is already served at {APP_URL}/. Plan 4 to 6 checks that together cover every
  feature in the description, including at least one edge case (empty or invalid input, or
  removing the last item).
- For each check, do what a user would with browser_click, browser_type, browser_press and
  browser_scroll, then call browser_screenshot with a `label` naming the check, and confirm
  exact values with browser_read where numbers or text matter. Add several items, not just
  one.
- If something is broken, confusing or ugly, fix the code, reload with browser_open, and verify
  again. Note each fix in finish's `fixed`.
- Everything you do in the browser is recorded and shown to the visitor as a video, so move
  with purpose: roughly 15 to 25 browser actions in total. finish is rejected below
  {MIN_QA_ACTIONS} actions or {MIN_QA_SCREENSHOTS} labeled screenshots.
- Finish with exactly one finish call listing the checks you ran.

Narrate briefly between tool calls (one sentence) so a viewer can follow along. After finish,
end with one or two plain sentences; finish already carries the details."""


def package_outputs(state: dict[str, Any], video: dict[str, Any] | None) -> None:
    files = [p for p in APP_DIR.rglob("*") if p.is_file()]
    files = files[:MAX_APP_FILES]
    with tarfile.open(WORKSPACE / "out.tar", "w") as tar:
        total = 0
        for p in files:
            total += p.stat().st_size
            if total > MAX_APP_BYTES:
                break
            tar.add(p, arcname=f"app/{p.relative_to(APP_DIR)}")
        if (WORKSPACE / "qa.mp4").exists():
            tar.add(WORKSPACE / "qa.mp4", arcname="qa.mp4")
        meta = WORKSPACE / "qa.json"
        meta.write_text(json.dumps({"report": state.get("report"), "video": video}))
        tar.add(meta, arcname="qa.json")


async def run(prompt: str) -> int:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(APP_PORT), "--bind", "127.0.0.1", "--directory", str(APP_DIR)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    session = BrowserSession()
    state: dict[str, Any] = {}

    # The egress proxy replaces this placeholder with the real router key on the way out.
    os.environ["ANTHROPIC_AUTH_TOKEN"] = os.environ["CHALK_ROUTER_API_KEY"]
    options = ClaudeAgentOptions(
        model=MODEL,
        cwd=str(APP_DIR),
        system_prompt=system_prompt(),
        # No shell: a static app needs none, and file writes through Write/Edit show up in the
        # viewer's activity feed.
        tools=["Read", "Write", "Edit", "Glob", "Grep"],
        allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"] + [f"mcp__browser__{name}" for name in BROWSER_TOOLS],
        mcp_servers={"browser": build_tools(session, state)},
        permission_mode="bypassPermissions",
        max_turns=MAX_TURNS,
        include_partial_messages=True,
        setting_sources=[],
    )
    pending_writes: dict[str, str] = {}
    try:
        async for message in query(prompt=f"Build this app:\n\n{prompt}", options=options):
            if isinstance(message, StreamEvent):
                delta = message.event.get("delta", {})
                if message.event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                    emit({"t": "text", "text": delta["text"]})
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_input = dict(block.input)
                        if block.name in ("Write", "Edit") and "file_path" in tool_input:
                            pending_writes[block.id] = tool_input["file_path"]
                            # File bodies are large and already visible in the Code tab.
                            for key in ("content", "old_string", "new_string"):
                                if key in tool_input:
                                    tool_input[key] = f"<{len(str(tool_input[key]))} chars>"
                        emit({"t": "tool_use", "id": block.id, "name": block.name, "input": tool_input})
                    elif isinstance(block, TextBlock):
                        emit({"t": "text_end"})
            elif isinstance(message, UserMessage) and isinstance(message.content, list):
                for block in message.content:
                    if not isinstance(block, ToolResultBlock):
                        continue
                    content = block.content
                    if isinstance(content, list):
                        content = "\n".join(
                            c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                        )
                    text = str(content or "")
                    emit(
                        {
                            "t": "tool_result",
                            "id": block.tool_use_id,
                            "is_error": bool(block.is_error),
                            "preview": text[:400],
                        }
                    )
                    path = pending_writes.pop(block.tool_use_id, None)
                    if path and not block.is_error and Path(path).is_file():
                        emit(
                            {
                                "t": "file",
                                "path": str(Path(path).relative_to(APP_DIR))
                                if Path(path).is_relative_to(APP_DIR)
                                else path,
                                "bytes": Path(path).stat().st_size,
                            }
                        )
            elif isinstance(message, ResultMessage):
                emit(
                    {
                        "t": "result",
                        "turns": message.num_turns,
                        "duration_ms": message.duration_ms,
                        "cost_usd": message.total_cost_usd,
                        "usage": message.usage,
                        "is_error": message.is_error,
                    }
                )
    finally:
        raw = await session.close()
        server.terminate()
        video: dict[str, Any] | None = None
        if raw is not None and raw.exists():
            try:
                video = cut_video(raw, session.windows, WORKSPACE / "qa.mp4")
                video["actions"] = len(session.windows)
                emit({"t": "video", **video})
            except subprocess.CalledProcessError as e:
                emit({"t": "error", "message": f"video encode failed: {e}"})
        package_outputs(state, video)
        shutil.rmtree(VIDEO_DIR, ignore_errors=True)
    return 0


def main() -> None:
    prompt = (WORKSPACE / "prompt.txt").read_text().strip()
    try:
        sys.exit(asyncio.run(run(prompt)))
    except Exception as exc:  # noqa: BLE001 - reported to the viewer, then exit nonzero
        emit({"t": "error", "message": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
