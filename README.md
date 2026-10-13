# App builder agent on Chalk

Describe a small web app and watch a coding agent build it, then test it in a real browser. The
agent runs the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) in a
Chalk sandbox. It writes the app, opens it in Chromium, clicks through every feature like a
human tester, fixes what it finds, and hands back three things: the app, its source, and a video
of the QA session.

The demo is mostly about where the agent runs and what it can touch:

1. **No-egress sandbox.** Each build gets its own gVisor sandbox with an empty network
   allowlist: no egress and no DNS. A probe to registry.npmjs.org fails before the agent starts,
   so the app is written from scratch, with no packages or CDNs.
2. **Chalk router.** The one host the sandbox can reach is the Chalk router, which serves the
   Anthropic Messages API that the Agent SDK speaks. The key inside the sandbox is a placeholder,
   and the egress proxy swaps in the real router key on requests to that host.
3. **A browser inside the sandbox.** The app is served on loopback, and the agent drives a
   Playwright Chromium through a handful of tools: `browser_open`, `browser_click`,
   `browser_type`, `browser_press`, `browser_scroll`, `browser_screenshot` (the model sees the
   page and its console errors) and `browser_read`.
4. **A recording you can watch.** The browser session is recorded with a visible cursor and
   click ripples. Idle time while the model thinks is cut out, so the video shows only the
   testing.
5. **A QA floor.** `finish` refuses to end QA before at least 12 browser actions and 4 labeled
   screenshots, so every build ships with real evidence that it works.

## Files

| File | Role |
| --- | --- |
| `pipeline.py` | Description → sandbox → agent → app + recording. Each step yields events. |
| `builder_agent.py` | Runs inside the sandbox: Claude Agent SDK, file tools, the browser tools, the recording. |
| `server.py` | FastAPI app: the page, a job per build, SSE, a replay cache, rate limits, and the built apps. |
| `index.html` | The demo page. `?embed=1` hides its header for embedding in an iframe. |
| `build_cli.py` | The same pipeline from a terminal. |
| `deploy.py` | Deploys the server as a `chalkcompute.ScalingGroup`. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment with the router enabled
(`chalk router status`). Credentials come from `chalk config` in the current directory.

```bash
chalk login

# One time: a router key for the sandbox. Paste the key `chalk router api-key create` prints.
chalk secrets set CHALK_ROUTER_API_KEY

# Build from the terminal; the app lands in ./out/app and the recording in ./out/qa.mp4...
./build_cli.py "A pomodoro timer with a small task list" --out out

# ...or serve the web demo from a scaling group, and open the printed URL.
./deploy.py
```

A typical build takes one to two minutes and costs about $0.10 to $0.15 in model usage.

## What a visitor can and can't do

- **Sandboxes:**
  - a restricted kernel (gVisor)
  - no network except the router host
  - no Chalk identity
  - a placeholder instead of the key
  - a hard `lifetime`
  - terminated when the build ends
- **Agent tools:** `Read`, `Write`, `Edit`, `Glob`, `Grep` and the browser tools. There is no
  shell.
- **Built apps** are untrusted script, so the server sends them with a Content-Security-Policy
  `sandbox` header. They run in an opaque origin even when opened directly, and the page shows
  them in a sandboxed iframe without `allow-same-origin`.
- **Server limits** (`server.py`): two concurrent fresh builds and 12 fresh builds per hour.
  - Finished builds are cached by their description on a Chalk volume.
  - A repeat request replays the recorded run instead of starting a new agent session.
  - The featured examples are built at startup, so visitors see them instantly.
- **Server credentials:** the server creates sandboxes with its own workload identity
  (`chalk_identity=True`), so no client secret is in the image.

## License

Apache-2.0.
