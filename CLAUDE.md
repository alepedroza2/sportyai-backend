# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SportyAI Golf — a single-endpoint FastAPI service that analyzes a golf swing video: it extracts pose landmarks with MediaPipe, computes joint angles, and sends the measured data to Claude (Anthropic API) to generate a scored coaching report. The entire service lives in `main.py`.

## Running locally

```bash
pip install -r requirements.txt
ANTHROPIC_KEY=sk-... uvicorn main:app --reload --port 8000
```

There is no test suite, linter, or build step configured in this repo.

## Deployment

Deploys via Docker (`Dockerfile`) to a platform that provides `ANTHROPIC_KEY` as an env var (Railway/Nixpacks config also present in `nixpacks.toml`, though the Dockerfile is the primary path per recent commit history — `Use Dockerfile for proper Linux deps`). The container installs OpenCV/MediaPipe's Linux system deps (`libgl1`, `libglib2.0-0`, etc.) via apt before `pip install`; these are required because `opencv-python-headless` and MediaPipe need them at import time and they are not present in a bare `python:3.12-slim` image. Port is hardcoded to 8000 (see commit history — shell-expanded `$PORT` env vars caused deploy failures on the target platform).

## Architecture

Request flow for `POST /analyze/golf` (`main.py`):

1. **Video ingestion** — uploaded video is written to a temp file (`tempfile.NamedTemporaryFile`), always cleaned up in a `finally` block.
2. **Pose extraction** (`analyze_golf_video`) — samples up to 15 evenly-spaced frames from the video (`np.linspace` over frame count), runs MediaPipe's `PoseLandmarker` (heavy model) on each frame in `IMAGE` mode, and computes joint angles (elbow, knee, spine) and shoulder/hip rotation via `calculate_angle`, which is a generic 3-point angle calculator (points are `[x, y]` in MediaPipe's normalized landmark coordinates, not pixels).
3. **Model download** (`get_model`) — the pose landmarker `.task` model file is lazily downloaded from a Google Cloud Storage URL into `/tmp/pose_landmarker.task` on first use per container instance (not bundled in the image).
4. **Benchmarking** — measured angles are compared against hardcoded elite-golfer benchmarks in `GOLF_BENCHMARKS`, keyed by shot type (`driver`/`iron`/`chip`/`putt`, matched via substring on the `shot_type` form field, defaulting to `driver`).
5. **LLM scoring** — the real measured data plus benchmarks are embedded into a prompt sent to `claude-haiku-4-5-20251001` (Anthropic Messages API), instructed to return only a JSON object with a score, per-metric classifications (good/warn/bad), percentile/pro comparisons, and feedback. The response is parsed with `json.loads` after stripping markdown code fences — there is no schema validation or retry if Claude returns malformed JSON.
6. The measured pose data (`measured_data`) is merged into the final JSON response alongside the LLM's output.

`GET /health` is a liveness check.

## Key constraints to keep in mind when editing

- `analyze_golf_video` imports `mediapipe` lazily (inside the function, not at module top) — keep it that way, since importing it eagerly slows down cold start / health checks.
- The Anthropic response-parsing step (`json.loads(...)`) is brittle by design (no retries, no schema validation) — if you touch the prompt, keep the "Return ONLY valid JSON" instruction and matching shape, or the parse will break silently for callers.
- CORS is wide open (`allow_origins=["*"]`) — this is intentional for the current client setup, not an oversight to "fix" without checking with the user first.
