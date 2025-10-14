# Python GUI Plan (No-Qt) — Live View + Recorder

## Overview
A Linux desktop GUI (no Qt) that:
1. Previews live camera frames.
2. Lets the user type a session (directory) name.
3. Starts/stops recording (PNG-8 or RAW-16) via button or shortcut.
4. Uses your camera API: `next(camera)`, `camera.record(dir, save_raw)`, and optional `camera.stop_record()`.

## Goals
- Minimal UI, high reliability, no Qt.
- Smooth live preview (8-bit BGR or 16-bit gray).
- Toggleable recording, session-based directory structure.
- Clean shutdown and recoverable errors.

## Non-Goals
- Multi-window complexity.
- Heavy overlay/annotation features (future work).
- Cross-platform polish beyond Linux (works elsewhere incidentally).

## Framework Choice (No-Qt)
- **Primary:** **Dear PyGui** — GPU-accelerated immediate-mode UI; minimal conflicts with `cv2`.
- **Alternatives (fallbacks):**
  - **PyGObject/GTK4** — native Linux feel; more boilerplate for textures.
  - **Tkinter** — simple, better for low FPS.
  - **wxPython** — mature, heavier footprint.
- **Decision:** Start with Dear PyGui for v1; keep GTK4 as Plan B if needed.

## High-Level Architecture
- **Process:** Single process.
- **Threads:**
  - **UI thread** runs the GUI loop, fetches the latest frame via `next(camera)`, converts to RGBA float32, updates GPU texture.
  - **Saver thread (optional)** only if not using `camera.record(...)`. Writes frames to disk while recording is ON.
- **Data Flow:**
  ```text
  camera.next() -> ndarray (gray16 or BGR8/16)
         UI thread:
           - Convert to preview texture (RGBA float32 [0–1])
           - Update dynamic texture
           - If fallback saver is used: enqueue frame copy into a bounded queue
         Saver thread (fallback):
           - Pop frames, write PNG(8) or RAW(16) to session dir
  ```
- **State Machine:**
  ```text
  IDLE -> (Toggle) -> RECORDING
  RECORDING -> (Toggle/Close) -> STOPPING -> IDLE
  ```

## UI Layout
- **Control Panel (top/left):**
  - Status: `● REC` / `○ idle`.
  - Base directory input + “Choose Dir” dialog.
  - Session name input (used as subdirectory).
  - Checkbox: **Save RAW16 (otherwise PNG8)**.
  - Button: **Toggle Record (Ctrl+S)**.
- **Main Area:**
  - Live preview image area.
  - First frame determines texture size; recreate texture on size change.
  - Keep aspect ratio via widget scaling (avoid CPU resizes).

## Recording Control
- **Preferred path:** Call `camera.record(<base>/<session>, save_raw: bool)` to start.
  - To stop: call `camera.stop_record()` if present.
- **Fallback path (if camera API can’t save continuously):**
  - Start a **saver thread** with a **bounded queue** (e.g., 3–10 frames) to decouple UI from disk I/O.
  - While recording: enqueue frame copies.
  - Saver writes:
    - **RAW16:** write `.raw` (binary dump) for gray16 frames.
    - **PNG8:**
      - Gray → PNG directly.
      - BGR8 → convert BGR→RGB first.
      - BGR16 → downshift to 8-bit (>> 8) before PNG.
  - Stop: set event, drain queue with timeout, join thread (bounded).

## File & Directory Conventions
- **Base dir (configurable):** default `~/recordings`.
- **Session dir:** `<base>/<session_name>/`.
- **Filenames:** zero-padded index `000000.png` / `000000.raw`.
- **Manifest (optional):** `manifest.json` with start time, mode (PNG/RAW), FPS cap, notes.

## Frame Handling & Preview
- **Supported frames:**
  - Gray: `(H, W)` dtype `uint8` or `uint16`.
  - Color: `(H, W, 3)` BGR `uint8` or `uint16`.
- **Preview normalization:**
  - `uint8` → divide by 255.0.
  - `uint16` → divide by 65535.0.
  - Gray → replicate to RGB; add alpha channel → RGBA float32 for Dear PyGui texture.
- **Resolution policy:** No CPU resizes for preview; let UI scale. Recreate texture if resolution changes.

## Hotkeys
- **Ctrl+S**: Toggle recording on/off.
- **Ctrl+D**: Open “Choose Dir” dialog.
- **Enter**: Commit session name from input box.

## Performance & Stability
- **UI loop acquisition:** Call `next(camera)` each tick; if it can block, wrap with short timeout or skip on delay.
- **Queue backpressure (fallback):** Bounded queue prevents RAM blow-up. On full: drop oldest or skip enqueue (configurable policy).
- **Disk throughput:** Sequential filenames. Avoid per-frame metadata files. Optional fsync at session end for durability.
- **GIL:** Image writers generally release the GIL; saver thread won’t stall the UI thread.
- **GPU:** Dear PyGui uses GPU for blitting; keep conversions to a minimum.

## Error Handling & UX
- **Visible status messages** for:
  - Base dir not writable.
  - Session dir creation failure.
  - Frame acquisition soft errors (skip frame).
  - Disk write errors (stop recording, notify user).
- **Graceful close:** If recording, stop cleanly before exit.
- **Recovery:** If saver crashes, stop recording and continue preview with a visible error banner.

## Configuration
- **Knobs (env or `.toml`):**
  - `base_dir`
  - `max_queue_len` (fallback saver)
  - `write_png_compression` (0–9)
  - `preview_fps_cap` (optional sleep)
  - `downshift_16_to_8_strategy` (bit-shift vs. tone-map)

## Test Plan
- **Functional:**
  - Synthetic camera: deterministic gray16, BGR8/16; verify preview + recording in each mode.
  - Toggle sequencing: IDLE ↔ RECORDING ↔ STOPPING transitions; status reflects instantly.
  - Session paths: base dir and session name create correct folder structure.
- **Load:**
  - 60 FPS for 5–10 minutes; check queue stats (drops/backpressure), written file count, and UI responsiveness.
  - RAW16 sustained write; measure throughput.
- **Edge Cases:**
  - Base dir permission denied.
  - Disk full simulation.
  - Missing `camera.stop_record()` (fallback behaves correctly).
  - Resolution change mid-session (texture recreation works).

## Packaging & Ops
- **Dependencies:** `opencv-python`, `dearpygui`, `numpy`, `imageio` (or `Pillow`).
- **Run:** `python app.py`.
- **Config:** `pyproject.toml` or `requirements.txt` for reproducible setup.
- **Logging:** stdout/stderr + rotating file in session dir; include timestamps, mode changes, and exceptions.
- **Versioning:** Include app version in `manifest.json` for traceability.

## Future Extensions
- Notes panel and hotkey annotations saved to CSV.
- Overlays: FPS, histogram, ROI guides (preview-only).
- Watermark frame index (preview-only).
- Multi-camera grid, per-camera sessions.
- JSONL metadata stream (timestamps) for downstream tools.

## Build Checklist
- [ ] Scaffold Dear PyGui window with: status, base dir picker, session input, RAW checkbox, Toggle button, preview area.
- [ ] Wire `next(camera)` into the UI loop; convert frames to RGBA float32; update dynamic texture.
- [ ] Implement recording toggle:
  - [ ] Preferred: `camera.record(dir, save_raw)` / `camera.stop_record()`.
  - [ ] Fallback: saver thread + bounded queue + writers.
- [ ] Add **Ctrl+S** and **Ctrl+D**; update status text immediately.
- [ ] Graceful shutdown on close; verify gray16 and BGR8/16 save paths.
- [ ] Minimal logs + optional `manifest.json`.
