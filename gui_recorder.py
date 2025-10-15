#!/usr/bin/env python3
"""
Live Camera Recorder GUI - No Qt Version
Uses Dear PyGui for GPU-accelerated live preview and recording control.
"""

import os
import sys
import time
import json
import threading
import queue
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import dearpygui.dearpygui as dpg

# Import camera API
from datagenerator import LeopardDMSCam


class RecorderState:
    """State machine for recording control."""
    IDLE = "idle"
    RECORDING = "recording"
    STOPPING = "stopping"


class CameraRecorderGUI:
    """Main GUI application for live camera preview and recording."""

    def __init__(self, camera=None):
        """Initialize the GUI application.

        Args:
            camera: Camera object with next() method. If None, creates default LeopardDMSCam.
        """
        # Camera
        self.camera = camera if camera is not None else LeopardDMSCam()

        # State
        self.state = RecorderState.IDLE
        self.base_dir = str(Path.home() / "recordings")
        self.session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_raw = False
        self.frame_index = 0

        # Recording FPS control
        self.recording_fps = 30  # Default FPS
        self.last_save_time = None
        self.frame_interval = None  # Will be set based on FPS

        # Recording
        self.save_queue = None
        self.save_thread = None
        self.save_stop_event = None
        self.session_dir = None
        self.recording_start_time = None

        # Preview
        self.texture_width = None
        self.texture_height = None
        self.texture_tag = "camera_texture"
        self.image_tag = "camera_image"
        self.rgba_buffer = None  # Reusable RGBA buffer

        # Status
        self.status_message = "Ready"
        self.frames_saved = 0

        # Configuration
        self.max_queue_len = 100
        self.png_compression = 3

        # GUI state
        self.running = True

    def frame_to_rgba_float32(self, frame, reuse_buffer=True):
        """Convert frame to RGBA float32 [0-1] for Dear PyGui texture.

        Args:
            frame: numpy array (H, W) gray or (H, W, 3) BGR, uint8 or uint16
            reuse_buffer: If True, reuse internal buffer to avoid allocations

        Returns:
            numpy array (H, W, 4) RGBA float32 in range [0, 1]
        """
        h, w = frame.shape[:2]

        # Create or reuse buffer
        if reuse_buffer:
            if self.rgba_buffer is None or self.rgba_buffer.shape[:2] != (h, w):
                self.rgba_buffer = np.ones((h, w, 4), dtype=np.float32)
            rgba = self.rgba_buffer
        else:
            rgba = np.ones((h, w, 4), dtype=np.float32)

        # Handle different frame types - optimize for uint8 grayscale (most common)
        if len(frame.shape) == 2:
            # Grayscale - most common case
            if frame.dtype == np.uint8:
                # Fast path: use numpy broadcasting directly into buffer
                gray_float = frame.astype(np.float32, copy=False) * (1.0 / 255.0)
            else:  # uint16
                gray_float = frame.astype(np.float32, copy=False) * (1.0 / 65535.0)

            # Write directly to RGB channels (avoid intermediate allocation)
            rgba[:, :, 0] = gray_float
            rgba[:, :, 1] = gray_float
            rgba[:, :, 2] = gray_float
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            # BGR color
            if frame.dtype == np.uint16:
                bgr_float = frame.astype(np.float32, copy=False) * (1.0 / 65535.0)
            else:
                bgr_float = frame.astype(np.float32, copy=False) * (1.0 / 255.0)
            # BGR to RGB: swap channels directly
            rgba[:, :, 0] = bgr_float[:, :, 2]  # R = B
            rgba[:, :, 1] = bgr_float[:, :, 1]  # G = G
            rgba[:, :, 2] = bgr_float[:, :, 0]  # B = R
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        return rgba

    def save_thread_worker(self):
        """Worker thread for saving frames to disk."""
        try:
            import tifffile as tiff
            have_tifffile = True
        except ImportError:
            have_tifffile = False
            if self.save_raw:
                self.update_status("ERROR: tifffile required for RAW16 saving")
                return

        try:
            while not self.save_stop_event.is_set() or not self.save_queue.empty():
                try:
                    frame_data = self.save_queue.get(timeout=0.1)
                    gray8, mono16, idx = frame_data

                    if self.save_raw and have_tifffile and mono16 is not None:
                        # Save 16-bit TIFF
                        save_path = self.session_dir / f"{idx:06d}.tiff"
                        tiff.imwrite(str(save_path), mono16, compression='lzw', predictor=2)
                    elif not self.save_raw and gray8 is not None:
                        # Save 8-bit PNG
                        save_path = self.session_dir / f"{idx:06d}.png"
                        cv2.imwrite(str(save_path), gray8, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression])
                    else:
                        print(f"Warning: Skipping frame {idx} - missing data")

                    self.frames_saved += 1
                    self.save_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    self.update_status(f"Save error: {e}")
                    print(f"Error saving frame: {e}")

        except Exception as e:
            self.update_status(f"Saver thread crashed: {e}")
            print(f"Saver thread error: {e}")

    def update_status(self, message):
        """Update status message (thread-safe)."""
        self.status_message = message

    def toggle_recording(self):
        """Toggle recording on/off."""
        if self.state == RecorderState.IDLE:
            # Start recording
            self.start_recording()
        elif self.state == RecorderState.RECORDING:
            # Stop recording
            self.stop_recording()

    def start_recording(self):
        """Start recording frames."""
        try:
            # Validate base directory
            base_path = Path(self.base_dir)
            if not base_path.exists():
                base_path.mkdir(parents=True, exist_ok=True)

            if not os.access(base_path, os.W_OK):
                self.update_status(f"ERROR: Base directory not writable: {base_path}")
                return

            # Create session directory
            self.session_dir = base_path / self.session_name
            if self.session_dir.exists() and list(self.session_dir.iterdir()):
                # Directory exists and not empty - create timestamped subdirectory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.session_dir = base_path / self.session_name / f"rec_{timestamp}"

            self.session_dir.mkdir(parents=True, exist_ok=True)

            # Initialize recording state
            self.frame_index = 0
            self.frames_saved = 0
            self.recording_start_time = time.time()
            self.last_save_time = time.time()
            self.state = RecorderState.RECORDING

            # Set frame interval based on FPS (None means MAX FPS)
            if self.recording_fps > 0:
                self.frame_interval = 1.0 / self.recording_fps
            else:
                self.frame_interval = None  # MAX FPS

            # Create manifest
            manifest = {
                "version": "1.0",
                "start_time": datetime.now().isoformat(),
                "mode": "raw16" if self.save_raw else "png8",
                "recording_fps": self.recording_fps if self.recording_fps > 0 else "MAX",
                "session_dir": str(self.session_dir),
                "base_dir": str(base_path),
            }
            with open(self.session_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # Start saver thread
            self.save_queue = queue.Queue(maxsize=self.max_queue_len)
            self.save_stop_event = threading.Event()
            self.save_thread = threading.Thread(target=self.save_thread_worker, daemon=True)
            self.save_thread.start()

            mode_str = "RAW16" if self.save_raw else "PNG8"
            fps_str = f"{self.recording_fps} FPS" if self.recording_fps > 0 else "MAX FPS"
            self.update_status(f"● RECORDING {mode_str} @ {fps_str} to {self.session_dir.name}")
            print(f"Recording started: {self.session_dir} ({fps_str})")

        except Exception as e:
            self.update_status(f"ERROR starting recording: {e}")
            self.state = RecorderState.IDLE
            print(f"Error starting recording: {e}")

    def stop_recording(self):
        """Stop recording frames."""
        if self.state != RecorderState.RECORDING:
            return

        self.state = RecorderState.STOPPING
        self.update_status("Stopping recording...")

        try:
            # Signal saver thread to stop
            if self.save_stop_event:
                self.save_stop_event.set()

            # Wait for saver thread to finish
            if self.save_thread and self.save_thread.is_alive():
                self.save_thread.join(timeout=5.0)

            # Update manifest with end time
            if self.session_dir:
                manifest_path = self.session_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)

                    elapsed = time.time() - self.recording_start_time
                    manifest["end_time"] = datetime.now().isoformat()
                    manifest["duration_seconds"] = elapsed
                    manifest["frames_saved"] = self.frames_saved
                    manifest["fps"] = self.frames_saved / elapsed if elapsed > 0 else 0

                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)

            elapsed = time.time() - self.recording_start_time
            fps = self.frames_saved / elapsed if elapsed > 0 else 0
            self.update_status(f"○ Stopped. Saved {self.frames_saved} frames in {elapsed:.1f}s ({fps:.1f} fps)")
            print(f"Recording stopped: {self.frames_saved} frames, {fps:.1f} fps")

        except Exception as e:
            self.update_status(f"ERROR stopping recording: {e}")
            print(f"Error stopping recording: {e}")
        finally:
            self.state = RecorderState.IDLE

    def open_directory_dialog(self):
        """Open directory picker dialog."""
        # Dear PyGui doesn't have built-in file dialogs, so we'll use a simple input
        # In production, you could use tkinter.filedialog or system dialog
        print("Directory dialog would open here. Current:", self.base_dir)
        # For now, just print - user can type path manually

    def create_gui(self):
        """Create the Dear PyGui interface."""
        dpg.create_context()

        # Setup window
        with dpg.window(label="Camera Recorder", tag="main_window", width=1200, height=900, no_close=True):

            # Control panel
            with dpg.group(horizontal=False):
                # Status
                dpg.add_text("Status:", color=(200, 200, 200))
                dpg.add_text(self.status_message, tag="status_text", color=(0, 255, 0))

                dpg.add_separator()

                # Base directory
                dpg.add_text("Base Directory:")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        default_value=self.base_dir,
                        tag="base_dir_input",
                        width=500,
                        callback=lambda s, d: setattr(self, 'base_dir', d)
                    )
                    dpg.add_button(label="Choose Dir (Ctrl+D)", callback=self.open_directory_dialog)

                # Session name
                dpg.add_text("Session Name:")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        default_value=self.session_name,
                        tag="session_name_input",
                        width=500,
                        callback=lambda s, d: setattr(self, 'session_name', d)
                    )

                # Save mode
                dpg.add_text("Save Mode:")
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Save RAW16 (TIFF, otherwise PNG8)",
                        tag="save_raw_checkbox",
                        default_value=self.save_raw,
                        callback=lambda s, d: setattr(self, 'save_raw', d)
                    )

                # Recording FPS
                dpg.add_text("Recording FPS:")
                with dpg.group(horizontal=True):
                    def fps_callback(sender, fps_value):
                        self.recording_fps = fps_value
                        if fps_value > 0:
                            print(f"Recording FPS set to: {fps_value}")
                        else:
                            print("Recording FPS set to: MAX")

                    dpg.add_radio_button(
                        items=["5 FPS", "10 FPS", "15 FPS", "30 FPS", "MAX"],
                        tag="fps_selector",
                        default_value="30 FPS",
                        horizontal=True,
                        callback=lambda s, d: fps_callback(s, {
                            "5 FPS": 5,
                            "10 FPS": 10,
                            "15 FPS": 15,
                            "30 FPS": 30,
                            "MAX": 0
                        }[d])
                    )

                dpg.add_separator()

                # Recording control
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Toggle Record (Ctrl+S)",
                        tag="toggle_record_button",
                        callback=lambda: self.toggle_recording(),
                        width=200,
                        height=40
                    )
                    dpg.add_text("", tag="record_status", color=(255, 100, 100))

                dpg.add_separator()

            # Preview area
            dpg.add_text("Live Preview:", color=(200, 200, 200))

            # Placeholder for camera image
            # Will be created dynamically when first frame arrives
            with dpg.group(tag="image_container"):
                dpg.add_text("Waiting for camera frames...", tag="waiting_text")

        # Setup viewport
        dpg.create_viewport(title="Camera Recorder - Live View", width=1280, height=960)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

        # Register keyboard handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_S, callback=self.handle_ctrl_s)
            dpg.add_key_press_handler(dpg.mvKey_D, callback=self.handle_ctrl_d)

    def handle_ctrl_s(self, sender, key):
        """Handle Ctrl+S hotkey."""
        # Check if Ctrl is pressed
        if dpg.is_key_down(dpg.mvKey_Control):
            self.toggle_recording()

    def handle_ctrl_d(self, sender, key):
        """Handle Ctrl+D hotkey."""
        # Check if Ctrl is pressed
        if dpg.is_key_down(dpg.mvKey_Control):
            self.open_directory_dialog()

    def update_frame(self):
        """Update the camera frame in the UI."""
        try:
            # Get next frame from camera
            result = next(self.camera)
            if result is None:
                self.update_status("ERROR: Cannot read frame")
                return

            gray8, mono16 = result

            # If recording, check FPS limit and enqueue frame if it's time
            if self.state == RecorderState.RECORDING:
                current_time = time.time()
                should_save = False

                # Check if enough time has passed based on FPS setting
                if self.frame_interval is None:
                    # MAX FPS - save every frame
                    should_save = True
                else:
                    # Limited FPS - check if interval has passed
                    time_since_last = current_time - self.last_save_time
                    if time_since_last >= self.frame_interval:
                        should_save = True
                        self.last_save_time = current_time

                if should_save:
                    try:
                        # Only copy what we need to save (avoid copying both if only saving one)
                        if self.save_raw:
                            self.save_queue.put_nowait((None, mono16.copy(), self.frame_index))
                        else:
                            self.save_queue.put_nowait((gray8.copy(), None, self.frame_index))
                        self.frame_index += 1

                        # Update recording status (do this less frequently to reduce overhead)
                        if self.frame_index % 10 == 0:
                            elapsed = time.time() - self.recording_start_time
                            fps = self.frame_index / elapsed if elapsed > 0 else 0
                            dpg.set_value("record_status", f"Frames: {self.frame_index} ({fps:.1f} fps)")

                    except queue.Full:
                        # Queue full - drop frame
                        print("Warning: Save queue full, dropping frame")
            else:
                dpg.set_value("record_status", "")

            # Resize preview to half size for performance (simple subsample by 2)
            gray8_preview = gray8[::2, ::2]

            # Convert to RGBA float32 for texture
            rgba_float = self.frame_to_rgba_float32(gray8_preview)
            h, w = rgba_float.shape[:2]

            # Create or update texture
            if self.texture_width != w or self.texture_height != h:
                # Recreate texture with new dimensions
                if dpg.does_item_exist(self.texture_tag):
                    dpg.delete_item(self.texture_tag)
                if dpg.does_item_exist(self.image_tag):
                    dpg.delete_item(self.image_tag)

                # Remove waiting text if it exists
                if dpg.does_item_exist("waiting_text"):
                    dpg.delete_item("waiting_text")

                # Create new texture
                with dpg.texture_registry():
                    dpg.add_raw_texture(
                        width=w,
                        height=h,
                        default_value=rgba_float.flatten(),
                        format=dpg.mvFormat_Float_rgba,
                        tag=self.texture_tag
                    )

                # Add image widget
                dpg.add_image(self.texture_tag, tag=self.image_tag, parent="image_container")

                self.texture_width = w
                self.texture_height = h
            else:
                # Update existing texture
                dpg.set_value(self.texture_tag, rgba_float.flatten())

        except Exception as e:
            self.update_status(f"Frame error: {e}")
            print(f"Error updating frame: {e}")

    def update_ui(self):
        """Update UI elements (status, etc.)."""
        # Update status text
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", self.status_message)

            # Color based on state
            if self.state == RecorderState.RECORDING:
                dpg.configure_item("status_text", color=(255, 0, 0))  # Red when recording
            elif "ERROR" in self.status_message:
                dpg.configure_item("status_text", color=(255, 100, 0))  # Orange for errors
            else:
                dpg.configure_item("status_text", color=(0, 255, 0))  # Green otherwise

    def run(self):
        """Main application loop."""
        self.create_gui()

        # Main loop
        while dpg.is_dearpygui_running() and self.running:
            # Update frame
            self.update_frame()

            # Update UI
            self.update_ui()

            # Render
            dpg.render_dearpygui_frame()

            # No sleep - let camera frame rate control the loop
            # The next(camera) call will block appropriately

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """Cleanup resources before exit."""
        print("Cleaning up...")

        # Stop recording if active
        if self.state == RecorderState.RECORDING:
            print("Stopping active recording...")
            self.stop_recording()

        # Release camera
        if hasattr(self.camera, 'release'):
            self.camera.release()

        # Cleanup Dear PyGui
        dpg.destroy_context()

        print("Cleanup complete")


def main():
    """Main entry point."""
    try:
        # Create camera instance
        print("Initializing camera...")
        camera = LeopardDMSCam(use_sig=True)

        # Create and run GUI
        print("Starting GUI...")
        app = CameraRecorderGUI(camera=camera)
        app.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
