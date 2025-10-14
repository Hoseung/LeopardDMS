import numpy as np
import cv2
import pyudev
import os
import threading
import queue
import time


def sigmoid(image, k=0.001):
    def _sigmoid(x, k=1.0):
        return 1 / (1 + np.exp(-k * x))
    ir_image_sigmoid = _sigmoid(image, k)
    ir_image_sigmoid = cv2.normalize(ir_image_sigmoid, None, 0, 255, cv2.NORM_MINMAX)
    ir_image_sigmoid = np.uint8(ir_image_sigmoid)
    return ir_image_sigmoid


class LeopardDMSCamBase:
    def __init__(self, product_name="LI-VG5761", vendor_id="2a0b", product_id="00b4",
                 width=1944, height=1204, sigmoid_k=0.004, default_index=0, use_sig=True):
        """
        Initializes the LeopardDMSCam, attempting to find the camera by product name/ID.
        Falls back to default_index if not found.

        Args:
            product_name (str): The product name of the camera to search for.
            vendor_id (str): The vendor ID of the camera (hex string).
            product_id (str): The product ID of the camera (hex string).
            width (int): Desired frame width.
            height (int): Desired frame height.
            sigmoid_k (float): Sigmoid curve parameter.
            default_index (int): Fallback camera index if detection by name/ID fails.
        """
        self._sigmoid_k = sigmoid_k
        self._apply_sig = use_sig
        self._flip_h = True  # Horizontal flip
        self._flip_v = False  # Vertical flip

        # --- Camera Detection Logic ---
        device_path = None
        try:
            context = pyudev.Context()
            print(f"Searching for camera '{product_name}' (ID: {vendor_id}:{product_id})...")
            for device in context.list_devices(subsystem='video4linux'):
                try:
                    # Prefer matching by Vendor ID and Product ID for reliability
                    if device.get('ID_VENDOR_ID', '').lower() == vendor_id.lower() and \
                       device.get('ID_MODEL_ID', '').lower() == product_id.lower():
                        device_path = device.device_node
                        print(f"Found camera '{product_name}' at {device_path}.")
                        break
                    # Fallback to product name if IDs don't match (less reliable)
                    elif device.get('ID_MODEL', '') == product_name:
                         device_path = device.device_node
                         print(f"Found camera '{product_name}' (by model name) at {device_path}.")
                         break
                except Exception as e:
                    # print(f"Could not retrieve device info for {device.device_node}: {e}")
                    pass # Silently skip devices that don't have all properties

        except Exception as e:
            print(f"Warning: pyudev error during camera detection: {e}. Falling back to default index.")
            device_path = None # Ensure device_path is None if pyudev fails

        if device_path:
            self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise Exception(f"Error: Cannot access the webcam at path {device_path}. "
                                f"Please check permissions or if the camera is in use.")
        else:
            print(f"Camera '{product_name}' (ID: {vendor_id}:{product_id}) not found. "
                  f"Attempting to open default camera index {default_index}.")
            self.cap = cv2.VideoCapture(default_index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise Exception(f"Error: Cannot access the webcam at default index {default_index}. "
                                f"No camera found or accessible.")

        # --- Set Camera Properties ---
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0) # This might be specific to your camera's raw format
        print(f"Camera opened successfully. Actual resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    @property
    def flip_h(self):
        """Get horizontal flip state."""
        return self._flip_h

    @flip_h.setter
    def flip_h(self, value):
        """Set horizontal flip state."""
        self._flip_h = bool(value)

    @property
    def flip_v(self):
        """Get vertical flip state."""
        return self._flip_v

    @flip_v.setter
    def flip_v(self, value):
        """Set vertical flip state."""
        self._flip_v = bool(value)

    def release(self):
        """Releases the camera capture."""
        if self.cap.isOpened():
            self.cap.release()
            print("Camera released.")
        else:
            print("Not releasing.. Camera is not Opened")

    def __next__(self):
        """Captures and processes the next frame from the camera.

        Returns:
            tuple: (gray8_display, mono16_raw) or just gray8_display depending on internal use
        """
        ret, raw2 = self.cap.read()
        if not ret:
            print("Error: Cannot read the frame. Camera might be disconnected or in use.")
            return None

        # --- Image Processing ---
        mono16 = raw2.view(np.uint16).reshape(raw2.shape[0], raw2.shape[1])

        # Apply flips to raw if needed
        if self._flip_h and self._flip_v:
            mono16 = cv2.flip(mono16, -1)
        elif self._flip_h:
            mono16 = cv2.flip(mono16, 1)
        elif self._flip_v:
            mono16 = cv2.flip(mono16, 0)

        # Simple tone-map to 8-bit for display
        if self._apply_sig:
            gray8 = sigmoid(mono16, k=0.001)
        else:
            gray8 = np.clip(mono16 * (255.0/4095.0), 0, 255).astype(np.uint8)

        return gray8, mono16


class LeopardDMSCam(LeopardDMSCamBase):
    """Extended camera class with visualization and saving capabilities."""

    def __init__(self, product_name="LI-VG5761", vendor_id="2a0b", product_id="00b4",
                 width=1944, height=1204, sigmoid_k=0.004, default_index=0, use_sig=True):
        """Initialize the camera with visualization and saving support."""
        super().__init__(product_name, vendor_id, product_id, width, height, sigmoid_k, default_index, use_sig)
        self._save_dir = './frames'
        self._save_raw = False  # Save mono16 (raw) or gray8 (processed)

    def set_save_raw(self, save_raw=True):
        """Set whether to save raw 16-bit mono16 or processed 8-bit gray8.

        Args:
            save_raw (bool): If True, save 16-bit PNG (lossless). If False, save 8-bit PNG.
        """
        self._save_raw = bool(save_raw)
        mode_str = "16-bit raw (mono16)" if self._save_raw else "8-bit processed (gray8)"
        print(f"Save mode set to: {mode_str}")

    def _capture_thread(self, frame_queue, stop_event):
        """Thread for continuous frame capture."""
        while not stop_event.is_set():
            result = next(self)
            if result is None:
                break
            try:
                frame_queue.put(result, timeout=0.01)
            except queue.Full:
                pass  # Skip frame if queue is full

    def _save_thread(self, save_queue, stop_event, save_dir, save_raw, stats):
        """Thread for saving frames to disk."""
        
        try:
            import tifffile as tiff
            self._have_tifffile = True
        except Exception:
            print("ERROR: tifffile is required. Install with: pip install tifffile")

        
        while not stop_event.is_set() or not save_queue.empty():
            try:
                frame_data = save_queue.get(timeout=0.1)
                gray8, mono16, frame_index = frame_data

                if save_raw:
                    # Save 16-bit TIFF (lossless compression)
                    save_path = os.path.join(save_dir, f"frame_{frame_index:05d}_raw.tiff")
                    tiff.imwrite(save_path, mono16, compression='lzw', predictor=2)
                    # cv2.imwrite(save_path, mono16)
                else:
                    # Save 8-bit PNG
                    save_path = os.path.join(save_dir, f"frame_{frame_index:05d}.png")
                    cv2.imwrite(save_path, gray8)

                stats['count'] += 1
                save_queue.task_done()
            except queue.Empty:
                continue

    def show(self, window_name='LeopardDMSCam'):
        """Opens a CV2 window and displays live camera frames with threaded capture and saving.

        Args:
            window_name (str): Name of the CV2 window.

        Press 'q' to quit, 'h' to toggle horizontal flip, 'v' to toggle vertical flip,
        's' to toggle save mode."""

        print("Controls: 'q' = quit, 'h' = horizontal flip, 'v' = vertical flip, 's' = save mode")

        # Threading components
        frame_queue = queue.Queue(maxsize=10)
        save_queue = queue.Queue(maxsize=100)
        stop_event = threading.Event()

        # Start capture thread
        capture_thread = threading.Thread(target=self._capture_thread, args=(frame_queue, stop_event))
        capture_thread.daemon = True
        capture_thread.start()

        # Variables
        save_mode = False
        save_thread = None
        save_stop_event = None
        save_stats = None
        save_start_time = None
        frame_index = 0

        while True:
            try:
                gray8, mono16 = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Add overlay text
            if save_mode:
                elapsed = time.time() - save_start_time
                fps = frame_index / elapsed if elapsed > 0 else 0
                cv2.putText(gray8, f"Saved: {frame_index} frames / {elapsed:.1f}s ({fps:.1f} fps)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                save_queue.put((gray8.copy(), mono16.copy(), frame_index))
                frame_index += 1

            cv2.imshow(window_name, gray8)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self._flip_h = not self._flip_h
                print(f"Horizontal flip: {'ON' if self._flip_h else 'OFF'}")
            elif key == ord('v'):
                self._flip_v = not self._flip_v
                print(f"Vertical flip: {'ON' if self._flip_v else 'OFF'}")
            elif key == ord('s'):
                save_mode = not save_mode
                if save_mode:
                    # Start saving
                    frame_index = 0
                    os.makedirs(self._save_dir, exist_ok=True)
                    save_stats = {'count': 0}
                    save_stop_event = threading.Event()
                    save_start_time = time.time()

                    save_thread = threading.Thread(
                        target=self._save_thread,
                        args=(save_queue, save_stop_event, self._save_dir, self._save_raw, save_stats)
                    )
                    save_thread.daemon = True
                    save_thread.start()
                    mode_str = "16-bit raw" if self._save_raw else "8-bit processed"
                    print(f"Save mode: ON - Saving {mode_str} frames to {self._save_dir}/")
                else:
                    # Stop saving and report stats
                    save_stop_event.set()
                    save_thread.join(timeout=2.0)
                    elapsed = time.time() - save_start_time
                    fps = save_stats['count'] / elapsed if elapsed > 0 else 0
                    print(f"Save mode: OFF - Saved {save_stats['count']} frames in {elapsed:.2f}s ({fps:.2f} fps)")

        # Cleanup
        stop_event.set()
        capture_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("Stopped.")

    def save(self, filename, format='png', save_raw=None):
        """
        Saves a single frame to a file.

        Args:
            filename (str): Output filename (with or without extension).
            format (str): Image format, either 'png' or 'bmp' (default: 'png').
            save_raw (bool): If True, save 16-bit raw. If False, save 8-bit processed.
                           If None, use self._save_raw setting.

        Returns:
            bool: True if save was successful, False otherwise.
        """
        result = next(self)
        if result is None:
            print("Error: Could not capture frame to save.")
            return False

        gray8, mono16 = result
        use_raw = self._save_raw if save_raw is None else save_raw

        # Add extension if not present
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"

        frame_to_save = mono16 if use_raw else gray8
        success = cv2.imwrite(filename, frame_to_save)
        if success:
            mode_str = "16-bit raw" if use_raw else "8-bit processed"
            print(f"Frame saved to {filename} ({mode_str})")
        else:
            print(f"Error: Failed to save frame to {filename}")
        return success


