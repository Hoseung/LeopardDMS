import numpy as np
import cv2
import pyudev
import os

        
def sigmoid(image, k=0.001):
    def _sigmoid(x, k=1.0):
        return 1 / (1 + np.exp(-k * x))
    ir_image_sigmoid = _sigmoid(image, k)
    ir_image_sigmoid = cv2.normalize(ir_image_sigmoid, None, 0, 255, cv2.NORM_MINMAX)
    ir_image_sigmoid = np.uint8(ir_image_sigmoid)
    return ir_image_sigmoid


class LeopardDMSCam:
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
        self._save_dir = './frames'

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
        """Captures and processes the next frame from the camera."""
        ret, raw2 = self.cap.read()
        if not ret:
            print("Error: Cannot read the frame. Camera might be disconnected or in use.")
            return None

        # --- Image Processing ---
        mono16 = raw2.view(np.uint16).reshape(raw2.shape[0], raw2.shape[1])
        # Simple tone-map to 8-bit; replace with your sigmoid if you like
        if self._apply_sig:
            gray8 = sigmoid(mono16, k=0.001)
        else:
            gray8 = np.clip(mono16 * (255.0/4095.0), 0, 255).astype(np.uint8)

        # Apply flips
        if self._flip_h and self._flip_v:
            gray8 = cv2.flip(gray8, -1)  # Both horizontal and vertical
        elif self._flip_h:
            gray8 = cv2.flip(gray8, 1)   # Horizontal only
        elif self._flip_v:
            gray8 = cv2.flip(gray8, 0)   # Vertical only

        return gray8
        # return cv2.merge([frame, frame, frame]) # Return as 3-channel grayscale for consistency with some CV ops

    def show(self, window_name='LeopardDMSCam'):
        """Opens a CV2 window and displays live camera frames.
        Press 'q' to quit, 'h' to toggle horizontal flip, 'v' to toggle vertical flip, 's' to toggle save mode."""
        print("Controls: 'q' = quit, 'h' = toggle horizontal flip, 'v' = toggle vertical flip, 's' = toggle save mode")

        save_mode = False
        frame_index = 0

        while True:
            frame = next(self)
            if frame is None:
                break

            # Add frame index text if in save mode
            # display_frame = frame.copy()
            if save_mode:
                save_path = os.path.join(self._save_dir, f"frame_{frame_index:05d}.png")
                cv2.imwrite(save_path, frame)
                cv2.putText(frame, f"Saving: Frame {frame_index}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                frame_index += 1

            cv2.imshow(window_name, frame)

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
                    frame_index = 0
                    os.makedirs(self._save_dir, exist_ok=True)
                    print(f"Save mode: ON - Saving frames to {self._save_dir}/frame_XXXXX.png")
                else:
                    print(f"Save mode: OFF - Saved {frame_index} frames to {self._save_dir}")

        cv2.destroyAllWindows()

    def save(self, filename, format='png'):
        """
        Saves a single frame to a file.

        Args:
            filename (str): Output filename (with or without extension).
            format (str): Image format, either 'png' or 'bmp' (default: 'png').

        Returns:
            bool: True if save was successful, False otherwise.
        """
        frame = next(self)
        if frame is None:
            print("Error: Could not capture frame to save.")
            return False

        # Add extension if not present
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"

        success = cv2.imwrite(filename, frame)
        if success:
            print(f"Frame saved to {filename}")
        else:
            print(f"Error: Failed to save frame to {filename}")
        return success


