# LeopardDMS Camera Interface

Python interface for the Leopard Imaging LI-VG5761 DMS (Driver Monitoring System) camera.

## Features

- Automatic camera detection via USB Vendor/Product ID
- 1944 × 1204 single-channel (grayscale) image capture
- Real-time live preview with CV2
- Image flip controls (horizontal/vertical)
- Frame saving (PNG/BMP/TIFF formats)
- Continuous recording mode with 16-bit raw or 8-bit processed output
- Threaded architecture for high-performance capture and saving

## Requirements

```bash
pip install numpy opencv-python pyudev tifffile imagecodecs
```

## Usage

### Basic Setup

```python
from datagenerator import LeopardDMSCam

# Initialize camera (auto-detects Leopard camera)
cam = LeopardDMSCam(2)  # or use default_index parameter

# Capture single frame
frame = next(cam)

# Release camera
cam.release()
```

### Live Preview

```python
# Show live camera feed
cam.show()
```

**Keyboard Controls:**
- `q` - Quit
- `h` - Toggle horizontal flip
- `v` - Toggle vertical flip
- `s` - Toggle save mode (saves frames to `./frames/`)

### Save Single Frame

```python
# Save as PNG (default)
cam.save("image.png")

# Save as BMP
cam.save("image.bmp", format='bmp')

# Save 16-bit raw data (lossless)
cam.set_save_raw(True)
cam.save("image_raw.png")
```

### Flip Control

```python
# Set flip properties
cam.flip_h = True   # Horizontal flip
cam.flip_v = True   # Vertical flip
```

### Continuous Recording Mode

```python
# Set to save 16-bit raw TIFF files (lossless, ~2.3MB per frame)
cam.set_save_raw(True)

# Show live preview and press 's' to start/stop recording
cam.show()
```

During live preview, press `s` to toggle recording. When recording is active:
- 16-bit raw mode: Saves lossless TIFF files (`frame_00000_raw.tiff`)
- 8-bit processed mode: Saves PNG files (`frame_00000.png`)
- Live FPS statistics shown on screen

## Camera Specifications

- **Model**: Leopard Imaging LI-VG5761
- **Resolution**: 1944 × 1204
- **Format**: Single-channel grayscale (8-bit)
- **USB ID**: 2a0b:00b4

## Class Parameters

```python
LeopardDMSCam(
    product_name="LI-VG5761",  # Camera model name
    vendor_id="2a0b",           # USB vendor ID
    product_id="00b4",          # USB product ID
    width=1944,                 # Frame width
    height=1204,                # Frame height
    sigmoid_k=0.004,            # Sigmoid tone-mapping parameter
    default_index=0,            # Fallback camera index
    use_sig=True                # Apply sigmoid tone-mapping
)
```

## Notes

- Camera detection uses pyudev to find the device by USB ID
- Falls back to default camera index if detection fails
- Frames are saved to `./frames/` directory when save mode is enabled
- Image processing includes optional sigmoid tone-mapping for better contrast
