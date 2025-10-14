# LeopardDMS Camera Interface

Python interface for the Leopard Imaging LI-VG5761 DMS (Driver Monitoring System) camera.

## Features

- Automatic camera detection via USB Vendor/Product ID
- 1944 × 1204 single-channel (grayscale) image capture
- Real-time live preview with CV2
- Image flip controls (horizontal/vertical)
- Frame saving (PNG/BMP formats)
- Continuous recording mode

## Requirements

```bash
pip install numpy opencv-python pyudev
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
```

### Flip Control

```python
# Set flip properties
cam.flip_h = True   # Horizontal flip
cam.flip_v = True   # Vertical flip
```

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
