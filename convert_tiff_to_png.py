#!/usr/bin/env python3
"""
Convert 16-bit raw TIFF files to 8-bit PNG with sigmoid tone-mapping.

This utility applies the same sigmoid processing used in live visualization
to convert raw 16-bit TIFF files to processed 8-bit PNG files.

Usage:
    python convert_tiff_to_png.py input.tiff [output.png]
    python convert_tiff_to_png.py frames/  # Convert entire directory
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def sigmoid(image, k=0.001):
    """Apply sigmoid tone-mapping to 16-bit image.

    Args:
        image: Input 16-bit image (numpy array)
        k: Sigmoid curve parameter (default: 0.001)

    Returns:
        8-bit numpy array with sigmoid tone-mapping applied
    """
    def _sigmoid(x, k=1.0):
        return 1 / (1 + np.exp(-k * x))

    ir_image_sigmoid = _sigmoid(image, k)
    ir_image_sigmoid = cv2.normalize(ir_image_sigmoid, None, 0, 255, cv2.NORM_MINMAX)
    ir_image_sigmoid = np.uint8(ir_image_sigmoid)
    return ir_image_sigmoid


def convert_tiff_to_png(input_path, output_path=None, sigmoid_k=0.001, verbose=True):
    """Convert a single 16-bit TIFF to 8-bit PNG with sigmoid processing.

    Args:
        input_path: Path to input TIFF file
        output_path: Path to output PNG file (optional, auto-generated if None)
        sigmoid_k: Sigmoid curve parameter
        verbose: Print status messages

    Returns:
        Tuple of (success: bool, input_path: str, output_path: str)
    """
    try:
        # Load 16-bit TIFF
        mono16 = tiff.imread(str(input_path))

        # Apply sigmoid tone-mapping
        gray8 = sigmoid(mono16, k=sigmoid_k)

        # Generate output path if not provided
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = input_path_obj.parent / f"{input_path_obj.stem}_processed.png"

        # Save 8-bit PNG
        success = cv2.imwrite(str(output_path), gray8)

        if verbose and success:
            print(f"Converted: {input_path} -> {output_path}")
        elif verbose:
            print(f"Error: Failed to save {output_path}")

        return (success, str(input_path), str(output_path))

    except Exception as e:
        if verbose:
            print(f"Error converting {input_path}: {e}")
        return (False, str(input_path), None)


def convert_directory(input_dir, output_dir=None, sigmoid_k=0.001, pattern="*.tiff", num_workers=None):
    """Convert all TIFF files in a directory to PNG using parallel processing.

    Args:
        input_dir: Directory containing TIFF files
        output_dir: Output directory (optional, uses input_dir if None)
        sigmoid_k: Sigmoid curve parameter
        pattern: File pattern to match (default: "*.tiff")
        num_workers: Number of parallel workers (default: cpu_count())

    Returns:
        Tuple of (successful_count, failed_count)
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching TIFF files
    tiff_files = sorted(input_dir.glob(pattern))

    if not tiff_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return 0, 0

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Found {len(tiff_files)} TIFF files to convert")
    print(f"Using {num_workers} parallel workers")

    success_count = 0
    fail_count = 0

    # Prepare conversion tasks
    tasks = []
    for tiff_file in tiff_files:
        output_file = output_dir / f"{tiff_file.stem}_processed.png"
        tasks.append((tiff_file, output_file, sigmoid_k))

    # Process files in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(convert_tiff_to_png, input_path, output_path, sk, False): (input_path, output_path)
            for input_path, output_path, sk in tasks
        }

        # Process results with progress bar
        with tqdm(total=len(futures), desc="Converting", unit="file") as pbar:
            for future in as_completed(futures):
                success, input_path, output_path = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    tqdm.write(f"Failed: {input_path}")
                pbar.update(1)

    print(f"\nConversion complete: {success_count} successful, {fail_count} failed")
    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert 16-bit TIFF files to 8-bit PNG with sigmoid tone-mapping"
    )
    parser.add_argument(
        "input",
        help="Input TIFF file or directory"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output PNG file or directory (optional, auto-generated if not provided)"
    )
    parser.add_argument(
        "-k", "--sigmoid-k",
        type=float,
        default=0.001,
        help="Sigmoid curve parameter (default: 0.001)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="*.tiff",
        help="File pattern for directory conversion (default: *.tiff)"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)

    # Handle directory conversion
    if input_path.is_dir():
        convert_directory(input_path, args.output, args.sigmoid_k, args.pattern, args.jobs)

    # Handle single file conversion
    elif input_path.is_file():
        success, _, _ = convert_tiff_to_png(input_path, args.output, args.sigmoid_k, verbose=True)
        if not success:
            sys.exit(1)

    else:
        print(f"Error: '{input_path}' is neither a file nor a directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
