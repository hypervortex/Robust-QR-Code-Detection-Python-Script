
# Robust QR Code Detector

## Overview
The **Robust QR Code Detector** is a Python-based tool designed to detect QR codes from images using multiple preprocessing techniques to improve accuracy. It supports batch processing of images in a directory and generates detailed reports on scan results.

## Features
- Detects QR codes using multiple preprocessing methods
- Saves detected QR code data in JSON format
- Logs scan statistics, including detection performance per method
- Supports batch processing of images in a directory
- Debug mode for detailed logging and image saving

## Dependencies
Ensure you have the following dependencies installed before running the script:

```sh
pip install opencv-python numpy pyzbar imghdr
```

## Usage
### Running the Script
Run the script to process all images in the current directory:

```sh
python script.py
```

### Debug Mode
To enable debug mode for detailed logs and intermediate image saving, modify the `debug` parameter:

```python
detector = RobustQRCodeDetector(debug=True)
```

## Output Files
- **QR Code Detection Results**: JSON file containing detected QR codes for each image.
- **Scan Summary**: `qr_scan_summary.json` with statistics on the scanning process.
- **Debug Images** (if enabled): Preprocessed images stored in `debug_preprocessed/`.

## License
This project is open-source and available under the MIT License.


