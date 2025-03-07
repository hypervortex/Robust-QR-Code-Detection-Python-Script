import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import json
import time
from datetime import datetime
import imghdr
from concurrent.futures import ThreadPoolExecutor, as_completed

class RobustQRCodeDetector:
    def __init__(self, debug=False, max_workers=None):
        """
        Initialize the QR code detector with performance optimizations
        
        Args:
            debug (bool): Enable detailed logging
            max_workers (int): Number of concurrent processing threads
        """
        self.debug = debug
        self.max_workers = max_workers or (os.cpu_count() or 4)
        

        self.scan_results = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_images_scanned': 0,
            'images_with_qr_codes': 0,
            'total_qr_codes_detected': 0,
            'detection_methods_performance': {},
            'processing_time': 0,
            'detection_rate': 0.0,
            'detected_qr_codes': []
        }
        
  
        self.preprocessing_methods = [
            ('original_grayscale', self._preprocess_original_grayscale),
            ('enhanced_contrast', self._preprocess_enhanced_contrast),
            ('otsu_threshold', self._preprocess_otsu_threshold),
            ('bilateral_filter', self._preprocess_bilateral_filter)
        ]

    def _preprocess_original_grayscale(self, image):
        """Convert image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    def _preprocess_enhanced_contrast(self, image):
        """Enhance image contrast"""
        gray = self._preprocess_original_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)

    def _preprocess_otsu_threshold(self, image):
        """Apply Otsu's thresholding"""
        gray = self._preprocess_original_grayscale(image)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _preprocess_bilateral_filter(self, image):
        """Apply bilateral filtering"""
        gray = self._preprocess_original_grayscale(image)
        return cv2.bilateralFilter(gray, 9, 75, 75)

    def _validate_image(self, file_path):
        """
        Validate if the file is a valid image
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            bool: True if valid image, False otherwise
        """
        try:
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                return False
            
            return imghdr.what(file_path) is not None
        except Exception:
            return False

    def detect_qr_codes(self, image_path):
        """
        Detect QR codes using multiple preprocessing techniques
        
        Args:
            image_path (str): Path to the input image
        
        Returns:
            list: Unique QR code detections
        """
        start_time = time.time()
        self.scan_results['total_images_scanned'] += 1
        
        try:
            # Validate and read image
            if not self._validate_image(image_path):
                return []

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                return []

     
            unique_qr_codes = {}
            method_detection_counts = {}
            
            for method_name, preprocess_func in self.preprocessing_methods:
                try:
                    preprocessed_img = preprocess_func(image)
                    decoded_objects = decode(preprocessed_img)
                    
                    method_detection_counts[method_name] = len(decoded_objects)
                    
                    for obj in decoded_objects:
                        qr_data = obj.data.decode('utf-8')
                        
                        # Only add if not already detected
                        if qr_data not in unique_qr_codes:
                            unique_qr_codes[qr_data] = {
                                'data': qr_data,
                                'image': os.path.basename(image_path),
                                'detection_method': method_name,
                                'detection_timestamp': datetime.now().isoformat()
                            }
                
                except Exception as e:
                    print(f"Error in {method_name} preprocessing: {e}")

   
            detected_codes = list(unique_qr_codes.values())
            if detected_codes:
                self.scan_results['images_with_qr_codes'] += 1
                self.scan_results['total_qr_codes_detected'] += len(detected_codes)
                self.scan_results['detected_qr_codes'].extend(detected_codes)
            
            for method, count in method_detection_counts.items():
                self.scan_results['detection_methods_performance'][method] = \
                    self.scan_results['detection_methods_performance'].get(method, 0) + count

            return detected_codes

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []

    def process_images_in_directory(self, directory):
        """
        Process images in directory using concurrent processing
        
        Args:
            directory (str): Path to the directory containing images
        
        Returns:
            list: Detected QR codes across all images
        """
        start_total_time = time.time()
        
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_paths = [
            os.path.join(directory, filename) 
            for filename in os.listdir(directory) 
            if filename.lower().endswith(supported_extensions)
        ]

    
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit detection tasks
            futures = [executor.submit(self.detect_qr_codes, path) for path in image_paths]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()

        total_processing_time = time.time() - start_total_time
        self.scan_results['processing_time'] = total_processing_time
       
        if self.scan_results['total_images_scanned'] > 0:
            self.scan_results['detection_rate'] = (
                self.scan_results['images_with_qr_codes'] / 
                self.scan_results['total_images_scanned']
            ) * 100

        return self.scan_results

def main():
  
    detector = RobustQRCodeDetector(debug=True)
    
    scan_results = detector.process_images_in_directory(os.getcwd())
    
    with open('qr_code_scan_results.json', 'w') as f:
        json.dump(scan_results, f, indent=4)
    
    print("\n--- QR Code Scan Summary ---")
    print(f"Total Images Scanned: {scan_results['total_images_scanned']}")
    print(f"Images with QR Codes: {scan_results['images_with_qr_codes']}")
    print(f"Total QR Codes Detected: {scan_results['total_qr_codes_detected']}")
    print(f"Detection Rate: {scan_results['detection_rate']:.2f}%")
    print(f"Total Processing Time: {scan_results['processing_time']:.2f} seconds")

if __name__ == "__main__":
    main()
