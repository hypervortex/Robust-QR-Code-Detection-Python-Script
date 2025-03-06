import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import json
import logging
import time
from datetime import datetime
import imghdr

class RobustQRCodeDetector:
    def __init__(self, debug=False):
        """
        Initialize the QR code detector with comprehensive tracking
        
        Args:
            debug (bool): Enable detailed logging and intermediate image saving
        """
        self.debug = debug
        self.logger = self._setup_logger()
        
        self.scan_stats = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_images_scanned': 0,
            'images_with_qr_codes': 0,
            'images_without_qr_codes': 0,
            'total_qr_codes_detected': 0,
            'preprocessing_methods_performance': {},
            'total_scan_time': 0,
            'average_image_processing_time': 0,
            'detection_details': []
        }

    def _setup_logger(self):
        """
        Set up logging for the detector
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('AdvancedQRCodeDetector')
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if self.debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def preprocess_image(self, image):
        """
        Advanced image preprocessing techniques
        
        Args:
            image (numpy.ndarray): Input image
        
        Returns:
            list of tuple: (method_name, preprocessed_image)
        """
        preprocessed_images = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        preprocessing_methods = [
            ('original_grayscale', gray),
            ('enhanced_contrast', self._enhance_contrast(gray)),
            ('adaptive_threshold', self._adaptive_threshold(gray)),
            ('otsu_threshold', self._otsu_threshold(gray)),
            ('gaussian_blur', self._gaussian_blur(gray)),
            ('median_blur', self._median_blur(gray)),
            ('bilateral_filter', self._bilateral_filter(gray)),
            ('equalized_histogram', self._equalize_histogram(gray)),
            ('morphological_opening', self._morphological_opening(gray)),
            ('laplacian_edge_enhancement', self._laplacian_edge_enhancement(gray))
        ]
        
        if self.debug:
            self._save_debug_images(preprocessing_methods)
        
        return preprocessing_methods

    def _enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

    def _adaptive_threshold(self, image):
        """Apply adaptive thresholding"""
        return cv2.adaptiveThreshold(
            image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

    def _otsu_threshold(self, image):
        """Apply Otsu's thresholding"""
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _gaussian_blur(self, image):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (5, 5), 0)

    def _median_blur(self, image):
        """Apply median blur"""
        return cv2.medianBlur(image, 5)

    def _bilateral_filter(self, image):
        """Apply bilateral filtering"""
        return cv2.bilateralFilter(image, 9, 75, 75)

    def _equalize_histogram(self, image):
        """Equalize image histogram"""
        return cv2.equalizeHist(image)

    def _morphological_opening(self, image):
        """Apply morphological opening"""
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def _laplacian_edge_enhancement(self, image):
        """Enhance edges using Laplacian"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    def _save_debug_images(self, preprocessed_images):
        """
        Save preprocessed images for debugging
        
        Args:
            preprocessed_images (list): List of (method_name, image) tuples
        """
        debug_dir = os.path.join(os.getcwd(), 'debug_preprocessed')
        os.makedirs(debug_dir, exist_ok=True)
        
        for method_name, img in preprocessed_images:
            cv2.imwrite(os.path.join(debug_dir, f'preprocessed_{method_name}.png'), img)

    def detect_qr_codes(self, image_path):
        """
        Detect QR codes using multiple preprocessing techniques
        
        Args:
            image_path (str): Path to the input image
        
        Returns:
            tuple: (detection_results, processing_time)
        """
        start_time = time.time()
        self.scan_stats['total_images_scanned'] += 1
        
        try:
            if not self._validate_image(image_path):
                self.logger.warning(f"Invalid image: {image_path}")
                return [], 0

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self.logger.error(f"Cannot read image: {image_path}")
                return [], 0

            preprocessed_images = self.preprocess_image(image)
            
            all_detected_qr_codes = []
            method_detection_counts = {}
            
            for method_name, preprocessed_img in preprocessed_images:
                decoded_objects = decode(preprocessed_img)
                
                method_detection_counts[method_name] = len(decoded_objects)
                
                for obj in decoded_objects:
                    qr_data = {
                        'filename': os.path.basename(image_path),
                        'data': obj.data.decode('utf-8'),
                        'type': obj.type,
                        'detection_method': method_name,
                        'polygon': [list(p) for p in obj.polygon],
                        'rect': [obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height]
                    }
                    all_detected_qr_codes.append(qr_data)
            
            unique_qr_codes = list({qr['data']: qr for qr in all_detected_qr_codes}.values())
            
            processing_time = time.time() - start_time
            self.scan_stats['total_scan_time'] += processing_time
            
            if unique_qr_codes:
                self.scan_stats['images_with_qr_codes'] += 1
                self.scan_stats['total_qr_codes_detected'] += len(unique_qr_codes)
                self.scan_stats['detection_details'].extend(unique_qr_codes)
            else:
                self.scan_stats['images_without_qr_codes'] += 1
            
            self.scan_stats['preprocessing_methods_performance'] = method_detection_counts
            
            return unique_qr_codes, processing_time

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return [], 0

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
            
            image_type = imghdr.what(file_path)
            return image_type is not None
        except Exception:
            return False

    def save_detection_results(self, results, image_path):
        """
        Save QR code detection results to a JSON file
        
        Args:
            results (list): QR code detection results
            image_path (str): Path to the original image
        """
        output_file = os.path.join(
            os.getcwd(), 
            os.path.splitext(os.path.basename(image_path))[0] + "_qr_results.json"
        )

        existing_results = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_results = json.load(f)

        existing_results.extend(results)

        with open(output_file, 'w') as f:
            json.dump(existing_results, f, indent=4)

    def save_scan_summary(self):
        """
        Save comprehensive scan summary to a JSON file
        """
        if self.scan_stats['total_images_scanned'] > 0:
            self.scan_stats['detection_percentage'] = (
                (self.scan_stats['images_with_qr_codes'] / self.scan_stats['total_images_scanned']) * 100
            )
            self.scan_stats['average_image_processing_time'] = (
                self.scan_stats['total_scan_time'] / self.scan_stats['total_images_scanned']
            )

        summary_file = os.path.join(os.getcwd(), 'qr_scan_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(self.scan_stats, f, indent=4)
        
        print("\n--- QR Code Scan Summary ---")
        print(f"Total Images Scanned: {self.scan_stats['total_images_scanned']}")
        print(f"Images with QR Codes: {self.scan_stats['images_with_qr_codes']}")
        print(f"Images without QR Codes: {self.scan_stats['images_without_qr_codes']}")
        print(f"Total QR Codes Detected: {self.scan_stats['total_qr_codes_detected']}")
        print(f"Detection Percentage: {self.scan_stats.get('detection_percentage', 0):.2f}%")
        print(f"Total Scan Time: {self.scan_stats['total_scan_time']:.2f} seconds")
        print(f"Average Image Processing Time: {self.scan_stats.get('average_image_processing_time', 0):.4f} seconds")

    def process_images_in_directory(self, directory):
        """
        Process all images in a given directory
        
        Args:
            directory (str): Path to the directory containing images
        """
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(supported_extensions):
                file_path = os.path.join(directory, filename)
                self.logger.info(f"Processing image: {filename}")
                
                qr_codes, processing_time = self.detect_qr_codes(file_path)
                
                if qr_codes:
                    for qr in qr_codes:
                        print(f"QR Code Data: {qr['data']}")
                    
                    self.save_detection_results(qr_codes, file_path)

def main():
    detector = RobustQRCodeDetector(debug=False)
    
    start_total_time = time.time()
    current_directory = os.getcwd()
    detector.process_images_in_directory(current_directory)
    
    detector.save_scan_summary()
    
    total_execution_time = time.time() - start_total_time
    print(f"\nTotal Execution Time: {total_execution_time:.2f} seconds")

if __name__ == "__main__":
    main()