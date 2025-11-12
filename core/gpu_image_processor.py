"""GPU-accelerated image processing utilities for Universal Game Bot."""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
from loguru import logger

from .gpu_manager import get_gpu_manager


class GPUImageProcessor:
    """GPU-accelerated image processing for snippet detection and OCR."""

    def __init__(self):
        self.gpu_manager = get_gpu_manager()
        self._template_cache: Dict[str, np.ndarray] = {}

    def preprocess_for_ocr(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """Preprocess image for better OCR results using GPU acceleration when available.
        
        Args:
            image: Input image (BGR or grayscale)
            enhance: Whether to apply enhancement filters
            
        Returns:
            Preprocessed image optimized for OCR
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = self.gpu_manager.gpu_cvt_color(image, cv2.COLOR_BGR2GRAY)
            if gray is None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if not enhance:
            return gray

        # GPU-accelerated preprocessing pipeline
        processed = gray

        try:
            # 1. Noise reduction with Gaussian blur
            blurred = self.gpu_manager.gpu_gaussian_blur(processed, (3, 3), 0.5)
            if blurred is not None:
                processed = blurred
            else:
                processed = cv2.GaussianBlur(processed, (3, 3), 0.5)

            # 2. Contrast enhancement using CLAHE (CPU-only for now)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)

            # 3. Adaptive thresholding for better text clarity
            binary = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # 4. Morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        except Exception as e:
            logger.debug("GPU preprocessing failed, using original image: {}", e)
            processed = gray

        return processed

    def template_match_gpu(self, image: np.ndarray, template: np.ndarray, 
                          threshold: float = 0.8, mask: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """Perform template matching with GPU acceleration.
        
        Args:
            image: Source image (grayscale)
            template: Template to match
            threshold: Matching threshold
            mask: Optional mask for template
            
        Returns:
            Dictionary with match information or None if no match found
        """
        # Ensure images are optimized for GPU
        image = self.gpu_manager.optimize_image_for_gpu(image)
        template = self.gpu_manager.optimize_image_for_gpu(template)

        # Try GPU template matching first
        result = self.gpu_manager.gpu_template_matching(image, template, cv2.TM_CCOEFF_NORMED, mask)
        
        if result is None:
            # Fallback to CPU template matching
            if mask is not None:
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            else:
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Find the best match
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            h, w = template.shape[:2]
            return {
                "x": int(max_loc[0]),
                "y": int(max_loc[1]),
                "w": w,
                "h": h,
                "score": float(max_val),
                "method": "template_gpu" if result is not None else "template_cpu"
            }
        
        return None

    def multiscale_template_match_gpu(self, image: np.ndarray, template: np.ndarray,
                                     threshold: float = 0.8, 
                                     scales: List[float] = None,
                                     mask: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """Perform multiscale template matching with GPU acceleration.
        
        Args:
            image: Source image
            template: Template to match
            threshold: Matching threshold
            scales: List of scale factors to try
            mask: Optional mask for template
            
        Returns:
            Best match information or None
        """
        if scales is None:
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]

        best_match = None
        best_score = 0.0

        for scale in scales:
            # Resize template with GPU if available
            new_w = int(template.shape[1] * scale)
            new_h = int(template.shape[0] * scale)
            
            if new_w < 5 or new_h < 5 or new_w > image.shape[1] or new_h > image.shape[0]:
                continue

            scaled_template = self.gpu_manager.gpu_resize(template, (new_w, new_h))
            if scaled_template is None:
                scaled_template = cv2.resize(template, (new_w, new_h))

            # Scale mask if provided
            scaled_mask = None
            if mask is not None:
                scaled_mask = self.gpu_manager.gpu_resize(mask, (new_w, new_h))
                if scaled_mask is None:
                    scaled_mask = cv2.resize(mask, (new_w, new_h))

            # Perform template matching
            match = self.template_match_gpu(image, scaled_template, threshold, scaled_mask)
            
            if match and match["score"] > best_score:
                best_score = match["score"]
                best_match = match
                best_match["scale"] = scale
                best_match["method"] = "multiscale_gpu"

        return best_match

    def enhance_for_feature_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better feature detection (ORB, SIFT, etc.).
        
        Args:
            image: Input grayscale image
            
        Returns:
            Enhanced image for feature detection
        """
        enhanced = image.copy()

        try:
            # GPU-accelerated enhancement pipeline
            # 1. Slight blur to reduce noise
            blurred = self.gpu_manager.gpu_gaussian_blur(enhanced, (3, 3), 0.8)
            if blurred is not None:
                enhanced = blurred
            else:
                enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.8)

            # 2. Histogram equalization for better contrast
            enhanced = cv2.equalizeHist(enhanced)

            # 3. Bilateral filter for edge preservation (CPU only)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        except Exception as e:
            logger.debug("Feature detection enhancement failed: {}", e)

        return enhanced

    def batch_template_matching(self, image: np.ndarray, templates: Dict[str, np.ndarray],
                               thresholds: Dict[str, float], 
                               masks: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Optional[Dict[str, Any]]]:
        """Perform batch template matching for multiple templates.
        
        Args:
            image: Source image
            templates: Dictionary mapping template names to template images
            thresholds: Dictionary mapping template names to thresholds
            masks: Optional dictionary mapping template names to masks
            
        Returns:
            Dictionary mapping template names to match results
        """
        results = {}
        
        # Optimize source image for GPU once
        optimized_image = self.gpu_manager.optimize_image_for_gpu(image)
        
        for name, template in templates.items():
            threshold = thresholds.get(name, 0.8)
            mask = masks.get(name) if masks else None
            
            try:
                result = self.template_match_gpu(optimized_image, template, threshold, mask)
                results[name] = result
                
                if result:
                    logger.debug("Template '{}' matched with score {:.3f}", name, result["score"])
                    
            except Exception as e:
                logger.warning("Template matching failed for '{}': {}", name, e)
                results[name] = None
        
        return results

    def create_snippet_mask(self, image: np.ndarray, method: str = "adaptive") -> Optional[np.ndarray]:
        """Create a mask for snippet detection using GPU acceleration.
        
        Args:
            image: Input image (BGR)
            method: Mask creation method ("adaptive", "canny", "combined")
            
        Returns:
            Binary mask or None if creation failed
        """
        try:
            # Convert to grayscale
            gray = self.gpu_manager.gpu_cvt_color(image, cv2.COLOR_BGR2GRAY)
            if gray is None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if method == "adaptive":
                # Adaptive threshold for mask creation
                mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
                
            elif method == "canny":
                # Edge-based mask using Canny
                blurred = self.gpu_manager.gpu_gaussian_blur(gray, (5, 5), 1.0)
                if blurred is None:
                    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
                mask = cv2.Canny(blurred, 50, 150)
                
            elif method == "combined":
                # Combined adaptive + edge mask
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
                
                blurred = self.gpu_manager.gpu_gaussian_blur(gray, (5, 5), 1.0)
                if blurred is None:
                    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
                edges = cv2.Canny(blurred, 50, 150)
                
                # Combine masks
                mask = cv2.bitwise_or(adaptive, edges)
                
            else:
                logger.warning("Unknown mask method: {}", method)
                return None

            # Post-processing to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)

            return mask

        except Exception as e:
            logger.error("Mask creation failed: {}", e)
            return None

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get GPU processing statistics and capabilities."""
        stats = self.gpu_manager.get_acceleration_info()
        stats.update({
            "template_cache_size": len(self._template_cache),
        })
        return stats

    def clear_caches(self):
        """Clear all processing caches."""
        self._template_cache.clear()
        self.gpu_manager.clear_gpu_cache()
        logger.debug("Image processing caches cleared")


# Global instance
_gpu_processor: Optional[GPUImageProcessor] = None


def get_gpu_image_processor() -> GPUImageProcessor:
    """Get the global GPU image processor instance."""
    global _gpu_processor
    if _gpu_processor is None:
        _gpu_processor = GPUImageProcessor()
    return _gpu_processor