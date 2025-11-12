"""GPU acceleration manager for Universal Game Bot.

This module provides GPU acceleration for OCR and image processing operations
when CUDA-compatible hardware is available.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# GPU acceleration imports
torch = None
cuda_available = False
cupy_available = False
cuda_cv2_available = False

try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info("PyTorch CUDA support detected: {} devices", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info("GPU {}: {} (Memory: {:.1f}GB)", i, props.name, props.total_memory / 1024**3)
except ImportError:
    logger.warning("PyTorch not available")

try:
    import cupy as cp
    cupy_available = cp.cuda.is_available()
    if cupy_available:
        logger.info("CuPy CUDA support detected")
except ImportError:
    logger.debug("CuPy not available (optional)")

# Check for OpenCV CUDA support
try:
    cuda_cv2_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if cuda_cv2_available:
        logger.info("OpenCV CUDA support detected: {} devices", cv2.cuda.getCudaEnabledDeviceCount())
except AttributeError:
    logger.debug("OpenCV compiled without CUDA support")
    cuda_cv2_available = False


class GPUManager:
    """Manages GPU acceleration for image processing and OCR operations."""

    def __init__(self, prefer_gpu: bool = True, device_id: int = 0):
        """Initialize GPU manager.
        
        Args:
            prefer_gpu: Whether to prefer GPU over CPU when available
            device_id: CUDA device ID to use (default: 0)
        """
        self.prefer_gpu = prefer_gpu
        self.device_id = device_id
        self.cuda_available = cuda_available and prefer_gpu
        self.cupy_available = cupy_available and prefer_gpu
        self.cuda_cv2_available = cuda_cv2_available and prefer_gpu
        
        # Initialize CUDA context if available
        if self.cuda_available and torch is not None:
            self.device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(device_id)
        else:
            self.device = torch.device('cpu') if torch is not None else None
            
        # GPU memory pools for reusing allocations
        self._gpu_image_cache: Dict[str, Any] = {}
        
        logger.info("GPU Manager initialized - CUDA: {}, CuPy: {}, OpenCV-CUDA: {}", 
                   self.cuda_available, self.cupy_available, self.cuda_cv2_available)

    def get_acceleration_info(self) -> Dict[str, Any]:
        """Get information about available GPU acceleration."""
        info = {
            "cuda_available": self.cuda_available,
            "cupy_available": self.cupy_available,
            "opencv_cuda_available": self.cuda_cv2_available,
            "device": str(self.device) if self.device else "cpu",
            "gpu_memory_gb": 0.0,
            "gpu_name": "N/A"
        }
        
        if self.cuda_available and torch is not None:
            try:
                props = torch.cuda.get_device_properties(self.device_id)
                info["gpu_memory_gb"] = props.total_memory / (1024**3)
                info["gpu_name"] = props.name
            except Exception as e:
                logger.warning("Failed to get GPU properties: {}", e)
                
        return info

    def create_easyocr_reader(self, languages: List[str] = None, **kwargs) -> Any:
        """Create EasyOCR reader with GPU acceleration if available.
        
        Args:
            languages: List of language codes (default: ['en'])
            **kwargs: Additional arguments for EasyOCR Reader
            
        Returns:
            EasyOCR Reader instance
        """
        if languages is None:
            languages = ['en']
            
        try:
            import easyocr
            
            # Configure EasyOCR to use GPU if available
            gpu_config = self.cuda_available and torch is not None
            
            # Create reader with GPU support
            reader = easyocr.Reader(
                languages, 
                gpu=gpu_config,
                **kwargs
            )
            
            logger.info("EasyOCR Reader created with GPU support: {}", gpu_config)
            return reader
            
        except ImportError as e:
            logger.error("EasyOCR not available: {}", e)
            raise
        except Exception as e:
            logger.warning("Failed to create EasyOCR with GPU, falling back to CPU: {}", e)
            # Fallback to CPU
            import easyocr
            return easyocr.Reader(languages, gpu=False, **kwargs)

    def gpu_template_matching(self, image: np.ndarray, template: np.ndarray, 
                            method: int = cv2.TM_CCOEFF_NORMED,
                            mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Perform template matching using GPU acceleration if available.
        
        Args:
            image: Source image (grayscale)
            template: Template to match
            method: OpenCV template matching method
            mask: Optional mask for template
            
        Returns:
            Result array or None if GPU acceleration failed
        """
        if not self.cuda_cv2_available:
            return None
            
        try:
            # Upload images to GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_template = cv2.cuda_GpuMat()
            gpu_result = cv2.cuda_GpuMat()
            
            gpu_image.upload(image)
            gpu_template.upload(template)
            
            if mask is not None:
                gpu_mask = cv2.cuda_GpuMat()
                gpu_mask.upload(mask)
                cv2.cuda.matchTemplate(gpu_image, gpu_template, gpu_result, method, gpu_mask)
            else:
                cv2.cuda.matchTemplate(gpu_image, gpu_template, gpu_result, method)
            
            # Download result from GPU
            result = gpu_result.download()
            return result
            
        except Exception as e:
            logger.debug("GPU template matching failed: {}", e)
            return None

    def gpu_gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int], 
                         sigma_x: float, sigma_y: float = 0) -> Optional[np.ndarray]:
        """Apply Gaussian blur using GPU acceleration if available."""
        if not self.cuda_cv2_available:
            return None
            
        try:
            gpu_image = cv2.cuda_GpuMat()
            gpu_result = cv2.cuda_GpuMat()
            
            gpu_image.upload(image)
            cv2.cuda.GaussianBlur(gpu_image, gpu_result, kernel_size, sigma_x, sigma_y)
            
            return gpu_result.download()
            
        except Exception as e:
            logger.debug("GPU Gaussian blur failed: {}", e)
            return None

    def gpu_resize(self, image: np.ndarray, size: Tuple[int, int], 
                  interpolation: int = cv2.INTER_LINEAR) -> Optional[np.ndarray]:
        """Resize image using GPU acceleration if available."""
        if not self.cuda_cv2_available:
            return None
            
        try:
            gpu_image = cv2.cuda_GpuMat()
            gpu_result = cv2.cuda_GpuMat()
            
            gpu_image.upload(image)
            cv2.cuda.resize(gpu_image, gpu_result, size, interpolation=interpolation)
            
            return gpu_result.download()
            
        except Exception as e:
            logger.debug("GPU resize failed: {}", e)
            return None

    def gpu_cvt_color(self, image: np.ndarray, code: int) -> Optional[np.ndarray]:
        """Convert color space using GPU acceleration if available."""
        if not self.cuda_cv2_available:
            return None
            
        try:
            gpu_image = cv2.cuda_GpuMat()
            gpu_result = cv2.cuda_GpuMat()
            
            gpu_image.upload(image)
            cv2.cuda.cvtColor(gpu_image, gpu_result, code)
            
            return gpu_result.download()
            
        except Exception as e:
            logger.debug("GPU color conversion failed: {}", e)
            return None

    def gpu_threshold(self, image: np.ndarray, thresh: float, max_val: float, 
                     thresh_type: int) -> Optional[Tuple[float, np.ndarray]]:
        """Apply threshold using GPU acceleration if available."""
        if not self.cuda_cv2_available:
            return None
            
        try:
            gpu_image = cv2.cuda_GpuMat()
            gpu_result = cv2.cuda_GpuMat()
            
            gpu_image.upload(image)
            ret_val = cv2.cuda.threshold(gpu_image, gpu_result, thresh, max_val, thresh_type)
            
            return ret_val, gpu_result.download()
            
        except Exception as e:
            logger.debug("GPU threshold failed: {}", e)
            return None

    def optimize_image_for_gpu(self, image: np.ndarray) -> np.ndarray:
        """Optimize image format for GPU processing."""
        # Ensure image is in contiguous memory layout
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        
        # Convert to appropriate data type for GPU processing
        if image.dtype != np.uint8 and image.dtype != np.float32:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        return image

    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        self._gpu_image_cache.clear()
        
        if self.cuda_available and torch is not None:
            try:
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
            except Exception as e:
                logger.warning("Failed to clear GPU cache: {}", e)


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        # Check environment variable for GPU preference
        prefer_gpu = os.getenv("GAME_BOT_USE_GPU", "1").lower() in ("1", "true", "yes")
        device_id = int(os.getenv("GAME_BOT_GPU_DEVICE", "0"))
        _gpu_manager = GPUManager(prefer_gpu=prefer_gpu, device_id=device_id)
    return _gpu_manager


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    manager = get_gpu_manager()
    return manager.cuda_available or manager.cuda_cv2_available