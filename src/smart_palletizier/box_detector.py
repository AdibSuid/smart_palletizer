"""
2D Box Detection Module

This module provides functionality for detecting boxes in color and depth images
using classical computer vision techniques such as contour detection, edge detection,
and morphological operations.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class BoxDetector:
    """
    A class for detecting boxes in 2D images using classical computer vision methods.
    
    This detector uses a combination of color segmentation, edge detection, and
    contour analysis to identify rectangular objects (boxes) in images.
    
    Attributes:
        min_area (int): Minimum contour area to consider as a valid box
        max_area (int): Maximum contour area to consider as a valid box
        aspect_ratio_range (Tuple[float, float]): Valid aspect ratio range for boxes
    """
    
    def __init__(self, min_area: int = 1000, max_area: int = 500000, 
                 aspect_ratio_range: Tuple[float, float] = (0.3, 3.0)):
        """
        Initialize the BoxDetector.
        
        Args:
            min_area: Minimum contour area in pixels
            max_area: Maximum contour area in pixels
            aspect_ratio_range: Tuple of (min_ratio, max_ratio) for valid box aspect ratios
        """
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for better box detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def detect_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the grayscale image using Canny edge detection.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Binary edge map
        """
        # Auto-compute Canny thresholds based on image median
        # This adaptive approach works better than fixed thresholds
        v = np.median(gray_image)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        
        edges = cv2.Canny(gray_image, lower, upper)
        
        # Dilate edges to connect nearby contours
        # Helps merge broken edges from low contrast areas
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def find_box_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        """
        Find contours that likely represent boxes.
        
        Args:
            edges: Binary edge map
            
        Returns:
            List of contours representing potential boxes
        """
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        box_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if approximation has 4 corners (rectangle)
            if len(approx) >= 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Check aspect ratio
                if (self.aspect_ratio_range[0] <= aspect_ratio <= 
                    self.aspect_ratio_range[1]):
                    box_contours.append(approx)
        
        return box_contours
    
    def detect_boxes_from_mask(self, mask: np.ndarray) -> List[Dict]:
        """
        Detect boxes from a binary mask image.
        
        Args:
            mask: Binary mask image where boxes are white (255)
            
        Returns:
            List of dictionaries containing box information
        """
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get rotated rectangle for better orientation estimation
            rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rect)
            box_points = np.intp(box_points)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            boxes.append({
                'id': i,
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'contour': contour,
                'rotated_rect': rect,
                'box_points': box_points,
                'area': area
            })
        
        return boxes
    
    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Detect boxes in the input image.
        
        Args:
            image: Input BGR image
            mask: Optional binary mask for guided detection
            
        Returns:
            List of detected boxes with their properties
        """
        if mask is not None:
            # Use mask-based detection if mask is provided
            return self.detect_boxes_from_mask(mask)
        
        # Preprocess image
        gray = self.preprocess_image(image)
        
        # Detect edges
        edges = self.detect_edges(gray)
        
        # Find box contours
        contours = self.find_box_contours(edges)
        
        # Extract box information
        boxes = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get rotated rectangle
            rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rect)
            box_points = np.intp(box_points)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            boxes.append({
                'id': i,
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'contour': contour,
                'rotated_rect': rect,
                'box_points': box_points,
                'area': cv2.contourArea(contour)
            })
        
        return boxes
    
    def draw_detections(self, image: np.ndarray, boxes: List[Dict],
                       box_type: str = "box") -> np.ndarray:
        """
        Draw detected boxes on the image with semi-transparent overlay.

        Args:
            image: Input image
            boxes: List of detected boxes
            box_type: Type label for the boxes

        Returns:
            Image with drawn detections
        """
        result = image.copy()
        overlay = image.copy()

        for i, box in enumerate(boxes):
            # Create semi-transparent filled overlay for rotated rectangle
            cv2.fillPoly(overlay, [box['box_points']], (0, 0, 255))  # Red fill

            # Draw rotated rectangle outline
            cv2.drawContours(result, [box['box_points']], 0, (0, 255, 0), 2)  # Green outline

            # Draw center
            cx, cy = box['center']
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)

            # Add label with confidence (100% since using masks)
            x, y, w, h = box['bbox']
            label = f"{box_type} 100%"

            # Add background rectangle for text
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (x, y - text_h - baseline - 5),
                         (x + text_w, y), (0, 0, 0), -1)

            # Draw text
            cv2.putText(result, label, (x, y - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Blend overlay with original image (30% transparency)
        result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)

        return result
