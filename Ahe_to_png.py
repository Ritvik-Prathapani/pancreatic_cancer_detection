import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_dicom_image(dicom_path):
    """
    Load a DICOM image and convert it to a numpy array
    
    Args:
        dicom_path (str): Path to the DICOM file
    
    Returns:
        numpy.ndarray: Pixel data of the DICOM image
    """
    # Read the DICOM file
    dicom_data = pydicom.dcmread('1-001.dcm')
    
    # Convert to numpy array
    image_array = dicom_data.pixel_array
    
    return image_array

def apply_clahe(image, clip_limit=8.0, tile_grid_size=(8,8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Args:
        image (numpy.ndarray): Input image
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (tuple): Size of grid for histogram equalization
    
    Returns:
        numpy.ndarray: Image after adaptive histogram equalization
    """
    # Normalize image to 8-bit unsigned integer
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, 
        tileGridSize=tile_grid_size
    )
    
    # Apply CLAHE
    equalized_image = clahe.apply(image_normalized)
    
    return equalized_image

def display_and_save_comparison(original, equalized, output_path='equalized_image.png'):
    """
    Display original and equalized images side by side and save the equalized image
    
    Args:
        original (numpy.ndarray): Original image
        equalized (numpy.ndarray): Equalized image
        output_path (str): Path to save the equalized image
    """
    # Save the equalized image
    cv2.imwrite(output_path, equalized)
    print(f"Equalized image saved to {output_path}")
    
    # Display images
    plt.figure(figsize=(12,6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    # Equalized image
    plt.subplot(1, 2, 2)
    plt.title('Adaptive Histogram Equalized Image')
    plt.imshow(equalized, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main(dicom_path='1-001.dcm', output_path='equalized_image.png'):
    """
    Main function to process DICOM image with adaptive histogram equalization
    
    Args:
        dicom_path (str): Path to the DICOM file
        output_path (str): Path to save the equalized image
    """
    # Load DICOM image
    dicom_image = load_dicom_image(dicom_path)
    
    # Apply adaptive histogram equalization
    equalized_image = apply_clahe(dicom_image)
    
    # Display and save results
    display_and_save_comparison(dicom_image, equalized_image, output_path)

# Call the main function
main()