"""
SDX - Computer Vision Utilities Module
Custom utilities for the Computer Vision course at Insper
"""

import numpy as np
import cv2 as cv
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image


def cv_imread(filename: str, flags: int = cv.IMREAD_COLOR, as_rgb=True) -> np.ndarray:
    """
    Read an image file using OpenCV with proper error handling
    
    Args:
        filename: Path to the image file
        flags: OpenCV imread flags (default: cv.IMREAD_COLOR)
    
    Returns:
        Image as numpy array in RGB format (or BGR if as_rgb==False)
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be read
    """
    img = cv.imread(filename, flags)
    
    if img is None:
        raise FileNotFoundError(f"Could not read image: {filename}")
    
    # Convert BGR to RGB for color images
    if as_rgb:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    return img


def cv_grayread(filename: str) -> np.ndarray:
    """
    Read an image file as grayscale
    
    Args:
        filename: Path to the image file
    
    Returns:
        Grayscale image as numpy array
    """
    return cv_imread(filename, cv.IMREAD_GRAYSCALE)


def cv_imshow(img: np.ndarray, title: str = "", figsize: Tuple[int, int] = (10, 10)) -> None:
    """
    Display an image using matplotlib (Colab-friendly)
    
    Args:
        img: Image array (can be grayscale or RGB)
        title: Optional title for the image
        figsize: Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)
    
    if len(img.shape) == 2:
        # Grayscale image
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        # Color image (already in RGB from cv_imread)
        plt.imshow(img)
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def cv_imwrite(filename: str, img: np.ndarray) -> bool:
    """
    Save an image to file
    
    Args:
        filename: Path where to save the image
        img: Image array to save
    
    Returns:
        True if successful, False otherwise
    """
    # Convert RGB to BGR if color image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    
    return cv.imwrite(filename, img)


def bgr2gray(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR or RGB image to grayscale
    
    Args:
        img: Color image array
    
    Returns:
        Grayscale image array
    """
    if len(img.shape) == 2:
        # Already grayscale
        return img
    
    # Assume RGB (since cv_imread converts to RGB)
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def rgb2gray(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale (alias for bgr2gray)
    
    Args:
        img: RGB image array
    
    Returns:
        Grayscale image array
    """
    return bgr2gray(img)


def show_multiple(images: list, titles: list = None, figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Display multiple images side by side
    
    Args:
        images: List of image arrays
        titles: Optional list of titles for each image
        figsize: Figure size (width, height) in inches
    """
    n = len(images)
    
    if titles is None:
        titles = [f"Image {i+1}" for i in range(n)]
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range
    
    Args:
        img: Input image
    
    Returns:
        Normalized image as uint8
    """
    img_normalized = img.astype(float)
    img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())
    img_normalized = (img_normalized * 255).astype(np.uint8)
    return img_normalized


def get_image_stats(img: np.ndarray) -> dict:
    """
    Get basic statistics about an image
    
    Args:
        img: Input image
    
    Returns:
        Dictionary with image statistics
    """
    return {
        'shape': img.shape,
        'dtype': img.dtype,
        'min': img.min(),
        'max': img.max(),
        'mean': img.mean(),
        'std': img.std()
    }


def print_image_info(img: np.ndarray, name: str = "Image") -> None:
    """
    Print detailed information about an image
    
    Args:
        img: Input image
        name: Name to display for the image
    """
    stats = get_image_stats(img)
    print(f"{name} Information:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Data type: {stats['dtype']}")
    print(f"  Min value: {stats['min']}")
    print(f"  Max value: {stats['max']}")
    print(f"  Mean value: {stats['mean']:.2f}")
    print(f"  Std deviation: {stats['std']:.2f}")


# Additional utilities that might be used in the course

def create_histogram(img: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create histogram for grayscale image
    
    Args:
        img: Grayscale image
        bins: Number of bins
    
    Returns:
        Tuple of (histogram, bin_edges)
    """
    hist, edges = np.histogram(img.flatten(), bins=bins, range=(0, 256))
    return hist, edges


def show_histogram(img: np.ndarray, bins: int = 256, title: str = "Histogram") -> None:
    """
    Display histogram of an image
    
    Args:
        img: Grayscale image
        bins: Number of bins
        title: Title for the plot
    """
    plt.figure(figsize=(10, 4))
    plt.hist(img.flatten(), bins=bins, range=(0, 256), color='gray')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def cv_gridshow(images: np.ndarray, start: int = 0, stop: int = 9, labels: Optional[np.ndarray] = None, 
                cols: int = 5, figsize: Tuple[int, int] = None) -> None:
    """
    Display a grid of images
    
    Args:
        images: Array of images
        start: Start index (default: 0)
        stop: Stop index (default: 9)
        labels: Optional array of labels to display
        cols: Number of columns in grid (default: 5)
        figsize: Figure size (auto-calculated if None)
    """
    # Get subset of images
    img_subset = images[start:stop+1]
    n_images = len(img_subset)
    
    # Calculate grid dimensions
    rows = (n_images + cols - 1) // cols
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (cols * 2, rows * 2)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Display images
    for idx in range(n_images):
        ax = axes[idx]
        img = img_subset[idx]
        
        # Display image
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(img)
        
        # Add label if provided
        if labels is not None:
            label = labels[start + idx]
            ax.set_title(f'{label}', fontsize=10)
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion(model, test_images: np.ndarray, test_labels: np.ndarray) -> None:
    """
    Plot confusion matrix for a PyTorch model
    
    Args:
        model: PyTorch model
        test_images: Test images (numpy array)
        test_labels: Test labels (numpy array)
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. This function requires PyTorch.")
        return
    
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("Error: scikit-learn not installed. Please install it: pip install scikit-learn")
        return
    
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
    # Convert to PyTorch tensors
    device = next(model.parameters()).device
    
    # Prepare data
    if isinstance(test_images, np.ndarray):
        # Normalize to [0, 1] if needed
        if test_images.max() > 1.0:
            test_images = test_images / 255.0
        test_images = torch.FloatTensor(test_images).unsqueeze(1)  # Add channel dimension
    
    # Get predictions
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 256
        for i in range(0, len(test_images), batch_size):
            batch = test_images[i:i+batch_size].to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(test_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if has_seaborn:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
    else:
        # Basic matplotlib version
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        
        plt.xticks(range(10), range(10))
        plt.yticks(range(10), range(10))
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


# Export all public functions
__all__ = [
    'cv_imread',
    'cv_grayread',
    'cv_imshow',
    'cv_imwrite',
    'bgr2gray',
    'rgb2gray',
    'show_multiple',
    'normalize_image',
    'get_image_stats',
    'print_image_info',
    'create_histogram',
    'show_histogram',
    'cv_gridshow',
    'plot_confusion'
]
