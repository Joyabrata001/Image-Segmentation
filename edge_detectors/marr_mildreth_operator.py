import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace


def detect_edges(image: Image.Image, threshold: float = None, sigma: float = 1.0) -> Image.Image:
    # Convert image to grayscale if it is not already
    image = image.convert("L")

    # Convert the grayscale image to a numpy array of floats for processing
    image_array = np.array(image, dtype=float)

    # Apply Gaussian filter to smooth the image
    smoothed_image = gaussian_filter(image_array, sigma=sigma)

    # Apply Laplacian operator to the smoothed image
    laplacian_image = laplace(smoothed_image)

    # Detect zero crossings in the Laplacian image
    zero_crossings = np.zeros_like(laplacian_image, dtype=np.uint8)

    # Check for zero-crossings by comparing signs of adjacent pixels
    zero_crossings[1:-1, 1:-1] = (
        ((laplacian_image[1:-1, 1:-1] * laplacian_image[:-2, 1:-1]) < 0)
        | ((laplacian_image[1:-1, 1:-1] * laplacian_image[2:, 1:-1]) < 0)
        | ((laplacian_image[1:-1, 1:-1] * laplacian_image[1:-1, :-2]) < 0)
        | ((laplacian_image[1:-1, 1:-1] * laplacian_image[1:-1, 2:]) < 0)
    )

    # If a threshold is provided, apply it to the zero crossings
    if threshold is not None:
        # Calculate gradient magnitude
        gradient_magnitude = np.abs(laplacian_image[1:-1, 1:-1])

        # Apply the threshold: keep only the zero-crossings where the gradient magnitude is above the threshold
        zero_crossings[1:-1, 1:-1] &= gradient_magnitude >= threshold

    # Convert the boolean array to 8-bit for proper image representation
    zero_crossings = (zero_crossings * 255).astype(np.uint8)

    # Convert the zero crossings array back to a PIL image
    edge_image = Image.fromarray(zero_crossings)

    return edge_image
