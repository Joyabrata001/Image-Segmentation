import numpy as np
from PIL import Image
from scipy.ndimage import convolve

"""
The Sobel operators are used for edge detection by calculating the gradient of the image intensity function. 
The kernels Gx and Gy are represented by the following matrices:

      _               _                   _                _
     |                 |                 |                  |
     | -1.0   0.0  1.0 |                 |  1.0   1.0   1.0 |
Gx = | -1.0   0.0  1.0 |    and     Gy = |  0.0   0.0   0.0 |
     | -1.0   0.0  1.0 |                 | -1.0  -1.0  -1.0 |
     |_               _|                 |_                _|
"""


def detect_edges(image: Image.Image) -> Image.Image:
    # Convert image to grayscale if it is not already
    image = image.convert("L")

    # Convert the grayscale image to a numpy array of floats for processing
    image_array = np.array(image, dtype=float)

    # Define Sobel kernels for edge detection
    # The Sobel X kernel detects changes in the horizontal direction
    sobel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)

    # The Sobel Y kernel detects changes in the vertical direction
    sobel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=float)

    # Apply convolution of the image with the Sobel X kernel
    # This highlights edges oriented vertically
    grad_x = convolve(image_array, sobel_x)

    # Apply convolution of the image with the Sobel Y kernel
    # This highlights edges oriented horizontally
    grad_y = convolve(image_array, sobel_y)

    # Compute the gradient magnitude from the x and y gradients
    # This combines the vertical and horizontal edge information
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize the gradient magnitude to the range [0, 255]
    # This step scales the image to use the full 8-bit range for better visualization
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(
        np.uint8
    )

    # Convert the numpy array with gradient magnitudes back to a PIL image
    gradient_image = Image.fromarray(gradient_magnitude)

    return gradient_image
