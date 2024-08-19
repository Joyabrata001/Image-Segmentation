import numpy as np
from PIL import Image
from scipy.ndimage import convolve

"""
The kernels Gx and Gy can be thought of as a differential operation in the "input_image" array in the directions x and y 
respectively. These kernels are represented by the following matrices:
      _         _                   _          _
     |           |                 |            |
     | 1.0   0.0 |                 |  0.0   1.0 |
Gx = | 0.0  -1.0 |    and     Gy = | -1.0   0.0 |
     |_         _|                 |_          _|
"""

def detect_edges(image: Image.Image) -> Image.Image:
    # Convert image to grayscale if it is not already
    image = image.convert("L")

    # Convert the grayscale image to a numpy array of floats for processing
    image_array = np.array(image, dtype=float)

    # Define Roberts cross kernels for edge detection
    # The vertical kernel detects changes in the vertical direction
    roberts_cross_v = np.array([[1, 0], [0, -1]], dtype=float)

    # The horizontal kernel detects changes in the horizontal direction
    roberts_cross_h = np.array([[0, 1], [-1, 0]], dtype=float)

    # Apply convolution of the image with the vertical kernel
    # This highlights edges oriented vertically
    grad_x = convolve(image_array, roberts_cross_v)

    # Apply convolution of the image with the horizontal kernel
    # This highlights edges oriented horizontally
    grad_y = convolve(image_array, roberts_cross_h)

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
