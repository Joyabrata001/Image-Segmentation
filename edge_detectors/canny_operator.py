import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel


def detect_edges(
    image: Image.Image,
    sigma: float = 1.0,
    low_threshold: float = 50,
    high_threshold: float = 150,
) -> Image.Image:
    # Convert image to grayscale if it is not already
    image = image.convert("L")

    # Convert the grayscale image to a numpy array of floats for processing
    image_array = np.array(image, dtype=float)

    # Step 1: Apply Gaussian filter to smooth the image
    smoothed_image = gaussian_filter(image_array, sigma=sigma)

    # Step 2: Calculate gradients using Sobel operator
    gradient_x = sobel(smoothed_image, axis=0)
    gradient_y = sobel(smoothed_image, axis=1)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Step 3: Non-maximum suppression
    suppressed = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    angle = gradient_direction * 180.0 / np.pi  # Convert direction to degrees
    angle[angle < 0] += 180  # Map to [0, 180]

    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            try:
                q = 255
                r = 255
                # Angle 0 degrees
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                # Angle 45 degrees
                elif 22.5 <= angle[i, j] < 67.5:
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                # Angle 90 degrees
                elif 67.5 <= angle[i, j] < 112.5:
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                # Angle 135 degrees
                elif 112.5 <= angle[i, j] < 157.5:
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    suppressed[i, j] = gradient_magnitude[i, j]
                else:
                    suppressed[i, j] = 0

            except IndexError:
                pass

    # Step 4: Double thresholding
    strong_edges = suppressed >= high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed < high_threshold)

    # Step 5: Edge tracking by hysteresis
    output_image = np.zeros_like(suppressed, dtype=np.uint8)
    output_image[strong_edges] = 255

    for i in range(1, output_image.shape[0] - 1):
        for j in range(1, output_image.shape[1] - 1):
            if weak_edges[i, j] and (
                output_image[i + 1, j - 1] == 255
                or output_image[i + 1, j] == 255
                or output_image[i + 1, j + 1] == 255
                or output_image[i, j - 1] == 255
                or output_image[i, j + 1] == 255
                or output_image[i - 1, j - 1] == 255
                or output_image[i - 1, j] == 255
                or output_image[i - 1, j + 1] == 255
            ):
                output_image[i, j] = 255

    # Convert the final edge map back to a PIL image
    edge_image = Image.fromarray(output_image)

    return edge_image
