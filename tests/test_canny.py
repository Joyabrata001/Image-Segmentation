import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from edge_detectors import canny_detect_edges


def test_canny(image_path, sigma_values, low_thresholds, high_thresholds, output_dir):
    """Test Canny edge detector with various sigma, low, and high threshold values."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = Image.open(image_path)

    for sigma in sigma_values:
        for low_threshold in low_thresholds:
            for high_threshold in high_thresholds:
                edge_image = canny_detect_edges(
                    image,
                    sigma=sigma,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                )
                output_path = os.path.join(
                    output_dir,
                    f"canny_sigma_{sigma}_low_{low_threshold}_high_{high_threshold}.png",
                )
                edge_image.save(output_path)
                # print(f"Saved Canny result to: {output_path}")


# Example parameters for testing
canny_sigma_values = [0.5, 1.0, 1.5]
canny_low_thresholds = [50, 100, 150]
canny_high_thresholds = [100, 150, 200]

# Paths
image_path = "data/grayscale-images/baboon.png"
canny_output_dir = "results/canny_comparison"

# Run tests
test_canny(
    image_path,
    canny_sigma_values,
    canny_low_thresholds,
    canny_high_thresholds,
    canny_output_dir,
)
