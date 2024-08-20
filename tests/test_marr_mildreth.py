import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from edge_detectors import marr_mildreth_detect_edges

def test_marr_mildreth(image_path, sigma_values, thresholds, output_dir):
    """Test Marr-Hildreth edge detector with various sigma and threshold values."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = Image.open(image_path)

    for sigma in sigma_values:
        for threshold in thresholds:
            edge_image = marr_mildreth_detect_edges(
                image, threshold=threshold, sigma=sigma
            )
            output_path = os.path.join(
                output_dir, f"marr_mildreth_sigma_{sigma}_threshold_{threshold}.png"
            )
            edge_image.save(output_path)
            print(f"Saved Marr-Hildreth result to: {output_path}")


# Example parameters for testing
marr_mildreth_sigma_values = [0.5, 1.0, 1.5]
marr_mildreth_thresholds = [10, 20, 30]

# Paths
image_path = "data/grayscale-images/baboon.png"
marr_mildreth_output_dir = "results/marr_mildreth_comparison"

# Run tests
test_marr_mildreth(
    image_path,
    marr_mildreth_sigma_values,
    marr_mildreth_thresholds,
    marr_mildreth_output_dir,
)
