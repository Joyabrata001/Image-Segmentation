import os
from PIL import Image
from edge_detectors import (
    roberts_detect_edges,
    sobel_detect_edges,
    prewitt_detect_edges,
    marr_mildreth_detect_edges,
    canny_detect_edges,
)


def process_images(input_folder: str, output_folder: str):

    # Create output directories if they don't exist
    roberts_folder = os.path.join(output_folder, "roberts")
    sobel_folder = os.path.join(output_folder, "sobel")
    prewitt_folder = os.path.join(output_folder, "prewitt")
    marr_mildreth_folder = os.path.join(output_folder, "marr_mildreth")
    canny_folder = os.path.join(output_folder, "canny")

    os.makedirs(roberts_folder, exist_ok=True)
    os.makedirs(sobel_folder, exist_ok=True)
    os.makedirs(prewitt_folder, exist_ok=True)
    os.makedirs(marr_mildreth_folder, exist_ok=True)
    os.makedirs(canny_folder, exist_ok=True)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        # Construct the full path to the image file
        file_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        if os.path.isfile(file_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
        ):
            try:
                # Load the image
                image = Image.open(file_path)

                # 1. Process the image with the Roberts operator
                edges_roberts = roberts_detect_edges(image)
                roberts_output_path = os.path.join(roberts_folder, filename)
                edges_roberts.save(roberts_output_path)

                # 2. Process the image with the Prewitt operator
                edges_prewitt = prewitt_detect_edges(image)
                prewitt_output_path = os.path.join(prewitt_folder, filename)
                edges_prewitt.save(prewitt_output_path)

                # 3. Process the image with the Sobel operator
                edges_sobel = sobel_detect_edges(image)
                sobel_output_path = os.path.join(sobel_folder, filename)
                edges_sobel.save(sobel_output_path)

                # 4. Process the image with the Marr_Mildreth operator
                edges_marr_mildreth = marr_mildreth_detect_edges(
                    image, threshold=10.0, sigma=1.0
                )
                marr_mildreth_output_path = os.path.join(marr_mildreth_folder, filename)
                edges_marr_mildreth.save(marr_mildreth_output_path)

                # 5. Process the image with the Canny operator
                edges_canny = canny_detect_edges(
                    image,
                    sigma=1.0,
                    low_threshold=50,
                    high_threshold=150,
                )
                canny_output_path = os.path.join(canny_folder, filename)
                edges_canny.save(canny_output_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    input_folder = "data/grayscale-images"
    output_folder = "results/edge-detectors"
    process_images(input_folder, output_folder)
