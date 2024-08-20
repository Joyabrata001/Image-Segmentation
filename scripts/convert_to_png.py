import os
from PIL import Image

def convert_images_to_png(directory_path):
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"The directory {directory_path} does not exist.")

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check if the file is a PNG
        if file_path.lower().endswith('.png'):
            # print(f"Skipping already PNG file: {file_path}")
            continue

        # Open the image file
        try:
            with Image.open(file_path) as img:
                # Define the new file path with .png extension
                new_file_path = os.path.splitext(file_path)[0] + ".png"

                # Convert and save the image as PNG
                img.save(new_file_path, format="PNG")

                # Remove the original file
                os.remove(file_path)

                print(f"Converted and replaced: {file_path} -> {new_file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Example usage
directory_path = "data/grayscale-images"
convert_images_to_png(directory_path)