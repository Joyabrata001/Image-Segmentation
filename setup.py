from setuptools import setup, find_packages

setup(
    name="edge_detectors",
    version="1.0.0",
    description="A Python package implementing various edge detectors including Roberts, Sobel, Laplacian of Gaussian, Marr-Hildreth, and Canny.",
    author="Joy",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/edge-detectors",
    packages=find_packages(include=["edge_detectors"]),
    install_requires=[
        "numpy",
        "Pillow",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "edge-detect=main:main",  # Ensure main.py has a main function
        ],
    },
)
