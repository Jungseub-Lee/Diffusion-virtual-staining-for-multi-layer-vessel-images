from setuptools import setup, find_packages

setup(
    name="depth-aware-vessel-analysis",
    version="1.0.0",
    author="Seungbum Baik",
    description="Depth-aware vessel network analysis for multi-color fluorescence microscopy",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "scipy>=1.7.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.4.0",
    ],
)
